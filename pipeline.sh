#!/bin/bash

# =============================================================================
# pipeline.sh — Puls-Events RAG Full Pipeline
# =============================================================================
# Runs the complete pipeline end to end:
#   Step 1 — Fetch events from Open Agenda
#   Step 2 — Chunk and embed events
#   Step 3 — Build FAISS index
#   Step 4 — Run unit tests
#   Step 5 — Build/verify the canonical LangChain RAG chain
#   Step 6 — Run RAGAS evaluation (uses the LangChain chain from Step 5)
#   Step 7 — Launch Streamlit chatbot (uses the LangChain chain from Step 5)
#
# LangChain is now the SINGLE implementation of retrieve+generate used by
# both the live demo (chatbot.py) and the RAGAS evaluation (evaluate.py) —
# Step 5 must run before Steps 6 and 7, which is why it is ordered here
# right after the index is built and tests pass.
#
# Usage:
#   bash pipeline.sh                  # smart run — skip completed steps
#   bash pipeline.sh --force          # force rebuild of all steps
#   bash pipeline.sh --skip-run       # run pipeline but do not launch chatbot
#   bash pipeline.sh --skip-test      # skip unit tests
#   bash pipeline.sh --skip-langchain # skip LangChain integration check
#   bash pipeline.sh --skip-eval      # skip RAGAS evaluation
# =============================================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$ROOT"

# --- Parse flags ---
FORCE=""
SKIP_RUN=false
SKIP_TEST=false
SKIP_LANGCHAIN=false
SKIP_EVAL=false

for arg in "$@"; do
    case $arg in
        --force)          FORCE="--force" ;;
        --skip-run)       SKIP_RUN=true ;;
        --skip-test)      SKIP_TEST=true ;;
        --skip-langchain) SKIP_LANGCHAIN=true ;;
        --skip-eval)       SKIP_EVAL=true ;;
    esac
done

# --- Logging helpers ---
log_step() {
    echo ""
    echo "=============================================="
    echo "  $1"
    echo "=============================================="
    echo ""
}

log_ok()   { echo "  ✓ $1"; }
log_skip() { echo "  → $1 (already exists — skipping)"; }
log_error() { echo ""; echo "  ERROR: $1"; echo ""; }

# --- Banner ---
echo ""
echo "=============================================="
echo "  Puls-Events RAG — Full Pipeline"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

[ -n "$FORCE" ]              && echo "  Mode      : FORCE — all steps rebuilt" \
                              || echo "  Mode      : SMART — completed steps skipped"
[ "$SKIP_RUN" = true ]       && echo "  Chatbot   : SKIPPED (--skip-run)"
[ "$SKIP_TEST" = true ]      && echo "  Tests     : SKIPPED (--skip-test)"
[ "$SKIP_LANGCHAIN" = true ] && echo "  LangChain : SKIPPED (--skip-langchain)"
[ "$SKIP_EVAL" = true ]      && echo "  Eval      : SKIPPED (--skip-eval)"
echo ""

# --- Guards ---
if [ ! -f "$ROOT/.env" ]; then
    log_error ".env not found. Run: cp .env.example .env"
    exit 1
fi

if [ ! -f "$ROOT/.venv/bin/python" ]; then
    log_error "Virtual environment not found. Run: poetry install"
    exit 1
fi

PIPELINE_START=$(date +%s)

# =============================================================================
# Step 1 — Fetch Events
# =============================================================================
log_step "Step 1/7 — Fetch Events"

# NOTE: fetch_metadata.json is required. It records the fetch timestamp
# used by 05_test.sh to validate event recency relative to fetch time (not
# relative to "now"). Datasets fetched before this file existed will trigger
# a re-fetch here automatically so the metadata sidecar gets created.
if [ ! -s "$ROOT/data/raw/events_paris.json" ] || \
   [ ! -s "$ROOT/data/raw/fetch_metadata.json" ] || \
   [ ! -s "$ROOT/src/project_11/ingestion/fetch_events.py" ] || \
   [ -n "$FORCE" ]; then
    bash "$SCRIPTS_DIR/01_fetch.sh" --force
    log_ok "Events fetched  → data/raw/events_paris.json"
    log_ok "Fetch metadata  → data/raw/fetch_metadata.json"
else
    log_skip "data/raw/events_paris.json"
    log_skip "data/raw/fetch_metadata.json"
fi

# =============================================================================
# Step 2 — Chunk & Embed
# =============================================================================
log_step "Step 2/7 — Chunk & Embed Events"

CHUNKS_DONE=false
EMBED_DONE=false
SCRIPTS_2_DONE=false
[ -s "$ROOT/data/processed/chunks.json" ]    && CHUNKS_DONE=true
[ -s "$ROOT/data/processed/embeddings.npy" ] && EMBED_DONE=true
[ -s "$ROOT/src/project_11/processing/chunker.py" ] && \
[ -s "$ROOT/src/project_11/processing/embedder.py" ] && SCRIPTS_2_DONE=true

if [ "$CHUNKS_DONE" = true ] && [ "$EMBED_DONE" = true ] && [ "$SCRIPTS_2_DONE" = true ] && [ -z "$FORCE" ]; then
    log_skip "data/processed/chunks.json"
    log_skip "data/processed/embeddings.npy"
else
    bash "$SCRIPTS_DIR/02_embed.sh" $FORCE
    log_ok "Chunks saved     → data/processed/chunks.json"
    log_ok "Embeddings saved → data/processed/embeddings.npy"
fi

# =============================================================================
# Step 3 — Build FAISS Index
# =============================================================================
log_step "Step 3/7 — Build FAISS Index"

INDEX_DONE=false
META_DONE=false
SCRIPT_3_DONE=false
[ -s "$ROOT/data/processed/faiss_index.idx" ] && INDEX_DONE=true
[ -s "$ROOT/data/processed/metadata.json" ]   && META_DONE=true
[ -s "$ROOT/src/project_11/vector_store/build_index.py" ] && SCRIPT_3_DONE=true

if [ "$INDEX_DONE" = true ] && [ "$META_DONE" = true ] && [ "$SCRIPT_3_DONE" = true ] && [ -z "$FORCE" ]; then
    log_skip "data/processed/faiss_index.idx"
    log_skip "data/processed/metadata.json"
else
    bash "$SCRIPTS_DIR/03_index.sh" $FORCE
    log_ok "FAISS index saved → data/processed/faiss_index.idx"
    log_ok "Metadata saved    → data/processed/metadata.json"
fi

# =============================================================================
# Step 4 — Unit Tests
# =============================================================================
log_step "Step 4/7 — Unit Tests"

if [ "$SKIP_TEST" = true ]; then
    echo "  Tests skipped (--skip-test)."
else
    bash "$SCRIPTS_DIR/05_test.sh"
    TEST_EXIT=$?
    if [ $TEST_EXIT -ne 0 ]; then
        echo ""
        echo "  ERROR: Unit tests failed. Fix before continuing."
        echo "  Run: bash 05_test.sh --verbose for details."
        exit $TEST_EXIT
    fi
    log_ok "All unit tests passed."
fi

# =============================================================================
# Step 5 — LangChain Integration (canonical chain — required before Steps 6/7)
# =============================================================================
log_step "Step 5/7 — LangChain Integration"

# NOTE: Required by the project brief — "integration of LangChain, Mistral,
# and Faiss." src/project_11/rag/langchain_chain.py is the SINGLE implementation of
# the RAG chain, imported by both chatbot.py (Step 7) and evaluate.py
# (Step 6). This step must succeed before either of those can run.
if [ "$SKIP_LANGCHAIN" = true ]; then
    echo "  LangChain check skipped (--skip-langchain)."
    if [ "$SKIP_EVAL" != true ] || [ "$SKIP_RUN" != true ]; then
        log_error "Cannot skip LangChain while running eval or chatbot — both depend on it."
        exit 1
    fi
else
    if [ ! -s "$ROOT/data/processed/faiss_index.idx" ]; then
        log_error "FAISS index missing or empty — cannot verify LangChain integration."
        exit 1
    fi
    bash "$SCRIPTS_DIR/07_langchain_integration.sh"
    LC_EXIT=$?
    if [ $LC_EXIT -ne 0 ]; then
        echo ""
        echo "  ERROR: LangChain integration check failed."
        echo "  Run: bash 07_langchain_integration.sh for details."
        exit $LC_EXIT
    fi
    log_ok "LangChain integration verified → src/project_11/rag/langchain_chain.py"
fi

# =============================================================================
# Pipeline Summary
# =============================================================================
PIPELINE_END=$(date +%s)
ELAPSED=$((PIPELINE_END - PIPELINE_START))
MINUTES=$((ELAPSED / 60))
SECONDS_REM=$((ELAPSED % 60))

echo ""
echo "=============================================="
echo "  Pipeline Summary"
echo "=============================================="
echo ""
echo "  Data artifacts:"

for f in \
    "data/raw/events_paris.json" \
    "data/raw/fetch_metadata.json" \
    "data/processed/chunks.json" \
    "data/processed/embeddings.npy" \
    "data/processed/faiss_index.idx" \
    "data/processed/metadata.json"; do
    if [ -f "$ROOT/$f" ]; then
        SIZE=$(du -sh "$ROOT/$f" 2>/dev/null | cut -f1)
        printf "    %-45s %s\n" "$f" "$SIZE"
    else
        printf "    %-45s %s\n" "$f" "MISSING"
    fi
done

echo ""
echo "  Total pipeline time: ${MINUTES}m ${SECONDS_REM}s"
echo ""

# =============================================================================
# Step 6 — RAGAS Evaluation (uses the canonical LangChain chain)
# =============================================================================
log_step "Step 6/7 — RAGAS Evaluation"

if [ "$SKIP_EVAL" = true ]; then
    echo "  Evaluation skipped (--skip-eval)."
else
    bash "$SCRIPTS_DIR/06_evaluate.sh"
    log_ok "Evaluation complete → data/processed/test_dataset.csv"
fi

if [ "$SKIP_RUN" = true ]; then
    echo "  Chatbot launch skipped (--skip-run)."
    echo ""
    echo "  To launch manually:"
    echo "  bash 04_run.sh"
    echo ""
    echo "=============================================="
    echo "  Pipeline complete."
    echo "=============================================="
    echo ""
    exit 0
fi

# =============================================================================
# Step 7 — Launch Chatbot (uses the canonical LangChain chain)
# =============================================================================
log_step "Step 7/7 — Launch Chatbot"
bash "$SCRIPTS_DIR/04_run.sh"
