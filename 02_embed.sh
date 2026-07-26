#!/bin/bash

# =============================================================================
# 02_embed.sh — Chunk events and generate Mistral embeddings
# =============================================================================
# Usage:
#   bash 02_embed.sh           # skip if embeddings already exist
#   bash 02_embed.sh --force   # force re-embed
# =============================================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=============================================="
echo "  Step 2 — Chunk & Embed Events"
echo "=============================================="
echo ""

if [ ! -f "$ROOT/.env" ]; then
    echo "ERROR: .env file not found at $ROOT/.env"
    echo "Run: cp .env.example .env and add your API keys"
    exit 1
fi

if [ ! -f "$ROOT/.venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Run: poetry install"
    exit 1
fi

if [ ! -f "$ROOT/data/raw/events_paris.json" ]; then
    echo "ERROR: data/raw/events_paris.json not found."
    echo "Run: bash 01_fetch.sh first."
    exit 1
fi

FORCE=""
if [ "$1" == "--force" ]; then
    FORCE="--force"
    echo "Force mode: re-embedding even if outputs exist."
fi

echo "Writing src/processing/chunker.py..."
mkdir -p "$ROOT/src/processing"
touch "$ROOT/src/processing/__init__.py"

cat > "$ROOT/src/processing/chunker.py" << 'EOF'
# =============================================================================
# src/processing/chunker.py
# =============================================================================
# Loads cleaned events, builds one enriched text chunk per event,
# and saves chunks with metadata to data/processed/chunks.json
#
# Chunking strategy: 1 chunk per event
# Rationale: descriptions are very short (median 50 chars) —
# no recursive splitting needed. Title + description + venue + date
# are concatenated into one rich text unit per event.
# =============================================================================

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "raw" / "events_paris.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "chunks.json"

BASE_URL = "https://openagenda.com/deciding-for-paris/events"


def build_chunk_text(event: dict) -> str:
    """Concatenate event fields into one rich text unit for embedding."""
    title       = event.get("title.fr") or ""
    description = event.get("description.fr") or ""
    venue       = event.get("location.name") or ""
    city        = event.get("location.city") or ""
    date        = event.get("dateRange.fr") or ""

    return (
        f"Événement : {title}\n"
        f"Description : {description}\n"
        f"Lieu : {venue}, {city}\n"
        f"Date : {date}"
    )


def build_chunks(events: list[dict]) -> list[dict]:
    """Build chunk list with text and metadata from events."""
    chunks = []
    for event in events:
        slug = event.get("slug", "")
        chunk = {
            "uid"       : event.get("uid"),
            "text"      : build_chunk_text(event),
            "title"     : event.get("title.fr"),
            "city"      : event.get("location.city"),
            "address"   : event.get("location.address"),
            "venue"     : event.get("location.name"),
            "date_begin": event.get("firstTiming.begin"),
            "date_end"  : event.get("lastTiming.end"),
            "date_range": event.get("dateRange.fr"),
            "slug"      : slug,
            "url"       : f"{BASE_URL}/{slug}",
        }
        chunks.append(chunk)
    return chunks


def run(force: bool = False) -> list[dict]:
    """Main chunker entry point."""
    if OUTPUT_PATH.exists() and not force:
        log.info(f"Chunks already exist: {OUTPUT_PATH} — skipping.")
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        events = json.load(f)
    log.info(f"Loaded {len(events)} events from {INPUT_PATH}")

    chunks = build_chunks(events)
    log.info(f"Built {len(chunks)} chunks")

    assert len(chunks) == len(events), "Chunk count mismatch"
    assert all(c["text"] for c in chunks), "Empty chunk text detected"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    log.info(f"Chunks saved → {OUTPUT_PATH}")

    return chunks
EOF

echo "    src/processing/chunker.py written."

echo "Writing src/processing/embedder.py..."

cat > "$ROOT/src/processing/embedder.py" << 'EOF'
# =============================================================================
# src/processing/embedder.py
# =============================================================================
# Loads chunks, generates Mistral embeddings in batches,
# and saves to data/processed/embeddings.npy
#
# Usage:
#   poetry run python src/processing/embedder.py
#   poetry run python src/processing/embedder.py --force
#
# Rate limit handling:
#   Batch size: 5 | Sleep: 3s | On 429: wait 60s and retry
# =============================================================================

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral
import os

from chunker import build_chunks, run as run_chunker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
ENV_PATH    = ROOT / ".env"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
EMBED_PATH  = ROOT / "data" / "processed" / "embeddings.npy"

EMBEDDING_MODEL = "mistral-embed"
BATCH_SIZE      = 5
SLEEP_BETWEEN   = 3.0
SLEEP_429       = 60


def load_client() -> Mistral:
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env")
    return Mistral(api_key=api_key)


def embed_batch(client: Mistral, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=texts
    )
    return [item.embedding for item in response.data]


def embed_all(client: Mistral, chunks: list[dict]) -> np.ndarray:
    """Embed all chunks with rate limit handling."""
    all_embeddings  = []
    total_batches   = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    log.info(f"Embedding {len(chunks)} chunks in {total_batches} batches...")
    log.info(f"Estimated time: ~{(total_batches * SLEEP_BETWEEN) / 60:.1f} minutes")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch       = chunks[i : i + BATCH_SIZE]
        batch_texts = [c["text"] for c in batch]
        batch_num   = (i // BATCH_SIZE) + 1

        try:
            embeddings = embed_batch(client, batch_texts)
            all_embeddings.extend(embeddings)

            if batch_num % 50 == 0 or batch_num == total_batches:
                log.info(f"  Batch {batch_num}/{total_batches} — "
                         f"cumulative: {len(all_embeddings)}/{len(chunks)}")

        except Exception as e:
            if "429" in str(e):
                log.warning(f"Rate limit at batch {batch_num} — "
                            f"waiting {SLEEP_429}s...")
                time.sleep(SLEEP_429)
                try:
                    embeddings = embed_batch(client, batch_texts)
                    all_embeddings.extend(embeddings)
                    log.info("Retry successful.")
                except Exception as retry_e:
                    log.error(f"Retry failed: {retry_e}")
                    raise
            else:
                log.error(f"Error at batch {batch_num}: {e}")
                raise

        time.sleep(SLEEP_BETWEEN)

    return np.array(all_embeddings, dtype=np.float32)


def main(force: bool = False) -> None:
    log.info("Starting embedder pipeline...")

    if EMBED_PATH.exists() and not force:
        log.info(f"Embeddings already exist: {EMBED_PATH} — skipping.")
        return

    import json
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks")

    client     = load_client()
    embeddings = embed_all(client, chunks)

    assert embeddings.shape[0] == len(chunks), \
        f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks"
    assert embeddings.shape[1] == 1024, \
        f"Wrong dimension: {embeddings.shape[1]}"

    EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBED_PATH, embeddings)

    log.info(f"Embeddings saved → {EMBED_PATH}")
    log.info(f"Shape: {embeddings.shape} | dtype: {embeddings.dtype}")
    log.info("Embedder pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mistral embeddings")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-embed even if output already exists"
    )
    args = parser.parse_args()
    main(force=args.force)
EOF

echo "    src/processing/embedder.py written."

echo ""
echo "Running src/processing/chunker.py..."
"$ROOT/.venv/bin/python" -c "
import sys
sys.path.insert(0, '$ROOT/src/processing')
from chunker import run
run(force=$( [ '$FORCE' == '--force' ] && echo 'True' || echo 'False' ))
print('Chunker complete.')
"

echo ""
echo "Running src/processing/embedder.py..."
"$ROOT/.venv/bin/python" -c "
import sys
sys.path.insert(0, '$ROOT/src/processing')
from embedder import main
main(force=$( [ '$FORCE' == '--force' ] && echo 'True' || echo 'False' ))
"

echo ""
echo "=============================================="
echo "  Step 2 complete."
echo "  Output: data/processed/chunks.json"
echo "          data/processed/embeddings.npy"
echo "  Next:   bash 03_index.sh"
echo "=============================================="
echo ""
