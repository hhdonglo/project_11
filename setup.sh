#!/bin/bash

# =============================================================================
# Puls-Events RAG — Project Setup Script
# =============================================================================
# Usage: bash setup_project.sh
# Run this from inside your already-initialized Poetry project directory.
# Assumes: Poetry project already created, pyproject.toml exists.
# Note: pandas is intentionally omitted — streamlit resolves it automatically.
# Note: mistralai pinned to 1.2.5 — 2.x is broken under Python 3.13
# =============================================================================

set -e

echo ""
echo "=============================================="
echo "  Puls-Events RAG — Project Initialization"
echo "  (Poetry project already initialized)"
echo "=============================================="
echo ""

# --- 1. Add dependencies ---
echo "[1/5] Adding dependencies..."
source .venv/bin/activate

# Add streamlit first — it pins pandas to a compatible version (<3)
poetry add streamlit

# Add remaining core dependencies
# pandas excluded — managed by streamlit
# mistralai pinned to 1.2.5 — 2.x broken under Python 3.13
poetry add \
  langchain \
  langchain-mistralai \
  "faiss-cpu>=1.13.2,<2.0.0" \
  "mistralai==1.2.5" \
  python-dotenv \
  requests \
  numpy

# Evaluation dependencies
poetry add ragas datasets

# Dev dependencies
poetry add --group dev \
  pytest \
  pytest-cov \
  black \
  ruff

echo "    All dependencies installed."

# --- 2. Create folder structure ---
echo "[2/5] Creating folder structure..."

mkdir -p src/ingestion
mkdir -p src/processing
mkdir -p src/vector_store
mkdir -p src/rag
mkdir -p src/app
mkdir -p tests
mkdir -p data/raw
mkdir -p data/processed
mkdir -p notebooks
mkdir -p reports
mkdir -p slides

# __init__.py for all Python packages
touch src/__init__.py
touch src/ingestion/__init__.py
touch src/processing/__init__.py
touch src/vector_store/__init__.py
touch src/rag/__init__.py
touch src/app/__init__.py
touch tests/__init__.py

# Placeholder module files
touch src/ingestion/fetch_events.py
touch src/ingestion/filter_events.py
touch src/ingestion/clean_events.py
touch src/processing/chunker.py
touch src/processing/embedder.py
touch src/vector_store/build_index.py
touch src/vector_store/faiss_store.py
touch src/rag/retriever.py
touch src/rag/prompt.py
touch src/rag/chain.py
touch src/app/chatbot.py
touch tests/test_data_pipeline.py

# Notebook placeholders
touch notebooks/01_explore_openagenda.ipynb
touch notebooks/02_chunking_embedding.ipynb
touch notebooks/03_faiss_index.ipynb
touch notebooks/04_rag_chain.ipynb

echo "    Folders and placeholder files created."

# --- 3. Write configuration files ---
echo "[3/5] Writing configuration files..."

cat > .env.example << 'EOF'
# =============================================================================
# Environment Variables — Puls-Events RAG
# Copy this file to .env and fill in your values. Never commit .env.
# =============================================================================

# Mistral AI
MISTRAL_API_KEY=your_mistral_api_key_here

# Open Agenda
OPENAGENDA_API_KEY=your_openagenda_api_key_here

# Project settings
AGENDA_UID=82290100
LOCATION=Paris
MAX_EVENT_AGE_DAYS=365
TOP_K_RESULTS=5
LLM_TEMPERATURE=0.1
EMBEDDING_MODEL=mistral-embed
LLM_MODEL=mistral-small-latest
EOF

cat > .gitignore << 'EOF'
# Environment
.env
.venv/
__pycache__/
*.pyc
*.pyo

# Raw data (not versioned — fetched on demand)
data/raw/

# Processed data (versioned for reproducibility)
# NOTE: chunks.json, embeddings.npy, faiss_index.idx, metadata.json
# are committed so the demo can run without re-embedding

# FAISS artifacts — keep these versioned
# *.idx
# metadata.json

# Large embedding file — exclude if too large for git
# data/processed/embeddings.npy

# Poetry
poetry.lock

# Reports & slides (version final versions only)
reports/*.pdf
slides/*.pptx

# Jupyter checkpoints
.ipynb_checkpoints/
notebooks/.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
EOF

echo "    .env.example and .gitignore written."

# --- 4. Write README.md ---
echo "[4/5] Writing README..."

cat > README.md << 'EOF'
# Puls-Events RAG — Cultural Event Assistant POC

A Retrieval-Augmented Generation (RAG) system for personalized cultural event
recommendations in Paris, powered by Open Agenda data, Mistral AI, FAISS, and LangChain.

---

## Project Structure

```
puls-events-rag/
├── src/
│   ├── ingestion/       # Open Agenda API client and data filtering
│   ├── processing/      # Text chunking and embedding
│   ├── vector_store/    # FAISS index construction and querying
│   ├── rag/             # Retriever, prompt templates, RAG chain
│   └── app/             # Streamlit chat interface
├── notebooks/           # Exploration and development notebooks
├── tests/               # Unit tests (pytest)
├── data/
│   ├── raw/             # Raw JSON from Open Agenda (not versioned)
│   └── processed/       # Chunks, embeddings, FAISS index
├── reports/             # Technical report
├── slides/              # Presentation
├── .env.example         # Environment variable template
└── pyproject.toml       # Poetry dependency management
```

---

## Setup

### Prerequisites
- Python 3.13
- Poetry
- Mistral AI API key (https://console.mistral.ai)
- Open Agenda API key (https://openagenda.com)

### Installation

```bash
git clone <repo-url>
cd puls-events-rag
poetry install
cp .env.example .env
# Edit .env with your API keys
```

---

## Usage

### 1. Fetch and filter events (Paris, past 12 months + upcoming)
```bash
poetry run python src/ingestion/fetch_events.py
```

### 2. Build chunks and embeddings
```bash
poetry run python src/processing/embedder.py
```

### 3. Build the FAISS vector index
```bash
poetry run python src/vector_store/build_index.py
```

### 4. Run the Streamlit chatbot
```bash
poetry run streamlit run src/app/chatbot.py
```

### 5. Run unit tests
```bash
poetry run pytest tests/ -v
```

---

## Data Source

- **Platform:** Open Agenda (https://openagenda.com)
- **Agenda:** Deciding for Paris (UID: 82290100)
- **Region:** Paris, Île-de-France
- **Filter:** Events from the past 12 months + upcoming
- **Volume:** ~1,740 events

---

## Technical Stack

| Component | Technology |
|---|---|
| Embedding model | mistral-embed (1,024 dimensions) |
| LLM | mistral-small-latest |
| Vector database | FAISS IndexFlatL2 |
| Orchestration | LangChain |
| Interface | Streamlit |
| Language | Python 3.13 |

---

## Evaluation

RAGAS metrics computed against annotated test dataset.

| Metric | Target |
|---|---|
| faithfulness | > 0.80 |
| answer_relevancy | > 0.75 |
| context_precision | > 0.70 |
| context_recall | > 0.70 |

---

## Known Constraints

- mistralai pinned to 1.2.5 — version 2.x incompatible with Python 3.13
- Free tier rate limits require batch_size=5 and sleep=3s during embedding
- Embedding ~1,740 events takes ~17 minutes on free tier

---

## Author

Hope — Freelance Data Engineer
EOF

echo "    README.md written."

# --- 5. Git commit ---
echo "[5/5] Committing scaffold to git..."
git add .
git commit -m "chore: initial project scaffold — puls-events-rag"
git checkout -b dev 2>/dev/null || git checkout dev

echo ""
echo "=============================================="
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "  1. cp .env.example .env"
echo "  2. Add your API keys to .env"
echo "  3. poetry shell"
echo "  4. Run notebooks in order: 01 → 02 → 03 → 04"
echo "  5. poetry run streamlit run src/app/chatbot.py"
echo "=============================================="
echo ""