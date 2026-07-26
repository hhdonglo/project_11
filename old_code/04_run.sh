#!/bin/bash

# =============================================================================
# 04_run.sh — Launch Puls-Events RAG Streamlit chatbot
# =============================================================================
# Usage:
#   bash 04_run.sh
# =============================================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=============================================="
echo "  Step 4 — Launch Chatbot"
echo "=============================================="
echo ""

# Guard — check .env exists
if [ ! -f "$ROOT/.env" ]; then
    echo "ERROR: .env file not found at $ROOT/.env"
    echo "Run: cp .env.example .env and add your API keys"
    exit 1
fi

# Guard — check venv exists
if [ ! -f "$ROOT/.venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Run: poetry install"
    exit 1
fi

# Guard — check FAISS index exists
if [ ! -f "$ROOT/data/processed/faiss_index.idx" ]; then
    echo "ERROR: data/processed/faiss_index.idx not found."
    echo "Run: bash 03_index.sh first."
    exit 1
fi

# Guard — check metadata exists
if [ ! -f "$ROOT/data/processed/metadata.json" ]; then
    echo "ERROR: data/processed/metadata.json not found."
    echo "Run: bash 03_index.sh first."
    exit 1
fi

# --- Write chatbot.py ---
echo "Writing src/app/chatbot.py..."
mkdir -p "$ROOT/src/app"

cat > "$ROOT/src/app/chatbot.py" << 'EOF'
# =============================================================================
# src/app/chatbot.py
# =============================================================================
# Puls-Events RAG — Cultural Event Chatbot
# Streamlit interface with FAISS retrieval and Mistral LLM generation.
#
# Usage:
#   poetry run streamlit run src/app/chatbot.py
# =============================================================================

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral
import os

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Paths ---
ROOT          = Path(__file__).resolve().parents[2]
ENV_PATH      = ROOT / ".env"
INDEX_PATH    = ROOT / "data" / "processed" / "faiss_index.idx"
META_PATH     = ROOT / "data" / "processed" / "metadata.json"

# --- Configuration ---
load_dotenv(dotenv_path=ENV_PATH)
MISTRAL_KEY     = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
LLM_MODEL       = os.getenv("LLM_MODEL", "mistral-small-latest")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
TOP_K           = int(os.getenv("TOP_K_RESULTS", "5"))

if not MISTRAL_KEY:
    st.error("MISTRAL_API_KEY not found in .env")
    st.stop()


# --- Load resources (cached across sessions) ---
@st.cache_resource
def load_resources():
    client   = Mistral(api_key=MISTRAL_KEY)
    index    = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    log.info(f"Resources loaded: {index.ntotal} vectors, {len(metadata)} events")
    return client, index, metadata


client, index, metadata = load_resources()


# --- Core RAG functions ---
def embed_query(text: str) -> np.ndarray:
    """Embed a single user query using Mistral."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        inputs=[text]
    )
    return np.array([response.data[0].embedding], dtype="float32")


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Search FAISS index and return top-k matching events."""
    query_vec          = embed_query(query)
    distances, indices = index.search(query_vec, k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        result             = metadata[idx].copy()
        result["distance"] = float(dist)
        results.append(result)
    return results


SYSTEM_PROMPT = """Tu es un assistant spécialisé dans les événements culturels à Paris.
Tu aides les utilisateurs à découvrir des événements en te basant UNIQUEMENT sur
les informations fournies dans le contexte ci-dessous.

Règles strictes :
- Réponds UNIQUEMENT en te basant sur les événements fournis dans le contexte.
- Si l'information n'est pas dans le contexte, dis clairement que tu ne sais pas.
- Ne génère JAMAIS d'informations inventées sur des événements.
- Réponds toujours en français.
- Sois concis, précis et utile.
- Pour chaque événement mentionné, indique le titre, la date et le lieu.
"""


def generate(query: str, context_events: list[dict]) -> str:
    """Generate a response using Mistral LLM with retrieved context."""
    context = "\n\n".join([
        f"Événement {i+1}:\n{event['text']}"
        for i, event in enumerate(context_events)
    ])
    user_message = f"""Contexte des événements disponibles :
{context}

Question de l'utilisateur : {query}

Réponds en te basant uniquement sur les événements fournis ci-dessus."""

    response = client.chat.complete(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ]
    )
    return response.choices[0].message.content


def rag(query: str) -> dict:
    """Full RAG pipeline: embed query → retrieve → generate."""
    retrieved = retrieve(query)
    answer    = generate(query, retrieved)
    return {"query": query, "answer": answer, "sources": retrieved}


# --- Streamlit UI ---
st.set_page_config(
    page_title="Puls-Events — Assistant Culturel Paris",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 Puls-Events — Assistant Culturel Paris")
st.caption("Découvrez les événements culturels à Paris grâce à l'IA")

# Sidebar — source transparency
with st.sidebar:
    st.header("📚 Sources récupérées")
    st.caption(
        "Les événements utilisés pour générer la réponse "
        "apparaîtront ici après chaque question."
    )
    source_placeholder = st.empty()

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input(
    "Que recherchez-vous ? Ex: concerts ce weekend, expositions d'art..."
):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            result = rag(prompt)
        st.markdown(result["answer"])

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": result["answer"]
    })

    # Update sidebar with sources
    with source_placeholder.container():
        for i, source in enumerate(result["sources"], 1):
            with st.expander(f"{i}. {source['title']}"):
                st.write(f"📅 {source['date_range']}")
                st.write(f"📍 {source['venue']}, {source['city']}")
                st.write(f"🔗 [Voir l'événement]({source['url']})")
EOF

echo "    src/app/chatbot.py written."

# --- Launch Streamlit ---
echo ""
echo "Launching Streamlit chatbot..."
echo "URL: http://localhost:8501"
echo "Press Ctrl+C to stop."
echo ""

"$ROOT/.venv/bin/python" -m streamlit run \
    "$ROOT/src/app/chatbot.py" \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false