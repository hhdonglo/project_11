#!/bin/bash

# =============================================================================
# 07_langchain_integration.sh — Genuine LangChain integration for the RAG chain
# =============================================================================
# Satisfies the project brief requirement: "integration of LangChain, Mistral,
# and Faiss" as a real, working component of the RAG system — not just a
# discussion point in the report/slides.
#
# Wraps the existing FAISS index and metadata (already built by 03_index.sh)
# inside LangChain's FAISS vector store wrapper, uses ChatMistralAI as the
# LLM, and assembles retrieval + generation into a LangChain Expression
# Language (LCEL) chain.
#
# Usage:
#   bash 07_langchain_integration.sh
# =============================================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=============================================="
echo "  Step 7 — LangChain Integration"
echo "=============================================="
echo ""

if [ ! -f "$ROOT/.env" ]; then
    echo "ERROR: .env not found."
    exit 1
fi

if [ ! -f "$ROOT/.venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found."
    exit 1
fi

for f in \
    "data/processed/faiss_index.idx" \
    "data/processed/metadata.json" \
    "data/processed/embeddings.npy" \
    "data/processed/chunks.json"; do
    if [ ! -f "$ROOT/$f" ]; then
        echo "ERROR: Missing $f — run pipeline.sh --skip-run first."
        exit 1
    fi
done

# --- Write langchain_chain.py to src/rag/ ---
echo "Writing src/rag/langchain_chain.py..."
mkdir -p "$ROOT/src/rag"

cat > "$ROOT/src/rag/langchain_chain.py" << 'EOF'
# =============================================================================
# src/rag/langchain_chain.py
# =============================================================================
# Genuine LangChain integration of the RAG pipeline, as required by the
# project brief: "Complete versioned RAG system code ... with integration
# of LangChain, Mistral, and Faiss."
#
# This module wraps the existing FAISS index (built by 03_index.sh) inside
# LangChain's FAISS vector store abstraction, uses ChatMistralAI as the
# generation model, and assembles retrieval + generation into a LangChain
# Expression Language (LCEL) chain.
#
# The existing custom chain in src/rag/evaluate.py and src/app/chatbot.py
# remains available and is used for the live demo, since it gives full
# visibility into each step (embed / retrieve / generate) for teaching and
# debugging purposes. This module demonstrates the LangChain-native path
# and is the one that should be used going forward in production, where
# RetrievalQA's built-in conversation memory and agent support become
# valuable.
#
# Usage:
#   poetry run python src/rag/langchain_chain.py
#   from src.rag.langchain_chain import build_chain, ask
# =============================================================================

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
import os

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Paths ---
ROOT        = Path(__file__).resolve().parents[2]
ENV_PATH    = ROOT / ".env"
INDEX_PATH  = ROOT / "data" / "processed" / "faiss_index.idx"
META_PATH   = ROOT / "data" / "processed" / "metadata.json"

# --- Config ---
load_dotenv(dotenv_path=ENV_PATH)
MISTRAL_KEY     = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL       = "mistral-small-latest"
LLM_TEMPERATURE = 0.1
TOP_K           = 5

SYSTEM_PROMPT = """Tu es un assistant specialise dans les evenements culturels a Paris.
Tu aides les utilisateurs a decouvrir des evenements en te basant UNIQUEMENT sur
les informations fournies dans le contexte ci-dessous.

Regles strictes :
- Reponds UNIQUEMENT en te basant sur les evenements fournis dans le contexte.
- Si l'information n'est pas dans le contexte, dis clairement que tu ne sais pas.
- Ne genere JAMAIS d'informations inventees sur des evenements.
- Reponds toujours en francais.
- Sois concis, precis et utile.
- Pour chaque evenement mentionne, indique le titre, la date et le lieu.

Contexte des evenements disponibles :
{context}

Question de l'utilisateur : {question}

Reponds en te basant uniquement sur les evenements fournis ci-dessus."""


def load_faiss_and_metadata() -> tuple[faiss.IndexFlatL2, list[dict]]:
    """Load the existing FAISS index and metadata built by 03_index.sh."""
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert index.ntotal == len(metadata), (
        f"Index/metadata mismatch: {index.ntotal} vs {len(metadata)}"
    )
    return index, metadata


def wrap_faiss_in_langchain(
    index: faiss.IndexFlatL2,
    metadata: list[dict],
    embeddings: MistralAIEmbeddings,
) -> LangChainFAISS:
    """
    Wrap the existing raw FAISS index inside LangChain's FAISS vector store
    abstraction, so LangChain's retriever interface can be used directly
    against the same index built by build_index.py, with no re-embedding.
    """
    docs = {}
    index_to_docstore_id = {}

    for i, record in enumerate(metadata):
        doc_id = str(record["uid"])
        docs[doc_id] = Document(
            page_content=record["text"],
            metadata={
                "uid"       : record["uid"],
                "title"     : record["title"],
                "city"      : record["city"],
                "venue"     : record["venue"],
                "date_range": record["date_range"],
                "url"       : record["url"],
            },
        )
        index_to_docstore_id[i] = doc_id

    docstore = InMemoryDocstore(docs)

    vector_store = LangChainFAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    return vector_store


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into the context block for the prompt."""
    return "\n\n".join(
        f"Evenement {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def build_chain():
    """
    Build the full LangChain LCEL chain: retriever -> prompt -> LLM -> parser.

    Returns both the chain (for generating answers) and the retriever
    (for inspecting sources separately, e.g. for the Streamlit sidebar).
    """
    assert MISTRAL_KEY, "MISTRAL_API_KEY not found in .env"

    embeddings = MistralAIEmbeddings(
        mistral_api_key=MISTRAL_KEY,
        model=EMBEDDING_MODEL,
    )

    index, metadata = load_faiss_and_metadata()
    vector_store = wrap_faiss_in_langchain(index, metadata, embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatMistralAI(
        mistral_api_key=MISTRAL_KEY,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(question: str) -> dict:
    """
    Run the LangChain RAG chain end to end for a single question.

    Returns the generated answer alongside the source documents used,
    matching the shape expected by the Streamlit sidebar.
    """
    chain, retriever = build_chain()

    sources = retriever.invoke(question)
    answer  = chain.invoke(question)

    return {
        "question": question,
        "answer"  : answer,
        "sources" : [
            {
                "uid"       : d.metadata["uid"],
                "title"     : d.metadata["title"],
                "city"      : d.metadata["city"],
                "venue"     : d.metadata["venue"],
                "date_range": d.metadata["date_range"],
                "url"       : d.metadata["url"],
            }
            for d in sources
        ],
    }


if __name__ == "__main__":
    log.info("Building LangChain RAG chain from existing FAISS index...")

    test_questions = [
        "Quels concerts de musique classique ont lieu a Paris ?",
        "Quel est le meilleur restaurant a Paris ?",
    ]

    for q in test_questions:
        log.info(f"\nQuestion: {q}")
        result = ask(q)
        log.info(f"Answer: {result['answer']}")
        log.info(f"Sources used: {len(result['sources'])}")
        for s in result["sources"]:
            log.info(f"  - {s['title']} ({s['date_range']})")

    log.info("\nLangChain integration verified successfully.")
EOF

echo "    src/rag/langchain_chain.py written."

# --- Run it as a smoke test ---
echo ""
echo "Running LangChain chain smoke test (2 sample questions)..."
echo ""

"$ROOT/.venv/bin/python" "$ROOT/src/rag/langchain_chain.py"

echo ""
echo "=============================================="
echo "  Step 7 complete."
echo "  Module: src/rag/langchain_chain.py"
echo "  This module now provides the LangChain +"
echo "  Mistral + FAISS integration required by"
echo "  the project brief, built on top of the"
echo "  same FAISS index used by the chatbot."
echo "=============================================="
echo ""