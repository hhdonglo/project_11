# =============================================================================
# src/project_11/rag/langchain_chain.py
# =============================================================================
# CANONICAL RAG chain — the single implementation of retrieve + generate
# used across the whole project, as required by the project brief:
# "Complete versioned RAG system code ... with integration of LangChain,
# Mistral, and Faiss."
#
# src/project_11/app/chatbot.py (Streamlit live demo) and src/project_11/rag/evaluate.py (RAGAS
# evaluation) both call ask() from this module. There is intentionally no
# second, hand-rolled retrieve/generate loop anywhere else in the codebase —
# a single source of truth for the RAG logic, per standard data engineering
# practice (DRY).
#
# This module wraps the existing FAISS index (built by 03_index.sh) inside
# LangChain's FAISS vector store abstraction, uses ChatMistralAI as the
# generation model, and assembles retrieval + generation into a LangChain
# Expression Language (LCEL) chain.
#
# Every Mistral API call made through this module (embeddings AND chat
# completions) goes through call_with_retry, which retries with exponential
# backoff on 429 / service_tier_capacity_exceeded errors. The free / shared
# Mistral tier occasionally throttles sustained sequential traffic — this
# is expected and absorbed automatically rather than crashing the caller.
#
# Usage:
#   poetry run python src/project_11/rag/langchain_chain.py
#   from src.project_11.rag.langchain_chain import ask, build_chain
# =============================================================================

import json
import logging
import time
from pathlib import Path

import faiss
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
ROOT        = Path(__file__).resolve().parents[3]
ENV_PATH    = ROOT / ".env"
INDEX_PATH  = ROOT / "data" / "processed" / "faiss_index.idx"
META_PATH   = ROOT / "data" / "processed" / "metadata.json"

# --- Config ---
load_dotenv(dotenv_path=ENV_PATH)
MISTRAL_KEY     = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL       = "mistral-small-latest"
LLM_TEMPERATURE = 0.1
TOP_K           = int(os.getenv("TOP_K_RESULTS", "5"))

# --- Retry configuration ---
MAX_RETRIES        = 5
RETRY_BACKOFF_BASE  = 2   # seconds; doubles each attempt: 2, 4, 8, 16, 32

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


def call_with_retry(fn, *args, **kwargs):
    """
    Call any LangChain/Mistral invocation with retry + exponential backoff
    on rate limit errors. LangChain wraps the underlying Mistral SDKError,
    so we match on the string content of the raised exception rather than
    a specific exception class, keeping this resilient across LangChain
    and mistralai version upgrades.
    """
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "capacity" in str(e).lower()
            last_exception = e

            if is_rate_limit and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE ** attempt
                log.warning(
                    f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            elif is_rate_limit:
                log.error(f"Rate limit persisted after {MAX_RETRIES} attempts.")
                raise
            else:
                raise

    raise last_exception


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


_cached_chain     = None
_cached_retriever = None


def build_chain(force_rebuild: bool = False):
    """
    Build the full LangChain LCEL chain: retriever -> prompt -> LLM -> parser.

    Cached at module level after first build so repeated calls to ask()
    from chatbot.py (once per Streamlit interaction) don't reload the
    FAISS index and reconstruct the chain on every user message.

    Returns both the chain (for generating answers) and the retriever
    (for inspecting sources separately, e.g. for the Streamlit sidebar).
    """
    global _cached_chain, _cached_retriever

    if _cached_chain is not None and not force_rebuild:
        return _cached_chain, _cached_retriever

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

    _cached_chain, _cached_retriever = chain, retriever
    return chain, retriever


def ask(question: str) -> dict:
    """
    Run the LangChain RAG chain end to end for a single question.

    This is the single entry point used by both the Streamlit chatbot
    (live demo) and the RAGAS evaluation script. Wrapped in retry logic
    on both the retrieval and generation calls.

    Returns the generated answer alongside the source documents used,
    including each source's raw text (needed by evaluate.py to build
    the "contexts" field required by RAGAS).
    """
    chain, retriever = build_chain()

    sources = call_with_retry(retriever.invoke, question)
    answer  = call_with_retry(chain.invoke, question)

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
                "text"      : d.page_content,
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
