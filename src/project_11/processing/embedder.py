# =============================================================================
# src/project_11/processing/embedder.py
# =============================================================================
# Loads chunks, generates Mistral embeddings in batches,
# and saves to data/processed/embeddings.npy
#
# Usage:
#   poetry run python src/project_11/processing/embedder.py
#   poetry run python src/project_11/processing/embedder.py --force
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

ROOT        = Path(__file__).resolve().parents[3]
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
