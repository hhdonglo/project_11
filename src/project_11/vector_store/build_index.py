# =============================================================================
# src/project_11/vector_store/build_index.py
# =============================================================================
# Loads chunks and embeddings, builds a FAISS IndexFlatL2 index,
# saves the index and a parallel metadata JSON file.
#
# Usage:
#   poetry run python src/project_11/vector_store/build_index.py
#   poetry run python src/project_11/vector_store/build_index.py --force
#
# Outputs:
#   data/processed/faiss_index.idx  — FAISS binary index
#   data/processed/metadata.json    — parallel metadata array
# =============================================================================

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[3]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
EMBED_PATH  = ROOT / "data" / "processed" / "embeddings.npy"
INDEX_PATH  = ROOT / "data" / "processed" / "faiss_index.idx"
META_PATH   = ROOT / "data" / "processed" / "metadata.json"

EMBEDDING_DIM = 1024


def load_artifacts() -> tuple[list[dict], np.ndarray]:
    """Load chunks and embeddings from disk."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    log.info(f"Chunks loaded     : {len(chunks)}")

    embeddings = np.load(EMBED_PATH).astype("float32")
    log.info(f"Embeddings shape  : {embeddings.shape}")

    assert len(chunks) == embeddings.shape[0], \
        f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
    assert embeddings.shape[1] == EMBEDDING_DIM, \
        f"Wrong dimension: {embeddings.shape[1]} expected {EMBEDDING_DIM}"

    log.info("Artifact assertions passed.")
    return chunks, embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build and populate FAISS IndexFlatL2."""
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    log.info(f"FAISS index built : {index.ntotal} vectors | dim: {index.d}")
    return index


def build_metadata(chunks: list[dict]) -> list[dict]:
    """Build metadata array parallel to FAISS index positions."""
    return [
        {
            "uid"       : c["uid"],
            "title"     : c["title"],
            "city"      : c["city"],
            "address"   : c["address"],
            "venue"     : c["venue"],
            "date_begin": c["date_begin"],
            "date_end"  : c["date_end"],
            "date_range": c["date_range"],
            "url"       : c["url"],
            "text"      : c["text"],
        }
        for c in chunks
    ]


def save_artifacts(
    index: faiss.IndexFlatL2,
    metadata: list[dict]
) -> None:
    """Save FAISS index and metadata to disk."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    log.info(f"FAISS index saved : {INDEX_PATH}")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log.info(f"Metadata saved    : {META_PATH}")


def verify_reload(expected_count: int) -> None:
    """Reload index from disk and verify vector count."""
    reloaded = faiss.read_index(str(INDEX_PATH))
    assert reloaded.ntotal == expected_count, \
        f"Reload mismatch: {reloaded.ntotal} vs {expected_count}"
    log.info(f"Reload verified   : {reloaded.ntotal} vectors — OK")


def main(force: bool = False) -> None:
    log.info("Starting build_index pipeline...")

    if INDEX_PATH.exists() and META_PATH.exists() and not force:
        log.info(f"Index already exists: {INDEX_PATH} — skipping.")
        log.info("Use --force to rebuild.")
        return

    chunks, embeddings = load_artifacts()
    index              = build_faiss_index(embeddings)
    metadata           = build_metadata(chunks)

    save_artifacts(index, metadata)
    verify_reload(len(chunks))

    log.info("=" * 50)
    log.info("SUMMARY")
    log.info(f"  Vectors indexed : {index.ntotal}")
    log.info(f"  Index type      : IndexFlatL2")
    log.info(f"  Dimension       : {index.d}")
    log.info(f"  Index file      : {INDEX_PATH}")
    log.info(f"  Metadata file   : {META_PATH}")
    log.info("=" * 50)
    log.info("build_index pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index already exists"
    )
    args = parser.parse_args()
    main(force=args.force)
