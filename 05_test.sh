#!/bin/bash

# =============================================================================
# 05_test.sh — Run unit tests for the RAG pipeline
# =============================================================================
# Usage:
#   bash 05_test.sh           # run all tests
#   bash 05_test.sh --verbose # run with full output
# =============================================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=============================================="
echo "  Step 4 — Unit Tests"
echo "=============================================="
echo ""

if [ ! -f "$ROOT/.venv/bin/python" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Run: poetry install"
    exit 1
fi

MISSING=false

for f in \
    "data/raw/events_paris.json" \
    "data/raw/fetch_metadata.json" \
    "data/processed/chunks.json" \
    "data/processed/embeddings.npy" \
    "data/processed/faiss_index.idx" \
    "data/processed/metadata.json"; do
    if [ ! -f "$ROOT/$f" ]; then
        echo "ERROR: Missing required artifact: $f"
        MISSING=true
    fi
done

if [ "$MISSING" = true ]; then
    echo ""
    echo "Run: bash 01_fetch.sh --force   (to regenerate fetch_metadata.json)"
    echo "Then: bash pipeline.sh --skip-run"
    exit 1
fi

PYTEST_FLAGS="-v"
if [ "$1" == "--verbose" ]; then
    PYTEST_FLAGS="-v --tb=long"
fi

echo "Writing tests/test_data_pipeline.py..."
mkdir -p "$ROOT/tests"
touch "$ROOT/tests/__init__.py"

cat > "$ROOT/tests/test_data_pipeline.py" << 'EOF'
# =============================================================================
# tests/test_data_pipeline.py
# =============================================================================
# Unit tests validating that data integrated into the vector database
# corresponds to events less than one year old in the Paris region.
#
# Date recency is validated RELATIVE TO THE FETCH TIMESTAMP stored in
# data/raw/fetch_metadata.json, not relative to datetime.now(). This keeps
# the suite passing regardless of how long ago the data was fetched —
# staleness of the DATA is a separate concern (re-run 01_fetch.sh --force)
# from correctness of the FILTERING LOGIC, which is what this test verifies.
#
# Usage:
#   poetry run pytest tests/test_data_pipeline.py -v
# =============================================================================

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

# --- Paths ---
ROOT         = Path(__file__).resolve().parents[1]
RAW_PATH     = ROOT / "data" / "raw" / "events_paris.json"
FETCH_META   = ROOT / "data" / "raw" / "fetch_metadata.json"
CHUNKS_PATH  = ROOT / "data" / "processed" / "chunks.json"
EMBED_PATH   = ROOT / "data" / "processed" / "embeddings.npy"
INDEX_PATH   = ROOT / "data" / "processed" / "faiss_index.idx"
META_PATH    = ROOT / "data" / "processed" / "metadata.json"

# --- Constants ---
EXPECTED_REGION      = "Paris"
DEFAULT_MAX_AGE_DAYS = 365
EMBEDDING_DIM        = 1024
MIN_EXPECTED_EVENTS  = 100


def _parse_event_date(value):
    """Parse a firstTiming.begin value that may be an int (ms) or ISO string."""
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="ms", utc=True)
    return pd.to_datetime(value, utc=True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def raw_events() -> list[dict]:
    """Load raw events from disk."""
    assert RAW_PATH.exists(), f"Raw events file not found: {RAW_PATH}"
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def fetch_reference_time() -> datetime:
    """
    Load the timestamp the data was actually fetched at.

    Falls back to now() with a warning if fetch_metadata.json is missing
    (e.g. data generated before this file existed) so the suite still runs,
    but recency checks will then be relative to test-run time as before.
    """
    if FETCH_META.exists():
        with open(FETCH_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return datetime.fromisoformat(meta["fetched_at"])
    return datetime.now(timezone.utc)


@pytest.fixture(scope="session")
def max_age_days() -> int:
    """Load the max event age used at fetch time, if recorded."""
    if FETCH_META.exists():
        with open(FETCH_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("max_event_age_days", DEFAULT_MAX_AGE_DAYS))
    return DEFAULT_MAX_AGE_DAYS


@pytest.fixture(scope="session")
def chunks() -> list[dict]:
    """Load processed chunks from disk."""
    assert CHUNKS_PATH.exists(), f"Chunks file not found: {CHUNKS_PATH}"
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def embeddings() -> np.ndarray:
    """Load embeddings array from disk."""
    assert EMBED_PATH.exists(), f"Embeddings file not found: {EMBED_PATH}"
    return np.load(EMBED_PATH).astype("float32")


@pytest.fixture(scope="session")
def faiss_index() -> faiss.IndexFlatL2:
    """Load FAISS index from disk."""
    assert INDEX_PATH.exists(), f"FAISS index not found: {INDEX_PATH}"
    return faiss.read_index(str(INDEX_PATH))


@pytest.fixture(scope="session")
def metadata() -> list[dict]:
    """Load metadata from disk."""
    assert META_PATH.exists(), f"Metadata file not found: {META_PATH}"
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Group 1 — Raw Data Tests
# =============================================================================

class TestRawData:
    """Tests on the raw ingested dataset."""

    def test_raw_file_exists(self):
        """Raw events file must exist on disk."""
        assert RAW_PATH.exists(), f"Missing: {RAW_PATH}"

    def test_fetch_metadata_exists(self):
        """Fetch metadata (timestamp) must exist alongside the raw data."""
        assert FETCH_META.exists(), (
            f"Missing: {FETCH_META}. Re-run: bash 01_fetch.sh --force"
        )

    def test_raw_events_not_empty(self, raw_events):
        """Dataset must contain at least MIN_EXPECTED_EVENTS events."""
        assert len(raw_events) >= MIN_EXPECTED_EVENTS, (
            f"Too few events: {len(raw_events)} < {MIN_EXPECTED_EVENTS}"
        )

    def test_all_events_have_title(self, raw_events):
        """Every event must have a non-empty French title."""
        missing = [
            e for e in raw_events
            if not e.get("title.fr") or not str(e["title.fr"]).strip()
        ]
        assert len(missing) == 0, (
            f"{len(missing)} events have missing or empty title.fr"
        )

    def test_all_events_have_description(self, raw_events):
        """Every event must have a non-empty French description."""
        missing = [
            e for e in raw_events
            if not e.get("description.fr") or not str(e["description.fr"]).strip()
        ]
        assert len(missing) == 0, (
            f"{len(missing)} events have missing or empty description.fr"
        )

    def test_all_events_have_date(self, raw_events):
        """Every event must have a firstTiming.begin date."""
        missing = [e for e in raw_events if not e.get("firstTiming.begin")]
        assert len(missing) == 0, (
            f"{len(missing)} events are missing firstTiming.begin"
        )


# =============================================================================
# Group 2 — Date Filter Tests (Jeremy's requirement)
# =============================================================================

class TestDateFilter:
    """
    Tests that all events were less than one year old AT FETCH TIME.

    This is the core requirement from Jeremy's brief. Recency is checked
    relative to fetch_reference_time (when 01_fetch.sh ran), not relative
    to whenever this test suite happens to be executed. A dataset fetched
    months ago will correctly still pass, because it WAS valid when fetched.
    If you want current data, re-run: bash 01_fetch.sh --force
    """

    def test_all_events_within_one_year(
        self, raw_events, fetch_reference_time, max_age_days
    ):
        """
        All events must have firstTiming.begin within max_age_days of the
        fetch timestamp, or in the future relative to it.
        """
        cutoff     = fetch_reference_time - timedelta(days=max_age_days + 1)
        violations = []

        for event in raw_events:
            date_str = event.get("firstTiming.begin")
            if not date_str:
                continue
            event_date = _parse_event_date(date_str)
            if event_date < cutoff:
                violations.append({
                    "uid"  : event.get("uid"),
                    "title": event.get("title.fr"),
                    "date" : str(event_date.date())
                })

        assert len(violations) == 0, (
            f"{len(violations)} events are older than {max_age_days} days "
            f"relative to fetch time ({fetch_reference_time.date()}):\n"
            + "\n".join(
                f"  - {v['title']} ({v['date']})"
                for v in violations[:5]
            )
        )

    def test_no_future_events_beyond_two_years(
        self, raw_events, fetch_reference_time
    ):
        """Events should not be more than 2 years in the future of fetch time."""
        upper_bound = fetch_reference_time + timedelta(days=730)
        violations  = []

        for event in raw_events:
            date_str = event.get("firstTiming.begin")
            if not date_str:
                continue
            event_date = _parse_event_date(date_str)
            if event_date > upper_bound:
                violations.append({
                    "uid"  : event.get("uid"),
                    "title": event.get("title.fr"),
                    "date" : str(event_date.date())
                })

        assert len(violations) == 0, (
            f"{len(violations)} events are more than 2 years in the future."
        )

    def test_date_range_is_reasonable(self, raw_events):
        """Date range must span at least 30 days."""
        dates = [
            _parse_event_date(e["firstTiming.begin"])
            for e in raw_events
            if e.get("firstTiming.begin")
        ]
        date_range = max(dates) - min(dates)
        assert date_range.days >= 30, (
            f"Date range too narrow: {date_range.days} days"
        )


# =============================================================================
# Group 3 — Location Filter Tests (Jeremy's requirement)
# =============================================================================

class TestLocationFilter:
    """
    Tests that all events are in the correct geographical region.
    This is the core requirement from Jeremy's brief.
    """

    def test_majority_events_in_paris(self, raw_events):
        """At least 90% of events must be located in Paris."""
        total       = len(raw_events)
        paris_count = sum(
            1 for e in raw_events
            if str(e.get("location.city", "")).strip() == EXPECTED_REGION
        )
        ratio = paris_count / total

        assert ratio >= 0.90, (
            f"Only {paris_count}/{total} ({ratio:.1%}) events are in Paris. "
            f"Expected at least 90%."
        )

    def test_all_events_have_city(self, raw_events):
        """Every event must have a non-null city after cleaning."""
        missing = [
            e for e in raw_events
            if not e.get("location.city") or
            str(e["location.city"]).strip() == ""
        ]
        assert len(missing) == 0, (
            f"{len(missing)} events have null or empty location.city"
        )

    def test_no_postal_code_anomalies(self, raw_events):
        """City field must not contain raw postal codes."""
        anomalies = [
            e for e in raw_events
            if str(e.get("location.city", "")).strip().startswith("75")
            and str(e.get("location.city", "")).strip() != "Paris"
        ]
        assert len(anomalies) == 0, (
            f"{len(anomalies)} events have postal codes in location.city"
        )


# =============================================================================
# Group 4 — Vector Database Tests
# =============================================================================

class TestVectorDatabase:
    """Tests on the FAISS index and its metadata."""

    def test_index_file_exists(self):
        """FAISS index file must exist on disk."""
        assert INDEX_PATH.exists(), f"Missing: {INDEX_PATH}"

    def test_metadata_file_exists(self):
        """Metadata file must exist on disk."""
        assert META_PATH.exists(), f"Missing: {META_PATH}"

    def test_index_not_empty(self, faiss_index):
        """FAISS index must contain at least MIN_EXPECTED_EVENTS vectors."""
        assert faiss_index.ntotal >= MIN_EXPECTED_EVENTS, (
            f"Index has only {faiss_index.ntotal} vectors"
        )

    def test_index_dimension(self, faiss_index):
        """FAISS index must have correct embedding dimension."""
        assert faiss_index.d == EMBEDDING_DIM, (
            f"Wrong dimension: {faiss_index.d} expected {EMBEDDING_DIM}"
        )

    def test_index_metadata_sync(self, faiss_index, metadata):
        """FAISS index and metadata must have identical counts."""
        assert faiss_index.ntotal == len(metadata), (
            f"Sync error: {faiss_index.ntotal} vectors vs "
            f"{len(metadata)} metadata records"
        )

    def test_embeddings_shape(self, embeddings):
        """Embeddings array must have correct shape."""
        assert embeddings.ndim == 2, "Embeddings must be 2D"
        assert embeddings.shape[1] == EMBEDDING_DIM, (
            f"Wrong dimension: {embeddings.shape[1]}"
        )

    def test_embeddings_dtype(self, embeddings):
        """Embeddings must be float32."""
        assert embeddings.dtype == np.float32, (
            f"Wrong dtype: {embeddings.dtype}"
        )

    def test_embeddings_chunks_sync(self, embeddings, chunks):
        """Embeddings count must match chunks count."""
        assert embeddings.shape[0] == len(chunks), (
            f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks"
        )

    def test_index_search_returns_results(self, faiss_index, embeddings):
        """A search on the index must return valid results."""
        query_vec          = embeddings[0:1]
        distances, indices = faiss_index.search(query_vec, 5)

        assert len(indices[0]) == 5, "Search must return 5 results"
        assert all(i >= 0 for i in indices[0]), "All indices must be valid"
        assert all(d >= 0 for d in distances[0]), \
            "All distances must be non-negative"

    def test_metadata_has_required_fields(self, metadata):
        """Every metadata record must have all required fields."""
        required = ["uid", "title", "city", "date_range", "url", "text"]
        for i, record in enumerate(metadata):
            for field in required:
                assert field in record and record[field] is not None, (
                    f"Record {i} missing field: {field}"
                )


# =============================================================================
# Group 5 — Chunk Quality Tests
# =============================================================================

class TestChunkQuality:
    """Tests on the text chunks used for embedding."""

    def test_chunks_not_empty(self, chunks):
        """All chunks must have non-empty text."""
        empty = [
            i for i, c in enumerate(chunks)
            if not c.get("text", "").strip()
        ]
        assert len(empty) == 0, f"{len(empty)} chunks have empty text"

    def test_chunks_contain_event_marker(self, chunks):
        """All chunks must contain the 'Événement :' marker."""
        missing = [
            i for i, c in enumerate(chunks)
            if "Événement :" not in c.get("text", "")
        ]
        assert len(missing) == 0, (
            f"{len(missing)} chunks missing 'Événement :' marker"
        )

    def test_chunks_have_url(self, chunks):
        """All chunks must have a non-empty URL."""
        missing = [
            i for i, c in enumerate(chunks)
            if not c.get("url", "").strip()
        ]
        assert len(missing) == 0, f"{len(missing)} chunks missing URL"
EOF

echo "    tests/test_data_pipeline.py written."

echo ""
echo "Running pytest..."
echo ""

"$ROOT/.venv/bin/python" -m pytest \
    "$ROOT/tests/test_data_pipeline.py" \
    $PYTEST_FLAGS \
    --tb=short

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=============================================="
    echo "  All tests passed."
    echo "=============================================="
else
    echo "=============================================="
    echo "  Some tests failed. See output above."
    echo "=============================================="
fi
echo ""

exit $EXIT_CODE
