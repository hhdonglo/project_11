#!/bin/bash

# =============================================================================
# 01_fetch.sh — Fetch Paris events from Open Agenda
# =============================================================================
# Usage:
#   bash 01_fetch.sh           # skip if data already exists
#   bash 01_fetch.sh --force   # force re-fetch
# =============================================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=============================================="
echo "  Step 1 — Fetch Events"
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

FORCE=""
if [ "$1" == "--force" ]; then
    FORCE="--force"
    echo "Force mode: re-fetching even if data exists."
fi

echo "Writing src/ingestion/fetch_events.py..."
mkdir -p "$ROOT/src/ingestion"
touch "$ROOT/src/__init__.py"
touch "$ROOT/src/ingestion/__init__.py"

cat > "$ROOT/src/ingestion/fetch_events.py" << 'EOF'
# =============================================================================
# src/ingestion/fetch_events.py
# =============================================================================
# Fetches cultural events from Open Agenda API, applies date and location
# filters, cleans the data, and saves to data/raw/events_paris.json
#
# Also writes data/raw/fetch_metadata.json containing the fetch timestamp.
# Downstream unit tests validate event recency RELATIVE TO THIS TIMESTAMP,
# not relative to "now" — this keeps tests passing even when run days or
# months after the data was originally fetched.
#
# Network resilience: each page request is wrapped in retry logic with
# exponential backoff. The Open Agenda API occasionally resets long-lived
# connections after dozens of sequential requests — this is transient and
# unrelated to the request itself, so a retry almost always succeeds.
#
# Usage:
#   poetry run python src/ingestion/fetch_events.py
#   poetry run python src/ingestion/fetch_events.py --force
# =============================================================================

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parents[2]
ENV_PATH      = ROOT / ".env"
OUTPUT_PATH   = ROOT / "data" / "raw" / "events_paris.json"
METADATA_PATH = ROOT / "data" / "raw" / "fetch_metadata.json"

COLS = [
    "uid",
    "title.fr",
    "description.fr",
    "location.city",
    "location.address",
    "location.name",
    "firstTiming.begin",
    "lastTiming.end",
    "dateRange.fr",
    "slug",
    "attendanceMode",
    "categories",
    "types-devenement",
]

MAX_RETRIES_PER_PAGE = 5
RETRY_BACKOFF_BASE    = 2  # seconds; doubles each retry: 2, 4, 8, 16, 32


def build_session() -> requests.Session:
    """
    Build a requests Session with automatic retries for transient
    HTTP-level failures (429, 500-504) baked into the adapter, on top
    of the manual retry loop in fetch_all_events which also catches
    lower-level connection resets that the adapter alone won't retry.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def load_config() -> dict:
    """Load configuration from environment variables."""
    load_dotenv(dotenv_path=ENV_PATH)

    api_key    = os.getenv("OPENAGENDA_API_KEY")
    agenda_uid = os.getenv("AGENDA_UID", "82290100")
    max_age    = int(os.getenv("MAX_EVENT_AGE_DAYS", "365"))

    if not api_key:
        raise ValueError("OPENAGENDA_API_KEY not found in .env")

    return {
        "api_key"   : api_key,
        "agenda_uid": agenda_uid,
        "max_age"   : max_age,
        "base_url"  : f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events",
    }


def fetch_page_with_retry(
    session: requests.Session, base_url: str, params: dict, page: int
) -> dict:
    """
    Fetch a single page with manual retry + exponential backoff.

    Catches connection-level failures (resets, chunked encoding errors,
    timeouts) that occur mid-stream and are not retried by urllib3's
    Retry adapter, since those only cover failures before a response
    is received, not failures while reading the response body.
    """
    last_exception = None

    for attempt in range(1, MAX_RETRIES_PER_PAGE + 1):
        try:
            response = session.get(base_url, params=params, timeout=30)

            if response.status_code != 200:
                raise RuntimeError(
                    f"API error {response.status_code}: {response.text[:200]}"
                )

            return response.json()

        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as e:
            last_exception = e
            if attempt < MAX_RETRIES_PER_PAGE:
                wait = RETRY_BACKOFF_BASE ** attempt
                log.warning(
                    f"Page {page}: network error on attempt {attempt}/"
                    f"{MAX_RETRIES_PER_PAGE} ({type(e).__name__}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                log.error(
                    f"Page {page}: failed after {MAX_RETRIES_PER_PAGE} attempts."
                )

    raise last_exception


def fetch_all_events(config: dict) -> list[dict]:
    """Fetch all events from Open Agenda API using cursor-based pagination."""
    session    = build_session()
    all_events = []
    cursor     = None
    page       = 1
    limit      = 100

    while True:
        params = {
            "key"        : config["api_key"],
            "limit"      : limit,
            "relative[0]": "passed",
            "relative[1]": "current",
            "relative[2]": "upcoming",
        }

        if cursor:
            params["after[0]"] = cursor[0]
            params["after[1]"] = cursor[1]
            params["after[2]"] = cursor[2]
            params["after[3]"] = cursor[3]

        data   = fetch_page_with_retry(session, config["base_url"], params, page)
        events = data.get("events", [])
        all_events.extend(events)

        log.info(f"Page {page} — {len(events)} fetched | cumulative: {len(all_events)}")

        if len(events) < limit:
            log.info("Last page reached.")
            break

        cursor = data.get("after")
        page  += 1
        time.sleep(0.5)

    return all_events


def filter_by_date(df: pd.DataFrame, max_age_days: int) -> pd.DataFrame:
    """Keep events within the past max_age_days and up to 12 months ahead."""
    df["firstTiming.begin"] = pd.to_datetime(df["firstTiming.begin"], utc=True)

    lower = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max_age_days)
    upper = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=365)

    filtered = df[
        (df["firstTiming.begin"] >= lower) &
        (df["firstTiming.begin"] <= upper)
    ].copy()

    log.info(f"Date filter: {len(df)} → {len(filtered)} events "
             f"(dropped {len(df) - len(filtered)})")
    return filtered


def select_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Select only fields required by the RAG pipeline."""
    available = [c for c in COLS if c in df.columns]
    missing   = [c for c in COLS if c not in df.columns]

    if missing:
        log.warning(f"Missing columns skipped: {missing}")

    return df[available].copy()


def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules for null values."""
    before = len(df)

    df = df.dropna(subset=["title.fr", "description.fr"])
    log.info(f"Dropped {before - len(df)} rows with null title/description")

    if "location.city" in df.columns:
        df["location.city"] = df.apply(
            lambda row: "En ligne" if row["attendanceMode"] == 2
            else ("Non renseigné" if pd.isnull(row["location.city"])
            else row["location.city"]),
            axis=1
        )
        df["location.city"] = df["location.city"].replace("75015 Paris", "Paris")

    return df


def validate(df: pd.DataFrame) -> None:
    """Run validation assertions before saving."""
    assert len(df) > 0, "No events after filtering"
    assert df["title.fr"].isnull().sum() == 0, "Null titles remain"
    assert df["description.fr"].isnull().sum() == 0, "Null descriptions remain"
    assert df["location.city"].isnull().sum() == 0, "Null cities remain"
    log.info("All validation checks passed.")


def save(df: pd.DataFrame, output_path: Path) -> None:
    """Save cleaned dataset to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    log.info(f"Saved {len(df)} events → {output_path}")


def save_fetch_metadata(metadata_path: Path, max_age_days: int) -> None:
    """
    Save the fetch timestamp alongside the data.

    Downstream unit tests (TestDateFilter) load this timestamp and validate
    event recency relative to it, rather than relative to datetime.now().
    """
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "fetched_at"       : datetime.now(timezone.utc).isoformat(),
        "max_event_age_days": max_age_days,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Fetch metadata saved → {metadata_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset summary for the pipeline log."""
    log.info("=" * 50)
    log.info("SUMMARY")
    log.info(f"  Total events : {len(df)}")
    log.info(f"  Date range   : {df['firstTiming.begin'].min().date()} "
             f"→ {df['firstTiming.begin'].max().date()}")
    log.info("  Top cities   :")
    for city, count in df["location.city"].value_counts().head(5).items():
        log.info(f"    {city:<30} {count}")
    log.info("=" * 50)


def main(force: bool = False) -> None:
    """Main entry point."""
    log.info("Starting fetch_events pipeline...")

    if OUTPUT_PATH.exists() and not force:
        log.info(f"Output already exists: {OUTPUT_PATH}")
        log.info("Use --force to re-fetch. Skipping.")
        return

    config = load_config()
    log.info(f"Agenda UID : {config['agenda_uid']}")
    log.info(f"Max age    : {config['max_age']} days")

    raw_events = fetch_all_events(config)
    log.info(f"Total fetched: {len(raw_events)}")

    df = pd.json_normalize(raw_events)
    df = filter_by_date(df, config["max_age"])
    df = select_fields(df)
    df = clean_nulls(df)
    validate(df)
    print_summary(df)
    save(df, OUTPUT_PATH)
    save_fetch_metadata(METADATA_PATH, config["max_age"])

    log.info("fetch_events pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Paris events from Open Agenda")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch even if output already exists"
    )
    args = parser.parse_args()
    main(force=args.force)
EOF

echo "    src/ingestion/fetch_events.py written."

echo ""
echo "Running src/ingestion/fetch_events.py..."
echo ""

"$ROOT/.venv/bin/python" "$ROOT/src/ingestion/fetch_events.py" $FORCE

echo ""
echo "=============================================="
echo "  Step 1 complete."
echo "  Output: data/raw/events_paris.json"
echo "          data/raw/fetch_metadata.json"
echo "  Next:   bash 02_embed.sh"
echo "=============================================="
echo ""
