# =============================================================================
# src/project_11/processing/chunker.py
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

ROOT       = Path(__file__).resolve().parents[3]
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
