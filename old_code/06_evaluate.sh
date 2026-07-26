#!/bin/bash

# =============================================================================
# 06_evaluate.sh — Build annotated test dataset and run RAGAS evaluation
# =============================================================================
# Usage:
#   bash 06_evaluate.sh
# =============================================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "=============================================="
echo "  Step 6 — Evaluation"
echo "=============================================="
echo ""

# Guards
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
    "data/processed/metadata.json"; do
    if [ ! -f "$ROOT/$f" ]; then
        echo "ERROR: Missing $f — run pipeline.sh --skip-run first."
        exit 1
    fi
done

# --- Write evaluation script ---
echo "Writing src/rag/evaluate.py..."
mkdir -p "$ROOT/src/rag"

cat > "$ROOT/src/rag/evaluate.py" << 'EOF'
# =============================================================================
# src/rag/evaluate.py
# =============================================================================
# Builds an annotated test dataset and runs RAGAS evaluation.
#
# Step 1: Run RAG pipeline on predefined questions
# Step 2: Save questions + generated answers + contexts to CSV
# Step 3: Run RAGAS metrics
# Step 4: Save evaluation results
#
# Rate limit handling: every Mistral API call (embeddings AND chat
# completions) is wrapped in retry logic with exponential backoff.
# The free / shared tier occasionally returns 429 service_tier_capacity_
# exceeded under sustained sequential load — this is expected and the
# retry wrapper absorbs it automatically rather than crashing the run.
#
# Usage:
#   poetry run python src/rag/evaluate.py
# =============================================================================

import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models import SDKError
import os

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
DATASET_PATH = ROOT / "data" / "processed" / "test_dataset.csv"
RESULTS_PATH = ROOT / "data" / "processed" / "ragas_results.csv"

# --- Config ---
load_dotenv(dotenv_path=ENV_PATH)
MISTRAL_KEY     = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL       = "mistral-small-latest"
LLM_TEMPERATURE = 0.1
TOP_K           = 5

# --- Retry configuration ---
MAX_RETRIES        = 5
RETRY_BACKOFF_BASE = 2   # seconds; doubles each attempt: 2, 4, 8, 16, 32
CALL_SLEEP         = 1.0 # baseline pause between successful API calls

# --- Annotated test dataset ---
# reference_answer: manually written ideal answer based on known event data
TEST_QUESTIONS = [
    # --- Specific event queries ---
    {
        "id"              : "Q01",
        "question"        : "Qu'est-ce que le concert Onestage à Paris ?",
        "reference_answer": "Le concert Onestage est un concert prévu le vendredi 27 mars à 16h00 à l'Église de La Madeleine à Paris.",
        "category"        : "specific_event"
    },
    {
        "id"              : "Q02",
        "question"        : "Parlez-moi du concert des 4 Saisons de Vivaldi à Paris.",
        "reference_answer": "Les 4 Saisons de Vivaldi et la Petite Musique de Nuit de Mozart sont interprétées à l'Église de La Madeleine à Paris. Plusieurs dates sont disponibles dont le mercredi 11 mars et le mardi 7 avril à 20h00.",
        "category"        : "specific_event"
    },
    {
        "id"              : "Q03",
        "question"        : "Qu'est-ce que le Concert Aria Baroque ?",
        "reference_answer": "Le Concert Aria Baroque propose des œuvres de Vivaldi, Haendel et Bach. Il a lieu le samedi 14 mars à 16h00 à l'Église Sainte-Elisabeth de Hongrie à Paris.",
        "category"        : "specific_event"
    },
    {
        "id"              : "Q04",
        "question"        : "Quand a lieu le Requiem de Mozart et Boléro de Ravel ?",
        "reference_answer": "Le Requiem de Mozart et Boléro de Ravel a lieu le samedi 11 avril 2025 à 20h45 à l'Église de La Madeleine à Paris.",
        "category"        : "specific_event"
    },
    {
        "id"              : "Q05",
        "question"        : "Qu'est-ce que le concert spirituel à Paris ?",
        "reference_answer": "Le Concert spirituel se tient les dimanches après la messe de 11h à la Basilique Sainte-Clotilde à Paris, du 30 novembre au 21 décembre 2025.",
        "category"        : "specific_event"
    },
    {
        "id"              : "Q06",
        "question"        : "Qu'est-ce que le concert de l'Orchestre de la Bastille ?",
        "reference_answer": "Le Concert de l'Orchestre de la Bastille dirigé par Émilie Postel-Vinay a lieu le samedi 5 avril 2025 à 20h30 à l'Église Saint Leu Saint Gilles à Paris.",
        "category"        : "specific_event"
    },
    # --- Date-based queries ---
    {
        "id"              : "Q07",
        "question"        : "Quels concerts ont lieu en avril à Paris ?",
        "reference_answer": "En avril à Paris, on peut assister notamment aux 4 Saisons de Vivaldi à l'Église de La Madeleine le 7 avril, et au Requiem de Mozart et Boléro de Ravel le 11 avril.",
        "category"        : "date_based"
    },
    {
        "id"              : "Q08",
        "question"        : "Quels événements se déroulent en mars à Paris ?",
        "reference_answer": "En mars à Paris, plusieurs concerts sont prévus dont le Concert Aria Baroque le 14 mars, les 4 Saisons de Vivaldi le 11 mars, et le concert Onestage le 27 mars.",
        "category"        : "date_based"
    },
    {
        "id"              : "Q09",
        "question"        : "Que faire ce weekend à Paris ?",
        "reference_answer": "Ce weekend à Paris, vous pouvez assister à plusieurs événements culturels disponibles dans notre agenda.",
        "category"        : "date_based"
    },
    {
        "id"              : "Q10",
        "question"        : "Y a-t-il des événements culturels en juillet 2025 à Paris ?",
        "reference_answer": "En juillet 2025 à Paris, on peut assister aux 4 Saisons de Vivaldi et la Petite Musique de Nuit de Mozart le 5 juillet à 20h45.",
        "category"        : "date_based"
    },
    {
        "id"              : "Q11",
        "question"        : "Quels événements ont lieu en décembre 2025 à Paris ?",
        "reference_answer": "En décembre 2025 à Paris, le Concert spirituel se tient les dimanches à la Basilique Sainte-Clotilde jusqu'au 21 décembre.",
        "category"        : "date_based"
    },
    # --- Category-based queries ---
    {
        "id"              : "Q12",
        "question"        : "Quels concerts de musique classique ont lieu à Paris ?",
        "reference_answer": "Plusieurs concerts de musique classique sont disponibles à Paris dont les 4 Saisons de Vivaldi à l'Église de La Madeleine, le Concert Aria Baroque à l'Église Sainte-Elisabeth de Hongrie, et le Requiem de Mozart.",
        "category"        : "category_based"
    },
    {
        "id"              : "Q13",
        "question"        : "Y a-t-il des événements religieux à Paris ?",
        "reference_answer": "Oui, plusieurs événements religieux ont lieu à Paris notamment le Concert spirituel à la Basilique Sainte-Clotilde et des pèlerinages paroissiaux.",
        "category"        : "category_based"
    },
    {
        "id"              : "Q14",
        "question"        : "Y a-t-il des événements gratuits à Paris ?",
        "reference_answer": "Je ne dispose pas d'informations suffisantes dans le contexte pour confirmer quels événements sont gratuits.",
        "category"        : "category_based"
    },
    {
        "id"              : "Q15",
        "question"        : "Quels événements musicaux sont disponibles à Paris ?",
        "reference_answer": "De nombreux événements musicaux sont disponibles à Paris incluant des concerts de musique classique, de musique baroque et des récitals d'orgue.",
        "category"        : "category_based"
    },
    {
        "id"              : "Q16",
        "question"        : "Y a-t-il des récitals d'orgue à Paris ?",
        "reference_answer": "Oui, des récitals d'orgue sont organisés à Paris dans le cadre des Dimanches Musicaux.",
        "category"        : "category_based"
    },
    # --- Venue-based queries ---
    {
        "id"              : "Q17",
        "question"        : "Quels événements ont lieu à l'Église de La Madeleine ?",
        "reference_answer": "L'Église de La Madeleine accueille plusieurs concerts dont les 4 Saisons de Vivaldi et la Petite Musique de Nuit de Mozart, ainsi que le Requiem de Mozart et Boléro de Ravel.",
        "category"        : "venue_based"
    },
    {
        "id"              : "Q18",
        "question"        : "Que se passe-t-il à la Basilique Sainte-Clotilde à Paris ?",
        "reference_answer": "La Basilique Sainte-Clotilde accueille le Concert spirituel les dimanches après la messe de 11h, du 30 novembre au 21 décembre 2025.",
        "category"        : "venue_based"
    },
    {
        "id"              : "Q19",
        "question"        : "Quels événements ont lieu à l'Église Sainte-Elisabeth de Hongrie ?",
        "reference_answer": "L'Église Sainte-Elisabeth de Hongrie accueille le Concert Aria Baroque avec des œuvres de Vivaldi, Haendel et Bach le samedi 14 mars à 16h00.",
        "category"        : "venue_based"
    },
    {
        "id"              : "Q20",
        "question"        : "Y a-t-il des événements à l'Église Saint Leu Saint Gilles ?",
        "reference_answer": "L'Église Saint Leu Saint Gilles accueille le concert de l'Orchestre de la Bastille dirigé par Émilie Postel-Vinay le samedi 5 avril 2025 à 20h30.",
        "category"        : "venue_based"
    },
    # --- Out of scope queries ---
    {
        "id"              : "Q21",
        "question"        : "Quel est le meilleur restaurant à Paris ?",
        "reference_answer": "Je ne sais pas. Cette information ne figure pas dans ma base de données d'événements culturels.",
        "category"        : "out_of_scope"
    },
    {
        "id"              : "Q22",
        "question"        : "Quel est ton nom ?",
        "reference_answer": "Je ne sais pas. Je suis un assistant spécialisé dans les événements culturels à Paris.",
        "category"        : "out_of_scope"
    },
    {
        "id"              : "Q23",
        "question"        : "Quelle est la météo à Paris aujourd'hui ?",
        "reference_answer": "Je ne sais pas. Je suis spécialisé dans les événements culturels à Paris et ne dispose pas d'informations météorologiques.",
        "category"        : "out_of_scope"
    },
    {
        "id"              : "Q24",
        "question"        : "Comment aller à Paris depuis Lyon ?",
        "reference_answer": "Je ne sais pas. Je suis spécialisé dans les événements culturels à Paris.",
        "category"        : "out_of_scope"
    },
    {
        "id"              : "Q25",
        "question"        : "Quel est le prix d'un billet de métro à Paris ?",
        "reference_answer": "Je ne sais pas. Cette information ne figure pas dans ma base de données d'événements culturels.",
        "category"        : "out_of_scope"
    },
    # --- No match queries ---
    {
        "id"              : "Q26",
        "question"        : "Y a-t-il des événements de corrida à Paris ?",
        "reference_answer": "Je ne sais pas. Aucun événement de corrida n'est référencé dans ma base de données.",
        "category"        : "no_match"
    },
    {
        "id"              : "Q27",
        "question"        : "Y a-t-il des matchs de football à Paris ?",
        "reference_answer": "Je ne sais pas. Aucun match de football n'est référencé dans ma base de données d'événements culturels.",
        "category"        : "no_match"
    },
    {
        "id"              : "Q28",
        "question"        : "Y a-t-il des festivals de jazz à Paris ?",
        "reference_answer": "Je ne sais pas. Aucun festival de jazz n'est référencé dans ma base de données pour le moment.",
        "category"        : "no_match"
    },
    {
        "id"              : "Q29",
        "question"        : "Y a-t-il des spectacles de cirque à Paris ?",
        "reference_answer": "Je ne sais pas. Aucun spectacle de cirque n'est référencé dans ma base de données d'événements culturels.",
        "category"        : "no_match"
    },
    {
        "id"              : "Q30",
        "question"        : "Y a-t-il des expositions de peinture contemporaine à Paris ?",
        "reference_answer": "Je ne sais pas. Aucune exposition de peinture contemporaine n'est référencée dans ma base de données pour le moment.",
        "category"        : "no_match"
    },
]


def load_resources():
    """Load FAISS index, metadata and Mistral client."""
    client = Mistral(api_key=MISTRAL_KEY)
    index  = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return client, index, metadata


def call_with_retry(fn, *args, **kwargs):
    """
    Call any Mistral API function with retry + exponential backoff on
    429 rate limit / service_tier_capacity_exceeded errors.

    The shared/free Mistral tier occasionally throttles sustained
    sequential traffic. Rather than crashing the entire 30-question
    evaluation run on a single transient 429, this wrapper pauses and
    retries, which resolves the vast majority of these errors.
    """
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except SDKError as e:
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
                # Non-rate-limit SDK error — don't retry, fail fast
                raise

    raise last_exception


def embed_query(client: Mistral, text: str) -> np.ndarray:
    def _call():
        return client.embeddings.create(model=EMBEDDING_MODEL, inputs=[text])

    response = call_with_retry(_call)
    return np.array([response.data[0].embedding], dtype="float32")


def retrieve(client: Mistral, index, metadata, query: str) -> list[dict]:
    query_vec          = embed_query(client, query)
    distances, indices = index.search(query_vec, TOP_K)
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


def generate(client: Mistral, query: str, context_events: list[dict]) -> str:
    context = "\n\n".join([
        f"Événement {i+1}:\n{event['text']}"
        for i, event in enumerate(context_events)
    ])
    user_message = f"""Contexte des événements disponibles :
{context}

Question de l'utilisateur : {query}

Réponds en te basant uniquement sur les événements fournis ci-dessus."""

    def _call():
        return client.chat.complete(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ]
        )

    response = call_with_retry(_call)
    return response.choices[0].message.content


def run_rag_on_dataset(
    client, index, metadata, questions: list[dict]
) -> list[dict]:
    """Run RAG pipeline on all test questions."""
    results = []
    total   = len(questions)

    for i, q in enumerate(questions, 1):
        log.info(f"Processing {i}/{total}: {q['id']} — {q['question'][:50]}...")

        retrieved = retrieve(client, index, metadata, q["question"])
        time.sleep(CALL_SLEEP)
        answer    = generate(client, q["question"], retrieved)

        results.append({
            "id"              : q["id"],
            "question"        : q["question"],
            "answer"          : answer,
            "reference_answer": q["reference_answer"],
            "category"        : q["category"],
            "contexts"        : json.dumps(
                [r["text"] for r in retrieved],
                ensure_ascii=False
            ),
            "source_uids"     : json.dumps(
                [r["uid"] for r in retrieved]
            ),
        })
        time.sleep(CALL_SLEEP)

    return results


def save_dataset(results: list[dict]) -> None:
    """Save annotated dataset to CSV."""
    df = pd.DataFrame(results)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8")
    log.info(f"Dataset saved → {DATASET_PATH} ({len(df)} rows)")


def run_ragas(results: list[dict]) -> None:
    """Run RAGAS evaluation on the generated answers."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from langchain_mistralai.chat_models import ChatMistralAI
        from langchain_mistralai.embeddings import MistralAIEmbeddings

        log.info("Running RAGAS evaluation...")

        eval_data = {
            "question"    : [r["question"] for r in results],
            "answer"      : [r["answer"] for r in results],
            "contexts"    : [json.loads(r["contexts"]) for r in results],
            "ground_truth": [r["reference_answer"] for r in results],
        }

        dataset = Dataset.from_dict(eval_data)

        llm        = ChatMistralAI(
            mistral_api_key=MISTRAL_KEY,
            model="mistral-small-latest",
            temperature=0.1
        )
        embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_KEY)

        ragas_results = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=llm,
            embeddings=embeddings
        )

        results_df = ragas_results.to_pandas()
        results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")

        log.info("=" * 50)
        log.info("RAGAS EVALUATION RESULTS")
        avg = results_df.mean(numeric_only=True)
        for metric, score in avg.items():
            log.info(f"  {metric:<25} : {score:.4f}")
        log.info("=" * 50)
        log.info(f"Full results saved → {RESULTS_PATH}")

    except Exception as e:
        log.warning(f"RAGAS evaluation failed: {e}")
        log.warning("Skipping RAGAS. Dataset saved successfully.")


def main() -> None:
    log.info("Starting evaluation pipeline...")

    client, index, metadata = load_resources()
    log.info(f"Resources loaded: {index.ntotal} vectors")

    results = run_rag_on_dataset(client, index, metadata, TEST_QUESTIONS)
    save_dataset(results)
    run_ragas(results)

    log.info("Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
EOF

echo "    src/rag/evaluate.py written."

# --- Run evaluation ---
echo ""
echo "Running evaluation pipeline..."
echo "(This will make ~30 API calls with retry/backoff — approx 2-5 minutes)"
echo ""

"$ROOT/.venv/bin/python" "$ROOT/src/rag/evaluate.py"

echo ""
echo "=============================================="
echo "  Step 6 complete."
echo "  Dataset : data/processed/test_dataset.csv"
echo "  Results : data/processed/ragas_results.csv"
echo "=============================================="
echo ""