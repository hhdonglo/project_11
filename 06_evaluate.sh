#!/bin/bash

# =============================================================================
# 06_evaluate.sh — Build annotated test dataset and run RAGAS evaluation
# =============================================================================
# The evaluation script now imports ask() from src/rag/langchain_chain.py —
# the SAME LangChain + Mistral + FAISS chain used by the live Streamlit
# demo. There is no separate hand-rolled retrieve/generate logic in this
# file anymore, so RAGAS is evaluating exactly what the chatbot serves to
# real users, not a parallel implementation that could silently drift.
#
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
        echo "ERROR: Missing $f — run bash 03_index.sh first."
        exit 1
    fi
done

if [ ! -f "$ROOT/src/rag/langchain_chain.py" ]; then
    echo "ERROR: src/rag/langchain_chain.py not found."
    echo "Run: bash 07_langchain_integration.sh first."
    exit 1
fi

# --- Write evaluation script ---
echo "Writing src/rag/evaluate.py..."
mkdir -p "$ROOT/src/rag"
touch "$ROOT/src/rag/__init__.py"

cat > "$ROOT/src/rag/evaluate.py" << 'EOF'
# =============================================================================
# src/rag/evaluate.py
# =============================================================================
# Builds an annotated test dataset and runs RAGAS evaluation.
#
# Uses ask() from src/rag/langchain_chain.py — the SAME LangChain +
# Mistral + FAISS chain that powers the live Streamlit chatbot. This
# guarantees RAGAS evaluates exactly what users interact with, rather
# than a second, independently-maintained implementation that could
# silently drift out of sync with the demo.
#
# Step 1: Run the canonical LangChain RAG chain on predefined questions
# Step 2: Save questions + generated answers + contexts to CSV
# Step 3: Run RAGAS metrics
# Step 4: Save evaluation results
#
# Rate limit handling is inherited from langchain_chain.ask(), which wraps
# every Mistral call (embeddings AND chat completions) in retry logic with
# exponential backoff. This script adds an additional pacing sleep between
# questions on top of that, since 30 sequential questions is enough
# sustained load to occasionally trigger 429 service_tier_capacity_exceeded
# on the free/shared tier even with per-call retries.
#
# Usage:
#   poetry run python src/rag/evaluate.py
# =============================================================================

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# --- Make the project root importable ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.langchain_chain import ask  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# --- Paths ---
DATASET_PATH = ROOT / "data" / "processed" / "test_dataset.csv"
RESULTS_PATH = ROOT / "data" / "processed" / "ragas_results.csv"

# --- Pacing between questions (on top of langchain_chain's own retries) ---
CALL_SLEEP = 1.5

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


def run_rag_on_dataset(questions: list[dict]) -> list[dict]:
    """
    Run the canonical LangChain RAG chain (src.rag.langchain_chain.ask)
    on all test questions.
    """
    results = []
    total   = len(questions)

    for i, q in enumerate(questions, 1):
        log.info(f"Processing {i}/{total}: {q['id']} — {q['question'][:50]}...")

        result = ask(q["question"])

        results.append({
            "id"              : q["id"],
            "question"        : q["question"],
            "answer"          : result["answer"],
            "reference_answer": q["reference_answer"],
            "category"        : q["category"],
            "contexts"        : json.dumps(
                [s["text"] for s in result["sources"]],
                ensure_ascii=False
            ),
            "source_uids"     : json.dumps(
                [s["uid"] for s in result["sources"]]
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
        import os

        log.info("Running RAGAS evaluation...")

        mistral_key = os.getenv("MISTRAL_API_KEY")

        eval_data = {
            "question"    : [r["question"] for r in results],
            "answer"      : [r["answer"] for r in results],
            "contexts"    : [json.loads(r["contexts"]) for r in results],
            "ground_truth": [r["reference_answer"] for r in results],
        }

        dataset = Dataset.from_dict(eval_data)

        llm        = ChatMistralAI(
            mistral_api_key=mistral_key,
            model="mistral-small-latest",
            temperature=0.1
        )
        embeddings = MistralAIEmbeddings(mistral_api_key=mistral_key)

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
    log.info("Starting evaluation pipeline (canonical LangChain chain)...")

    results = run_rag_on_dataset(TEST_QUESTIONS)
    save_dataset(results)
    run_ragas(results)

    log.info("Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
EOF

echo "    src/rag/evaluate.py written (now backed by LangChain)."

# --- Run evaluation ---
echo ""
echo "Running evaluation pipeline..."
echo "(This will make ~30+ API calls with retry/backoff — approx 2-6 minutes)"
echo ""

"$ROOT/.venv/bin/python" "$ROOT/src/rag/evaluate.py"

echo ""
echo "=============================================="
echo "  Step 6 complete."
echo "  Dataset : data/processed/test_dataset.csv"
echo "  Results : data/processed/ragas_results.csv"
echo "=============================================="
echo ""
