import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

load_dotenv()

DATASET_FILE = Path("data/evaluation/ragas_dataset.json")
RESULTS_FILE = Path("data/evaluation/ragas_results.json")
CSV_FILE = Path("data/evaluation/ragas_results.csv")

MODEL = os.getenv("GROQ_MODEL_ID", "qwen/qwen3.8-27b")
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing")

with open(DATASET_FILE, encoding="utf-8") as f:
    dataset = json.load(f)

print("=" * 70)
print("FULL RAGAS EVALUATION")
print("=" * 70)
print(f"Samples: {len(dataset)}")
print(f"Model: {MODEL}")
print("Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
print()


# ============================================================
# Load previous results
# ============================================================

if RESULTS_FILE.exists():
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)

    print(f"Existing results found: {len(results)}")
else:
    results = []

completed = {item["id"] for item in results}

print(f"Already completed: {len(completed)}")
print(f"Remaining: {len(dataset) - len(completed)}")
print()


# ============================================================
# Groq async client
# ============================================================

groq_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

judge_llm = llm_factory(
    model=MODEL,
    client=groq_client,
    temperature=0,
)


# ============================================================
# Local BGE-M3
# ============================================================

print("Loading BGE-M3...")

judge_embeddings = HuggingFaceEmbeddings(
    model="BAAI/bge-m3",
    device="cpu",
    normalize_embeddings=True,
)


# ============================================================
# Metrics
# ============================================================

faithfulness = Faithfulness(
    llm=judge_llm
)

answer_relevancy = AnswerRelevancy(
    llm=judge_llm,
    embeddings=judge_embeddings,
)

context_precision = ContextPrecision(
    llm=judge_llm
)

context_recall = ContextRecall(
    llm=judge_llm
)

print("RAGAS metrics initialized.")
print()


# ============================================================
# Save helpers
# ============================================================

def save_results():
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    if results:
        fields = [
            "id",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "expected_articles",
            "retrieved_article_numbers",
            "generation_provider",
            "generation_model",
        ]

        with open(CSV_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
                extrasaction="ignore",
            )

            writer.writeheader()

            for item in results:
                writer.writerow(item)


# ============================================================
# Evaluate
# ============================================================

for index, sample in enumerate(dataset, start=1):

    qid = sample["id"]

    if qid in completed:
        continue

    print("=" * 70)
    print(f"[{index}/{len(dataset)}] {qid}")
    print("=" * 70)

    print("Question:")
    print(sample["user_input"])
    print()

    try:

        # ----------------------------------------------------
        # Faithfulness
        # ----------------------------------------------------

        print("  [1/4] Faithfulness...")

        faith = faithfulness.score(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
        )

        faith_score = float(faith.value)

        print(f"        {faith_score:.4f}")


        # ----------------------------------------------------
        # Answer Relevancy
        # ----------------------------------------------------

        print("  [2/4] Answer Relevancy...")

        relevancy = answer_relevancy.score(
            user_input=sample["user_input"],
            response=sample["response"],
        )

        relevancy_score = float(relevancy.value)

        print(f"        {relevancy_score:.4f}")


        # ----------------------------------------------------
        # Context Precision
        # ----------------------------------------------------

        print("  [3/4] Context Precision...")

        precision = context_precision.score(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )

        precision_score = float(precision.value)

        print(f"        {precision_score:.4f}")


        # ----------------------------------------------------
        # Context Recall
        # ----------------------------------------------------

        print("  [4/4] Context Recall...")

        recall = context_recall.score(
            user_input=sample["user_input"],
            reference=sample["reference"],
            retrieved_contexts=sample["retrieved_contexts"],
        )

        recall_score = float(recall.value)

        print(f"        {recall_score:.4f}")


        # ----------------------------------------------------
        # Save immediately
        # ----------------------------------------------------

        result = {
            "id": qid,
            "faithfulness": faith_score,
            "answer_relevancy": relevancy_score,
            "context_precision": precision_score,
            "context_recall": recall_score,
            "expected_articles": sample["expected_articles"],
            "retrieved_article_numbers": sample[
                "retrieved_article_numbers"
            ],
            "generation_provider": sample.get(
                "generation_provider"
            ),
            "generation_model": sample.get(
                "generation_model"
            ),
        }

        results.append(result)

        save_results()

        print()
        print("  SAVED")

    except Exception as e:

        print()
        print("  ERROR:")
        print(f"  {type(e).__name__}: {e}")
        print()

        print("  Existing results were already saved.")
        print("  Re-run the same command to resume.")

        save_results()

        raise

    # Small delay between samples
    time.sleep(2)


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("RAGAS EVALUATION COMPLETE")
print("=" * 70)

print(f"Completed: {len(results)}/{len(dataset)}")


def average(key):
    values = [
        float(x[key])
        for x in results
        if x.get(key) is not None
    ]

    return sum(values) / len(values) if values else 0.0


faith_avg = average("faithfulness")
relevancy_avg = average("answer_relevancy")
precision_avg = average("context_precision")
recall_avg = average("context_recall")


print()
print(f"Faithfulness:      {faith_avg:.4f}")
print(f"Answer Relevancy:  {relevancy_avg:.4f}")
print(f"Context Precision: {precision_avg:.4f}")
print(f"Context Recall:    {recall_avg:.4f}")

print()
print("Faithfulness gate:")
print(
    "PASS"
    if faith_avg >= 0.75
    else "FAIL"
)

print()
print(f"JSON: {RESULTS_FILE}")
print(f"CSV:  {CSV_FILE}")

print("=" * 70)
