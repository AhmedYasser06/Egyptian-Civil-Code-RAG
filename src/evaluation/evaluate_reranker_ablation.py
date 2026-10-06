import json
import os
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from src.retrieval.retriever import LegalRetriever

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = PROJECT_ROOT / "data/evaluation/retrieval_eval.json"
OUTPUT_FILE = PROJECT_ROOT / "data/evaluation/reranker_ablation_results.json"

TOP_K = 5
CANDIDATE_K = max(TOP_K * 4, 15)


def article_numbers(results):
    return [
        int(result["article_number"])
        for result in results
    ]


def reciprocal_rank(retrieved, expected):
    for rank, article in enumerate(retrieved, start=1):
        if article in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved, expected, k):
    retrieved_k = set(retrieved[:k])
    expected_set = set(expected)

    if not expected_set:
        return 0.0

    return len(retrieved_k & expected_set) / len(expected_set)


def top1_consistency(ar_results, en_results):
    ar_top1 = ar_results[0] if ar_results else None
    en_top1 = en_results[0] if en_results else None

    return (
        ar_top1 is not None
        and en_top1 is not None
        and ar_top1 == en_top1
    )


def run_baseline(retriever, question):
    """
    Baseline:
        BGE-M3 embedding
        -> Qdrant
        -> article grouping
        -> NO reranker

    We intentionally call the existing retriever's semantic retrieval
    and grouping methods instead of changing production retrieval code.
    """

    article_number = retriever._detect_article_number(question)

    if article_number is not None:
        exact_results = retriever._retrieve_article(article_number)

        if exact_results:
            return retriever.group_by_article(exact_results)[:TOP_K]

    results = retriever._retrieve_semantic(
        query=question,
        top_k=CANDIDATE_K,
    )

    articles = retriever.group_by_article(results)

    # group_by_article preserves the semantic retrieval ordering.
    return articles[:TOP_K]


def run_reranked(retriever, question):
    """
    Production:
        BGE-M3
        -> Qdrant
        -> article grouping
        -> BGE-reranker-v2-m3
    """

    return retriever.retrieve(
        query=question,
        top_k=TOP_K,
    )


def evaluate_configuration(retriever, data, mode):
    rows = []

    for item in data:
        question = item["question"]
        expected = item["expected_articles"]

        if mode == "baseline":
            results = run_baseline(
                retriever,
                question,
            )
        else:
            results = run_reranked(
                retriever,
                question,
            )

        retrieved = article_numbers(results)

        rows.append(
            {
                "id": item["id"],
                "language": item.get("language"),
                "question": question,
                "expected_articles": expected,
                "retrieved_article_numbers": retrieved,
                "recall_at_1": recall_at_k(
                    retrieved,
                    expected,
                    1,
                ),
                "recall_at_3": recall_at_k(
                    retrieved,
                    expected,
                    3,
                ),
                "recall_at_5": recall_at_k(
                    retrieved,
                    expected,
                    5,
                ),
                "mrr": reciprocal_rank(
                    retrieved,
                    expected,
                ),
            }
        )

    return rows


def calculate_metrics(rows):
    return {
        "recall_at_1": mean(
            row["recall_at_1"]
            for row in rows
        ),
        "recall_at_3": mean(
            row["recall_at_3"]
            for row in rows
        ),
        "recall_at_5": mean(
            row["recall_at_5"]
            for row in rows
        ),
        "mrr": mean(
            row["mrr"]
            for row in rows
        ),
    }


def calculate_bilingual_consistency(data, rows):
    by_pair = {}

    for item, row in zip(data, rows):
        pair_id = item.get("pair_id")

        if pair_id is None:
            continue

        by_pair.setdefault(
            pair_id,
            {}
        )[item.get("language")] = row

    if not by_pair:
        return None

    consistent = 0

    for pair in by_pair.values():
        ar = pair.get("ar")
        en = pair.get("en")

        if not ar or not en:
            continue

        ar_top1 = (
            ar["retrieved_article_numbers"][0]
            if ar["retrieved_article_numbers"]
            else None
        )

        en_top1 = (
            en["retrieved_article_numbers"][0]
            if en["retrieved_article_numbers"]
            else None
        )

        if ar_top1 == en_top1:
            consistent += 1

    return consistent / len(by_pair)


def print_metrics(name, metrics, consistency):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Recall@1: {metrics['recall_at_1']:.4f}"
    )
    print(
        f"Recall@3: {metrics['recall_at_3']:.4f}"
    )
    print(
        f"Recall@5: {metrics['recall_at_5']:.4f}"
    )
    print(
        f"MRR:      {metrics['mrr']:.4f}"
    )

    if consistency is not None:
        print(
            "Bilingual Top-1 consistency: "
            f"{consistency:.4f}"
        )


def main():
    print("=" * 70)
    print("RERANKER ABLATION")
    print("=" * 70)

    with open(
        EVAL_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    print(
        f"[INFO] Evaluation questions: {len(data)}"
    )

    print(
        "[INFO] Loading LegalRetriever "
        "(BGE-M3 + Qdrant + local reranker)..."
    )

    retriever = LegalRetriever()

    print()
    print("[INFO] Running BASELINE...")
    baseline_rows = evaluate_configuration(
        retriever,
        data,
        mode="baseline",
    )

    baseline_metrics = calculate_metrics(
        baseline_rows
    )

    baseline_consistency = (
        calculate_bilingual_consistency(
            data,
            baseline_rows,
        )
    )

    print_metrics(
        "BASELINE — BGE-M3 + Qdrant",
        baseline_metrics,
        baseline_consistency,
    )

    print()
    print("[INFO] Running RERANKED...")
    reranked_rows = evaluate_configuration(
        retriever,
        data,
        mode="reranked",
    )

    reranked_metrics = calculate_metrics(
        reranked_rows
    )

    reranked_consistency = (
        calculate_bilingual_consistency(
            data,
            reranked_rows,
        )
    )

    print_metrics(
        "RERANKED — BGE-M3 + Qdrant + BGE-Reranker-v2-M3",
        reranked_metrics,
        reranked_consistency,
    )

    print()
    print("=" * 70)
    print("IMPROVEMENT")
    print("=" * 70)

    for metric in (
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
    ):
        improvement = (
            reranked_metrics[metric]
            - baseline_metrics[metric]
        )

        print(
            f"{metric}: "
            f"{improvement:+.4f}"
        )

    if (
        baseline_consistency is not None
        and reranked_consistency is not None
    ):
        print(
            "bilingual_top1_consistency: "
            f"{reranked_consistency - baseline_consistency:+.4f}"
        )

    output = {
        "config": {
            "embedding_model": os.getenv(
                "EMBEDDING_MODEL_ID",
                "BAAI/bge-m3",
            ),
            "reranker_model": os.getenv(
                "RERANKER_MODEL_ID",
                "BAAI/bge-reranker-v2-m3",
            ),
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "device": os.getenv(
                "RERANKER_DEVICE",
                "cpu",
            ),
        },
        "baseline": {
            "description": (
                "BGE-M3 + Qdrant without reranking"
            ),
            "metrics": baseline_metrics,
            "bilingual_top1_consistency": (
                baseline_consistency
            ),
            "results": baseline_rows,
        },
        "reranked": {
            "description": (
                "BGE-M3 + Qdrant + "
                "BAAI/bge-reranker-v2-m3"
            ),
            "metrics": reranked_metrics,
            "bilingual_top1_consistency": (
                reranked_consistency
            ),
            "results": reranked_rows,
        },
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"[OK] Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
