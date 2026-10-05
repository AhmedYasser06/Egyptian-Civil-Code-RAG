import json
import os
from pathlib import Path
from statistics import mean

from src.retrieval.retriever import LegalRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_eval.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_results.json"
)

TOP_K = 5


def load_dataset():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_articles, expected_articles, k):
    retrieved = retrieved_articles[:k]

    return int(
        any(
            article in expected_articles
            for article in retrieved
        )
    )


def reciprocal_rank(retrieved_articles, expected_articles):
    for rank, article in enumerate(retrieved_articles, start=1):
        if article in expected_articles:
            return 1.0 / rank

    return 0.0


def main():

    print("=" * 70)
    print("LEGAL RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    dataset = load_dataset()

    print(f"Evaluation questions: {len(dataset)}")
    print(f"Evaluation file: {EVAL_FILE}")

    retriever = LegalRetriever()

    results = []

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        expected = set(item["expected_articles"])

        print("\n" + "-" * 70)
        print(
            f"[{index}/{len(dataset)}] "
            f"{item['id']} "
            f"({item['language']})"
        )
        print(f"Question: {question}")
        print(f"Expected: {sorted(expected)}")

        retrieved = retriever.retrieve(
            query=question,
            top_k=TOP_K,
        )

        retrieved_articles = [
            article["article_number"]
            for article in retrieved
        ]

        print(
            f"Retrieved: {retrieved_articles}"
        )

        scores = [
            article.get("rerank_score")
            for article in retrieved
        ]

        formatted_scores = [
            round(score, 4) if score is not None else None
            for score in scores
        ]

        print(f"Rerank scores: {formatted_scores}")

        result = {
            "id": item["id"],
            "pair_id": item["pair_id"],
            "language": item["language"],
            "question": question,
            "expected_articles": sorted(expected),
            "retrieved_articles": retrieved_articles,
            "recall_at_1": recall_at_k(
                retrieved_articles,
                expected,
                1,
            ),
            "recall_at_3": recall_at_k(
                retrieved_articles,
                expected,
                3,
            ),
            "recall_at_5": recall_at_k(
                retrieved_articles,
                expected,
                5,
            ),
            "mrr": reciprocal_rank(
                retrieved_articles,
                expected,
            ),
            "rerank_scores": formatted_scores,
        }

        results.append(result)

    # ------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------

    recall_1 = mean(
        result["recall_at_1"]
        for result in results
    )

    recall_3 = mean(
        result["recall_at_3"]
        for result in results
    )

    recall_5 = mean(
        result["recall_at_5"]
        for result in results
    )

    mrr = mean(
        result["mrr"]
        for result in results
    )

    # ------------------------------------------------------------
    # Bilingual consistency
    # ------------------------------------------------------------

    pairs = {}

    for result in results:
        pairs.setdefault(
            result["pair_id"],
            {}
        )[result["language"]] = result

    bilingual_matches = 0
    bilingual_total = 0

    bilingual_results = []

    for pair_id, pair in pairs.items():

        if "ar" not in pair or "en" not in pair:
            continue

        ar_top1 = (
            pair["ar"]["retrieved_articles"][0]
            if pair["ar"]["retrieved_articles"]
            else None
        )

        en_top1 = (
            pair["en"]["retrieved_articles"][0]
            if pair["en"]["retrieved_articles"]
            else None
        )

        same = ar_top1 == en_top1

        bilingual_total += 1

        if same:
            bilingual_matches += 1

        bilingual_results.append(
            {
                "pair_id": pair_id,
                "ar_top1": ar_top1,
                "en_top1": en_top1,
                "same_top1": same,
            }
        )

    bilingual_consistency = (
        bilingual_matches / bilingual_total
        if bilingual_total
        else 0.0
    )

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 70)

    print(
        f"Recall@1 : {recall_1:.4f}"
    )

    print(
        f"Recall@3 : {recall_3:.4f}"
    )

    print(
        f"Recall@5 : {recall_5:.4f}"
    )

    print(
        f"MRR      : {mrr:.4f}"
    )

    print(
        f"Bilingual Top-1 Consistency : "
        f"{bilingual_consistency:.4f}"
    )

    print(
        f"Bilingual pairs             : "
        f"{bilingual_matches}/{bilingual_total}"
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    output = {
        "config": {
            "top_k": TOP_K,
            "embedding_model": os.getenv(
                "EMBEDDING_MODEL_ID",
                "BAAI/bge-m3",
            ),
            "reranker_model": os.getenv(
                "RERANKER_MODEL_ID",
                "BAAI/bge-reranker-v2-m3",
            ),
        },
        "metrics": {
            "recall_at_1": recall_1,
            "recall_at_3": recall_3,
            "recall_at_5": recall_5,
            "mrr": mrr,
            "bilingual_top1_consistency": (
                bilingual_consistency
            ),
        },
        "bilingual_results": bilingual_results,
        "results": results,
    }

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

    print("\nResults saved to:")
    print(OUTPUT_FILE)

    print("\nDone.")


if __name__ == "__main__":
    main()