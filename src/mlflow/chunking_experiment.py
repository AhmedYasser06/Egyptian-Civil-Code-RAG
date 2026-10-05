from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.retrieval.reranker import LegalReranker


# ============================================================
# PROJECT CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/final-articles-v2.json"

EXPERIMENT_DIR = ROOT / "data/mlflow/chunking"

RETRIEVAL_EVAL_FILE = ROOT / "data/evaluation/retrieval_eval.json"

RAGAS_RESULTS_FILE = ROOT / "data/evaluation/ragas_results.json"

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

EXPERIMENT_NAME = "Egyptian-Civil-Code-RAG"

EMBEDDING_MODEL = "BAAI/bge-m3"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

TOP_K = 5

CANDIDATE_K = 15


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================
#
# These are NEW experiment outputs.
# The existing final-chunks-bgem3-v2.json is NOT touched.
#
# ============================================================

CONFIGS = [
    {
        "name": "article_256",
        "strategy": "article",
        "max_tokens": 256,
        "overlap": 0,
    },
    {
        "name": "article_384",
        "strategy": "article",
        "max_tokens": 384,
        "overlap": 0,
    },
    {
        "name": "article_512",
        "strategy": "article",
        "max_tokens": 512,
        "overlap": 0,
    },
    {
        "name": "article_768",
        "strategy": "article",
        "max_tokens": 768,
        "overlap": 0,
    },
    {
        "name": "window_512",
        "strategy": "window",
        "max_tokens": 512,
        "overlap": 0,
    },
]


# ============================================================
# HELPERS
# ============================================================


def run_splitter(config: dict) -> tuple[Path, Path]:
    """Run the EXISTING splitter without modifying it."""

    output_dir = EXPERIMENT_DIR / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "chunks.json"
    stats_file = output_dir / "chunk_stats.json"

    command = [
        sys.executable,
        str(ROOT / "src/scripts/text_splitter.py"),
        "--input",
        str(INPUT_FILE),
        "--output",
        str(output_file),
        "--stats",
        str(stats_file),
        "--strategy",
        config["strategy"],
        "--max-tokens",
        str(config["max_tokens"]),
        "--overlap",
        str(config["overlap"]),
        "--context-header",
        "full",
        "--tokenizer",
        EMBEDDING_MODEL,
    ]

    print()
    print("=" * 80)
    print(f"RUNNING SPLITTER: {config['name']}")
    print("=" * 80)

    subprocess.run(command, cwd=ROOT, check=True)

    return output_file, stats_file


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ragas_faithfulness() -> float | None:
    """
    Read already-computed RAGAS results if available.

    We do NOT invent a score.
    If RAGAS results are unavailable, return None.
    """

    if not RAGAS_RESULTS_FILE.exists():
        return None

    data = load_json(RAGAS_RESULTS_FILE)

    values = []

    if isinstance(data, list):
        for item in data:
            value = item.get("faithfulness")

            if isinstance(value, (int, float)):
                values.append(float(value))

    elif isinstance(data, dict):
        results = data.get("results", [])

        for item in results:
            value = item.get("faithfulness")

            if isinstance(value, (int, float)):
                values.append(float(value))

    if not values:
        return None

    return float(np.mean(values))


def article_map():
    articles = load_json(INPUT_FILE)

    return {
        int(article["article_number"]): article
        for article in articles
    }


def load_eval_questions():
    return load_json(RETRIEVAL_EVAL_FILE)


def build_article_passage(article: dict) -> str:
    return (
        f"Article {article['article_number']}\n"
        f"Arabic:\n{article.get('text_ar') or ''}\n"
        f"English:\n{article.get('text_en') or ''}"
    )


# ============================================================
# RETRIEVAL EVALUATION
# ============================================================


def evaluate_retrieval(
    chunks: list[dict],
    articles: dict[int, dict],
    embedding_model: SentenceTransformer,
    reranker: LegalReranker,
    questions: list[dict],
) -> dict:

    texts = [
        chunk["embed_text"]
        for chunk in chunks
        if chunk.get("embed_text")
    ]

    chunk_records = [
        chunk
        for chunk in chunks
        if chunk.get("embed_text")
    ]

    print(f"[EMBED] Encoding {len(texts)} chunks...")

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=16,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    recall_1 = []
    recall_3 = []
    recall_5 = []
    reciprocal_ranks = []

    bilingual = {}

    for index, item in enumerate(questions, start=1):

        question_id = item["id"]
        question = item["question"]

        expected = {
            int(x)
            for x in item.get("expected_articles", [])
        }

        language = item.get("language", "unknown")

        print(
            f"[RETRIEVAL {index}/{len(questions)}] "
            f"{question_id}"
        )

        query_embedding = embedding_model.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        scores = embeddings @ query_embedding

        candidate_indices = np.argsort(scores)[::-1][
            :CANDIDATE_K
        ]

        # Group chunks by article.
        candidate_articles = {}

        for idx in candidate_indices:
            chunk = chunk_records[int(idx)]

            article_number = int(
                chunk["article_number"]
            )

            score = float(scores[int(idx)])

            candidate_articles[article_number] = max(
                candidate_articles.get(article_number, -1.0),
                score,
            )

        # Keep strongest article candidates.
        ranked_articles = sorted(
            candidate_articles.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:CANDIDATE_K]

        # Rerank using the same legal reranker used by the
        # existing production retriever.
        rerank_articles = []

        for article_number, retrieval_score in ranked_articles:

            if article_number not in articles:
                continue

            article = dict(
                articles[article_number]
            )

            article["_retrieval_score"] = retrieval_score

            rerank_articles.append(article)

        reranked = reranker.rerank(
            question,
            rerank_articles,
            TOP_K,
        )

        retrieved = [
            int(article["article_number"])
            for article in reranked
        ]

        hit_1 = bool(expected & set(retrieved[:1]))
        hit_3 = bool(expected & set(retrieved[:3]))
        hit_5 = bool(expected & set(retrieved[:5]))

        recall_1.append(float(hit_1))
        recall_3.append(float(hit_3))
        recall_5.append(float(hit_5))

        rr = 0.0

        for rank, article_number in enumerate(
            retrieved,
            start=1,
        ):
            if article_number in expected:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

        # Store bilingual top-1 article.
        pair_id = question_id.rsplit("_", 1)[0]

        bilingual.setdefault(pair_id, {})[
            language
        ] = retrieved[0] if retrieved else None

    bilingual_pairs = 0
    bilingual_consistent = 0

    for pair in bilingual.values():

        if "ar" in pair and "en" in pair:
            bilingual_pairs += 1

            if pair["ar"] == pair["en"]:
                bilingual_consistent += 1

    return {
        "recall_at_1": float(np.mean(recall_1)),
        "recall_at_3": float(np.mean(recall_3)),
        "recall_at_5": float(np.mean(recall_5)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "bilingual_top1_consistency": (
            bilingual_consistent / bilingual_pairs
            if bilingual_pairs
            else 0.0
        ),
        "bilingual_pairs": bilingual_pairs,
        "evaluation_questions": len(questions),
    }


# ============================================================
# MLflow
# ============================================================


def main():

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    print()
    print("=" * 80)
    print("MLFLOW CHUNKING ABLATION")
    print("=" * 80)

    print(f"Tracking URI : {TRACKING_URI}")
    print(f"Experiment   : {EXPERIMENT_NAME}")
    print(f"Embedding    : {EMBEDDING_MODEL}")
    print(f"Reranker     : {RERANKER_MODEL}")

    print()
    print("Loading BGE-M3...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    print("Loading legal reranker...")

    reranker = LegalReranker(
        model_id=RERANKER_MODEL,
        device="cpu",
    )

    articles = article_map()

    questions = load_eval_questions()

    ragas_faithfulness = load_ragas_faithfulness()

    print()

    if ragas_faithfulness is not None:
        print(
            f"[RAGAS] Existing mean faithfulness: "
            f"{ragas_faithfulness:.4f}"
        )
    else:
        print(
            "[RAGAS] No completed RAGAS scores available. "
            "Will not invent a value."
        )

    all_results = []

    for config in CONFIGS:

        output_file, stats_file = run_splitter(
            config
        )

        chunks = load_json(output_file)
        stats = load_json(stats_file)

        run_name = (
            f"{config['name']}"
            f"__bge-m3"
        )

        with mlflow.start_run(
            run_name=run_name
        ):

            # ------------------------------------------------
            # Parameters
            # ------------------------------------------------

            mlflow.log_params(
                {
                    "strategy": config["strategy"],
                    "chunk_size": config["max_tokens"],
                    "overlap": config["overlap"],
                    "context_header": "full",
                    "embedding_model": EMBEDDING_MODEL,
                    "reranker_model": RERANKER_MODEL,
                    "top_k": TOP_K,
                    "candidate_k": CANDIDATE_K,
                    "tokenizer": EMBEDDING_MODEL,
                }
            )

            # ------------------------------------------------
            # Chunk statistics
            # ------------------------------------------------

            mlflow.log_metrics(
                {
                    "total_chunks": stats["total_chunks"],
                    "articles_split": stats["articles_split"],
                    "token_min": stats["tokens"]["min"],
                    "token_median": stats["tokens"]["median"],
                    "token_mean": stats["tokens"]["mean"],
                    "token_p95": stats["tokens"]["p95"],
                    "token_max": stats["tokens"]["max"],
                }
            )

            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            retrieval_metrics = evaluate_retrieval(
                chunks=chunks,
                articles=articles,
                embedding_model=embedding_model,
                reranker=reranker,
                questions=questions,
            )

            mlflow.log_metrics(
                {
                    "recall_at_1": retrieval_metrics[
                        "recall_at_1"
                    ],
                    "recall_at_3": retrieval_metrics[
                        "recall_at_3"
                    ],
                    "recall_at_5": retrieval_metrics[
                        "recall_at_5"
                    ],
                    "mrr": retrieval_metrics["mrr"],
                    "bilingual_top1_consistency": retrieval_metrics[
                        "bilingual_top1_consistency"
                    ],
                }
            )

            # ------------------------------------------------
            # RAGAS
            # ------------------------------------------------

            if ragas_faithfulness is not None:
                mlflow.log_metric(
                    "ragas_faithfulness",
                    ragas_faithfulness,
                )

                mlflow.set_tag(
                    "ragas_status",
                    "existing_evaluation",
                )

            else:
                mlflow.set_tag(
                    "ragas_status",
                    "pending_full_evaluation",
                )

            # ------------------------------------------------
            # Artifacts
            # ------------------------------------------------

            mlflow.log_artifact(
                str(stats_file),
                artifact_path="chunking",
            )

            # Do not log the huge chunk file by default.
            # Save the configuration/summary instead.

            summary = {
                "config": config,
                "embedding_model": EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
                "chunk_stats": stats,
                "retrieval_metrics": retrieval_metrics,
                "ragas_faithfulness": ragas_faithfulness,
                "output_file": str(output_file),
            }

            summary_file = (
                output_file.parent
                / "experiment_summary.json"
            )

            summary_file.write_text(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            mlflow.log_artifact(
                str(summary_file),
                artifact_path="results",
            )

            run_id = mlflow.active_run().info.run_id

            result = {
                "run_id": run_id,
                "name": run_name,
                **config,
                **retrieval_metrics,
                "ragas_faithfulness": ragas_faithfulness,
                "total_chunks": stats["total_chunks"],
                "articles_split": stats["articles_split"],
            }

            all_results.append(result)

            print()
            print(
                f"[MLFLOW] Run: {run_name}"
            )
            print(
                f"Recall@1 = "
                f"{retrieval_metrics['recall_at_1']:.4f}"
            )
            print(
                f"Recall@3 = "
                f"{retrieval_metrics['recall_at_3']:.4f}"
            )
            print(
                f"Recall@5 = "
                f"{retrieval_metrics['recall_at_5']:.4f}"
            )
            print(
                f"MRR      = "
                f"{retrieval_metrics['mrr']:.4f}"
            )

    # ========================================================
    # Comparison artifact
    # ========================================================

    results_file = (
        EXPERIMENT_DIR
        / "mlflow_chunking_comparison.json"
    )

    results_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_file.write_text(
        json.dumps(
            all_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Best configuration based on Recall@5 then MRR
    # --------------------------------------------------------

    best = max(
        all_results,
        key=lambda x: (
            x["recall_at_5"],
            x["mrr"],
            x["bilingual_top1_consistency"],
        ),
    )

    best_file = (
        EXPERIMENT_DIR
        / "best_chunking_config.json"
    )

    best_file.write_text(
        json.dumps(
            best,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("MLFLOW CHUNKING EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"Runs created: {len(all_results)}"
    )

    print()
    print("BEST CONFIGURATION")
    print("-" * 80)

    for key in [
        "name",
        "strategy",
        "max_tokens",
        "overlap",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "bilingual_top1_consistency",
        "ragas_faithfulness",
    ]:
        print(
            f"{key}: {best.get(key)}"
        )

    print()
    print(
        f"Comparison: {results_file}"
    )

    print(
        f"Best config: {best_file}"
    )


if __name__ == "__main__":
    main()
