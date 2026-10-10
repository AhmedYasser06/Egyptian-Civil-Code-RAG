import json
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "reranker_ablation_results.json"
)

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "Egyptian-Civil-Code-RAG"


def main():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    config = data["config"]
    baseline = data["baseline"]
    reranked = data["reranked"]

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="reranker_ablation") as run:

        # Configuration
        mlflow.log_param(
            "embedding_model",
            config["embedding_model"],
        )
        mlflow.log_param(
            "reranker_model",
            config["reranker_model"],
        )
        mlflow.log_param("top_k", config["top_k"])
        mlflow.log_param("candidate_k", config["candidate_k"])
        mlflow.log_param("device", config["device"])

        mlflow.log_param("evaluation_questions", 20)
        mlflow.log_param("evaluation_pairs", 10)

        # Baseline metrics
        mlflow.log_metrics({
            "baseline_recall_at_1": baseline["metrics"]["recall_at_1"],
            "baseline_recall_at_3": baseline["metrics"]["recall_at_3"],
            "baseline_recall_at_5": baseline["metrics"]["recall_at_5"],
            "baseline_mrr": baseline["metrics"]["mrr"],
            "baseline_bilingual_top1": baseline["bilingual_top1_consistency"],
        })

        # Reranked metrics
        mlflow.log_metrics({
            "reranked_recall_at_1": reranked["metrics"]["recall_at_1"],
            "reranked_recall_at_3": reranked["metrics"]["recall_at_3"],
            "reranked_recall_at_5": reranked["metrics"]["recall_at_5"],
            "reranked_mrr": reranked["metrics"]["mrr"],
            "reranked_bilingual_top1": reranked["bilingual_top1_consistency"],
        })

        # Improvement
        mlflow.log_metrics({
            "improvement_recall_at_1": (
                reranked["metrics"]["recall_at_1"]
                - baseline["metrics"]["recall_at_1"]
            ),
            "improvement_mrr": (
                reranked["metrics"]["mrr"]
                - baseline["metrics"]["mrr"]
            ),
            "improvement_bilingual_top1": (
                reranked["bilingual_top1_consistency"]
                - baseline["bilingual_top1_consistency"]
            ),
        })

        # Store the original evaluation JSON as an artifact
        mlflow.log_artifact(
            str(RESULTS_PATH),
            artifact_path="evaluation",
        )

        print("=" * 70)
        print("RERANKER ABLATION LOGGED TO MLFLOW")
        print("=" * 70)
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment: {EXPERIMENT_NAME}")
        print()
        print("Baseline:")
        print(f"  Recall@1: {baseline['metrics']['recall_at_1']:.4f}")
        print(f"  MRR:      {baseline['metrics']['mrr']:.4f}")
        print(
            f"  Bilingual: "
            f"{baseline['bilingual_top1_consistency']:.4f}"
        )
        print()
        print("Reranked:")
        print(f"  Recall@1: {reranked['metrics']['recall_at_1']:.4f}")
        print(f"  MRR:      {reranked['metrics']['mrr']:.4f}")
        print(
            f"  Bilingual: "
            f"{reranked['bilingual_top1_consistency']:.4f}"
        )


if __name__ == "__main__":
    main()
