"""Chunking experiment with *RAGAS faithfulness per configuration* + Model Registry promotion.

    python -m src.mlflow.chunking_ragas_sweep \
        --questions data/evaluation/rag_eval.json --n 20 --device cuda

For each of the 5 configs in ``chunking_experiment.CONFIGS`` (chunk_size 256/384/512/768 + a
fixed-window baseline) it: re-chunks the corpus with the production splitter, embeds the chunks,
answers N questions with the configured LLM (LLM_PROVIDER=VLLM on Kaggle), scores them with RAGAS
and logs one MLflow run with chunk_size, overlap, embedding_model, faithfulness and the other
metrics. The winner (highest faithfulness, ties broken by article hit rate) is registered as
``civil-code-rag-chunking`` and promoted to Production. The winner's N-question scores are also
written to ``reports/ragas_ci.json``: the file the CI quality gate reads (faithfulness >= 0.75).

Re-running resumes: finished configs are cached in ``eval_out/sweep/<config>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import mlflow
from src.evaluation.dataset import load_eval
from src.evaluation.ragas_runner import METRICS, RagasJudge, aggregate
from src.mlflow.chunking_experiment import (
    CONFIGS,
    EMBEDDING_MODEL,
    INPUT_FILE,
    RERANKER_MODEL,
    run_splitter,
)
from src.mlflow.registry import register_best
from src.rag import RAGPipeline, make_llm_from_env
from src.retrieval.memory_retriever import InMemoryRetriever, load_articles


def md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def pick_subset(items: list[dict], n: int) -> list[dict]:
    """n evenly spaced questions (keeps the language / category mix of the full set)."""
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def run_config(config, items, articles, embed_model, reranker, llm_bundle, judge, args) -> dict:
    cache = Path(args.cache_dir) / f"{config['name']}.json"
    if cache.exists():
        print(f"[sweep] {config['name']}: cached")
        return json.loads(cache.read_text(encoding="utf-8"))

    chunks_file, stats_file = run_splitter(config)
    chunks = json.loads(Path(chunks_file).read_text(encoding="utf-8"))
    stats = json.loads(Path(stats_file).read_text(encoding="utf-8"))
    t0 = time.time()
    retriever = InMemoryRetriever(chunks, articles, embed_model, reranker, candidate_k=15)
    index_s = time.time() - t0

    llm, provider, model_id = llm_bundle
    pipeline = RAGPipeline(retriever, llm, provider=provider, model_id=model_id, drift=False)

    rows = []
    for i, item in enumerate(items, 1):
        result = pipeline.answer(item["question"], args.top_k)
        sample = {
            "id": item["id"],
            "user_input": item["question"],
            "response": result.answer,
            "retrieved_contexts": result.contexts,
            "reference": item["reference"],
        }
        scores = judge.score(sample)
        expected = set(item["expected_articles"])
        rows.append(
            {
                "id": item["id"],
                "article_hit": bool(expected & set(result.retrieved_articles))
                if expected
                else None,
                "retrieved_articles": result.retrieved_articles,
                "answer": result.answer,
                "tokens": result.input_tokens + result.output_tokens,
                **scores,
            }
        )
        print(f"[sweep] {config['name']} {i}/{len(items)} faith={scores.get('faithfulness')}")

    agg = aggregate(rows, METRICS)
    hits = [r["article_hit"] for r in rows if r["article_hit"] is not None]
    agg["article_hit_rate"] = sum(hits) / len(hits) if hits else 0.0
    out = {
        "config": config,
        "chunk_stats": stats,
        "aggregate": agg,
        "rows": rows,
        "index_s": index_s,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def log_run(
    result: dict, args: argparse.Namespace, llm_model: str, judge_model: str, n: int
) -> None:
    cfg, agg, stats = result["config"], result["aggregate"], result["chunk_stats"]
    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.log_params(
            {
                "strategy": cfg["strategy"],
                "chunk_size": cfg["max_tokens"],
                "overlap": cfg["overlap"],
                "embedding_model": EMBEDDING_MODEL,
                "reranker_model": RERANKER_MODEL,
                "llm_model": llm_model,
                "judge_model": judge_model,
                "top_k": args.top_k,
                "n_questions": n,
            }
        )
        mlflow.log_metrics({k: v for k, v in agg.items() if v is not None})
        mlflow.log_metrics(
            {
                "total_chunks": stats["total_chunks"],
                "token_mean": stats["tokens"]["mean"],
                "token_p95": stats["tokens"]["p95"],
                "token_max": stats["tokens"]["max"],
            }
        )
        out = Path(args.cache_dir) / f"{cfg['name']}.json"
        mlflow.log_artifact(str(out), artifact_path="per_question")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default="data/evaluation/rag_eval.json")
    ap.add_argument("--n", type=int, default=20, help="questions per config (rubric CI gate: 20)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cpu", help="device for BGE-M3 / the reranker (cuda)")
    ap.add_argument("--experiment", default="civil-code-rag-chunking")
    ap.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    ap.add_argument("--cache-dir", default="eval_out/sweep")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--only", nargs="*", help="config names to run (default: all 5)")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    from src.retrieval.reranker import LegalReranker

    items = pick_subset(load_eval(args.questions), args.n)
    articles = load_articles(INPUT_FILE)
    print(f"[sweep] {len(items)} questions, {len(articles)} articles")

    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=args.device)
    reranker = LegalReranker(RERANKER_MODEL, device=args.device)
    llm_bundle = make_llm_from_env()
    judge = RagasJudge(embed_device=args.device)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    results = []
    for config in CONFIGS:
        if args.only and config["name"] not in args.only:
            continue
        result = run_config(config, items, articles, embed_model, reranker, llm_bundle, judge, args)
        log_run(result, args, llm_bundle[2], judge.model, len(items))
        results.append(result)

    if not results:
        raise SystemExit("no configuration ran")
    best = max(
        results,
        key=lambda r: (r["aggregate"]["faithfulness"] or 0.0, r["aggregate"]["article_hit_rate"]),
    )
    registered = register_best(best["config"], best["aggregate"])
    print(
        f"[sweep] best = {best['config']['name']} "
        f"faithfulness={best['aggregate']['faithfulness']:.3f} -> {registered}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best": best["config"],
        "registered": registered,
        "runs": [
            {
                "config": r["config"]["name"],
                "chunk_size": r["config"]["max_tokens"],
                "overlap": r["config"]["overlap"],
                **r["aggregate"],
            }
            for r in results
        ],
    }
    (out_dir / "chunking_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # The CI quality gate reads this file (scripts/ragas_gate.py)
    ci_report = {
        "meta": {
            "n_questions": len(items),
            "questions_md5": md5(args.questions),
            "corpus_md5": md5(INPUT_FILE),
            "config": best["config"]["name"],
            "llm_model": llm_bundle[2],
            "judge_model": judge.model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "aggregate": best["aggregate"],
        "rows": [{k: v for k, v in r.items() if k != "answer"} for r in best["rows"]],
    }
    (out_dir / "ragas_ci.json").write_text(
        json.dumps(ci_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[sweep] wrote {out_dir / 'chunking_sweep.json'} and {out_dir / 'ragas_ci.json'}")


if __name__ == "__main__":
    main()
