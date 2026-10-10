"""RAGAS monitoring run: >= 50 questions, 4 metrics, MLflow trend, Langfuse scores, alert.

    python -m src.monitoring.ragas_monitor \
        --questions data/evaluation/rag_eval_gemini_50.json --label session-6

For every question it (1) answers through the same RAGPipeline the API uses, which creates a
Langfuse trace, (2) scores faithfulness / answer relevancy / context precision / context recall
with RAGAS, (3) attaches those four scores to the question's Langfuse trace, then (4) logs the means
to MLflow (one run per monitoring session -> the trend across sessions is the MLflow chart view),
(5) pushes them to Prometheus (Grafana panel) and (6) alerts when mean faithfulness < threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from src.evaluation.dataset import load_eval
from src.evaluation.ragas_runner import METRICS, RagasJudge, aggregate
from src.monitoring import alerts
from src.monitoring.tracing import get_tracer
from src.rag import RAGPipeline


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_reused(args: argparse.Namespace) -> dict:
    """Questions already answered + judged by ``evaluate_ragas`` (the 30-sample run): reuse them
    instead of paying for generation and judging again."""
    ds_p, res_p = Path(args.reuse_dataset), Path(args.reuse_results)
    if not (ds_p.exists() and res_p.exists()):
        return {}
    ds = {r["id"]: r for r in json.loads(ds_p.read_text(encoding="utf-8"))}
    out = {}
    for r in json.loads(res_p.read_text(encoding="utf-8")):
        d = ds.get(r["id"])
        if not d or any(r.get(m) is None for m in METRICS):
            continue
        out[r["id"]] = {
            "id": r["id"],
            "question": d["user_input"],
            "answer": d["response"],
            "sources": [],
            "retrieved_articles": [int(x) for x in d.get("retrieved_article_numbers", [])],
            "trace_id": None,
            "cost_usd": 0.0,
            "tokens": 0,
            "latency_s": 0.0,
            **{m: float(r[m]) for m in METRICS},
        }
    return out


def run(args: argparse.Namespace) -> dict:
    items = load_eval(args.questions, limit=args.n)
    if len(items) < args.min_questions:
        raise SystemExit(
            f"{len(items)} questions in {args.questions}; the rubric needs >= {args.min_questions}"
        )

    tracer = get_tracer()
    reused = _load_reused(args) if not args.no_reuse else {}
    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    # rows with a missing metric (e.g. Groq daily token limit) are re-done on the next run
    cache = {k: v for k, v in cache.items() if all(v.get(m) is not None for m in METRICS)}
    pipeline = judge = None  # built lazily: nothing is loaded if every question is reusable

    rows, samples = [], []
    for i, item in enumerate(items, 1):
        t0 = time.time()
        qid = item["id"]
        expected = set(item["expected_articles"])
        if qid in reused or qid in cache:
            row = {**(reused.get(qid) or cache[qid])}
            row["expected_articles"] = item["expected_articles"]
            row["article_hit"] = (
                bool(expected & set(row.get("retrieved_articles") or [])) if expected else None
            )
            rows.append(row)
            print(f"[{i}/{len(items)}] {qid} reused ({'30-set' if qid in reused else 'cache'})")
            continue
        if pipeline is None:
            pipeline = RAGPipeline.from_env()
            judge = RagasJudge(embed_device=args.embed_device)
        result = pipeline.answer(item["question"], args.top_k)
        sample = {
            "id": qid,
            "user_input": item["question"],
            "response": result.answer,
            "retrieved_contexts": result.contexts,
            "reference": item["reference"],
        }
        scores = judge.score(sample)
        for name, value in scores.items():
            if value is not None:
                tracer.score(result.trace_id, name, value, comment=f"RAGAS monitor {args.label}")
        rows.append(
            {
                "id": qid,
                "question": item["question"],
                "answer": result.answer,
                "sources": result.sources,
                "retrieved_articles": result.retrieved_articles,
                "expected_articles": item["expected_articles"],
                "article_hit": bool(expected & set(result.retrieved_articles))
                if expected
                else None,
                "trace_id": result.trace_id,
                "cost_usd": result.cost_usd,
                "tokens": result.input_tokens + result.output_tokens,
                "latency_s": round(result.total_s, 2),
                **scores,
            }
        )
        samples.append(sample)
        cache[qid] = rows[-1]  # resume point: Ctrl-C / quota errors lose nothing
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{i}/{len(items)}] {qid} faith={scores.get('faithfulness')} ({time.time() - t0:.1f}s)"
        )
    tracer.flush()

    agg = aggregate(rows, METRICS)
    hits = [r["article_hit"] for r in rows if r["article_hit"] is not None]
    agg["article_hit_rate"] = sum(hits) / len(hits) if hits else None
    agg["total_cost_usd"] = sum(r["cost_usd"] for r in rows)
    agg["total_tokens"] = sum(r["tokens"] for r in rows)
    agg["n_questions"] = len(rows)

    report = {
        "meta": {
            "label": args.label,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": _git_sha(),
            "llm_provider": pipeline.provider if pipeline else os.getenv("LLM_PROVIDER", "reused"),
            "llm_model": pipeline.model_id if pipeline else "reused",
            "judge_model": judge.model if judge else "reused",
            "top_k": args.top_k,
            "questions_file": args.questions,
        },
        "aggregate": agg,
        "rows": rows,
    }
    out = Path(args.out or f"reports/ragas_monitor_{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))

    log_to_mlflow(args, report, out)
    scores = {k: agg[k] for k in METRICS if agg.get(k) is not None}
    gateway = args.pushgateway or os.getenv("PUSHGATEWAY_URL")
    if gateway:
        try:
            alerts.push_to_gateway(scores, gateway)
        except Exception as exc:
            print(f"[monitor] pushgateway unavailable ({exc}); skipping")
    report["alert"] = alerts.check_and_alert(scores, args.threshold)
    return report


def log_to_mlflow(args: argparse.Namespace, report: dict, out: Path) -> None:
    import mlflow

    mlflow.set_tracking_uri(
        args.mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment(args.experiment)
    meta, agg = report["meta"], report["aggregate"]
    with mlflow.start_run(run_name=meta["label"]):
        mlflow.log_params({k: v for k, v in meta.items() if k != "timestamp"})
        mlflow.set_tag("session", meta["label"])
        mlflow.log_metrics(
            {k: v for k, v in agg.items() if isinstance(v, (int, float)) and v is not None}
        )
        mlflow.log_metric("faithfulness_alert_threshold", args.threshold)
        mlflow.log_artifact(str(out), artifact_path="monitor")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default="data/evaluation/rag_eval_gemini_50.json")
    ap.add_argument("--n", type=int, default=None, help="limit (default: all questions)")
    ap.add_argument("--min-questions", type=int, default=50)
    ap.add_argument("--label", default=time.strftime("monitor-%Y%m%d-%H%M"))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=alerts.FAITHFULNESS_ALERT_THRESHOLD)
    ap.add_argument("--experiment", default="civil-code-rag-monitoring")
    ap.add_argument("--mlflow-uri", default=None)
    ap.add_argument("--pushgateway", default=None)
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse-dataset", default="data/evaluation/ragas_dataset.json")
    ap.add_argument("--reuse-results", default="data/evaluation/ragas_results.json")
    ap.add_argument("--no-reuse", action="store_true", help="answer + judge every question again")
    ap.add_argument("--cache", default="data/evaluation/ragas_monitor_cache.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
