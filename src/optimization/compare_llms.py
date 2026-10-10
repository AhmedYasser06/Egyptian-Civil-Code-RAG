"""Optimization: AWQ 4-bit vs fp16 generative model (and optionally a distilled reranker),
RAGAS before vs after, latency and memory.

Run one vLLM server at a time (two 8B models do not fit on 2x T4), so it is three steps:

    # server = Qwen/Qwen3-8B-AWQ
    python -m src.optimization.compare_llms generate --label awq --model Qwen/Qwen3-8B-AWQ \
        --vllm-log vllm_awq.log
    # server = Qwen/Qwen3-8B (fp16)
    python -m src.optimization.compare_llms generate --label fp16 --model Qwen/Qwen3-8B \
        --vllm-log vllm_fp16.log
    # still on the fp16 server, which acts as the judge for BOTH answer sets
    python -m src.optimization.compare_llms score --labels fp16 awq --judge-model Qwen/Qwen3-8B
    python -m src.optimization.compare_llms report --baseline fp16 --optimized awq

Use ``--reranker models/reranker-student --label awq_student`` to measure the full optimized
stack (AWQ generator + distilled reranker). Retrieval is cached per reranker, so every label that
uses the same reranker sees exactly the same contexts and differences come from the LLM only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import time
from pathlib import Path

from src.evaluation.dataset import load_eval
from src.evaluation.ragas_runner import METRICS, aggregate

OUT = Path("eval_out/quant")


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


class CachedRetriever:
    """Wraps the real retriever; answers from a JSON cache so all labels share contexts."""

    def __init__(self, path: Path, build_base):
        self.path, self.build_base = path, build_base
        self.cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.base = None
        self.embedding_provider = None

    def retrieve(self, query: str, top_k: int = 5):
        if query not in self.cache:
            if self.base is None:
                self.base = self.build_base()
            self.cache[query] = self.base.retrieve(query, top_k)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        return self.cache[query]


def build_base_retriever(args):
    from sentence_transformers import SentenceTransformer

    from src.mlflow.chunking_experiment import EMBEDDING_MODEL, INPUT_FILE
    from src.retrieval.memory_retriever import InMemoryRetriever, load_articles
    from src.retrieval.reranker import LegalReranker

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    articles = load_articles(INPUT_FILE)
    model = SentenceTransformer(EMBEDDING_MODEL, device=args.device)
    reranker = LegalReranker(args.reranker, device=args.device)
    return InMemoryRetriever(chunks, articles, model, reranker)


def vllm_weight_gib(log_path: str | None) -> float | None:
    if not log_path or not Path(log_path).exists():
        return None
    match = re.search(r"Model loading took ([\d.]+) GiB", Path(log_path).read_text(errors="ignore"))
    return float(match.group(1)) if match else None


# ----------------------------------------------------------------------------- generate
def cmd_generate(args) -> None:
    os.environ.update(LLM_PROVIDER="VLLM", VLLM_BASE_URL=args.base_url, VLLM_MODEL_ID=args.model)
    from src.rag import RAGPipeline, make_llm_from_env

    items = load_eval(args.questions, limit=args.n)
    tag = re.sub(r"\W+", "_", args.reranker)
    retriever = CachedRetriever(OUT / f"retrieval_{tag}.json", lambda: build_base_retriever(args))
    llm, provider, model_id = make_llm_from_env()
    pipeline = RAGPipeline(retriever, llm, provider=provider, model_id=model_id, drift=False)

    async def sequential():
        rows = []
        for i, item in enumerate(items, 1):
            events = [e async for e in pipeline.astream(item["question"], args.top_k)]
            done = events[-1]["data"]
            rows.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "reference": item["reference"],
                    "expected_articles": item["expected_articles"],
                    "answer": done["answer"],
                    "contexts": done["contexts"],
                    "retrieved_articles": done["retrieved_articles"],
                    "latency_s": done["total_s"],
                    "generation_s": done["generation_s"],
                    "ttft_s": done["ttft_s"],
                    "input_tokens": done["input_tokens"],
                    "output_tokens": done["output_tokens"],
                }
            )
            print(f"[gen:{args.label}] {i}/{len(items)} {done['total_s']:.1f}s")
        return rows

    async def concurrent():
        sem = asyncio.Semaphore(args.concurrency)

        async def one(item):
            async with sem:
                return await pipeline.aanswer(item["question"], args.top_k)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(i) for i in items))
        wall = time.perf_counter() - t0
        return sum(r.output_tokens for r in results) / wall, wall

    rows = asyncio.run(sequential())
    tok_s, wall = asyncio.run(concurrent())
    lat = [r["latency_s"] for r in rows]
    ttft = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    out = {
        "label": args.label,
        "model": args.model,
        "reranker": args.reranker,
        "n": len(rows),
        "latency_p50_s": pct(lat, 0.5),
        "latency_p95_s": pct(lat, 0.95),
        "ttft_p50_s": pct(ttft, 0.5),
        "throughput_tokens_per_s": tok_s,
        "concurrent_wall_s": wall,
        "weights_gib_per_gpu": vllm_weight_gib(args.vllm_log),
        "mean_output_tokens": statistics.fmean(r["output_tokens"] for r in rows),
        "rows": rows,
    }
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.label}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print({k: v for k, v in out.items() if k != "rows"})


# ----------------------------------------------------------------------------- score
def cmd_score(args) -> None:
    from src.evaluation.ragas_runner import RagasJudge

    judge = RagasJudge(
        base_url=args.judge_base_url,
        api_key="EMPTY",
        model=args.judge_model,
        embed_device=args.device,
    )
    for label in args.labels:
        data = json.loads((OUT / f"{label}.json").read_text(encoding="utf-8"))
        samples = [
            {
                "id": r["id"],
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"],
                "reference": r["reference"],
            }
            for r in data["rows"]
        ]
        rows = judge.score_many(samples)
        res = {
            "label": label,
            "judge": args.judge_model,
            "aggregate": aggregate(rows),
            "rows": rows,
        }
        (OUT / f"{label}_scores.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[score] {label}: {res['aggregate']}")


# ----------------------------------------------------------------------------- report
def cmd_report(args) -> None:
    import mlflow

    labels = [args.baseline] + [args.optimized] + args.extra
    data = {lb: json.loads((OUT / f"{lb}.json").read_text(encoding="utf-8")) for lb in labels}
    scores = {
        lb: json.loads((OUT / f"{lb}_scores.json").read_text(encoding="utf-8")) for lb in labels
    }

    header = (
        "| label | model | reranker | faithfulness | answer_rel | ctx_prec | ctx_recall | "
        "p50 s | p95 s | TTFT p50 s | tok/s (conc.) | weights GiB/GPU |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    f3 = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    rows = ""
    for lb in labels:
        d, a = data[lb], scores[lb]["aggregate"]
        rows += (
            f"| {lb} | {d['model']} | {Path(d['reranker']).name} | {f3(a['faithfulness'])} | "
            f"{f3(a['answer_relevancy'])} | {f3(a['context_precision'])} | "
            f"{f3(a['context_recall'])} | {d['latency_p50_s']:.2f} | {d['latency_p95_s']:.2f} | "
            f"{d['ttft_p50_s']:.2f} | {d['throughput_tokens_per_s']:.1f} | "
            f"{f3(d['weights_gib_per_gpu'])} |\n"
        )
    base, opt = scores[args.baseline]["aggregate"], scores[args.optimized]["aggregate"]
    drop = (base["faithfulness"] or 0) - (opt["faithfulness"] or 0)
    verdict = "PASS" if drop < args.max_drop else "FAIL"
    speed = data[args.baseline]["latency_p50_s"] / data[args.optimized]["latency_p50_s"]
    md = (
        f"# Optimization: {args.baseline} -> {args.optimized}\n\n"
        f"{header}{rows}\n"
        f"- faithfulness drop: **{drop:+.3f}** (limit {args.max_drop}) -> **{verdict}**\n"
        f"- p50 latency speed-up: **{speed:.2f}x**\n"
        f"- judge: {scores[args.baseline]['judge']} for every label; "
        f"{data[args.baseline]['n']} questions\n"
    )
    Path("reports").mkdir(exist_ok=True)
    Path("reports/quantization.md").write_text(md, encoding="utf-8")
    Path("reports/quantization.json").write_text(
        json.dumps(
            {
                "labels": {
                    lb: {
                        **{k: v for k, v in data[lb].items() if k != "rows"},
                        "ragas": scores[lb]["aggregate"],
                    }
                    for lb in labels
                },
                "faithfulness_drop": drop,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(md)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("civil-code-rag-optimization")
    for lb in labels:
        with mlflow.start_run(run_name=lb):
            d, a = data[lb], scores[lb]["aggregate"]
            mlflow.log_params({"model": d["model"], "reranker": d["reranker"], "n": d["n"]})
            mlflow.log_metrics({k: a[k] for k in METRICS if a[k] is not None})
            mlflow.log_metrics(
                {
                    k: d[k]
                    for k in (
                        "latency_p50_s",
                        "latency_p95_s",
                        "ttft_p50_s",
                        "throughput_tokens_per_s",
                    )
                    if d[k] == d[k]
                }
            )
            if d["weights_gib_per_gpu"]:
                mlflow.log_metric("weights_gib_per_gpu", d["weights_gib_per_gpu"])
            mlflow.log_artifact("reports/quantization.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--label", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--base-url", default="http://localhost:8000/v1")
    g.add_argument("--questions", default="data/evaluation/rag_eval.json")
    g.add_argument("--chunks", default="data/processed/final-chunks-bgem3-v2.json")
    g.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    g.add_argument("--device", default="cpu")
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--top-k", type=int, default=5)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--vllm-log", default=None)
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("score")
    s.add_argument("--labels", nargs="+", required=True)
    s.add_argument("--judge-model", required=True)
    s.add_argument("--judge-base-url", default="http://localhost:8000/v1")
    s.add_argument("--device", default="cpu")
    s.set_defaults(fn=cmd_score)

    r = sub.add_parser("report")
    r.add_argument("--baseline", default="fp16")
    r.add_argument("--optimized", default="awq")
    r.add_argument("--extra", nargs="*", default=[])
    r.add_argument("--max-drop", type=float, default=0.03)
    r.set_defaults(fn=cmd_report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
