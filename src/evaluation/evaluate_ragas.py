"""Full RAGAS evaluation (all 4 metrics) with a resumable results file.

Run it as a *module* (no ``.py`` at the end)::

    python -m src.evaluation.evaluate_ragas                                # Groq judge, 30 samples
    python -m src.evaluation.evaluate_ragas --provider gemini              # Gemini judge
    python -m src.evaluation.evaluate_ragas \\
        --dataset data/evaluation/ragas_dataset_gemini_50.json \\
        --results data/evaluation/ragas_results_gemini_50.json --provider gemini
    python -m src.evaluation.evaluate_ragas --retry-failed                 # re-score only the gaps

What changed compared with the first version (which died with ``IncompleteOutputException``):

* the judge gets ``max_tokens=4096`` (was 900) and **doubles it automatically** (up to 16384)
  when the provider reports ``finish_reason=length`` -- see ``RagasJudge._call``;
* a metric that still fails is stored as ``null`` and the run **continues** with the next sample
  instead of crashing; rate limits (429) are retried with back-off;
* resume works per sample *and* per metric: rows with a ``null`` metric are re-scored with
  ``--retry-failed`` (or automatically on the next run), completed rows are never paid for twice;
* Groq Qwen3 "thinking" can be switched off with ``JUDGE_EXTRA_BODY='{"reasoning_effort":"none"}'``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from src.evaluation.ragas_runner import METRICS, RagasJudge, aggregate

load_dotenv()

FIELDS = [
    "id",
    *METRICS,
    "expected_articles",
    "retrieved_article_numbers",
    "generation_provider",
    "generation_model",
    "judge_provider",
    "judge_model",
]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _save(results: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = json_path.with_suffix(".tmp")  # write-then-rename: a Ctrl-C never corrupts the file
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(json_path)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _is_complete(row: dict, metrics: tuple[str, ...]) -> bool:
    return all(row.get(m) is not None for m in metrics)


def run(args: argparse.Namespace) -> list[dict]:
    dataset = _load(Path(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]
    results_path = Path(args.results)
    csv_path = results_path.with_suffix(".csv")
    metrics = tuple(args.metrics)

    results = _load(results_path)
    by_id = {r["id"]: r for r in results}
    todo = [s for s in dataset if s["id"] not in by_id or not _is_complete(by_id[s["id"]], metrics)]
    if not args.retry_failed:
        # without the flag only never-seen samples are scored (old behaviour)
        todo = [s for s in dataset if s["id"] not in by_id] or todo

    judge = RagasJudge(provider=args.provider, metrics=metrics, embed_device=args.embed_device)
    print("=" * 70)
    print(f"RAGAS | samples={len(dataset)} done={len(dataset) - len(todo)} todo={len(todo)}")
    print(f"judge={judge.provider}:{judge.model} max_tokens={judge.base_max_tokens}")
    print(f"endpoint={judge.base_url}")
    print("=" * 70)

    for i, sample in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {sample['id']}: {sample['user_input'][:70]}")
        scores = judge.score(sample)
        row = {
            "id": sample["id"],
            **scores,
            "expected_articles": sample.get("expected_articles"),
            "retrieved_article_numbers": sample.get("retrieved_article_numbers"),
            "generation_provider": sample.get("generation_provider"),
            "generation_model": sample.get("generation_model"),
            "judge_provider": judge.provider.upper(),
            "judge_model": judge.model,
        }
        old = by_id.get(sample["id"], {})
        for m in metrics:  # never overwrite a good old score with a failed new one
            if row.get(m) is None and old.get(m) is not None:
                row[m] = old[m]
        by_id[sample["id"]] = row
        order = {s["id"]: n for n, s in enumerate(dataset)}
        results = sorted(by_id.values(), key=lambda r: order.get(r["id"], 10**6))
        _save(results, results_path, csv_path)  # save after every sample
        print("   ", {m: (None if row[m] is None else round(row[m], 4)) for m in metrics})
        time.sleep(args.delay)

    results = sorted(
        by_id.values(),
        key=lambda r: {s["id"]: n for n, s in enumerate(dataset)}.get(r["id"], 10**6),
    )
    _save(results, results_path, csv_path)

    agg = aggregate(results, metrics)
    failed = [r["id"] for r in results if not _is_complete(r, metrics)]
    print("=" * 70)
    print(f"completed rows: {len(results)}/{len(dataset)}")
    for m in metrics:
        v = agg[m]
        print(f"{m:<18} {'n/a' if v is None else f'{v:.4f}'}   (n={agg[m + '_n']})")
    if "faithfulness" in metrics and agg["faithfulness"] is not None:
        print("Faithfulness gate (>=0.75):", "PASS" if agg["faithfulness"] >= 0.75 else "FAIL")
    if failed:
        print(f"{len(failed)} sample(s) have a missing metric: {failed}")
        print("-> re-run with --retry-failed to fill only the gaps.")
    print(f"JSON: {results_path}\nCSV:  {csv_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", default="data/evaluation/ragas_dataset.json")
    p.add_argument("--results", default="data/evaluation/ragas_results.json")
    p.add_argument("--provider", default=None, help="groq | gemini | openrouter | openai | vllm")
    p.add_argument("--metrics", nargs="+", default=list(METRICS), choices=list(METRICS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--delay", type=float, default=2.0, help="seconds between samples")
    p.add_argument("--embed-device", default="cpu")
    p.add_argument("--retry-failed", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
