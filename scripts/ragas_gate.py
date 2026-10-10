#!/usr/bin/env python
"""CI quality gate: fail when RAGAS faithfulness < 0.75 on the 20-question test set.

The 20-question RAGAS run needs an LLM (GPU/Kaggle or an API), so it is produced by
``python -m src.mlflow.chunking_ragas_sweep`` and committed as ``reports/ragas_ci.json``. This gate
then protects ``main``:

1. the report must exist and cover >= 20 questions,
2. mean faithfulness must be >= the threshold (0.75),
3. the report must not be stale: it records the md5 of the corpus and of the question file it was
   computed on, and the gate compares them with ``dvc.lock`` and the question file. Change the
   corpus or the questions without re-running the evaluation and CI goes red.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def corpus_md5_from_lock(lock_path: Path, target: str = "data/processed/final-articles-v2.json"):
    import yaml

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    for stage in lock.get("stages", {}).values():
        for out in stage.get("outs", []):
            if out.get("path") == target:
                return out.get("md5")
    return None


def evaluate(
    report: dict,
    threshold: float,
    min_questions: int,
    lock_md5: str | None,
    questions_md5: str | None,
) -> list[str]:
    """Returns the list of problems (empty = gate passes)."""
    problems = []
    meta, agg = report.get("meta", {}), report.get("aggregate", {})
    n = meta.get("n_questions", 0)
    faith = agg.get("faithfulness")
    if n < min_questions:
        problems.append(f"only {n} questions were evaluated, need >= {min_questions}")
    if faith is None:
        problems.append("no faithfulness score in the report")
    elif faith < threshold:
        problems.append(f"faithfulness {faith:.3f} < threshold {threshold:.2f}")
    scored = agg.get("faithfulness_n", n)
    if n and scored < 0.8 * n:
        problems.append(f"faithfulness scored on only {scored}/{n} questions (judge failures?)")
    if lock_md5 and meta.get("corpus_md5") and meta["corpus_md5"] != lock_md5:
        problems.append("STALE: the corpus changed since the evaluation (dvc.lock md5 differs)")
    if questions_md5 and meta.get("questions_md5") and meta["questions_md5"] != questions_md5:
        problems.append("STALE: data/evaluation/rag_eval.json changed since the evaluation")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="reports/ragas_ci.json")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--min-questions", type=int, default=20)
    ap.add_argument("--questions", default="data/evaluation/rag_eval.json")
    ap.add_argument("--lock", default="dvc.lock")
    args = ap.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(
            f"GATE FAILED: {report_path} not found. Run the chunking sweep and commit its report."
        )
        return 1
    report = json.loads(report_path.read_text(encoding="utf-8"))

    lock_md5 = corpus_md5_from_lock(Path(args.lock)) if Path(args.lock).exists() else None
    q_md5 = md5(Path(args.questions)) if Path(args.questions).exists() else None
    problems = evaluate(report, args.threshold, args.min_questions, lock_md5, q_md5)

    agg = report.get("aggregate", {})
    lines = [
        "### RAGAS quality gate",
        f"- questions: {report.get('meta', {}).get('n_questions')}",
        f"- faithfulness: {agg.get('faithfulness')} (threshold {args.threshold})",
        f"- answer_relevancy: {agg.get('answer_relevancy')}  "
        f"context_precision: {agg.get('context_precision')}  "
        f"context_recall: {agg.get('context_recall')}",
        f"- result: {'PASS' if not problems else 'FAIL - ' + '; '.join(problems)}",
    ]
    print("\n".join(lines))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        Path(summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
