#!/usr/bin/env python
"""Turn the output of ``python -m src.evaluation.evaluate_ragas`` into ``reports/ragas_ci.json``,
the file the CI quality gate (``scripts/ragas_gate.py``) reads. No GPU needed.

    python scripts/make_ragas_ci_report.py   # reads data/evaluation/ragas_results.json
    python scripts/ragas_gate.py             # check locally what CI will check
    git add reports/ragas_ci.json && git commit -m "RAGAS CI report"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ragas_gate import corpus_md5_from_lock  # noqa: E402

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/evaluation/ragas_results.json")
    ap.add_argument("--questions", default="data/evaluation/rag_eval.json")
    ap.add_argument("--lock", default="dvc.lock")
    ap.add_argument("--out", default="reports/ragas_ci.json")
    a = ap.parse_args()

    rows = json.loads(Path(a.results).read_text(encoding="utf-8"))
    agg: dict = {}
    for m in METRICS:
        vals = [float(r[m]) for r in rows if r.get(m) is not None]
        agg[m] = statistics.fmean(vals) if vals else None
        agg[f"{m}_n"] = len(vals)
    q, lock = Path(a.questions), Path(a.lock)
    report = {
        "meta": {
            "n_questions": len(rows),
            "source": a.results,
            "judge_model": rows[0].get("judge_model") if rows else None,
            "questions_md5": hashlib.md5(q.read_bytes()).hexdigest() if q.exists() else None,
            "corpus_md5": corpus_md5_from_lock(lock) if lock.exists() else None,
        },
        "aggregate": agg,
        "rows": rows,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}: n={len(rows)} faithfulness={agg['faithfulness']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
