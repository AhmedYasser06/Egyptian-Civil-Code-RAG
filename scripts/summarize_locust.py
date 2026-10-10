#!/usr/bin/env python
"""Turn Locust's ``--csv`` output into a Markdown table with p50 / p95 / p99 for the report.

python scripts/summarize_locust.py reports/locust_u1 reports/locust_u50 --out reports/load_test.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read(prefix: str) -> list[dict]:
    with open(f"{prefix}_stats.csv", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["Name"] != "Aggregated"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--out", default="reports/load_test.md")
    args = ap.parse_args()

    lines = [
        "| run | endpoint | requests | failures | req/s | avg (s) | p50 (s) | p95 (s) | p99 (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prefix in args.prefixes:
        for r in read(prefix):
            sec = lambda key: float(r[key]) / 1000  # noqa: E731
            lines.append(
                f"| {Path(prefix).name} | {r['Name']} | {r['Request Count']} | "
                f"{r['Failure Count']} | {float(r['Requests/s']):.2f} | "
                f"{sec('Average Response Time'):.2f} | {sec('50%'):.2f} | "
                f"{sec('95%'):.2f} | {sec('99%'):.2f} |"
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
