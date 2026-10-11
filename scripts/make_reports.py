#!/usr/bin/env python
"""Generate the evidence files for the rubric from the results already in the repo.

    python scripts/make_reports.py            # writes reports/*.png + reports/SUMMARY.md

Sections whose input is missing are skipped (and listed under "Still missing"), so it is safe to
re-run after every step (Kaggle outputs, Locust, MLflow runs ...).

NOTE: ``mlflow_comparison.png`` / ``ragas_trend.png`` here are charts *drawn from the MLflow
database*. The rubric asks for a screenshot of the MLflow UI: take that one too (mlflow ui ->
select runs -> Compare) and save it as reports/mlflow_comparison_ui.png.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path("reports")
EVAL = Path("data/evaluation")
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
INK, MUTED, BLUE, AMBER, GRID = "#1f2933", "#6b7280", "#2f6fdb", "#d97706", "#e5e7eb"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "axes.axisbelow": True,
    }
)
missing: list[str] = []
lines: list[str] = ["# Project evidence summary", ""]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def latest_monitor() -> Path | None:
    files = sorted(REPORTS.glob("ragas_monitor_*.json"), key=lambda p: p.stat().st_mtime)
    # prefer a run that really generated + judged answers (re-runs that only reuse the cache say "reused")
    real = [f for f in files if (load(f) or {}).get("meta", {}).get("llm_model") != "reused"]
    return (real or files or [None])[-1]


def save(fig, name: str) -> None:
    REPORTS.mkdir(exist_ok=True)
    fig.savefig(REPORTS / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", REPORTS / name)


def ragas_section() -> None:
    p = latest_monitor()
    rep = load(p) if p else None
    if not rep:
        missing.append("RAGAS monitor report (python -m src.monitoring.ragas_monitor ...)")
        return
    agg, rows, meta = rep["aggregate"], rep["rows"], rep["meta"]
    n = agg.get("n_questions", len(rows))
    lines[:] += [
        f"## RAGAS on {n} questions (`{p.name}`)",
        "",
        f"Generator: {meta.get('llm_provider')} / {meta.get('llm_model')} - "
        f"judge: {meta.get('judge_model')} - top_k {meta.get('top_k')}",
        "",
        "| metric | mean | scored |",
        "|---|---|---|",
    ]
    for m in METRICS:
        lines.append(f"| {m} | {agg[m]:.3f} | {agg.get(m + '_n')} |")
    lines[:] += [
        "",
        f"article hit rate: {agg.get('article_hit_rate')}  |  total tokens: {agg.get('total_tokens')}"
        f"  |  cost USD: {agg.get('total_cost_usd', 0):.4f}",
        "",
        f"Alert rule: faithfulness < 0.80 -> {'ALERT' if agg['faithfulness'] < 0.8 else 'ok'}"
        f" (mean {agg['faithfulness']:.3f}); CI gate: faithfulness >= 0.75 on >= 20 questions.",
        "",
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [agg[m] for m in METRICS]
    bars = ax.bar(METRICS, vals, color=BLUE, width=0.55)
    ax.axhline(0.80, color=AMBER, lw=1.5)
    ax.text(3.45, 0.81, "alert 0.80", color=AMBER, ha="right", va="bottom", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"RAGAS means on {n} questions", loc="left", color=INK)
    ax.set_xticks(range(len(METRICS)), [m.replace("_", "\n") for m in METRICS])
    save(fig, "ragas_metrics.png")

    scored = [(r["id"], r["faithfulness"]) for r in rows if r.get("faithfulness") is not None]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(range(len(scored)), [v for _, v in scored], color=BLUE, width=0.7)
    ax.axhline(0.80, color=AMBER, lw=1.5)
    ax.set_xticks(range(len(scored)), [i for i, _ in scored], rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("Faithfulness per question (line = 0.80 alert)", loc="left", color=INK)
    save(fig, "ragas_faithfulness_per_question.png")
    weak = sorted(scored, key=lambda t: t[1])[:5]
    lines[:] += [
        "Lowest-faithfulness questions: " + ", ".join(f"{i} ({v:.2f})" for i, v in weak),
        "",
    ]


def drift_section() -> None:
    d = load(REPORTS / "drift_demo.json")
    if not d:
        missing.append("drift report (python -m src.monitoring.drift_demo ...)")
        return
    names = ["in_domain", "family_law"]
    vals = [d[k]["drift"] for k in names]
    thr = d.get("threshold", 0.15)
    lines[:] += [
        "## Embedding (cosine) drift",
        "",
        "| query set | drift | similarity | alert |",
        "|---|---|---|---|",
        *[
            f"| {k} | {d[k]['drift']:.3f} | {d[k]['similarity']:.3f} | {d[k]['alert']} |"
            for k in names
        ],
        "",
        f"Threshold {thr}: contract-law questions stay below it; family-law questions "
        "(a different jurisdiction of the same code) cross it - the semantic shift the handbook describes.",
        "",
    ]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.bar(
        ["in-domain\n(civil law)", "family law\n(shifted)"], vals, color=[BLUE, AMBER], width=0.5
    )
    ax.axhline(thr, color=INK, lw=1, ls="--")
    ax.text(1.45, thr + 0.004, f"threshold {thr}", ha="right", fontsize=9, color=INK)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", color=INK)
    ax.set_title("Cosine drift vs reference", loc="left", color=INK)
    save(fig, "drift_demo.png")


def retrieval_section() -> None:
    d = load(EVAL / "reranker_ablation_results.json")
    if not d:
        missing.append("data/evaluation/reranker_ablation_results.json")
        return
    keys = ["recall_at_1", "recall_at_3", "recall_at_5", "mrr"]
    b, r = d["baseline"], d["reranked"]
    lines[:] += [
        "## Retrieval: BGE-M3 vs BGE-M3 + reranker (20 bilingual questions)",
        "",
        "| metric | no reranker | with reranker |",
        "|---|---|---|",
        *[f"| {k} | {b['metrics'][k]:.3f} | {r['metrics'][k]:.3f} |" for k in keys],
        f"| bilingual top-1 consistency | {b['bilingual_top1_consistency']:.2f} "
        f"| {r['bilingual_top1_consistency']:.2f} |",
        "",
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = range(len(keys))
    ax.bar(
        [x - 0.18 for x in xs],
        [b["metrics"][k] for k in keys],
        0.34,
        color=MUTED,
        label="no reranker",
    )
    ax.bar(
        [x + 0.18 for x in xs],
        [r["metrics"][k] for k in keys],
        0.34,
        color=BLUE,
        label="with reranker",
    )
    ax.set_xticks(list(xs), keys)
    ax.set_ylim(0.8, 1.03)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Retrieval quality (axis starts at 0.8)", loc="left", color=INK)
    save(fig, "reranker_ablation.png")


def mlflow_section() -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    if uri.startswith("sqlite:///") and not Path(uri.replace("sqlite:///", "")).exists():
        missing.append(
            "mlflow.db (run the Kaggle sweep / ragas_monitor locally to create MLflow runs)"
        )
        return
    import mlflow

    mlflow.set_tracking_uri(uri)
    try:
        runs = mlflow.search_runs(experiment_names=["civil-code-rag-chunking"])
    except Exception:
        runs = None
    if runs is None or runs.empty or "metrics.faithfulness" not in runs:
        missing.append("MLflow chunking runs (>= 5) - run notebooks/kaggle_1_mlflow_ragas_sweep")
    else:
        runs = runs.sort_values("metrics.faithfulness", ascending=False)
        lines[:] += [
            "## MLflow chunking runs",
            "",
            "| run | chunk_size | overlap | embedding_model | faithfulness |",
            "|---|---|---|---|---|",
            *[
                f"| {r['tags.mlflow.runName']} | {r.get('params.chunk_size')} | {r.get('params.overlap')} "
                f"| {r.get('params.embedding_model')} | {r['metrics.faithfulness']:.3f} |"
                for _, r in runs.iterrows()
            ],
            "",
        ]
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        ax.barh(runs["tags.mlflow.runName"][::-1], runs["metrics.faithfulness"][::-1], color=BLUE)
        ax.set_xlim(0, 1.05)
        ax.set_title("MLflow: faithfulness per chunking config", loc="left", color=INK)
        save(fig, "mlflow_comparison.png")
        if len(runs) < 5:
            missing.append(f"only {len(runs)} chunking runs in MLflow, the rubric needs >= 5")
    try:
        mon = mlflow.search_runs(experiment_names=["civil-code-rag-monitoring"])
    except Exception:
        mon = None
    if mon is not None and not mon.empty and "metrics.faithfulness" in mon:
        mon = mon.sort_values("start_time")
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.plot(
            mon["tags.mlflow.runName"], mon["metrics.faithfulness"], color=BLUE, lw=2, marker="o"
        )
        ax.axhline(0.80, color=AMBER, lw=1.5)
        ax.set_ylim(0, 1.05)
        ax.set_title("RAGAS faithfulness across monitoring sessions", loc="left", color=INK)
        save(fig, "ragas_trend.png")


def optional_sections() -> None:
    for name, hint in {
        "quantization.md": "AWQ vs fp16 + reranker distillation (notebook 2)",
        "reranker_distillation.md": "reranker distillation (notebook 2)",
        "load_test.md": "Locust p50/p95/p99 at 50 users (notebook 3)",
        "locust_u50.html": "Locust report at 50 users (notebook 3)",
        "streaming_demo.txt": "streaming curl proof (notebook 3)",
        "ragas_ci.json": "CI gate report (python scripts/make_ragas_ci_report.py)",
    }.items():
        if (REPORTS / name).exists():
            lines.append(f"- found `reports/{name}`")
        else:
            missing.append(f"reports/{name} - {hint}")
    for name, hint in {
        "grafana_faithfulness.png": "screenshot of the Grafana faithfulness panel",
        "langfuse_trace.png": "screenshot of a Langfuse trace with RAGAS scores",
        "mlflow_comparison_ui.png": "screenshot of the MLflow UI compare page",
    }.items():
        if not (REPORTS / name).exists() and not (REPORTS / "grafana_dashboard.png").exists():
            missing.append(f"reports/{name} - {hint} (manual screenshot)")
        elif not (REPORTS / name).exists() and name != "grafana_faithfulness.png":
            missing.append(f"reports/{name} - {hint} (manual screenshot)")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    for section in (ragas_section, drift_section, retrieval_section, mlflow_section):
        try:
            section()
        except Exception as exc:  # one broken input must not block the others
            missing.append(f"{section.__name__}: {type(exc).__name__}: {exc}")
    lines.append("## Files")
    lines.append("")
    optional_sections()
    lines.extend(["", "## Still missing", ""])
    lines.extend([f"- {m}" for m in missing] or ["- nothing"])
    (REPORTS / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote reports/SUMMARY.md")


if __name__ == "__main__":
    main()
