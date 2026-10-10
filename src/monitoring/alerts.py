"""Alerting for the RAGAS monitor: faithfulness below the threshold => notification."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

FAITHFULNESS_ALERT_THRESHOLD = float(os.getenv("FAITHFULNESS_ALERT_THRESHOLD", "0.80"))


def check_and_alert(
    scores: dict[str, float],
    threshold: float = FAITHFULNESS_ALERT_THRESHOLD,
    webhook_url: str | None = None,
    log_path: str | Path = "reports/alerts.log",
) -> bool:
    """Return True (and notify) when mean faithfulness < threshold.

    ``webhook_url`` (or $ALERT_WEBHOOK_URL) is any Slack / Discord / Teams compatible incoming
    webhook; the payload carries ``text`` and ``content`` so all three render it.
    The alert is always appended to ``log_path`` too, so it is visible even without a webhook.
    """
    faithfulness = scores.get("faithfulness")
    if faithfulness is None or faithfulness >= threshold:
        return False

    message = (
        f"ALERT: RAGAS faithfulness {faithfulness:.3f} is below {threshold:.2f} "
        f"(answer_relevancy={scores.get('answer_relevancy')}, "
        f"context_precision={scores.get('context_precision')}, "
        f"context_recall={scores.get('context_recall')})"
    )
    print(message)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), "message": message, "scores": scores}) + "\n")

    url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")
    if url:
        try:
            requests.post(url, json={"text": message, "content": message}, timeout=10)
        except requests.RequestException as exc:  # an alert must never crash the monitor
            print(f"[alert] webhook failed: {exc}")
    return True


def push_to_gateway(scores: dict[str, float], gateway: str, job: str = "ragas_monitor") -> None:
    """Expose the RAGAS scores to Prometheus/Grafana through a Pushgateway."""
    from prometheus_client import CollectorRegistry, Gauge
    from prometheus_client import push_to_gateway as _push

    registry = CollectorRegistry()
    for name, value in scores.items():
        if value is None:
            continue
        Gauge(
            f"rag_ragas_{name}", f"RAGAS {name} (mean of last monitor run)", registry=registry
        ).set(value)
    Gauge(
        "rag_ragas_last_run_timestamp", "Unix time of the last RAGAS monitor run", registry=registry
    ).set(time.time())
    _push(gateway, job=job, registry=registry)
