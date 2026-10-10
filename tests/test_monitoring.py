import json

import numpy as np

from src.monitoring import alerts
from src.monitoring.cost import compute_cost
from src.monitoring.drift import CosineDriftMonitor
from src.monitoring.tracing import Tracer


def _cluster(center, n, noise, rng):
    return center + noise * rng.normal(size=(n, center.shape[0]))


def test_drift_is_low_in_domain_and_high_out_of_domain():
    rng = np.random.default_rng(0)
    topic_a, topic_b = rng.normal(size=32), rng.normal(size=32)  # two unrelated "topics"
    reference = _cluster(topic_a, 60, 0.3, rng)

    in_domain = CosineDriftMonitor.from_embeddings(reference, min_samples=10)
    for v in _cluster(topic_a, 30, 0.3, rng):
        stats_in = in_domain.observe(v)

    shifted = CosineDriftMonitor.from_embeddings(reference, min_samples=10)
    for v in _cluster(topic_b, 30, 0.3, rng):
        stats_out = shifted.observe(v)

    assert stats_in["drift"] < 0.05 and not stats_in["alert"]
    assert stats_out["drift"] > 0.5 and stats_out["alert"]
    assert stats_out["similarity"] < stats_in["similarity"]


def test_drift_needs_min_samples_and_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    m = CosineDriftMonitor.from_embeddings(rng.normal(size=(20, 8)), min_samples=5)
    assert m.observe(rng.normal(size=8)) is None
    m.save(tmp_path / "ref.npz")
    loaded = CosineDriftMonitor.load(tmp_path / "ref.npz")
    assert np.allclose(loaded.centroid, m.centroid)


def test_cost_uses_provider_prices(monkeypatch):
    monkeypatch.setenv("PRICE_VLLM_INPUT_PER_1M", "1.0")
    monkeypatch.setenv("PRICE_VLLM_OUTPUT_PER_1M", "2.0")
    assert compute_cost("VLLM", 1_000_000, 500_000) == 2.0
    assert compute_cost("OLLAMA", 10_000, 10_000) == 0.0


def test_alert_fires_below_threshold_and_writes_log(tmp_path):
    log = tmp_path / "alerts.log"
    assert alerts.check_and_alert({"faithfulness": 0.79}, 0.80, log_path=log) is True
    assert "0.790" in json.loads(log.read_text().splitlines()[0])["message"]


def test_alert_silent_at_or_above_threshold(tmp_path):
    log = tmp_path / "alerts.log"
    assert alerts.check_and_alert({"faithfulness": 0.80}, 0.80, log_path=log) is False
    assert alerts.check_and_alert({"faithfulness": 0.91}, 0.80, log_path=log) is False
    assert not log.exists()


def test_alert_posts_to_webhook(tmp_path, monkeypatch):
    sent = {}
    monkeypatch.setattr(
        alerts.requests, "post", lambda url, json, timeout: sent.update(url=url, **json)
    )
    alerts.check_and_alert(
        {"faithfulness": 0.5}, 0.8, webhook_url="http://hook", log_path=tmp_path / "a"
    )
    assert sent["url"] == "http://hook" and "0.500" in sent["text"]


def test_tracer_is_a_silent_noop_without_keys():
    tracer = Tracer()
    assert tracer.enabled is False
    span = tracer.start("ask", input={"q": "x"})
    child = span.child("retrieval", as_type="retriever")
    child.update(output=1)
    child.end()
    span.end()
    tracer.score(span.trace_id, "faithfulness", 0.9)  # must not raise
