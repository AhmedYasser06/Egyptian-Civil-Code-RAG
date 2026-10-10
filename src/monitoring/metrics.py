"""Prometheus metrics shared by the FastAPI app and the BentoML service."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60)

REQUESTS = Counter("rag_requests_total", "Requests to /ask", ["endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds", "End-to-end /ask latency", ["endpoint"], buckets=_BUCKETS
)
STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds", "Latency per pipeline stage", ["stage"], buckets=_BUCKETS
)
TOKENS = Counter("rag_tokens_total", "LLM tokens", ["type"])  # type=input|output
COST = Counter("rag_cost_usd_total", "Estimated LLM cost in USD", ["provider"])
TTFT = Histogram(
    "rag_time_to_first_token_seconds", "Streaming time to first token", buckets=_BUCKETS
)
EMPTY_RETRIEVALS = Counter("rag_empty_retrievals_total", "Questions with no retrieved article")

DRIFT = Gauge("rag_query_cosine_drift", "1 - cos(reference centroid, recent-query centroid)")
SIMILARITY = Gauge(
    "rag_query_similarity_to_reference", "Mean cosine similarity of recent queries to reference"
)
DRIFT_ALERT = Gauge("rag_query_drift_alert", "1 when cosine drift is above its threshold")


def observe_llm(provider: str, input_tokens: int, output_tokens: int, cost: float) -> None:
    TOKENS.labels("input").inc(input_tokens)
    TOKENS.labels("output").inc(output_tokens)
    COST.labels(provider).inc(cost)


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


ONLINE_FAITHFULNESS = Gauge(
    "rag_online_faithfulness",
    "Exponential moving average of RAGAS faithfulness on sampled live traffic",
)
