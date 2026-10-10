"""Langfuse tracing (self-hosted). Every /ask creates one trace with child spans.

Trace layout::

    ask                       (span: input=question, output=answer)
    ├── retrieval             (retriever: output=article numbers)
    └── generation            (generation: model, token usage, cost)

When LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set the tracer is a silent no-op, so the API
and the tests run without a Langfuse server. Uses the explicit observation API (not context
managers) because the streaming endpoint yields from inside the request.
"""

from __future__ import annotations

import os
from typing import Any


class _NullSpan:
    trace_id: str | None = None

    def child(self, *_, **__) -> _NullSpan:
        return self

    def update(self, **_) -> None:
        pass

    def end(self) -> None:
        pass


class _Span:
    def __init__(self, observation, client):
        self._obs = observation
        self._client = client
        self.trace_id: str | None = getattr(observation, "trace_id", None)

    def child(self, name: str, as_type: str = "span", **kwargs: Any) -> _Span:
        return _Span(
            self._obs.start_observation(name=name, as_type=as_type, **kwargs), self._client
        )

    def update(self, **kwargs: Any) -> None:
        try:
            self._obs.update(**kwargs)
        except Exception as exc:  # tracing must never break a request
            print(f"[langfuse] update failed: {exc}")

    def end(self) -> None:
        try:
            self._obs.end()
        except Exception as exc:
            print(f"[langfuse] end failed: {exc}")


class Tracer:
    def __init__(self) -> None:
        self.enabled = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
        self._client = None
        if self.enabled:
            try:
                from langfuse import Langfuse

                # reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from the env
                self._client = Langfuse()
            except Exception as exc:
                print(f"[langfuse] disabled: {exc}")
                self.enabled = False

    def start(self, name: str, **kwargs: Any) -> _Span | _NullSpan:
        if not self.enabled:
            return _NullSpan()
        try:
            return _Span(
                self._client.start_observation(name=name, as_type="span", **kwargs), self._client
            )
        except Exception as exc:
            print(f"[langfuse] start failed: {exc}")
            return _NullSpan()

    def score(
        self, trace_id: str | None, name: str, value: float, comment: str | None = None
    ) -> None:
        """Attach a numeric score (e.g. RAGAS faithfulness) to an existing trace."""
        if not (self.enabled and trace_id):
            return
        try:
            self._client.create_score(
                name=name,
                value=float(value),
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=comment,
            )
        except Exception as exc:
            print(f"[langfuse] score failed: {exc}")

    def flush(self) -> None:
        if self.enabled:
            try:
                self._client.flush()
            except Exception as exc:
                print(f"[langfuse] flush failed: {exc}")


_TRACER: Tracer | None = None


def get_tracer() -> Tracer:
    global _TRACER
    if _TRACER is None:
        _TRACER = Tracer()
    return _TRACER
