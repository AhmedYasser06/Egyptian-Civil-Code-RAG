"""Score RAGAS faithfulness on live traffic and attach it to the request's Langfuse trace.

ONLINE_FAITHFULNESS_SAMPLE_RATE=1.0 scores every /ask (one judge call each, run in a background
thread so the user never waits); 0.1 samples 10 %; 0 (default) switches it off. The batch monitor
(``ragas_monitor.py``) always attaches all four metrics to the traces it creates.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor

from src.monitoring import metrics
from src.monitoring.tracing import get_tracer

_EMA_ALPHA = 0.2


class OnlineFaithfulness:
    def __init__(self, sample_rate: float | None = None, judge=None):
        self.rate = (
            float(os.getenv("ONLINE_FAITHFULNESS_SAMPLE_RATE", "0"))
            if sample_rate is None
            else sample_rate
        )
        self._judge = judge
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="online-faithfulness")
        self.ema: float | None = None

    def _get_judge(self):
        if self._judge is None:
            from src.evaluation.ragas_runner import RagasJudge

            self._judge = RagasJudge(metrics=("faithfulness",))
        return self._judge

    def maybe_score(self, question: str, result) -> bool:
        """Schedule scoring for this RAGResult. Returns True when it was scheduled."""
        if self.rate <= 0 or not result.contexts or random.random() > self.rate:
            return False
        self._pool.submit(self._score, question, result)
        return True

    def _score(self, question: str, result) -> None:
        try:
            scores = self._get_judge().score(
                {
                    "user_input": question,
                    "response": result.answer,
                    "retrieved_contexts": result.contexts,
                    "reference": "",
                }
            )
            value = scores.get("faithfulness")
            if value is None:
                return
            self.ema = (
                value if self.ema is None else (1 - _EMA_ALPHA) * self.ema + _EMA_ALPHA * value
            )
            metrics.ONLINE_FAITHFULNESS.set(self.ema)
            get_tracer().score(result.trace_id, "faithfulness", value, comment="RAGAS online")
        except Exception as exc:  # never let evaluation errors surface to users
            print(f"[online-eval] failed: {exc}")
