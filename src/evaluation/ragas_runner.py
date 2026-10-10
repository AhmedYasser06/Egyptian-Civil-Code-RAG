"""Shared RAGAS runner (ragas 0.4.x ``metrics.collections`` API, same construction as
``evaluate_ragas.py``). Used by the chunking sweep, the quantisation comparison, the monitor
and the online faithfulness scoring.

The judge is any OpenAI-compatible endpoint: Groq (default, as in the original scripts), a vLLM
server, or OpenAI itself. Set JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Callable, Iterable

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# Judge presets: every one is an OpenAI-compatible endpoint (ragas talks to it through instructor).
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL_ID",
        "model": "qwen/qwen3.8-27b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GOOGLE_API_KEY",
        "model_env": "GOOGLE_RAGAS_JUDGE_MODEL",
        "model": "gemini-2.5-flash",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL_ID",
        "model": "qwen/qwen3-235b-a22b-2507",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL_ID",
        "model": "gpt-4o-mini",
    },
    # a local / Kaggle vLLM server (see notebooks/kaggle_2_optimization.ipynb)
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "key_env": "VLLM_API_KEY",
        "model_env": "VLLM_MODEL_ID",
        "model": "Qwen/Qwen2.5-7B-Instruct-AWQ",
    },
}


def provider_config(name: str, explicit: bool = False) -> dict:
    """base_url / api_key / model for a preset.

    JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL override the preset, except when the provider was
    chosen explicitly (``--provider gemini``): then a leftover JUDGE_* line in ``.env`` (e.g. the
    Groq URL) must not send Gemini requests to the wrong server (that gave ``NotFoundError``).
    The ``vllm`` preset always honours them (the Kaggle notebooks rely on that).
    """
    use_env = (not explicit) or name == "vllm"
    if name not in PROVIDERS:
        raise ValueError(f"unknown judge provider {name!r}; choose from {sorted(PROVIDERS)}")
    p = PROVIDERS[name]
    return {
        "base_url": (use_env and os.getenv("JUDGE_BASE_URL")) or p["base_url"],
        "api_key": (use_env and os.getenv("JUDGE_API_KEY")) or os.getenv(p["key_env"]) or "EMPTY",
        "model": (use_env and os.getenv("JUDGE_MODEL")) or os.getenv(p["model_env"]) or p["model"],
    }


def is_truncation(exc: BaseException) -> bool:
    """instructor raises IncompleteOutputException when the judge hit finish_reason='length'
    (typical for reasoning models such as Qwen3 whose hidden thinking eats max_tokens)."""
    name = type(exc).__name__
    return (
        name == "IncompleteOutputException" or "finish_reason" in str(exc) and "length" in str(exc)
    )


def is_rate_limit(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return (
        "ratelimit" in text or "429" in text or "rate limit" in text or "resource_exhausted" in text
    )


class RagasJudge:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        embed_model: str = "BAAI/bge-m3",
        embed_device: str = "cpu",
        metrics: Iterable[str] = METRICS,
        provider: str | None = None,
        max_tokens: int | None = None,
    ):
        from openai import AsyncOpenAI
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        self.provider = provider or os.getenv("JUDGE_PROVIDER", "groq")
        cfg = provider_config(self.provider, explicit=provider is not None)
        base_url = base_url or cfg["base_url"]
        self.base_url = base_url
        api_key = api_key or cfg["api_key"]
        self.model = model or cfg["model"]
        self.metric_names = tuple(metrics)

        # The 900-token default used to crash with IncompleteOutputException: the judge's answer
        # (plus any hidden reasoning) did not fit.  Start generous and grow on demand.
        self.base_max_tokens = int(max_tokens or os.getenv("JUDGE_MAX_TOKENS", "4096"))
        self.max_tokens_cap = int(os.getenv("JUDGE_MAX_TOKENS_CAP", "16384"))
        self.retries = int(os.getenv("JUDGE_RETRIES", "4"))

        kwargs: dict = {"temperature": 0, "max_tokens": self.base_max_tokens}
        extra = os.getenv("JUDGE_EXTRA_BODY")  # e.g. {"reasoning_effort": "none"} for Groq Qwen3
        if extra:
            kwargs["extra_body"] = json.loads(extra)

        client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=3, timeout=120)
        llm = llm_factory(model=self.model, client=client, **kwargs)
        self._llm = llm

        self._metrics = {}
        if "faithfulness" in self.metric_names:
            self._metrics["faithfulness"] = Faithfulness(llm=llm)
        if "answer_relevancy" in self.metric_names:
            embeddings = HuggingFaceEmbeddings(
                model=embed_model, device=embed_device, normalize_embeddings=True
            )
            self._metrics["answer_relevancy"] = AnswerRelevancy(llm=llm, embeddings=embeddings)
        if "context_precision" in self.metric_names:
            self._metrics["context_precision"] = ContextPrecision(llm=llm)
        if "context_recall" in self.metric_names:
            self._metrics["context_recall"] = ContextRecall(llm=llm)

    def _set_max_tokens(self, value: int) -> None:
        self._llm.model_args["max_tokens"] = value

    def _call(self, name: str, fn: Callable):
        """Run one metric; grow max_tokens on truncation, back off on rate limits."""
        tokens = self.base_max_tokens
        last: BaseException | None = None
        for attempt in range(1, self.retries + 1):
            try:
                self._set_max_tokens(tokens)
                return float(fn().value)
            except Exception as exc:
                last = exc
                if "request too large" in str(exc).lower() and tokens > 512:
                    # provider caps output tokens per minute (e.g. Groq OTPM 1000): ask for less
                    tokens = max(512, tokens // 2)
                    print(f"[ragas] {name}: request too large -> retry with max_tokens={tokens}")
                elif is_truncation(exc) and tokens < self.max_tokens_cap:
                    tokens = min(tokens * 2, self.max_tokens_cap)
                    print(
                        f"[ragas] {name}: judge output truncated -> retry with max_tokens={tokens}"
                    )
                elif is_rate_limit(exc):
                    wait = min(60, 5 * 2 ** (attempt - 1))
                    print(f"[ragas] {name}: rate limited -> sleeping {wait}s")
                    time.sleep(wait)
                elif attempt < self.retries:
                    print(f"[ragas] {name}: {type(exc).__name__} -> retry {attempt}/{self.retries}")
                    time.sleep(2)
                else:
                    break
        self._set_max_tokens(self.base_max_tokens)
        raise last  # type: ignore[misc]

    def score(self, sample: dict) -> dict:
        """sample = {user_input, response, retrieved_contexts, reference}; failed metric -> None."""
        m = self._metrics
        self._set_max_tokens(self.base_max_tokens)
        calls = {
            "faithfulness": lambda: m["faithfulness"].score(
                user_input=sample["user_input"],
                response=sample["response"],
                retrieved_contexts=sample["retrieved_contexts"],
            ),
            "answer_relevancy": lambda: m["answer_relevancy"].score(
                user_input=sample["user_input"], response=sample["response"]
            ),
            "context_precision": lambda: m["context_precision"].score(
                user_input=sample["user_input"],
                reference=sample["reference"],
                retrieved_contexts=sample["retrieved_contexts"],
            ),
            "context_recall": lambda: m["context_recall"].score(
                user_input=sample["user_input"],
                reference=sample["reference"],
                retrieved_contexts=sample["retrieved_contexts"],
            ),
        }
        out: dict = {}
        for name in self.metric_names:
            try:
                out[name] = self._call(name, calls[name])
            except Exception as exc:  # one flaky judge call must not lose the whole run
                print(f"[ragas] {name} failed for {sample.get('id')}: {type(exc).__name__}: {exc}")
                out[name] = None
        return out

    def score_many(self, samples: list[dict]) -> list[dict]:
        rows = []
        for i, sample in enumerate(samples, 1):
            row = {"id": sample.get("id"), **self.score(sample)}
            rows.append(row)
            print(f"[ragas] {i}/{len(samples)} {row}")
        return rows


def aggregate(rows: list[dict], metrics: Iterable[str] = METRICS) -> dict:
    """Mean per metric over the rows where it was scored, plus how many were scored."""
    out: dict = {}
    for name in metrics:
        values = [r[name] for r in rows if r.get(name) is not None]
        out[name] = statistics.fmean(values) if values else None
        out[f"{name}_n"] = len(values)
    return out
