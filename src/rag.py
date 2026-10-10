"""The RAG pipeline: ``ingest`` (JSON -> vector store) and ``query`` (question -> cited answer).

One class, :class:`RAGPipeline`, is used by the FastAPI app (``main.py``), the BentoML service
(``src/serving/bento_service.py``) and every evaluation script, so what is evaluated is exactly
what is served.

CLI::

    civil-rag ingest --chunks data/processed/embedded-chunks-v2.json
    civil-rag query "ما هي شروط العقد؟"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from src.monitoring import metrics
from src.monitoring.cost import compute_cost, estimate_tokens
from src.monitoring.tracing import get_tracer

NO_SOURCES_AR = "لم يتم العثور على مصادر قانونية كافية للإجابة عن السؤال."

LEGAL_SYSTEM_PROMPT = """
You are an Egyptian Civil Code legal research assistant.

Answer the user's question using ONLY the legal sources provided
in the context.

Rules:

1. Do not use outside knowledge.
2. Do not invent legal provisions.
3. Do not make claims that are not supported by the provided sources.
4. Cite the relevant Egyptian Civil Code article numbers.
5. If an article is repealed, clearly state that it is repealed.
6. If the retrieved sources are insufficient to answer the question,
   say that the available sources are insufficient.
7. Answer in Arabic.
8. If the Arabic source text is unavailable but an English translation
   is available, use the available English source and do not invent
   an Arabic version.
9. Be concise and legally precise.
"""


# --------------------------------------------------------------------------- prompt helpers
def build_legal_context(sources: list[dict]) -> str:
    parts = []
    for source in sources:
        number = source.get("article_number")
        citation = source.get("citation") or f"Egyptian Civil Code, Article {number}"
        parts.append(
            f"""
--- SOURCE ---
Citation: {citation}
Article: {number}
Repealed: {source.get("is_repealed", False)}
Source status: {source.get("source_status") or "normal"}

Arabic:
{source.get("text_ar") or ""}

English:
{source.get("text_en") or ""}
--- END SOURCE ---
"""
        )
    return "\n".join(parts)


def build_prompt(question: str, sources: list[dict]) -> str:
    return f"""
{LEGAL_SYSTEM_PROMPT}

LEGAL SOURCES:
{build_legal_context(sources)}

USER QUESTION:
{question}

Answer the question in Arabic.
Mention the relevant article number(s).
Use only the provided legal sources.
"""


def source_context(source: dict) -> str:
    """The passage RAGAS sees as one retrieved context (same layout as the eval scripts)."""
    return (
        f"Article {source.get('article_number')}\n"
        f"Arabic:\n{source.get('text_ar') or ''}\n\n"
        f"English:\n{source.get('text_en') or ''}"
    )


def citations(sources: list[dict]) -> list[str]:
    """Article citations (not chunk ids), de-duplicated, in rank order."""
    seen, out = set(), []
    for s in sources:
        c = s.get("citation") or f"Egyptian Civil Code, Article {s.get('article_number')}"
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove Qwen3-style <think>…</think> reasoning from a complete answer."""
    text = _THINK_RE.sub("", text)
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


class ThinkFilter:
    """Streaming version of :func:`strip_think` (tags can be split across token deltas)."""

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False
        self._started = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = ""
        while True:
            tag = "</think>" if self._inside else "<think>"
            idx = self._buf.find(tag)
            if idx != -1:
                if not self._inside:
                    out += self._buf[:idx]
                self._buf = self._buf[idx + len(tag) :]
                self._inside = not self._inside
                continue
            # keep a possible partial tag at the end of the buffer
            keep = 0
            for k in range(min(len(tag) - 1, len(self._buf)), 0, -1):
                if tag.startswith(self._buf[-k:]):
                    keep = k
                    break
            if not self._inside:
                out += self._buf[: len(self._buf) - keep]
            self._buf = self._buf[len(self._buf) - keep :] if keep else ""
            return self._emit(out)

    def _emit(self, out: str) -> str:
        if not self._started:  # drop the blank line(s) a reasoning model leaves after </think>
            out = out.lstrip()
            self._started = bool(out)
        return out

    def flush(self) -> str:
        out = "" if self._inside else self._buf
        self._buf = ""
        return self._emit(out)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return str(content or "")


# --------------------------------------------------------------------------- result object
@dataclass
class RAGResult:
    answer: str
    sources: list[str]
    contexts: list[str] = field(default_factory=list)
    retrieved_articles: list[int] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    total_s: float = 0.0
    ttft_s: float | None = None
    trace_id: str | None = None

    def to_response(self) -> dict:
        return {"answer": self.answer, "sources": self.sources}


# --------------------------------------------------------------------------- pipeline
class RAGPipeline:
    def __init__(
        self,
        retriever,
        llm,
        *,
        provider: str | None = None,
        model_id: str | None = None,
        default_top_k: int = 5,
        drift=None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.provider = (provider or os.getenv("LLM_PROVIDER", "GROQ")).upper()
        self.model_id = model_id or ""
        self.default_top_k = default_top_k
        self.drift = drift if drift is not None else load_drift_monitor()
        self.tracer = get_tracer()

    # ----- construction -------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> RAGPipeline:
        from src.retrieval.retriever import LegalRetriever

        llm, provider, model_id = make_llm_from_env()
        return cls(LegalRetriever(), llm, provider=provider, model_id=model_id)

    # ----- stages -------------------------------------------------------------------------
    def _retrieve(self, question: str, top_k: int) -> list[dict]:
        return self.retriever.retrieve(query=question, top_k=top_k)

    def _observe_drift(self, question: str) -> None:
        if not self.drift:  # None (not configured) or False (disabled, e.g. in eval scripts)
            return
        try:
            embedder = self.retriever.embedding_provider
            self.drift.observe(embedder.embed_query(question))
        except Exception as exc:  # monitoring must never break answering
            print(f"[drift] skipped: {exc}")

    async def _retrieve_traced(self, question: str, top_k: int, root) -> list[dict]:
        span = root.child(
            "retrieval", as_type="retriever", input={"question": question, "top_k": top_k}
        )
        t0 = time.perf_counter()
        sources = await asyncio.to_thread(self._retrieve, question, top_k)
        elapsed = time.perf_counter() - t0
        metrics.STAGE_LATENCY.labels("retrieval").observe(elapsed)
        span.update(
            output={"articles": [s.get("article_number") for s in sources]},
            metadata={
                "latency_s": round(elapsed, 3),
                "scores": [s.get("rerank_score") for s in sources],
            },
        )
        span.end()
        asyncio.get_running_loop().run_in_executor(None, self._observe_drift, question)
        return sources

    def _finish_llm(self, prompt: str, text: str, usage: dict | None) -> tuple[int, int, float]:
        usage = usage or {}
        n_in = int(usage.get("input_tokens") or estimate_tokens(prompt))
        n_out = int(usage.get("output_tokens") or estimate_tokens(text))
        cost = compute_cost(self.provider, n_in, n_out)
        metrics.observe_llm(self.provider, n_in, n_out, cost)
        return n_in, n_out, cost

    # ----- public API ---------------------------------------------------------------------
    async def aanswer(self, question: str, top_k: int | None = None) -> RAGResult:
        top_k = top_k or self.default_top_k
        t_start = time.perf_counter()
        root = self.tracer.start("ask", input={"question": question, "top_k": top_k})
        try:
            sources = await self._retrieve_traced(question, top_k, root)
            retrieval_s = time.perf_counter() - t_start
            if not sources:
                metrics.EMPTY_RETRIEVALS.inc()
                root.update(output={"answer": NO_SOURCES_AR, "sources": []})
                return RAGResult(
                    NO_SOURCES_AR,
                    [],
                    retrieval_s=retrieval_s,
                    total_s=retrieval_s,
                    trace_id=root.trace_id,
                )

            prompt = build_prompt(question, sources)
            gen = root.child(
                "generation",
                as_type="generation",
                model=self.model_id,
                input=prompt,
                model_parameters={"temperature": 0},
            )
            t_gen = time.perf_counter()
            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(prompt)
            else:
                response = await asyncio.to_thread(self.llm.invoke, prompt)
            generation_s = time.perf_counter() - t_gen
            metrics.STAGE_LATENCY.labels("generation").observe(generation_s)

            answer = strip_think(_text(response.content))
            n_in, n_out, cost = self._finish_llm(
                prompt, answer, getattr(response, "usage_metadata", None)
            )
            gen.update(
                output=answer,
                usage_details={"input": n_in, "output": n_out},
                cost_details={"total": cost},
            )
            gen.end()

            result = RAGResult(
                answer=answer,
                sources=citations(sources),
                contexts=[source_context(s) for s in sources],
                retrieved_articles=[int(s["article_number"]) for s in sources],
                input_tokens=n_in,
                output_tokens=n_out,
                cost_usd=cost,
                retrieval_s=retrieval_s,
                generation_s=generation_s,
                total_s=time.perf_counter() - t_start,
                trace_id=root.trace_id,
            )
            root.update(
                output=result.to_response(), metadata={"cost_usd": cost, "tokens": n_in + n_out}
            )
            return result
        except Exception as exc:
            root.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            root.end()

    async def astream(self, question: str, top_k: int | None = None) -> AsyncIterator[dict]:
        """Yield ``{"event": "token", "data": str}`` deltas, then one ``{"event": "sources", ...}``
        and a final ``{"event": "done", "data": <RAGResult as dict>}``."""
        top_k = top_k or self.default_top_k
        t_start = time.perf_counter()
        root = self.tracer.start("ask_stream", input={"question": question, "top_k": top_k})
        try:
            sources = await self._retrieve_traced(question, top_k, root)
            if not sources:
                metrics.EMPTY_RETRIEVALS.inc()
                yield {"event": "token", "data": NO_SOURCES_AR}
                yield {"event": "sources", "data": []}
                root.update(output={"answer": NO_SOURCES_AR, "sources": []})
                return

            prompt = build_prompt(question, sources)
            gen = root.child(
                "generation",
                as_type="generation",
                model=self.model_id,
                input=prompt,
                model_parameters={"temperature": 0},
            )
            t_gen = time.perf_counter()
            ttft: float | None = None
            think = ThinkFilter()
            parts: list[str] = []
            usage: dict | None = None

            if hasattr(self.llm, "astream"):
                stream = self.llm.astream(prompt)
            else:  # a non-streaming LLM: emit the whole answer as one chunk

                async def _one():
                    resp = await asyncio.to_thread(self.llm.invoke, prompt)
                    yield resp

                stream = _one()

            async for chunk in stream:
                if getattr(chunk, "usage_metadata", None):
                    usage = chunk.usage_metadata
                delta = think.feed(_text(chunk.content))
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t_gen
                        metrics.TTFT.observe(ttft)
                    parts.append(delta)
                    yield {"event": "token", "data": delta}
            tail = think.flush()
            if tail:
                parts.append(tail)
                yield {"event": "token", "data": tail}

            generation_s = time.perf_counter() - t_gen
            metrics.STAGE_LATENCY.labels("generation").observe(generation_s)
            answer = "".join(parts).strip()
            n_in, n_out, cost = self._finish_llm(prompt, answer, usage)
            gen.update(
                output=answer,
                usage_details={"input": n_in, "output": n_out},
                cost_details={"total": cost},
                metadata={"ttft_s": ttft},
            )
            gen.end()

            result = RAGResult(
                answer=answer,
                sources=citations(sources),
                contexts=[source_context(s) for s in sources],
                retrieved_articles=[int(s["article_number"]) for s in sources],
                input_tokens=n_in,
                output_tokens=n_out,
                cost_usd=cost,
                generation_s=generation_s,
                ttft_s=ttft,
                total_s=time.perf_counter() - t_start,
                trace_id=root.trace_id,
            )
            root.update(output=result.to_response(), metadata={"cost_usd": cost, "ttft_s": ttft})
            yield {"event": "sources", "data": result.sources}
            yield {"event": "done", "data": result.__dict__}
        except Exception as exc:
            root.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            root.end()

    def answer(self, question: str, top_k: int | None = None) -> RAGResult:
        """Blocking helper for scripts and notebooks."""
        return asyncio.run(self.aanswer(question, top_k))


# --------------------------------------------------------------------------- factories
def make_llm_from_env():
    """``(llm, provider, model_id)`` for LLM_PROVIDER = OLLAMA | GROQ | GOOGLE_GENAI | VLLM."""
    from src.llm.LLMProviderFactory import LLMProviderFactory

    provider = os.getenv("LLM_PROVIDER", "OLLAMA").upper()
    defaults = {
        "OLLAMA": ("OLLAMA_MODEL_ID", "qwen3:1.7b"),
        "GROQ": ("GROQ_MODEL_ID", "qwen/qwen3.8-27b"),
        "GOOGLE_GENAI": ("GOOGLE_MODEL_ID", "gemini-2.5-flash"),
        "VLLM": ("VLLM_MODEL_ID", "Qwen/Qwen3-8B-AWQ"),
    }
    if provider not in defaults:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    env_key, default = defaults[provider]
    model_id = os.getenv(env_key, default)
    try:
        llm = LLMProviderFactory(config={}).create(
            provider=provider, model_id=model_id, model_temperature=0.0
        )
    except Exception as exc:  # e.g. missing API key: keep /health and /docs alive
        llm = UnavailableLLM(provider, str(exc))
    return llm, provider, model_id


class UnavailableLLM:
    """Stands in when the configured LLM cannot be created, and fails loudly on first use."""

    def __init__(self, provider: str, reason: str):
        self.message = (
            f"LLM provider {provider} is not usable: {reason}. "
            "Set the API key / URL in .env (see .env.example) and restart."
        )

    def invoke(self, *_, **__):
        raise RuntimeError(self.message)

    async def ainvoke(self, *_, **__):
        raise RuntimeError(self.message)


def load_drift_monitor():
    path = os.getenv("DRIFT_REFERENCE_PATH", "reports/drift_reference.npz")
    if not os.path.exists(path):
        return None
    from src.monitoring.drift import CosineDriftMonitor

    return CosineDriftMonitor.load(
        path,
        window=int(os.getenv("DRIFT_WINDOW", "100")),
        threshold=float(os.getenv("DRIFT_THRESHOLD", "0.15")),
    )


# --------------------------------------------------------------------------- ingest / query
def ingest(
    chunks_path: str,
    qdrant_url: str | None = None,
    collection: str | None = None,
    recreate: bool = True,
) -> int:
    """Embedded chunks JSON -> Qdrant. Returns the number of points in the collection."""
    from qdrant_client import QdrantClient

    from src.retrieval import indexing

    client = QdrantClient(url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333"))
    collection = collection or os.getenv("QDRANT_COLLECTION", indexing.DEFAULT_COLLECTION)
    with open(chunks_path, encoding="utf-8") as fh:
        chunks = json.load(fh)
    indexing.ensure_collection(client, collection, recreate=recreate)
    indexing.upsert_chunks(client, collection, chunks)
    return indexing.count_chunks(client, collection)


def query(question: str, top_k: int = 5) -> dict:
    return RAGPipeline.from_env().answer(question, top_k).to_response()


def cli() -> None:
    parser = argparse.ArgumentParser(prog="civil-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ing = sub.add_parser("ingest", help="embedded chunks JSON -> Qdrant")
    p_ing.add_argument("--chunks", default="data/processed/embedded-chunks-v2.json")
    p_ing.add_argument("--no-recreate", action="store_true")
    p_q = sub.add_parser("query", help="ask one question")
    p_q.add_argument("question")
    p_q.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.cmd == "ingest":
        n = ingest(args.chunks, recreate=not args.no_recreate)
        print(f"indexed {n} chunks")
    else:
        print(json.dumps(query(args.question, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
