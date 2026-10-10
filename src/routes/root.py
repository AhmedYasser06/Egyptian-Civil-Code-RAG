"""Root-level API required by the spec: ``POST /ask``, ``GET /health`` (+ streaming, metrics).

The older ``/api/v1/legal/*`` routes stay available. Handlers are ``async``: retrieval/reranking
run in a worker thread so the event loop keeps serving other requests, and the LLM is awaited.
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from src.monitoring import metrics
from src.monitoring.online_eval import OnlineFaithfulness
from src.rag import RAGPipeline
from src.retrieval import indexing
from src.routes import legal
from src.schemas.ask import AskResponse
from src.schemas.legal_query import LegalQueryRequest

router = APIRouter(tags=["Legal RAG"])

_cache: dict = {}
online_eval = OnlineFaithfulness()  # off unless ONLINE_FAITHFULNESS_SAMPLE_RATE > 0


def get_pipeline() -> RAGPipeline:
    """Built from the retriever/LLM already loaded by ``legal`` (models are loaded once).
    Rebuilt if those objects are swapped (tests monkeypatch them)."""
    key = (id(legal.retriever), id(legal.llm))
    if _cache.get("key") != key:
        _cache["key"] = key
        _cache["pipeline"] = RAGPipeline(
            legal.retriever,
            legal.llm,
            provider=legal.llm_provider,
            model_id=legal.llm_model,
        )
    return _cache["pipeline"]


# ----------------------------------------------------------------------------- /ask
@router.post("/ask", response_model=AskResponse)
async def ask(request: LegalQueryRequest, response: Response) -> AskResponse:
    started = time.perf_counter()
    try:
        result = await get_pipeline().aanswer(request.question, request.top_k)
    except Exception as exc:
        metrics.REQUESTS.labels("/ask", "error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    metrics.REQUESTS.labels("/ask", "ok").inc()
    metrics.REQUEST_LATENCY.labels("/ask").observe(time.perf_counter() - started)
    online_eval.maybe_score(request.question, result)
    if result.trace_id:
        response.headers["X-Trace-Id"] = result.trace_id
    return AskResponse(answer=result.answer, sources=result.sources)


@router.post("/ask/stream")
async def ask_stream(request: LegalQueryRequest) -> StreamingResponse:
    """Server-sent events: ``event: token`` per delta, then ``event: sources``. Use ``curl -N``."""
    pipeline = get_pipeline()

    async def events():
        started = time.perf_counter()
        try:
            async for item in pipeline.astream(request.question, request.top_k):
                if item["event"] == "done":
                    continue
                data = json.dumps(item["data"], ensure_ascii=False)
                yield f"event: {item['event']}\ndata: {data}\n\n"
            metrics.REQUESTS.labels("/ask/stream", "ok").inc()
        except Exception as exc:
            metrics.REQUESTS.labels("/ask/stream", "error").inc()
            yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
        finally:
            metrics.REQUEST_LATENCY.labels("/ask/stream").observe(time.perf_counter() - started)

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


# ----------------------------------------------------------------------------- /health
_health_cache: dict = {"at": 0.0, "n": None}


def _count_articles() -> int:
    retriever = legal.retriever
    return indexing.count_articles(retriever.client, retriever.collection_name)


@router.get("/health")
async def health():
    """``{"status": "healthy", "documents_indexed": N}`` where N = articles in the vector store."""
    try:
        if _health_cache["n"] is None or time.time() - _health_cache["at"] > 60:
            _health_cache["n"] = await asyncio.to_thread(_count_articles)
            _health_cache["at"] = time.time()
        n = _health_cache["n"]
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "documents_indexed": 0, "detail": str(exc)},
        )
    return {"status": "healthy" if n > 0 else "unhealthy", "documents_indexed": n}


# ----------------------------------------------------------------------------- ops
@router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@router.get("/drift")
def drift_status():
    monitor = get_pipeline().drift
    if not monitor:
        return {"enabled": False}
    return {"enabled": True, "threshold": monitor.threshold, "stats": monitor.stats()}
