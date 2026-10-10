"""BentoML service wrapping the RAG pipeline. vLLM serves the generative model.

    # terminal 1: the LLM (OpenAI-compatible API on :8001)
    vllm serve Qwen/Qwen3-8B-AWQ --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.85
    # terminal 2: the RAG service (needs Qdrant on :6333, see docker-compose.yml)
    LLM_PROVIDER=VLLM VLLM_BASE_URL=http://localhost:8001/v1 \
        bentoml serve src.serving.bento_service:LegalRAG --port 3000

    # streaming: tokens appear progressively (-N = no curl buffering)
    curl -N -X POST localhost:3000/ask -H 'Content-Type: application/json' \
         -d '{"question": "ما هي شروط العقد؟"}'

Endpoints: ``/ask`` (async, streaming text: answer tokens then a ``Sources:`` line),
``/ask_json`` (async, ``{"answer", "sources"}``) and ``/health``.
"""

from __future__ import annotations

import importlib
import os
from typing import Annotated, AsyncGenerator

import bentoml
from pydantic import Field

Question = Annotated[str, Field(min_length=2, description="Legal question, Arabic or English")]
TopK = Annotated[int, Field(ge=1, le=20)]


def build_pipeline():
    """RAGPipeline.from_env(), or ``module:function`` from RAG_PIPELINE_FACTORY (tests / dev)."""
    factory = os.getenv("RAG_PIPELINE_FACTORY")
    if factory:
        module, func = factory.split(":")
        return getattr(importlib.import_module(module), func)()
    from src.rag import RAGPipeline

    return RAGPipeline.from_env()


@bentoml.service(
    name="legal_rag",
    traffic={"timeout": 300, "concurrency": int(os.getenv("BENTO_CONCURRENCY", "64"))},
)
class LegalRAG:
    def __init__(self) -> None:
        self.pipeline = build_pipeline()

    @bentoml.api
    async def ask(self, question: Question, top_k: TopK = 5) -> AsyncGenerator[str, None]:
        """Stream the answer token by token, then the article citations."""
        async for event in self.pipeline.astream(question, top_k):
            if event["event"] == "token":
                yield event["data"]
            elif event["event"] == "sources":
                yield "\n\nSources: " + "; ".join(event["data"]) + "\n"

    @bentoml.api
    async def ask_json(self, question: Question, top_k: TopK = 5) -> dict:
        result = await self.pipeline.aanswer(question, top_k)
        return result.to_response()

    @bentoml.api
    def health(self) -> dict:
        return {"status": "healthy", "service": "legal_rag"}
