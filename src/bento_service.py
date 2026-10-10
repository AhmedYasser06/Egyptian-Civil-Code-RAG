import os
from collections.abc import Generator

import bentoml
from pydantic import BaseModel

from src.llm.LLMProviderFactory import LLMProviderFactory
from src.retrieval.retriever import LegalRetriever
from src.routes.legal import LEGAL_SYSTEM_PROMPT, build_legal_context


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@bentoml.service(
    resources={"cpu": "4"},
    traffic={"timeout": 120},
)
class LegalRAGService:

    def __init__(self):
        # Retrieval
        self.retriever = LegalRetriever()

        # LLM configuration
        self.llm_provider = os.getenv(
            "LLM_PROVIDER",
            "GOOGLE_GENAI",
        ).upper()

        if self.llm_provider == "GOOGLE_GENAI":
            model = os.getenv(
                "GOOGLE_MODEL_ID",
                "gemini-2.5-flash",
            )

        elif self.llm_provider == "GROQ":
            model = os.getenv(
                "GROQ_MODEL_ID",
                "qwen/qwen3.8-27b",
            )

        elif self.llm_provider == "OLLAMA":
            model = os.getenv(
                "OLLAMA_MODEL_ID",
                "qwen3:1.7b",
            )

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {self.llm_provider}"
            )

        self.llm_model = model

        factory = LLMProviderFactory(config={})

        self.llm = factory.create(
            provider=self.llm_provider,
            model_id=self.llm_model,
            model_temperature=0.0,
        )

    def _build_prompt(self, question: str, sources: list[dict]) -> str:
        context = build_legal_context(sources)

        return f"""
{LEGAL_SYSTEM_PROMPT}

LEGAL SOURCES:
{context}

USER QUESTION:
{question}

Answer the question in Arabic.
Mention the relevant article number(s).
Use only the provided legal sources.
"""

    def _get_citations(self, sources: list[dict]) -> list[str]:
        article_numbers = [
            source.get("article_number")
            for source in sources
        ]

        return [
            f"Egyptian Civil Code, Article {article}"
            for article in article_numbers
        ]

    @bentoml.api
    def ask(self, question: str, top_k: int = 3) -> AskResponse:

        sources = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        if not sources:
            return AskResponse(
                answer=(
                    "لم يتم العثور على مصادر قانونية "
                    "كافية للإجابة عن السؤال."
                ),
                sources=[],
            )

        prompt = self._build_prompt(question, sources)

        response = self.llm.invoke(prompt)

        citations = self._get_citations(sources)

        return AskResponse(
            answer=response.content,
            sources=citations,
        )

    @bentoml.api
    def ask_stream(
        self,
        question: str,
        top_k: int = 3,
    ) -> Generator[str, None, None]:

        sources = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        if not sources:
            yield (
                "لم يتم العثور على مصادر قانونية "
                "كافية للإجابة عن السؤال."
            )
            return

        prompt = self._build_prompt(question, sources)

        for chunk in self.llm.stream(prompt):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content