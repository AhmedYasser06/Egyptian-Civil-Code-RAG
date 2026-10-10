from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from src.rag import (  # noqa: F401  (re-exported: tests and notebooks import them from here)
    LEGAL_SYSTEM_PROMPT,
    build_legal_context,
    make_llm_from_env,
)
from src.retrieval.retriever import LegalRetriever
from src.schemas.ask import AskResponse
from src.schemas.legal_query import LegalQueryRequest
from src.schemas.response import LegalResponse
from src.schemas.sources import LegalSource

load_dotenv()


router = APIRouter(
    prefix="/api/v1/legal",
    tags=["Legal RAG"],
)

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_indexed": 1149,
    }


# RETRIEVER
retriever = LegalRetriever()


# LLM (OLLAMA | GROQ | GOOGLE_GENAI | VLLM, chosen by LLM_PROVIDER; see src/rag.py)
llm, llm_provider, llm_model = make_llm_from_env()


# RETRIEVAL ENDPOINT
@router.post("/retrieve")
def retrieve_legal_chunks(
    request: LegalQueryRequest,
):

    try:

        results = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        return {
            "question": request.question,
            "results": results,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
        
            
# RAG QUERY ENDPOINT
@router.post(
    "/ask",
    response_model=AskResponse,
)
def ask_legal(
    request: LegalQueryRequest,
):
    try:

        sources = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        if not sources:
            return AskResponse(
                answer=(
                    "لم يتم العثور على مصادر قانونية "
                    "كافية للإجابة عن السؤال."
                ),
                sources=[],
            )

        context = build_legal_context(sources)

        prompt = f"""
{LEGAL_SYSTEM_PROMPT}

LEGAL SOURCES:
{context}

USER QUESTION:
{request.question}

Answer the question in Arabic.
Mention the relevant article number(s).
Use only the provided legal sources.
"""

        response = llm.invoke(prompt)

        legal_sources = [
            LegalSource(**source)
            for source in sources
        ]

        citations = [
            source.citation
            for source in legal_sources
            if source.citation
        ]

        return AskResponse(
            answer=response.content,
            sources=citations,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
        
@router.post(
    "/query",
    response_model=LegalResponse,
)
def query_legal(
    request: LegalQueryRequest,
):
    try:

        sources = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        if not sources:
            return LegalResponse(
                answer_ar=(
                    "لم يتم العثور على مصادر قانونية "
                    "كافية للإجابة عن السؤال."
                ),
                sources=[],
                cannot_answer=True,
            )

        context = build_legal_context(sources)

        prompt = f"""
{LEGAL_SYSTEM_PROMPT}

LEGAL SOURCES:
{context}

USER QUESTION:
{request.question}

Answer the question in Arabic.
Mention the relevant article number(s).
Use only the provided legal sources.
"""

        response = llm.invoke(prompt)

        legal_sources = [
            LegalSource(**source)
            for source in sources
        ]

        return LegalResponse(
            answer_ar=response.content,
            sources=legal_sources,
            cannot_answer=False,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )