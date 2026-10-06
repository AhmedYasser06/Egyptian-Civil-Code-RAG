import os

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from src.llm.LLMProviderFactory import LLMProviderFactory
from src.retrieval.retriever import LegalRetriever
from src.schemas.legal_query import LegalQueryRequest
from src.schemas.response import LegalResponse
from src.schemas.sources import LegalSource

load_dotenv()


router = APIRouter(
    prefix="/api/v1/legal",
    tags=["Legal RAG"],
)


# RETRIEVER
retriever = LegalRetriever()


# LLM
llm_provider = os.getenv(
    "LLM_PROVIDER",
    "OLLAMA",
).upper()


if llm_provider == "OLLAMA":

    llm_model = os.getenv(
        "OLLAMA_MODEL_ID",
        "qwen3:1.7b",
    )

elif llm_provider == "GROQ":

    llm_model = os.getenv(
        "GROQ_MODEL_ID",
        "qwen/qwen3.8-27b",
    )

elif llm_provider == "GOOGLE_GENAI":

    llm_model = os.getenv(
        "GOOGLE_MODEL_ID",
        "gemini-2.5-flash",
    )

else:

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {llm_provider}"
    )


llm_factory = LLMProviderFactory(
    config={}
)


llm = llm_factory.create(
    provider=llm_provider,
    model_id=llm_model,
    model_temperature=0.0,
)


# prompt
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

# CONTEXT
def build_legal_context(
    sources: list[dict],
) -> str:

    context_parts = []

    for source in sources:

        article_number = source.get(
            "article_number"
        )

        text_ar = source.get(
            "text_ar"
        ) or ""

        text_en = source.get(
            "text_en"
        ) or ""

        citation = source.get(
            "citation"
        ) or f"Egyptian Civil Code, Article {article_number}"

        is_repealed = source.get(
            "is_repealed",
            False,
        )

        source_status = source.get(
            "source_status"
        ) or "normal"

        context_parts.append(
            f"""
--- SOURCE ---
Citation: {citation}
Article: {article_number}
Repealed: {is_repealed}
Source status: {source_status}

Arabic:
{text_ar}

English:
{text_en}
--- END SOURCE ---
"""
        )

    return "\n".join(context_parts)




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
    "/query",
    response_model=LegalResponse,
)
def query_legal(
    request: LegalQueryRequest,
):

    try:

        # 1. Retrieve
        sources = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
        )

        # 2. No sources
        if not sources:

            return LegalResponse(
                answer_ar=(
                    "لم يتم العثور على مصادر قانونية "
                    "كافية للإجابة عن السؤال."
                ),
                sources=[],
                cannot_answer=True,
            )

        # 3. Build context
        context = build_legal_context(
            sources
        )

        # 4. Prompt
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

        # 5. Generate
        response = llm.invoke(prompt)

        answer = response.content

        # 6. Pydantic sources
        legal_sources = [
            LegalSource(**source)
            for source in sources
        ]

        # 7. Response
        return LegalResponse(
            answer_ar=answer,
            sources=legal_sources,
            cannot_answer=False,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )