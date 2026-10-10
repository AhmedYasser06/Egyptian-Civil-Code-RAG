from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from src.monitoring.tracing import get_tracer  # noqa: E402
from src.routes.legal import router as legal_router  # noqa: E402
from src.routes.root import router as root_router  # noqa: E402


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    get_tracer().flush()  # make sure the last Langfuse traces are sent


app = FastAPI(
    title="Arabic Legal Document RAG",
    version="0.2.0",
    description="Egyptian Civil Code Q&A with article citations. POST /ask, GET /health.",
    lifespan=lifespan,
)

app.include_router(root_router)    # /ask  /ask/stream  /health  /metrics  /drift
app.include_router(legal_router)   # /api/v1/legal/*  (kept for backward compatibility)
