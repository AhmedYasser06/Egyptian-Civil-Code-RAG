from dotenv import load_dotenv
from fastapi import FastAPI

from src.routes.legal import router as legal_router

load_dotenv()

app = FastAPI(
    title="Arabic Legal Document RAG"
)

app.include_router(legal_router)