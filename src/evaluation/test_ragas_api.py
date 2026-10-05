import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


load_dotenv()

DATASET_FILE = Path("data/evaluation/ragas_dataset.json")

with open(DATASET_FILE, encoding="utf-8") as f:
    data = json.load(f)

sample = data[0]

print("=" * 70)
print("RAGAS API SMOKE TEST")
print("=" * 70)

print("Sample:", sample["id"])
print("Question:", sample["user_input"])
print("Expected articles:", sample["expected_articles"])
print("Retrieved articles:", sample["retrieved_article_numbers"])


# ============================================================
# Groq client
# ============================================================

model = os.getenv("GROQ_MODEL_ID", "qwen/qwen3.8-27b")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

print()
print("Groq model:", model)

groq_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# RAGAS modern LLM
# ============================================================

print("Creating RAGAS InstructorLLM...")

judge_llm = llm_factory(
    model=model,
    client=groq_client,
    temperature=0,
)


# ============================================================
# Local BGE-M3 embeddings
# ============================================================

print("Loading BGE-M3 embeddings...")

judge_embeddings = HuggingFaceEmbeddings(
    model="BAAI/bge-m3",
    device="cpu",
    normalize_embeddings=True,
)


# ============================================================
# Metrics
# ============================================================

print()
print("Initializing metrics...")

faithfulness = Faithfulness(
    llm=judge_llm
)

answer_relevancy = AnswerRelevancy(
    llm=judge_llm,
    embeddings=judge_embeddings,
)

context_precision = ContextPrecision(
    llm=judge_llm
)

context_recall = ContextRecall(
    llm=judge_llm
)

print("Metrics initialized successfully.")


# ============================================================
# 1. Faithfulness
# ============================================================

print()
print("[1/4] Faithfulness...")

faith = faithfulness.score(
    user_input=sample["user_input"],
    response=sample["response"],
    retrieved_contexts=sample["retrieved_contexts"],
)

print("Faithfulness:", faith.value)


# ============================================================
# 2. Answer Relevancy
# ============================================================

print()
print("[2/4] Answer Relevancy...")

relevancy = answer_relevancy.score(
    user_input=sample["user_input"],
    response=sample["response"],
)

print("Answer Relevancy:", relevancy.value)


# ============================================================
# 3. Context Precision
# ============================================================

print()
print("[3/4] Context Precision...")

precision = context_precision.score(
    user_input=sample["user_input"],
    reference=sample["reference"],
    retrieved_contexts=sample["retrieved_contexts"],
)

print("Context Precision:", precision.value)


# ============================================================
# 4. Context Recall
# ============================================================

print()
print("[4/4] Context Recall...")

recall = context_recall.score(
    user_input=sample["user_input"],
    reference=sample["reference"],
    retrieved_contexts=sample["retrieved_contexts"],
)

print("Context Recall:", recall.value)


# ============================================================
# Done
# ============================================================

print()
print("=" * 70)
print("RAGAS SMOKE TEST PASSED")
print("=" * 70)
