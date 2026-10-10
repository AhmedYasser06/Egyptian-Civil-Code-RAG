import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.retrieval.retriever import LegalRetriever

load_dotenv()

DATASET_FILE = Path("data/evaluation/rag_eval_gemini_50.json")
OUTPUT_FILE = Path("data/evaluation/rag_results_gemini_50.json")


LEGAL_SYSTEM_PROMPT = """
You are an Egyptian Civil Code legal research assistant.

Answer the user's question using ONLY the legal sources provided in the context.

Rules:
1. Do not use outside knowledge.
2. Do not invent legal provisions.
3. Do not make claims that are not supported by the provided sources.
4. Cite the relevant Egyptian Civil Code article numbers.
5. If an article is repealed, clearly state that it is repealed.
6. If the retrieved sources are insufficient to answer the question, say that the available sources are insufficient.
7. Answer in the same language as the user.
8. If the Arabic source text is unavailable but an English translation is available, use the available English source and do not invent an Arabic version.
9. Be concise and legally precise.
"""


def build_legal_context(sources):
    context_parts = []
    for source in sources:
        article_number = source.get("article_number")
        text_ar = source.get("text_ar") or ""
        text_en = source.get("text_en") or ""
        citation = source.get("citation") or f"Egyptian Civil Code, Article {article_number}"
        is_repealed = source.get("is_repealed", False)
        source_status = source.get("source_status") or "normal"

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


TOP_K = int(os.getenv("RAG_EVAL_TOP_K", "3"))
DELAY_SECONDS = float(os.getenv("RAG_EVAL_DELAY_SECONDS", "1.0"))
MODEL_ID = os.getenv("GOOGLE_MODEL_ID", "gemini-2.5-flash")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(OUTPUT_FILE)


def source_for_output(source):
    return {
        "article_number": source.get("article_number"),
        "text_ar": source.get("text_ar") or "",
        "text_en": source.get("text_en") or "",
        "citation": source.get("citation"),
        "is_repealed": source.get("is_repealed", False),
        "source_status": source.get("source_status", "normal"),
        "rerank_score": source.get("rerank_score"),
    }


def main():
    dataset = load_json(DATASET_FILE)

    completed = {}
    if OUTPUT_FILE.exists():
        try:
            completed = {x["id"]: x for x in load_json(OUTPUT_FILE)}
        except Exception:
            completed = {}

    retriever = LegalRetriever()

    llm = ChatGoogleGenerativeAI(
        model=MODEL_ID,
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    print("=" * 70)
    print("GEMINI RAG GENERATION EVALUATION")
    print("=" * 70)
    print(f"Total questions: {len(dataset)}")
    print(f"Gemini model: {MODEL_ID}")
    print(f"Top-K: {TOP_K}")
    print(f"Already completed: {len(completed)}")

    for index, item in enumerate(dataset, start=1):
        qid = item["id"]

        if qid in completed:
            print(f"[{index}/{len(dataset)}] {qid} already completed. Skipping.")
            continue

        question = item["question"]

        print("-" * 70)
        print(f"[{index}/{len(dataset)}] {qid}")
        print(f"Question: {question}")

        try:
            sources = retriever.retrieve(
                query=question,
                top_k=TOP_K,
            )

            article_numbers = [
                int(s["article_number"]) for s in sources if s.get("article_number") is not None
            ]

            print(f"Retrieved: {article_numbers}")

            context = build_legal_context(sources)

            prompt = f"""
{LEGAL_SYSTEM_PROMPT}

LEGAL SOURCES:
{context}

USER QUESTION:
{question}

Answer the question in the same language as the user.
Mention the relevant Egyptian Civil Code article number(s).
Use ONLY the provided legal sources.
"""

            print("[GEMINI] Generating answer...")
            response = llm.invoke(prompt)

            answer = (
                response.content if isinstance(response.content, str) else str(response.content)
            )

            result = {
                "id": qid,
                "question": question,
                "answer": answer,
                "expected_articles": item["expected_articles"],
                "reference_answer": item["reference_answer"],
                "retrieved_article_numbers": article_numbers,
                "retrieved_articles": [source_for_output(source) for source in sources],
                "provider": "GOOGLE_GENAI",
                "model": MODEL_ID,
            }

            completed[qid] = result
            save_json(list(completed.values()))

            print(f"[SAVED] {OUTPUT_FILE}")

        except Exception as exc:
            print(f"[ERROR] {qid}: {exc}")
            print("Continuing with the next question...")
            continue

        time.sleep(DELAY_SECONDS)

    ordered = [completed[item["id"]] for item in dataset if item["id"] in completed]
    save_json(ordered)

    print("=" * 70)
    print("GEMINI GENERATION FINISHED")
    print(f"Completed: {len(ordered)}/{len(dataset)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
