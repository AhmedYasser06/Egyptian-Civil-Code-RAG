import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.retrieval.retriever import LegalRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_eval.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "rag_results.json"
)

load_dotenv()

PROVIDER = "groq"
MODEL_ID = os.getenv(
    "GROQ_MODEL_ID",
    "qwen/qwen3.8-27b",
)

TEMPERATURE = 0.0
TOP_K = 3


def load_dataset():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results():
    if not OUTPUT_FILE.exists():
        return {}

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            item["id"]: item
            for item in data.get("results", [])
        }

    except Exception:
        return {}


def build_context(articles):
    contexts = []

    for article in articles:
        article_number = article["article_number"]

        text_ar = article.get("text_ar") or ""
        text_en = article.get("text_en") or ""

        context = (
            f"ARTICLE {article_number}\n"
            f"Arabic:\n{text_ar}\n\n"
            f"English:\n{text_en}"
        )

        contexts.append(context)

    return "\n\n---\n\n".join(contexts)


def build_prompt(question, context):
    return f"""
You are a legal question-answering assistant for the Egyptian Civil Code.

Answer the user's question ONLY using the provided legal context.

Rules:
1. Do not use outside knowledge.
2. Do not invent legal rules.
3. If the context does not contain enough information, say so.
4. Cite the relevant article number(s).
5. Give a concise answer.
6. Answer in the same language as the question.

Question:
{question}

Legal context:
{context}

Answer:
""".strip()


def create_llm():
    return ChatGroq(
        model=MODEL_ID,
        temperature=TEMPERATURE,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def extract_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if text:
                    parts.append(text)

        return "\n".join(parts).strip()

    return str(content).strip()

def save_results(results):
    output = {
        "config": {
            "provider": PROVIDER,
            "model": MODEL_ID,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
        },
        "results": results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of NEW questions to process.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Dataset index to start from.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LEGAL RAG GENERATION EVALUATION")
    print("=" * 70)

    dataset = load_dataset()

    print(f"Total evaluation questions: {len(dataset)}")
    print(f"Groq model: {MODEL_ID}")
    print(f"Top-K context: {TOP_K}")

    existing = load_existing_results()

    print(
        f"Already completed: {len(existing)}"
    )

    retriever = LegalRetriever()
    llm = create_llm()

    results = list(existing.values())

    processed_new = 0

    for index, item in enumerate(
        dataset[args.start:],
        start=args.start + 1,
    ):

        item_id = item["id"]

        if item_id in existing:
            print(
                f"\n[{index}/{len(dataset)}] "
                f"{item_id} already completed. Skipping."
            )
            continue

        if (
            args.limit is not None
            and processed_new >= args.limit
        ):
            break

        question = item["question"]

        print("\n" + "-" * 70)
        print(
            f"[{index}/{len(dataset)}] "
            f"{item_id}"
        )
        print(f"Question: {question}")

        try:

            # --------------------------------------------------
            # Retrieval
            # --------------------------------------------------

            retrieved = retriever.retrieve(
                query=question,
                top_k=TOP_K,
            )

            retrieved_articles = [
                article["article_number"]
                for article in retrieved
            ]

            print(
                f"Retrieved: {retrieved_articles}"
            )

            context = build_context(retrieved)

            prompt = build_prompt(
                question,
                context,
            )

            print("[GROQ] Generating answer...")

            response = llm.invoke(prompt)

            answer = extract_text(response)

            print(
                f"Answer: {answer}"
            )

            result = {
                "id": item_id,
                "pair_id": item["pair_id"],
                "language": item["language"],
                "question": question,
                "expected_articles": item["expected_articles"],
                "reference_answer": item["reference_answer"],
                "retrieved_articles": retrieved_articles,
                "context": context,
                "answer": answer,

                "provider": PROVIDER,
                "model": MODEL_ID,
            }

            results.append(result)

            # SAVE IMMEDIATELY
            save_results(results)

            print(
                f"[SAVED] {OUTPUT_FILE}"
            )

            processed_new += 1

            # Small delay to be conservative with free quota
            time.sleep(2)

        except Exception as e:

            print("\n[ERROR]")
            print(type(e).__name__)
            print(str(e))

            print(
                "\nStopping safely."
                "\nAlready completed results were saved."
            )

            save_results(results)

            break

    print("\n" + "=" * 70)
    print("RAG GENERATION EVALUATION FINISHED")
    print("=" * 70)

    print(
        f"Completed results: {len(results)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()