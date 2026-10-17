"""Generate answers for data/evaluation/rag_eval.json with the SAME pipeline the API serves.

Uses ``RAGPipeline`` (src/rag.py), so the prompt (LEGAL_SYSTEM_PROMPT), retrieval, reranking and
think-stripping are identical to production and cannot drift. The provider comes from
LLM_PROVIDER (default GROQ here, as before); the model from the provider's *_MODEL_ID variable.

Output: data/evaluation/rag_results.json (read by prepare_ragas_dataset.py).
Resumable: questions already in the output file are skipped.
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LLM_PROVIDER", "GROQ")  # this script historically used Groq

from src.rag import RAGPipeline, make_llm_from_env  # noqa: E402
from src.retrieval.retriever import LegalRetriever  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = PROJECT_ROOT / "data" / "evaluation" / "rag_eval.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "evaluation" / "rag_results.json"

TEMPERATURE = 0.0  # recorded in the output; the pipeline's LLM factory sets the real value
TOP_K = 3

PROVIDER = None  # filled in main() from make_llm_from_env()
MODEL_ID = None


def load_dataset():
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results():
    if not OUTPUT_FILE.exists():
        return {}

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {item["id"]: item for item in data.get("results", [])}

    except Exception:
        return {}


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

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main():
    global PROVIDER, MODEL_ID

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

    llm, PROVIDER, MODEL_ID = make_llm_from_env()
    pipeline = RAGPipeline(
        LegalRetriever(),
        llm,
        provider=PROVIDER,
        model_id=MODEL_ID,
        drift=False,  # no drift monitoring in offline evaluation
    )

    print(f"Total evaluation questions: {len(dataset)}")
    print(f"Provider / model: {PROVIDER} / {MODEL_ID}")
    print(f"Top-K context: {TOP_K}")

    existing = load_existing_results()
    print(f"Already completed: {len(existing)}")

    results = list(existing.values())
    processed_new = 0

    for index, item in enumerate(dataset[args.start:], start=args.start + 1):

        item_id = item["id"]

        if item_id in existing:
            print(f"\n[{index}/{len(dataset)}] {item_id} already completed. Skipping.")
            continue

        if args.limit is not None and processed_new >= args.limit:
            break

        question = item["question"]

        print("\n" + "-" * 70)
        print(f"[{index}/{len(dataset)}] {item_id}")
        print(f"Question: {question}")

        try:
            # Retrieval + prompt + generation all happen inside the shared pipeline.
            rag = pipeline.answer(question, top_k=TOP_K)

            retrieved_articles = rag.retrieved_articles
            context = "\n\n---\n\n".join(rag.contexts)
            answer = rag.answer

            print(f"Retrieved: {retrieved_articles}")
            print(f"Answer: {answer}")

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
            print(f"[SAVED] {OUTPUT_FILE}")

            processed_new += 1

            # Small delay to be conservative with free quota
            time.sleep(2)

        except Exception as e:
            print("\n[ERROR]")
            print(type(e).__name__)
            print(str(e))
            print("\nStopping safely.\nAlready completed results were saved.")
            save_results(results)
            break

    print("\n" + "=" * 70)
    print("RAG GENERATION EVALUATION FINISHED")
    print("=" * 70)
    print(f"Completed results: {len(results)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()