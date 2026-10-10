import json
from pathlib import Path

RESULTS_FILE = Path("data/evaluation/rag_results_gemini_50.json")
EVAL_FILE = Path("data/evaluation/rag_eval_gemini_50.json")
ARTICLES_FILE = Path("data/processed/final-articles-v2.json")
OUTPUT_FILE = Path("data/evaluation/ragas_dataset_gemini_50.json")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def article_context(article):
    number = article["article_number"]
    ar = article.get("text_ar") or ""
    en = article.get("text_en") or ""
    return f"Article {number}\nArabic:\n{ar}\n\nEnglish:\n{en}"


def main():
    results = load_json(RESULTS_FILE)
    eval_questions = load_json(EVAL_FILE)
    articles = load_json(ARTICLES_FILE)

    article_map = {int(a["article_number"]): a for a in articles}
    results_map = {item["id"]: item for item in results}

    dataset = []

    for eval_item in eval_questions:
        qid = eval_item["id"]

        if qid not in results_map:
            print(f"[WARNING] Missing generated result: {qid}")
            continue

        result = results_map[qid]
        expected_articles = [int(x) for x in eval_item["expected_articles"]]

        retrieved_contexts = []

        for item in result.get("retrieved_articles", []):
            number = item.get("article_number")
            if number is not None and int(number) in article_map:
                retrieved_contexts.append(
                    article_context(article_map[int(number)])
                )

        if not retrieved_contexts:
            raise ValueError(f"{qid}: cannot construct retrieved_contexts")

        reference_contexts = []
        for number in expected_articles:
            if number not in article_map:
                raise ValueError(
                    f"{qid}: expected Article {number} not found in corpus"
                )
            reference_contexts.append(article_context(article_map[number]))

        dataset.append({
            "id": qid,
            "user_input": result["question"],
            "response": result["answer"],
            "reference": eval_item["reference_answer"],
            "retrieved_contexts": retrieved_contexts,
            "reference_contexts": reference_contexts,
            "expected_articles": expected_articles,
            "retrieved_article_numbers": result.get(
                "retrieved_article_numbers", []
            ),
            "generation_provider": result.get("provider"),
            "generation_model": result.get("model"),
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("GEMINI RAGAS DATASET READY")
    print("=" * 70)
    print(f"Samples: {len(dataset)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
