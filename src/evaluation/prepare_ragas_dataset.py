import json
from pathlib import Path


RESULTS_FILE = Path("data/evaluation/rag_results.json")
EVAL_FILE = Path("data/evaluation/rag_eval.json")
ARTICLES_FILE = Path("data/processed/final-articles-v2.json")
OUTPUT_FILE = Path("data/evaluation/ragas_dataset.json")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_list(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["results", "items", "questions", "data"]:
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Cannot find list inside {type(data)}")


def normalize_article_numbers(value):
    if value is None:
        return []

    if isinstance(value, int):
        return [value]

    if isinstance(value, str):
        return [int(value)]

    return [int(x) for x in value]


def article_context(article):
    number = int(article["article_number"])

    ar = article.get("text_ar") or ""
    en = article.get("text_en") or ""

    return (
        f"Article {number}\n"
        f"Arabic:\n{ar}\n\n"
        f"English:\n{en}"
    )


print("=" * 70)
print("PREPARING RAGAS DATASET")
print("=" * 70)

results = get_list(load_json(RESULTS_FILE))
eval_questions = get_list(load_json(EVAL_FILE))
articles = get_list(load_json(ARTICLES_FILE))

print(f"RAG results: {len(results)}")
print(f"Evaluation questions: {len(eval_questions)}")
print(f"Articles: {len(articles)}")


# ============================================================
# Article lookup
# ============================================================

article_map = {
    int(a["article_number"]): a
    for a in articles
}


results_map = {
    item["id"]: item
    for item in results
}


dataset = []


# ============================================================
# Build dataset
# ============================================================

for eval_item in eval_questions:

    qid = eval_item["id"]

    if qid not in results_map:
        print(f"[WARNING] Missing generated result: {qid}")
        continue

    result = results_map[qid]

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question = (
        result.get("question")
        or eval_item.get("question")
        or eval_item.get("user_input")
    )

    if not question:
        raise ValueError(f"{qid}: missing question")


    # --------------------------------------------------------
    # Generated answer
    # --------------------------------------------------------

    response = (
        result.get("answer")
        or result.get("response")
    )

    if not response:
        raise ValueError(f"{qid}: missing response")


    # --------------------------------------------------------
    # Expected articles
    # --------------------------------------------------------

    expected_articles = (
        eval_item.get("expected_articles")
        or eval_item.get("ground_truth_articles")
        or eval_item.get("reference_articles")
    )

    expected_articles = normalize_article_numbers(
        expected_articles
    )

    if not expected_articles:
        raise ValueError(
            f"{qid}: no expected_articles found"
        )


    # --------------------------------------------------------
    # Reference answer
    # --------------------------------------------------------

    reference = (
        eval_item.get("reference_answer")
        or eval_item.get("ground_truth")
        or eval_item.get("reference")
    )

    if not reference:
        raise ValueError(
            f"{qid}: no reference answer found"
        )


    # ========================================================
    # Retrieved article numbers
    # ========================================================

    retrieved_numbers = []

    retrieved_articles = result.get(
        "retrieved_articles",
        []
    )


    for item in retrieved_articles:

        # Case 1:
        # retrieved_articles = [5, 988, 51]

        if isinstance(item, int):
            retrieved_numbers.append(item)
            continue


        # Case 2:
        # retrieved_articles = ["5", "988", "51"]

        if isinstance(item, str):
            try:
                retrieved_numbers.append(int(item))
            except ValueError:
                pass

            continue


        # Case 3:
        # retrieved_articles = [
        #     {"article_number": 5},
        #     ...
        # ]

        if isinstance(item, dict):

            number = (
                item.get("article_number")
                or item.get("article")
                or item.get("number")
            )

            if number is not None:
                retrieved_numbers.append(
                    int(number)
                )


    # --------------------------------------------------------
    # Fallback fields
    # --------------------------------------------------------

    if not retrieved_numbers:

        fallback = (
            result.get("retrieved_article_numbers")
            or result.get("retrieved_articles_numbers")
            or result.get("retrieved")
        )

        if fallback:
            retrieved_numbers = normalize_article_numbers(
                fallback
            )


    # Remove duplicates while preserving order

    retrieved_numbers = list(
        dict.fromkeys(retrieved_numbers)
    )


    if not retrieved_numbers:
        raise ValueError(
            f"{qid}: cannot determine retrieved article numbers"
        )


    # ========================================================
    # Build FULL retrieved contexts
    # ========================================================

    retrieved_contexts = []

    for number in retrieved_numbers:

        if number not in article_map:
            raise ValueError(
                f"{qid}: retrieved Article {number} "
                f"not found in corpus"
            )

        retrieved_contexts.append(
            article_context(article_map[number])
        )


    # ========================================================
    # Build ground-truth contexts
    # ========================================================

    reference_contexts = []

    for number in expected_articles:

        if number not in article_map:
            raise ValueError(
                f"{qid}: expected Article {number} "
                f"not found in corpus"
            )

        reference_contexts.append(
            article_context(article_map[number])
        )


    # ========================================================
    # Dataset record
    # ========================================================

    sample = {
        "id": qid,
        "user_input": question,
        "response": response,
        "reference": reference,

        # IMPORTANT:
        # These are FULL article texts, NOT article numbers.
        "retrieved_contexts": retrieved_contexts,

        "reference_contexts": reference_contexts,

        "expected_articles": expected_articles,

        "retrieved_article_numbers": retrieved_numbers,

        "generation_provider": result.get(
            "provider"
        ),

        "generation_model": result.get(
            "model"
        ),
    }


    dataset.append(sample)


# ============================================================
# Save
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# Validation
# ============================================================

print()
print("=" * 70)
print("DATASET READY")
print("=" * 70)

print(f"Samples: {len(dataset)}")
print(f"Output: {OUTPUT_FILE}")


# Verify first sample

first = dataset[0]

print()
print("VALIDATION - FIRST SAMPLE")
print("-" * 70)

print("ID:")
print(first["id"])

print()
print("Expected articles:")
print(first["expected_articles"])

print()
print("Retrieved article numbers:")
print(first["retrieved_article_numbers"])

print()
print("First retrieved context preview:")
print(
    repr(first["retrieved_contexts"][0][:500])
)

print()
print("First reference context preview:")
print(
    repr(first["reference_contexts"][0][:500])
)


# Hard validation

if first["retrieved_contexts"][0].isdigit():
    raise RuntimeError(
        "ERROR: retrieved_contexts still contain "
        "article numbers instead of article text."
    )

if not first["retrieved_contexts"][0].startswith("Article"):
    raise RuntimeError(
        "ERROR: retrieved context does not look "
        "like canonical article text."
    )


print()
print("VALIDATION PASSED")
