"""Load the evaluation question sets (tolerant of the few shapes used in this repo)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _as_list(data: Any) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("results", "items", "questions", "data"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"cannot find a list of questions inside a {type(data).__name__}")


def _articles(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, str)):
        value = [value]
    out = []
    for v in value:
        if isinstance(v, dict):
            v = v.get("article_number")
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            pass
    return out


def _find_articles(row: dict) -> list[int]:
    """The gold article(s) of a question, whatever the key is called in the eval file."""
    for key in (
        "expected_articles",
        "ground_truth_articles",
        "reference_articles",
        "expected_article",
        "expected_article_numbers",
        "relevant_articles",
        "gold_articles",
        "article_number",
        "article",
        "articles",
    ):
        if row.get(key) is not None:
            return _articles(row[key])
    for key, value in row.items():  # last resort: any "...article..." field holding numbers
        if "article" in key.lower() and "citation" not in key.lower() and value is not None:
            found = _articles(value)
            if found:
                return found
    return []


def load_eval(path: str | Path, limit: int | None = None) -> list[dict]:
    """Normalise to ``{id, question, reference, expected_articles, language, category}``."""
    raw = _as_list(json.loads(Path(path).read_text(encoding="utf-8")))
    items = []
    for i, row in enumerate(raw):
        question = (
            row.get("question")
            or row.get("user_input")
            or row.get("query")
            or row.get("question_ar")
        )
        if not question:
            raise ValueError(f"{path}: row {i} has no question")
        items.append(
            {
                "id": row.get("id") or f"q{i + 1:03d}",
                "question": question,
                "reference": row.get("reference_answer")
                or row.get("ground_truth")
                or row.get("reference")
                or "",
                "expected_articles": _find_articles(row),
                "language": row.get("language"),
                "category": row.get("category"),
            }
        )
    return items[:limit] if limit else items


if __name__ == "__main__":  # python -m src.evaluation.dataset data/evaluation/rag_eval.json
    import sys

    for path in sys.argv[1:] or ["data/evaluation/rag_eval.json"]:
        items = load_eval(path)
        with_gold = sum(1 for i in items if i["expected_articles"])
        with_ref = sum(1 for i in items if i["reference"])
        print(
            f"{path}: {len(items)} questions, {with_gold} with gold articles, "
            f"{with_ref} with reference answers"
        )
        print("  first:", items[0])
