"""Corpus validation (the rubric's "validate before you embed"). Runs after `dvc pull`/`dvc repro`;
skipped when the corpus is not present so a bare checkout still passes."""

import json
from pathlib import Path

import pytest

ARTICLES = Path("data/processed/final-articles-v2.json")
pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(not ARTICLES.exists(), reason="corpus not pulled (run `dvc pull`)"),
]

MAX_ARABIC_CHARS = 8000  # a longer record means the article splitter failed


@pytest.fixture(scope="module")
def articles():
    data = json.loads(ARTICLES.read_text(encoding="utf-8"))
    return data["articles"] if isinstance(data, dict) and "articles" in data else data


def test_article_numbers_are_integers_and_unique(articles):
    numbers = [a["article_number"] for a in articles]
    assert all(isinstance(n, int) for n in numbers)
    assert len(numbers) == len(set(numbers))


def test_article_numbers_are_contiguous(articles):
    numbers = sorted(a["article_number"] for a in articles)
    missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
    assert not missing, f"unexplained gaps in article numbering: {missing[:20]}"


def test_every_live_article_has_arabic_text(articles):
    empty = [
        a["article_number"]
        for a in articles
        if not a.get("is_repealed") and not (a.get("text_ar") or "").strip()
    ]
    # tolerated only if the record says why (e.g. source_status = "english_only")
    unexplained = [
        a["article_number"]
        for a in articles
        if a["article_number"] in empty and (a.get("source_status") or "normal") == "normal"
    ]
    assert not unexplained, f"articles without Arabic text and no source_status: {unexplained[:20]}"


def test_no_record_is_suspiciously_long(articles):
    too_long = [
        a["article_number"] for a in articles if len(a.get("text_ar") or "") > MAX_ARABIC_CHARS
    ]
    assert not too_long, f"likely failed split (giant records): {too_long}"


def test_repealed_articles_are_flagged_not_deleted(articles):
    numbers = {a["article_number"] for a in articles}
    repealed = [a for a in articles if a.get("is_repealed")]
    assert repealed, "no repealed article is flagged"
    assert {a["article_number"] for a in repealed} <= numbers
    assert all(isinstance(a["is_repealed"], bool) for a in articles)


def test_every_record_has_a_citation(articles):
    missing = [
        a["article_number"]
        for a in articles
        if not (a.get("citation") or "").startswith("Egyptian Civil Code, Article")
    ]
    assert not missing, f"records without a proper citation: {missing[:20]}"
