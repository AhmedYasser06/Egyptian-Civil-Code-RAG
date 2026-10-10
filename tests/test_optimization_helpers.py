from src.optimization.compare_llms import pct
from src.optimization.distill_reranker import first_sentence, passage, synth_queries


def test_percentiles():
    assert pct([1, 2, 3, 4, 5], 0.5) == 3
    assert pct(list(range(1, 101)), 0.95) > 95


def test_synthetic_queries_cover_topic_and_both_languages():
    article = {
        "article_number": 147,
        "topic": "The Effects of a Contract",
        "text_en": "The contract makes the law of the parties. It can be revoked only by consent.",
        "text_ar": "العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين.",
    }
    queries = synth_queries(article)
    assert queries[0] == "The Effects of a Contract" and len(queries) == 3
    assert first_sentence(article["text_en"]) == "The contract makes the law of the parties."
    assert passage(article).startswith("Article 147")
