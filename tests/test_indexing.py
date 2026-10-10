from qdrant_client import QdrantClient

from src.retrieval import indexing


def chunk(article, lang, idx=0, repealed=False):
    return {
        "chunk_id": f"a{article}-{lang}-{idx}",
        "article_number": article,
        "language": lang,
        "chunk_index": idx,
        "chunk_type": "article",
        "text": f"text {article} {lang}",
        "embedding": [0.1 * (article % 7 + 1)] * indexing.VECTOR_SIZE,
        "metadata": {
            "book": "B",
            "citation": f"Egyptian Civil Code, Article {article}",
            "is_repealed": repealed,
            "source_page": 3,
        },
    }


def test_chunk_to_point_keeps_the_citation_payload():
    point = indexing.chunk_to_point(chunk(147, "ar"))
    assert point.payload["article_number"] == 147
    assert point.payload["citation"] == "Egyptian Civil Code, Article 147"
    assert point.id == indexing.make_point_id("a147-ar-0")


def test_upsert_is_idempotent_and_counts_distinct_articles():
    client = QdrantClient(":memory:")
    indexing.ensure_collection(client, "c", recreate=True)
    chunks = [chunk(1, "ar"), chunk(1, "en"), chunk(2, "ar"), chunk(3, "en", repealed=True)]
    indexing.upsert_chunks(client, "c", chunks)
    indexing.upsert_chunks(client, "c", chunks)  # again: same ids, no duplicates
    assert indexing.count_chunks(client, "c") == 4
    assert indexing.count_articles(client, "c") == 3
