import json
from pathlib import Path

from qdrant_client import QdrantClient

from src.retrieval import indexing
from src.scripts import reindex_documents as rx

SAMPLE = Path(__file__).parent.parent / "data" / "new" / "sample_new_article.json"


def fake_embed(texts):
    return [[0.01 * (i % 5 + 1)] * indexing.VECTOR_SIZE for i, _ in enumerate(texts)]


def test_new_document_is_indexed_idempotently_and_verified():
    client = QdrantClient(":memory:")
    records = json.loads(SAMPLE.read_text(encoding="utf-8"))

    first = rx.reindex(records, client, "c", fake_embed)
    assert first["points_after"] > first["points_before"] == 0
    assert rx.verify(client, "c", [9001]) == []

    again = rx.reindex(records, client, "c", fake_embed)  # re-run: same ids, nothing duplicated
    assert again["points_after"] == first["points_after"]
    assert rx.verify(client, "c", [123456]) == [123456]


def test_dry_run_does_not_touch_the_index():
    client = QdrantClient(":memory:")
    records = json.loads(SAMPLE.read_text(encoding="utf-8"))
    report = rx.reindex(records, client, "c", fake_embed, dry_run=True)
    assert report["dry_run"] and report["chunks"] > 0
    assert "c" not in {c.name for c in client.get_collections().collections}
