"""Qdrant indexing helpers shared by ``rag.ingest``, ``ensure_index`` and ``reindex_documents``.

The payload layout is identical to ``src/scripts/index_civil_code.py`` (the DVC stage), so a
collection built by either path is interchangeable.
"""

from __future__ import annotations

import uuid
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

VECTOR_SIZE = 1024
DEFAULT_COLLECTION = "egyptian_civil_code"


def make_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def chunk_to_point(chunk: dict, vector_size: int = VECTOR_SIZE) -> PointStruct:
    embedding = chunk["embedding"]
    if len(embedding) != vector_size:
        raise ValueError(f"{chunk['chunk_id']}: vector size {len(embedding)} != {vector_size}")
    meta = chunk.get("metadata", {})
    payload = {
        "chunk_id": chunk["chunk_id"],
        "article_number": chunk.get("article_number"),
        "language": chunk.get("language"),
        "chunk_index": chunk.get("chunk_index"),
        "chunk_type": chunk.get("chunk_type"),
        "provision_index": chunk.get("provision_index"),
        "text": chunk.get("text"),
        "book": meta.get("book"),
        "chapter": meta.get("chapter"),
        "section": meta.get("section"),
        "topic": meta.get("topic"),
        "is_repealed": meta.get("is_repealed", False),
        "source_page": meta.get("source_page"),
        "citation": meta.get("citation"),
        "source_status": meta.get("source_status"),
        "parent_article_id": meta.get("parent_article_id"),
    }
    return PointStruct(id=make_point_id(chunk["chunk_id"]), vector=embedding, payload=payload)


def ensure_collection(
    client: QdrantClient, collection: str, recreate: bool = False, vector_size: int = VECTOR_SIZE
) -> None:
    exists = collection in {c.name for c in client.get_collections().collections}
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient, collection: str, chunks: Iterable[dict], batch_size: int = 128
) -> int:
    """Upsert embedded chunks (idempotent: point ids are derived from chunk ids)."""
    points = [chunk_to_point(c) for c in chunks if c.get("embedding") is not None]
    for start in range(0, len(points), batch_size):
        client.upsert(collection, points=points[start : start + batch_size], wait=True)
    return len(points)


def count_chunks(client: QdrantClient, collection: str) -> int:
    return client.count(collection, exact=True).count


def count_articles(client: QdrantClient, collection: str) -> int:
    """Number of distinct articles in the collection (what /health reports as documents_indexed)."""
    seen: set[int] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection,
            limit=512,
            offset=offset,
            with_payload=["article_number"],
            with_vectors=False,
        )
        seen.update(p.payload["article_number"] for p in points if p.payload)
        if offset is None:
            return len(seen)
