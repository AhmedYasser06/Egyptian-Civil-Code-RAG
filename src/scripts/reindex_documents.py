"""Batch re-indexing: add (or update) documents in the live Qdrant collection without a rebuild.

    python -m src.scripts.reindex_documents --input data/new/new_articles.json
    python -m src.scripts.reindex_documents --input data/new/ --dry-run

``--input`` is a JSON list of article records in the corpus schema (a file, or a directory of
such files): ``article_number``, ``text_ar``, ``text_en``, ``book``/``chapter``/``section``/
``topic``, ``is_repealed``, ``citation``. Each record is chunked with the production splitter,
embedded with BGE-M3 and upserted. Point ids derive from chunk ids, so re-running the same file
is idempotent and an edited article replaces its old chunks. Chunks that no longer exist for an
updated article are deleted. Exits non-zero if the new documents are not retrievable afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.retrieval import indexing
from src.scripts.text_splitter import ChunkConfig, TokenCounter, split_articles


def load_records(path: Path) -> list[dict]:
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    records: list[dict] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{f}: expected a JSON list of article records")
        records += data
    return records


def reindex(
    records: list[dict],
    client: QdrantClient,
    collection: str,
    embed: Callable[[list[str]], list[list[float]]],
    cfg: ChunkConfig | None = None,
    counter: TokenCounter | None = None,
    dry_run: bool = False,
) -> dict:
    cfg = cfg or ChunkConfig(strategy="article", max_tokens=512, overlap_tokens=0)
    chunks = split_articles(records, cfg, counter or TokenCounter(None))
    to_embed = [c for c in chunks if c.get("chunk_type") != "article_status"]
    report = {
        "articles": len(records),
        "chunks": len(to_embed),
        "article_numbers": sorted({int(r["article_number"]) for r in records}),
    }
    if dry_run:
        return {**report, "dry_run": True}

    vectors = embed([c["embed_text"] for c in to_embed])
    for chunk, vector in zip(to_embed, vectors):
        chunk["embedding"] = vector

    indexing.ensure_collection(client, collection, recreate=False)
    before = indexing.count_chunks(client, collection)
    # an updated article may now have fewer chunks: drop its old ones first
    for number in report["article_numbers"]:
        client.delete(
            collection,
            points_selector=Filter(
                must=[FieldCondition(key="article_number", match=MatchValue(value=number))]
            ),
            wait=True,
        )
    indexing.upsert_chunks(client, collection, to_embed)
    after = indexing.count_chunks(client, collection)
    return {**report, "points_before": before, "points_after": after}


def verify(client: QdrantClient, collection: str, numbers: list[int]) -> list[int]:
    """Article numbers that are still missing from the collection."""
    missing = []
    for n in numbers:
        points, _ = client.scroll(
            collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="article_number", match=MatchValue(value=n))]
            ),
            limit=1,
        )
        if not points:
            missing.append(n)
    return missing


def hf_embedder(device: str = "cpu") -> Callable[[list[str]], list[list[float]]]:
    from src.llm.providers.HuggingFaceEmbedding import HuggingFaceEmbedding

    provider = HuggingFaceEmbedding(model_id="BAAI/bge-m3", device=device)
    return provider.embed_documents


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument(
        "--collection", default=os.getenv("QDRANT_COLLECTION", indexing.DEFAULT_COLLECTION)
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = load_records(args.input)
    client = QdrantClient(url=args.qdrant_url)
    report = reindex(
        records, client, args.collection, hf_embedder(args.device), dry_run=args.dry_run
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0
    missing = verify(client, args.collection, report["article_numbers"])
    if missing:
        print(f"[reindex] FAILED: articles not found after indexing: {missing}", file=sys.stderr)
        return 1
    print("[reindex] OK: all new articles are in the index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
