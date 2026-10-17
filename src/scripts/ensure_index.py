"""Container entrypoint helper: make sure Qdrant holds the corpus, then exit.

Waits for Qdrant, and if the collection is missing or empty loads the embedded chunks that are
baked into the image. Idempotent, so ``docker compose up`` works on a clean machine and restarts
are instant. Index creation is retried because on small machines Qdrant can time out while it is
still warming up.
"""

from __future__ import annotations

import json
import os
import sys
import time

from qdrant_client import QdrantClient

from src.retrieval import indexing

CHUNKS = os.getenv("EMBEDDED_CHUNKS_PATH", "data/processed/embedded-chunks-v2.json")
URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", indexing.DEFAULT_COLLECTION)
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "120"))  # seconds per request
INDEX_ATTEMPTS = 3


def wait_for_qdrant(client: QdrantClient, seconds: int = 120) -> None:
    deadline = time.time() + seconds
    while True:
        try:
            client.get_collections()
            return
        except Exception as exc:
            if time.time() > deadline:
                raise SystemExit(f"[ensure_index] Qdrant not reachable at {URL}: {exc}") from exc
            time.sleep(2)


def build_index(client: QdrantClient, chunks: list[dict]) -> int:
    """Create the collection and upsert the chunks, retrying on timeouts.

    Safe to repeat: the collection is recreated each time and point ids are deterministic.
    """
    last_exc: Exception | None = None
    for attempt in range(1, INDEX_ATTEMPTS + 1):
        try:
            indexing.ensure_collection(client, COLLECTION, recreate=True)
            return indexing.upsert_chunks(client, COLLECTION, chunks)
        except Exception as exc:
            last_exc = exc
            print(
                f"[ensure_index] attempt {attempt}/{INDEX_ATTEMPTS} failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if attempt < INDEX_ATTEMPTS:
                time.sleep(5 * attempt)
    raise SystemExit(f"[ensure_index] giving up after {INDEX_ATTEMPTS} attempts: {last_exc}")


def main() -> int:
    client = QdrantClient(url=URL, timeout=QDRANT_TIMEOUT)
    wait_for_qdrant(client)
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing and indexing.count_chunks(client, COLLECTION) > 0:
        print(
            f"[ensure_index] '{COLLECTION}' already holds "
            f"{indexing.count_chunks(client, COLLECTION)} chunks - nothing to do"
        )
        return 0
    if not os.path.exists(CHUNKS):
        print(
            f"[ensure_index] {CHUNKS} missing: run `dvc pull` before building the image",
            file=sys.stderr,
        )
        return 1
    with open(CHUNKS, encoding="utf-8") as fh:
        chunks = json.load(fh)
    n = build_index(client, chunks)
    print(f"[ensure_index] indexed {n} chunks into '{COLLECTION}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())