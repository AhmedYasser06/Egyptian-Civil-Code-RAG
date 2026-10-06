import json
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/processed/embedded-chunks-v2.json"

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "egyptian_civil_code",
)

VECTOR_SIZE = 1024

BATCH_SIZE = 128

RECREATE_COLLECTION = os.getenv(
    "RECREATE_COLLECTION",
    "true",
).lower() == "true"

# ============================================================
# HELPERS
# ============================================================

def make_point_id(chunk_id: str) -> str:
    """
    Convert our chunk_id into a deterministic UUID.

    Qdrant does not accept arbitrary strings as point IDs,
    so we generate a stable UUID from the chunk ID.
    """

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            chunk_id,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EGYPTIAN CIVIL CODE → QDRANT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    print(f"[INFO] Loading: {INPUT_FILE}")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        chunks = json.load(f)

    print(
        f"[INFO] Loaded chunks: {len(chunks)}"
    )

    if not chunks:
        raise ValueError(
            "No chunks found."
        )

    # --------------------------------------------------------
    # Connect to Qdrant
    # --------------------------------------------------------

    print(
        f"[INFO] Connecting to Qdrant: "
        f"{QDRANT_URL}"
    )

    client = QdrantClient(
        url=QDRANT_URL
    )

    # Test connection

    collections = client.get_collections()

    print(
        "[OK] Connected to Qdrant"
    )

    # --------------------------------------------------------
    # Recreate collection
    # --------------------------------------------------------

    existing_collections = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME in existing_collections:

        if RECREATE_COLLECTION:
            print(
                f"[INFO] Collection '{COLLECTION_NAME}' "
                "already exists."
            )

            print("[INFO] Recreating collection...")

            client.delete_collection(
                collection_name=COLLECTION_NAME
            )

            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

            print(
                f"[OK] Recreated collection: "
                f"{COLLECTION_NAME}"
            )

        else:
            collection_info = client.get_collection(
                COLLECTION_NAME
            )

            print(
                f"[INFO] Collection '{COLLECTION_NAME}' "
                "already exists with "
                f"{collection_info.points_count} points."
            )

            print("[OK] Skipping re-indexing.")
            return

    else:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"[OK] Created collection: "
            f"{COLLECTION_NAME}"
        )

    print(
        f"[OK] Created collection: "
        f"{COLLECTION_NAME}"
    )

    # --------------------------------------------------------
    # Prepare points
    # --------------------------------------------------------

    points = []

    for chunk in chunks:

        chunk_id = chunk["chunk_id"]

        embedding = chunk.get(
            "embedding"
        )

        if embedding is None:
            print(
                f"[WARNING] Missing embedding: "
                f"{chunk_id}"
            )
            continue

        if len(embedding) != VECTOR_SIZE:

            raise ValueError(
                f"Invalid vector size for "
                f"{chunk_id}: "
                f"{len(embedding)}"
            )

        metadata = chunk.get(
            "metadata",
            {}
        )

        # ----------------------------------------------------
        # Payload
        # ----------------------------------------------------

        payload = {
            "chunk_id": chunk_id,

            "article_number":
                chunk.get("article_number"),

            "language":
                chunk.get("language"),

            "chunk_index":
                chunk.get("chunk_index"),

            "chunk_type":
                chunk.get("chunk_type"),

            "provision_index":
                chunk.get("provision_index"),

            "text":
                chunk.get("text"),

            # Legal metadata
            "book":
                metadata.get("book"),

            "chapter":
                metadata.get("chapter"),

            "section":
                metadata.get("section"),

            "topic":
                metadata.get("topic"),

            "is_repealed":
                metadata.get(
                    "is_repealed",
                    False,
                ),

            "source_page":
                metadata.get("source_page"),

            "citation":
                metadata.get("citation"),

            "source_status":
                metadata.get("source_status"),

            "parent_article_id":
                metadata.get(
                    "parent_article_id"
                ),
        }

        point = PointStruct(
            id=make_point_id(chunk_id),

            vector=embedding,

            payload=payload,
        )

        points.append(point)

    print(
        f"[INFO] Points prepared: "
        f"{len(points)}"
    )

    # --------------------------------------------------------
    # Upload in batches
    # --------------------------------------------------------

    total = len(points)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = points[
            start:end
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,

            points=batch,

            wait=True,
        )

        print(
            f"[INFO] Uploaded "
            f"{end}/{total}"
        )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    collection_info = (
        client.get_collection(
            COLLECTION_NAME
        )
    )

    print()
    print("=" * 70)
    print("[OK] QDRANT INDEXING COMPLETE")
    print("=" * 70)

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Vectors: "
        f"{collection_info.points_count}"
    )

    print(
        f"Vector dimension: "
        f"{VECTOR_SIZE}"
    )

    print(
        "Distance: COSINE"
    )


if __name__ == "__main__":
    main()