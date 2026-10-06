import json
from pathlib import Path

from src.llm.EmbeddingProviderFactory import EmbeddingProviderFactory
from src.llm.Enums import EmbeddingEnums

INPUT_FILE = Path(
    "data/processed/final-chunks-bgem3-v2.json"
)

OUTPUT_FILE = Path(
    "data/processed/embedded-chunks-v2.json"
)


def main():

    print("=" * 60)
    print("CIVIL CODE EMBEDDING")
    print("=" * 60)

    print(f"[INFO] Loading chunks from: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[INFO] Total chunks: {len(chunks)}")

    # Remove status chunks

    chunks_to_embed = [
        chunk
        for chunk in chunks
        if chunk.get("chunk_type") != "article_status"
    ]

    print(
        f"[INFO] Chunks to embed: {len(chunks_to_embed)}"
    )

    print(
        f"[INFO] Status chunks skipped: "
        f"{len(chunks) - len(chunks_to_embed)}"
    )

    # Create embedding provider

    factory = EmbeddingProviderFactory(config={})

    embedding_provider = factory.create(
        provider=EmbeddingEnums.HUGGINGFACE.value,
        model_name="BAAI/bge-m3",
        device="cpu",
    )

    # Extract texts

    texts = [
        chunk["embed_text"]
        for chunk in chunks_to_embed
    ]

    # Generate embeddings

    print("[INFO] Generating embeddings...")

    embeddings = embedding_provider.embed_documents(texts)

    print(
        f"[INFO] Generated embeddings: {len(embeddings)}"
    )

    if embeddings:
        print(
            f"[INFO] Embedding dimension: "
            f"{len(embeddings[0])}"
        )

    # Attach embeddings

    for chunk, embedding in zip(
        chunks_to_embed,
        embeddings,
    ):
        chunk["embedding"] = embedding

    # Save

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            chunks_to_embed,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 60)
    print("[OK] Embeddings saved")
    print("=" * 60)

    print(f"Output: {OUTPUT_FILE}")
    print(f"Chunks: {len(chunks_to_embed)}")

    if embeddings:
        print(
            f"Dimension: {len(embeddings[0])}"
        )


if __name__ == "__main__":
    main()