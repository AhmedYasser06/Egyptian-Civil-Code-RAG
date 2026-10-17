from .Enums import EmbeddingEnums
from .providers.HuggingFaceEmbedding import HuggingFaceEmbedding
from .providers.LocalEmbeddingProvider import LocalEmbeddingProvider

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_SIZE = 1024  # BGE-M3 dense dimension


class EmbeddingProviderFactory:

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def create(
        self,
        provider: str,
        model_name: str | None = None,
        device: str = "cpu",
    ):
        if provider == EmbeddingEnums.LOCAL_EMBEDDING.value:
            return LocalEmbeddingProvider()

        if provider == EmbeddingEnums.HUGGINGFACE.value:
            return HuggingFaceEmbedding(
                model_id=model_name or self.config.get("model_name", DEFAULT_EMBEDDING_MODEL),
                device=device,
                normalize_embeddings=True,
                embedding_size=self.config.get("embedding_size", DEFAULT_EMBEDDING_SIZE),
            )

        raise ValueError(
            f"Unknown embedding provider: {provider!r}. "
            f"Expected one of: {[e.value for e in EmbeddingEnums]}"
        )