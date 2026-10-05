from .providers.LocalEmbeddingProvider import LocalEmbeddingProvider
from .providers.HuggingFaceEmbedding import HuggingFaceEmbedding
from .Enums import EmbeddingEnums


class EmbeddingProviderFactory:

    def __init__(self, config: dict):
        self.config = config

    def create(
        self,
        provider: str,
        model_name: str = None,
        device: str = "cpu",
    ):

        if provider == EmbeddingEnums.LOCAL_EMBEDDING.value:
            return LocalEmbeddingProvider()

        elif provider == EmbeddingEnums.HUGGINGFACE.value:

            model_name = model_name or "BAAI/bge-m3"

            return HuggingFaceEmbedding(
                model_id=model_name,
                device=device,
                normalize_embeddings=True,
                embedding_size=1024,
            )

        return None