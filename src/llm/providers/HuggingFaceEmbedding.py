from typing import List, Optional

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedding(Embeddings):

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        embedding_size: int = 1024,
    ):
        self.model_id = model_id
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.embedding_size = embedding_size

        self.embedding_model: Optional[SentenceTransformer] = None

    def _init_model(self):

        print(f"[INFO] Loading embedding model: {self.model_id}")
        print(f"[INFO] Device: {self.device}")

        self.embedding_model = SentenceTransformer(
            self.model_id,
            device=self.device,
        )

        return self.embedding_model

    def _get_model(self):

        if self.embedding_model is None:
            self._init_model()

        return self.embedding_model

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:

        model = self._get_model()

        embeddings = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        text: str,
    ) -> List[float]:

        model = self._get_model()

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embedding.tolist()