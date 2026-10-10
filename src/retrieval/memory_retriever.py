"""Qdrant-free retriever with the same ``retrieve(query, top_k)`` contract as ``LegalRetriever``.

Used where a vector server is overkill or impossible: the chunking sweep (one index per chunking
config), the quantisation comparison and the notebooks on Kaggle. Pipeline = exact article lookup
-> dense BGE-M3 retrieval over chunks -> group by article -> optional cross-encoder rerank.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.retrieval.retriever import LegalRetriever


def load_articles(path: str | Path) -> dict[int, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("articles") or data.get("data") or list(data.values())
    return {int(a["article_number"]): a for a in data}


class _Embedder:
    def __init__(self, model):
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True).tolist()


class InMemoryRetriever:
    def __init__(
        self,
        chunks: list[dict],
        articles: dict[int, dict],
        embed_model,
        reranker=None,
        candidate_k: int = 15,
        batch_size: int = 32,
        chunk_embeddings: np.ndarray | None = None,
    ):
        self.articles = articles
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.model = embed_model
        self.embedding_provider = _Embedder(embed_model)
        self.chunks = [c for c in chunks if c.get("embed_text")]
        if chunk_embeddings is None:
            chunk_embeddings = embed_model.encode(
                [c["embed_text"] for c in self.chunks],
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        self.embeddings = np.asarray(chunk_embeddings, dtype=np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        exact = LegalRetriever._detect_article_number(query)
        if exact is not None and exact in self.articles:
            return [self._article(exact)]

        q = np.asarray(self.embedding_provider.embed_query(query), dtype=np.float32)
        scores = self.embeddings @ q
        best: dict[int, float] = {}
        for idx in np.argsort(scores)[::-1]:
            number = int(self.chunks[int(idx)]["article_number"])
            if number in self.articles and number not in best:
                best[number] = float(scores[int(idx)])
            if len(best) >= self.candidate_k:
                break

        candidates = [self._article(n) for n in best]
        if self.reranker is None:
            return candidates[:top_k]
        return self.reranker.rerank(query, candidates, top_k)

    def _article(self, number: int) -> dict:
        article = dict(self.articles[number])
        article.setdefault("citation", f"Egyptian Civil Code, Article {number}")
        article["article_number"] = number
        return article
