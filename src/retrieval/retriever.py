import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.llm.providers.HuggingFaceEmbedding import (
    HuggingFaceEmbedding,
)
from src.retrieval.reranker import LegalReranker

load_dotenv()


class LegalRetriever:

    def __init__(
        self,
        qdrant_url: str | None = None,
        collection_name: str = "egyptian_civil_code",
    ):
        if qdrant_url is None:
            qdrant_url = os.getenv(
                "QDRANT_URL",
                "http://localhost:6333",
            )

        self.reranker = LegalReranker(
            model_id=os.getenv(
                "RERANKER_MODEL_ID",
                "BAAI/bge-reranker-v2-m3",
            ),
            device=os.getenv(
                "RERANKER_DEVICE",
                "cpu",
            ),
        )

        self.collection_name = collection_name

        self.client = QdrantClient(
            url=qdrant_url,
        )
        
        embedding_model = os.getenv(
            "EMBEDDING_MODEL_ID",
            "BAAI/bge-m3",
        )

        embedding_provider = os.getenv(
            "EMBEDDING_MODEL_PROVIDER",
            "HUGGINGFACE",
        )

        if embedding_provider != "HUGGINGFACE":
            raise ValueError(
                "For the current Qdrant collection, "
                "EMBEDDING_MODEL_PROVIDER must be "
                "'HUGGINGFACE' because the collection "
                "uses BAAI/bge-m3 embeddings."
            )

        self.embedding_provider = HuggingFaceEmbedding(
            model_id=embedding_model,
            device="cpu",
            normalize_embeddings=True,
            embedding_size=1024,
        )

    # ============================================================
    # ARTICLE NUMBER DETECTION
    # ============================================================

    @staticmethod
    def _normalize_digits(text: str) -> str:
        """
        Convert Arabic-Indic and Eastern Arabic-Indic digits
        to Western digits.
        """

        translation = str.maketrans(
            "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
            "01234567890123456789",
        )

        return text.translate(translation)

    @classmethod
    def _detect_article_number(
        cls,
        query: str,
    ) -> Optional[int]:
        """
        Detect explicit article references such as:

        Article 1022
        article 1022
        Art. 1022
        المادة 1022
        مادة 1022
        المادة ١٠٢٢
        """

        normalized = cls._normalize_digits(query)

        pattern = re.compile(
            r"""
            (?:
                \barticle\b
                |
                \bart\.?\b
                |
                المادة
                |
                مادة
            )
            \s*
            (?:\#\s*)?
            (\d+)
            \b
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        match = pattern.search(normalized)

        if not match:
            return None

        article_number = int(match.group(1))

        if not 1 <= article_number <= 1149:
            return None

        return article_number

    # ============================================================
    # PAYLOAD FORMATTER
    # ============================================================

    @staticmethod
    def _format_result(
        payload: Dict[str, Any],
        score: Optional[float] = None,
    ) -> Dict[str, Any]:

        return {
            "score": score,
            "chunk_id": payload.get("chunk_id"),
            "article_number": payload.get(
                "article_number"
            ),
            "language": payload.get(
                "language"
            ),
            "chunk_index": payload.get(
                "chunk_index"
            ),
            "chunk_type": payload.get(
                "chunk_type"
            ),
            "text": payload.get(
                "text"
            ),
            "book": payload.get(
                "book"
            ),
            "chapter": payload.get(
                "chapter"
            ),
            "section": payload.get(
                "section"
            ),
            "topic": payload.get(
                "topic"
            ),
            "is_repealed": payload.get(
                "is_repealed",
                False,
            ),
            "source_page": payload.get(
                "source_page"
            ),
            "citation": payload.get(
                "citation"
            ),
            "source_status": payload.get(
                "source_status"
            ),
            "parent_article_id": payload.get(
                "parent_article_id"
            ),
        }

    # ============================================================
    # EXACT ARTICLE RETRIEVAL
    # ============================================================

    def _retrieve_article(
        self,
        article_number: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks belonging to an exact article number.

        This does not use embeddings.
        """

        article_filter = Filter(
            must=[
                FieldCondition(
                    key="article_number",
                    match=MatchValue(
                        value=article_number
                    ),
                )
            ]
        )

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=article_filter,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        # Sort by language first, then chunk index.
        # This keeps article parts in the correct order.
        points = sorted(
            points,
            key=lambda point: (
                point.payload.get("language", ""),
                point.payload.get("chunk_index", 0),
            ),
        )

        retrieved = []

        for point in points:

            payload = point.payload or {}

            retrieved.append(
                self._format_result(
                    payload=payload,
                    score=1.0,
                )
            )

        return retrieved

    # ============================================================
    # SEMANTIC RETRIEVAL
    # ============================================================

    def _retrieve_semantic(
        self,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:

        query_vector = (
            self.embedding_provider.embed_query(
                query
            )
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        retrieved = []

        for result in results:

            payload = result.payload or {}

            retrieved.append(
                self._format_result(
                    payload=payload,
                    score=result.score,
                )
            )

        return retrieved

    # ============================================================
    # GROUP BY ARTICLE
    # ============================================================
    def group_by_article(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        articles = {}

        for result in results:

            article_number = result.get("article_number")

            if article_number is None:
                continue

            if article_number not in articles:
                articles[article_number] = {
                    "article_number": article_number,
                    "score": result.get("score"),
                    "book": result.get("book"),
                    "chapter": result.get("chapter"),
                    "section": result.get("section"),
                    "topic": result.get("topic"),
                    "text_ar": [],
                    "text_en": [],
                    "is_repealed": result.get(
                        "is_repealed",
                        False,
                    ),
                    "source_page": result.get(
                        "source_page"
                    ),
                    "citation": result.get(
                        "citation"
                    ),
                    "source_status": result.get(
                        "source_status"
                    ),
                    "rerank_score":result.get(
                        "rerank_score"
                    ),
                }

            article = articles[article_number]

            # Keep the best retrieval score for this article
            result_score = result.get("score")

            if (
                result_score is not None
                and (
                    article["score"] is None
                    or result_score > article["score"]
                )
            ):
                article["score"] = result_score            
                        
            language = result.get("language")
            text = result.get("text") or ""
            chunk_index = result.get("chunk_index", 0)

            if language == "ar":
                article["text_ar"].append(
                    (chunk_index, text)
                )

            elif language == "en":
                article["text_en"].append(
                    (chunk_index, text)
                )

        for article in articles.values():

            article["text_ar"] = " ".join(
                text
                for _, text in sorted(
                    article["text_ar"],
                    key=lambda x: x[0],
                )
            ).strip()

            article["text_en"] = " ".join(
                text
                for _, text in sorted(
                    article["text_en"],
                    key=lambda x: x[0],
                )
            ).strip()

        return list(articles.values())
    
    # ============================================================
    # PUBLIC RETRIEVAL METHOD
    # ============================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieval strategy:

        1. If the user explicitly mentions an article number,
        perform exact article lookup.

        2. Otherwise, perform BGE-M3 semantic retrieval
        with an expanded candidate pool.

        3. Group retrieved chunks by article.

        4. Rerank the candidate articles using the
        multilingual legal reranker.

        5. Return the final top_k articles.
        """

        article_number = self._detect_article_number(query)

        # --------------------------------------------------------
        # Exact article retrieval
        # --------------------------------------------------------

        if article_number is not None:

            results = self._retrieve_article(
                article_number
            )

            # Article exists
            if results:
                return self.group_by_article(results)

        # --------------------------------------------------------
        # Semantic retrieval
        # --------------------------------------------------------

        # Retrieve more candidates than we finally return.
        candidate_k = max(top_k * 4, 15)

        results = self._retrieve_semantic(
            query=query,
            top_k=candidate_k,
        )

        articles = self.group_by_article(results)

        reranked_articles = self.reranker.rerank(
            query=query,
            articles=articles,
            top_k=top_k,
        )

        print("\n[RERANKER] Results:")
        for article in reranked_articles:
            print(
                f"Article {article['article_number']} "
                f"score={article.get('rerank_score', 0.0):.4f}"
            )

        return reranked_articles