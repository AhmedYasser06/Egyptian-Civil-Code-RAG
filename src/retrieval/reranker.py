from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class LegalReranker:
    """
    Cross-encoder reranker for Arabic/English legal retrieval.

    The model receives:
        (query, article_text)

    and directly scores their relevance.
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.max_length = max_length

        print(f"[RERANKER] Loading model: {model_id}")
        print(f"[RERANKER] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id
        )

        self.model.to(self.device)
        self.model.eval()

        print("[RERANKER] Model loaded successfully.")

    @torch.inference_mode()
    def score(
        self,
        query: str,
        passages: List[str],
    ) -> List[float]:

        if not passages:
            return []

        pairs = [[query, passage] for passage in passages]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        logits = self.model(**inputs).logits.view(-1)

        # Convert logits to [0, 1]
        scores = torch.sigmoid(logits)

        return scores.cpu().tolist()

    def rerank(
        self,
        query: str,
        articles: List[Dict],
        top_k: int,
    ) -> List[Dict]:

        if not articles:
            return []

        passages = []

        for article in articles:

            text_ar = article.get("text_ar") or ""
            text_en = article.get("text_en") or ""

            passage = (
                f"Article {article['article_number']}\n"
                f"Arabic:\n{text_ar}\n"
                f"English:\n{text_en}"
            )

            passages.append(passage)

        scores = self.score(
            query=query,
            passages=passages,
        )

        for article, score in zip(articles, scores):

            article["rerank_score"] = float(score)

        articles.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return articles[:top_k]