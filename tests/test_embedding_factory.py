import pytest

pytest.importorskip("sentence_transformers")

from src.llm.EmbeddingProviderFactory import EmbeddingProviderFactory  # noqa: E402


def test_unknown_embedding_provider_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown embedding provider: 'TYPO'"):
        EmbeddingProviderFactory().create("TYPO")