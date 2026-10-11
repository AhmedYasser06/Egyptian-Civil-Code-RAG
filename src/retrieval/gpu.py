"""Small helpers so the GPU notebooks fit next to a vLLM server on the same card."""

from __future__ import annotations


def load_sentence_transformer(name: str, device: str):
    """SentenceTransformer, in float16 on CUDA (BGE-M3 fp32 = 2.3 GB, fp16 = 1.1 GB).

    On Kaggle vLLM already occupies most of each T4; fp32 copies of the embedder, the reranker
    and the judge's embedder caused ``torch.OutOfMemoryError`` in the chunking sweep.
    """
    from sentence_transformers import SentenceTransformer

    if str(device).startswith("cuda"):
        import torch

        return SentenceTransformer(name, device=device, model_kwargs={"torch_dtype": torch.float16})
    return SentenceTransformer(name, device=device)
