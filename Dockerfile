# syntax=docker/dockerfile:1
# API image: code + corpus + embedded chunks are baked in; Qdrant runs as its own compose service
# and is filled from the baked-in embeddings at start-up (src/scripts/ensure_index.py).
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r requirements-api.txt

# Optional: bake the model weights in (~4.5 GB) for fully offline start-up.
#   docker compose build --build-arg PRELOAD_MODELS=true
ARG PRELOAD_MODELS=false
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-m3')" && \
      python -c "from transformers import AutoModelForSequenceClassification as M, AutoTokenizer as T; \
n='BAAI/bge-reranker-v2-m3'; T.from_pretrained(n); M.from_pretrained(n)"; \
    fi

COPY main.py pyproject.toml ./
COPY src ./src
COPY docker/entrypoint.sh ./entrypoint.sh
COPY reports ./reports
# The DVC-tracked corpus (run `dvc pull` before building)
COPY data/processed/final-articles-v2.json data/processed/final-articles-v2.json
COPY data/processed/final-chunks-bgem3-v2.json data/processed/final-chunks-bgem3-v2.json
COPY data/processed/embedded-chunks-v2.json data/processed/embedded-chunks-v2.json

RUN chmod +x entrypoint.sh
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
ENTRYPOINT ["./entrypoint.sh"]
