# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

COPY main.py .
COPY pyproject.toml .
COPY src ./src
COPY data/processed/final-articles-v2.json ./data/processed/final-articles-v2.json
COPY data/processed/final-chunks-bgem3-v2.json ./data/processed/final-chunks-bgem3-v2.json
COPY data/processed/embedded-chunks-v2.json ./data/processed/embedded-chunks-v2.json

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
