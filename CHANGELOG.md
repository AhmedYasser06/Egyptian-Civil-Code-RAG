# Changelog (per course session)

## Session 7 - merge with local tree, RAGAS fix
- RAGAS judge: max_tokens 4096 with auto-doubling on `IncompleteOutputException`, 429 back-off, per-metric null instead of crash, `--retry-failed`; Groq/Gemini/OpenRouter/OpenAI/vLLM presets.
- Eval paths now `data/evaluation/*` (30 / 50 / 20 question sets); 50-question monitor defaults to `rag_eval_gemini_50.json`.
- `scripts/make_ragas_ci_report.py` builds the CI gate report from `ragas_results.json`.
- `src/scripts/compact_embeddings.py` (pack/unpack embeddings), `.dvc/` restored.

## Session 6 — Observability
- Langfuse tracing (`src/monitoring/tracing.py`): one trace per `/ask`, spans `retrieval` / `generation`, tokens and cost.
- RAGAS monitor on ≥ 50 questions (`src.monitoring.ragas_monitor`): 4 metrics → MLflow, Langfuse scores on each trace, Prometheus, alert at faithfulness < 0.80.
- Cosine drift (`src/monitoring/drift.py`, demo with family-law queries), token cost, Grafana dashboard, Prometheus alert rules.

## Session 5 — Optimization
- AWQ 4-bit Qwen3-8B vs fp16 with RAGAS before/after (`src.optimization.compare_llms`).
- Re-ranker distillation 568M → 118M (`src.optimization.distill_reranker`).

## Session 4 — Serving
- BentoML async + streaming service, vLLM backend (`VLLM` provider), Locust at 50 users, canary rollout config, batch re-indexing script.
- Docker compose with Qdrant and self-indexing API image; root `/ask`, real `/health`, `/metrics`.

## Session 3 — CI/CD
- GitHub Actions: lint → test → RAGAS gate (≥ 0.75, staleness check) → rebuild index → build/push image to GHCR.
- Shared DVC remote (DagsHub).

## Session 2 — Tracking & versioning
- MLflow chunking sweep with RAGAS faithfulness per config; best config in the Model Registry.
- DVC pipeline: extract → chunk → embed → index.

## Session 1 — Package & API
- Corpus extraction/validation, article-aware chunking, BGE-M3 + Qdrant, FastAPI, `pyproject.toml`.
