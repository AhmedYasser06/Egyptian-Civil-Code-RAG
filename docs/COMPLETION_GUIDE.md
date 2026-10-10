# Finish-the-project guide (your real tree, merged)

Everything below was checked against **your** zip: 49 tests pass (incl. the 6 that read your real
`final-articles-v2.json`), `ruff` is clean, the 3 eval files load (30 / 50 / 20 questions), your
chunk schema (`chunk_id, metadata, embedding[1024]`) indexes into Qdrant, and the RAGAS fix was
reproduced and verified against a fake judge that truncates like your Groq run.
**Not testable here (no GPU / Docker / GitHub / HuggingFace):** real BGE-M3 & vLLM runs, `docker compose`,
GitHub Actions, DagsHub, Langfuse server. Those steps are marked ▶ below.

## 0. Your RAGAS crash (`IncompleteOutputException`) — cause and fix
* Cause: the judge (Qwen3 on Groq, which "thinks" before answering) hit `max_tokens=900`, the API
  answered `finish_reason=length`, `instructor` raised, and your script `raise`d out of the loop.
* Also: run it as a module **without** `.py` → `python -m src.evaluation.evaluate_ragas`.
* Fix (already in `src/evaluation/evaluate_ragas.py` + `ragas_runner.py`): `max_tokens=4096`,
  auto-doubling to 16384 on truncation, retries + back-off on 429, a failed metric becomes `null`
  and the run continues, finished rows are kept, `--retry-failed` fills only gaps.
* Optional: switch off Qwen3 thinking on Groq: `JUDGE_EXTRA_BODY={"reasoning_effort":"none"}` in `.env`.
* Alternative judges (no code change): `--provider gemini | openrouter | openai | vllm`
  (your `evaluate_ragas_gemini*.py` still work, now as wrappers of the same resilient code).

## 1. Install the merged files
Unzip this archive **over** your project folder (it contains only code/config; your `data/`,
`mlruns/`, `.env` are untouched). Then:
```bash
conda activate mini-rag-TA-py311
pip install -r requirements.txt && pip install -e .
cp .env.example .env     # only if you do not have .env; add GROQ_API_KEY / GOOGLE_API_KEY / LANGFUSE_*
```
Things that differ from your old tree: `src/routes/legal.py` now delegates to `src/rag.py` (the handbook's
`src/rag.py`), Langfuse tracing lives in `src/monitoring/tracing.py` (same spans you wrote: retrieval,
generation + token usage + RAGAS score); your `src/bento_service.py` is superseded by
`src/serving/bento_service.py` (streaming + async) – delete yours if you like. `.dvc/` is restored.

## 2. Finish RAGAS on the 30 samples, then the CI gate
```bash
python -m src.evaluation.evaluate_ragas                 # resumes at 22/30, never crashes on one sample
python -m src.evaluation.evaluate_ragas --retry-failed  # only if the summary lists missing metrics
python scripts/make_ragas_ci_report.py                  # -> reports/ragas_ci.json (the CI gate reads it)
python scripts/ragas_gate.py                            # should print PASS (faithfulness >= 0.75)
```
(`reports/ragas_ci.json` in this zip was generated from your 21 finished rows: faithfulness 0.964, PASS.
Regenerate it after step 1 so it covers 30.)

## 3. ≥ 50-question RAGAS monitor (simplest path – no Gemini needed)
The 50-question set = your 30 (already answered **and** judged) + 20 new ones (q16–q25). One command
reuses the 30 and only answers + judges the 20 new ones with your normal LLM/judge (Groq):
```bash
docker compose up -d qdrant && python -m src.scripts.ensure_index   # the pipeline needs Qdrant
python -m src.monitoring.ragas_monitor --questions data/evaluation/rag_eval_gemini_50.json --label session-6
```
It is resumable (`data/evaluation/ragas_monitor_cache.json`), logs the 4 means to MLflow, attaches scores
to Langfuse traces, pushes to Prometheus if `PUSHGATEWAY_URL` is set and alerts if faithfulness < 0.80.
`evaluate_ragas_gemini_50` is no longer needed. If you do use Gemini as judge and get `NotFoundError`, a
stale `JUDGE_BASE_URL` in `.env` was the likely cause (fixed: `--provider gemini` now ignores it).

## 4. DVC (your `.dvc/` folder was missing from the zip – restored here)
`.dvc/config` has a DagsHub remote `origin` (reachable by a reviewer) and your old MinIO as `local_minio`.
```bash
# one-time: create a DagsHub repo by "Connect a repository" from your GitHub repo, then
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-user>
dvc remote modify origin --local password <dagshub-token>
dvc commit embed_corpus -f   # only if status lists 'modified: src/llm/Enums.py' (comment-level change, embeddings unaffected)
dvc status            # should say "Data and pipelines are up to date" (embedded-chunks-v2.json is local on your PC)
dvc push              # uploads corpus + embeddings; reviewer then does: git clone && dvc pull
```
The embedded file is only too big for chat. For Kaggle/other machines you can also
`python -m src.scripts.compact_embeddings pack` (≈55 MB → ≈5 MB `.npz`, float16 vectors, same top-k),
upload the `.npz` as a Kaggle dataset and `unpack` there – or just `dvc pull`.

## 5. Run the service locally (▶ needs Docker / Qdrant)
```bash
docker compose up -d qdrant
python -m src.scripts.ensure_index                  # loads embedded-chunks-v2.json into Qdrant if empty
uvicorn main:app --port 8000
curl -s localhost:8000/health
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"ما هي شروط العقد؟"}'
curl -N localhost:8000/ask/stream -H 'content-type: application/json' -d '{"question":"ما هي شروط العقد؟"}'
```
Full stack (API + Qdrant, 3-command README) → `docker compose up`; monitoring (Prometheus, Grafana,
Langfuse, alerts) → `docker compose -f docker-compose.monitoring.yml up -d`; see `docs/MONITORING.md`.

## 6. Kaggle (GPU) notebooks – use mine instead of the friend's
`notebooks/kaggle_1_mlflow_ragas_sweep` (≥5 chunking runs → MLflow → Registry),
`kaggle_2_optimization` (AWQ vs fp16 + reranker distillation, RAGAS before/after),
`kaggle_3_serving_loadtest` (vLLM + BentoML + Locust 50 users). They clone **your** repo
(AhmedYasser06/Egyptian-Civil-Code-RAG), so push first; the friend's notebooks point at a different repo.
They now use vLLM with Qwen3 thinking disabled and `JUDGE_MAX_TOKENS=4096` (same truncation issue).
Download `eval_out/` from each run and commit `reports/` (mlflow screenshot, locust html, sweep json).

## 7. Push & CI (▶ GitHub)
```bash
git checkout -b complete-project && git add -A && git commit -m "Complete MLOps project" && git push -u origin complete-project
```
Open a PR to `main`: lint → tests → RAGAS gate (reads `reports/ragas_ci.json`) → rebuild index → Docker push (main).
Repository secrets needed: `DVC_USER`, `DVC_PASSWORD` (DagsHub token) – see `.github/workflows/ci.yml`.

## 8. Remaining manual rubric items
Grafana screenshot → README, MLflow comparison screenshot → `reports/`, peer review (≥300 words), final
architecture diagram check in README. The checklist mapping is in `README.md`.
