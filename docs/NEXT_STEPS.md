# Next steps (in order)

## 0. Why GitHub Actions and Kaggle failed
Your DVC remote is a MinIO at `localhost:9000` = your laptop. GitHub's and Kaggle's machines cannot reach it
(`Unable to locate credentials`, `Could not connect to the endpoint URL`). Fix = the data also lives in git as
a small snapshot (`data/release/`, ~10 MB: corpus + chunks as JSON, embeddings as float16 `.npz`);
`python -m src.scripts.fetch_data` restores `data/processed/` from it everywhere. No DVC remote, secrets or
Kaggle dataset upload needed. (DVC stays in the project for `dvc repro`; see step 6 for a shared remote.)

## 1. On your PC (WSL, project folder)
```bash
unzip -o update.zip                                    # this archive, over the project
python -m src.scripts.fetch_data --make-release        # data/processed -> data/release (needs your local embedded-chunks-v2.json)
python -m pytest -q && ruff check src main.py scripts tests
python scripts/make_ragas_ci_report.py && python scripts/ragas_gate.py       # 30 questions, must print PASS
python scripts/make_reports.py                         # charts + reports/SUMMARY.md
git add -A && git commit -m "Data snapshot, fetch_data, Kaggle/CI fixes, reports" && git push
```
Open the Actions tab: `Lint`, `RAGAS gate`, `Pull corpus ... rebuild index` should be green; `Build image ...`
runs only on `main` (it builds the Docker image, downloads BGE-M3 and pushes to GHCR - the first run is slow;
if the push is denied: repo Settings > Actions > General > Workflow permissions > *Read and write*).

## 2. Kaggle (GPU T4 x2, Internet ON) - no secrets needed any more
Notebooks 1, 2, 3 in `notebooks/`. Upload each (File > Import notebook), Settings: Accelerator *GPU T4 x2*, Internet *On*.
Cell 1 clones the repo, cell 3 runs `fetch_data`. Run **after step 1's push** (they clone GitHub).
Order: 1 (chunking sweep, >= 5 MLflow runs + Registry) -> 2 (AWQ + distillation) -> 3 (vLLM + BentoML + Locust 50 users).
Download each `*_results.zip`, unzip into the project root, and commit only `reports/` (not `mlflow.db`/`mlruns`).

## 3. Screenshots (manual, save in reports/)
`grafana_faithfulness.png` (http://localhost:3001, admin/admin), `langfuse_trace.png`, `mlflow_comparison_ui.png`
(mlflow ui --backend-store-uri sqlite:///mlflow.db -> select >= 5 runs -> Compare). Then re-run
`python scripts/make_reports.py`: its "Still missing" list is your checklist.

## 4. README + rubric
Put the screenshots/charts and the Locust p95 in the README, keep the 3-command quick start, then the peer review
(>= 300 words: setup, code, strengths, 2 improvements, extension).

## 5. Merge
PR -> Actions green -> merge to `main` -> image pushed to ghcr.io.

## 6. Optional: a real shared DVC remote (rubric wording "dvc pull fetches them")
Create a DagsHub repo from your GitHub repo, then
`dvc remote default origin && dvc remote modify origin --local auth basic && dvc remote modify origin --local user <u> && dvc remote modify origin --local password <token> && dvc push`.
`dvc status` may show `embed_corpus: changed deps: src/llm/Enums.py` - harmless: `dvc commit embed_corpus -f`.
