# Project evidence summary

## RAGAS on 50 questions (`ragas_monitor_session-6.json`)

Generator: GROQ / qwen/qwen3.8-27b - judge: qwen/qwen3.8-27b - top_k 5

| metric | mean | scored |
|---|---|---|
| faithfulness | 0.874 | 50 |
| answer_relevancy | 0.732 | 50 |
| context_precision | 0.934 | 50 |
| context_recall | 0.920 | 50 |

article hit rate: 0.98  |  total tokens: 25872  |  cost USD: 0.0082

Alert rule: faithfulness < 0.80 -> ok (mean 0.874); CI gate: faithfulness >= 0.75 on >= 20 questions.

Lowest-faithfulness questions: q14_ar (0.20), q15_ar (0.45), q11_en (0.50), q19_ar (0.50), q19_en (0.50)

## Embedding (cosine) drift

| query set | drift | similarity | alert |
|---|---|---|---|
| in_domain | 0.032 | 0.715 | False |
| family_law | 0.227 | 0.567 | True |

Threshold 0.15: contract-law questions stay below it; family-law questions (a different jurisdiction of the same code) cross it - the semantic shift the handbook describes.

## Retrieval: BGE-M3 vs BGE-M3 + reranker (20 bilingual questions)

| metric | no reranker | with reranker |
|---|---|---|
| recall_at_1 | 0.950 | 1.000 |
| recall_at_3 | 1.000 | 1.000 |
| recall_at_5 | 1.000 | 1.000 |
| mrr | 0.975 | 1.000 |
| bilingual top-1 consistency | 0.90 | 1.00 |

## Files

- found `reports/ragas_ci.json`

## Still missing

- MLflow chunking runs (>= 5) - run notebooks/kaggle_1_mlflow_ragas_sweep
- reports/quantization.md - AWQ vs fp16 + reranker distillation (notebook 2)
- reports/reranker_distillation.md - reranker distillation (notebook 2)
- reports/load_test.md - Locust p50/p95/p99 at 50 users (notebook 3)
- reports/locust_u50.html - Locust report at 50 users (notebook 3)
- reports/streaming_demo.txt - streaming curl proof (notebook 3)
- reports/langfuse_trace.png - screenshot of a Langfuse trace with RAGAS scores (manual screenshot)
- reports/mlflow_comparison_ui.png - screenshot of the MLflow UI compare page (manual screenshot)
