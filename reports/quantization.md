# Optimization: fp16 -> awq

| label | model | reranker | faithfulness | answer_rel | ctx_prec | ctx_recall | p50 s | p95 s | TTFT p50 s | tok/s (conc.) | weights GiB/GPU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | Qwen/Qwen3-8B | bge-reranker-v2-m3 | 0.932 | 0.740 | 0.919 | 0.983 | 9.11 | 26.03 | 0.86 | 52.0 | 7.640 |
| awq | Qwen/Qwen3-8B-AWQ | bge-reranker-v2-m3 | 0.972 | 0.720 | 0.915 | 0.983 | 10.72 | 41.28 | 0.88 | 58.8 | 2.850 |
| awq_student | Qwen/Qwen3-8B-AWQ | reranker-student | 0.894 | 0.737 | 0.896 | 1.000 | 9.81 | 40.59 | 0.77 | 67.2 | 2.850 |

- faithfulness drop: **-0.040** (limit 0.03) -> **PASS**
- p50 latency speed-up: **0.85x**
- judge: Qwen/Qwen3-8B for every label; 30 questions
