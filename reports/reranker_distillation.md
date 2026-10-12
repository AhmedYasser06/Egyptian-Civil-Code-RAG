| reranker | R@1 | R@3 | R@5 | MRR | ms/query (mean) | ms p95 | params (M) |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_rerank | 0.950 | 1.000 | 1.000 | 0.975 | 0 | 0 | 0 |
| teacher | 1.000 | 1.000 | 1.000 | 1.000 | 226 | 242 | 568 |
| student | 0.850 | 0.950 | 1.000 | 0.912 | 38 | 52 | 118 |

student is 5.9x faster and 4.8x smaller; keeps 100% of teacher recall@5 on 20 questions.
