| run | endpoint | requests | failures | req/s | avg (s) | p50 (s) | p95 (s) | p99 (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| locust_u1 | /ask_json | 7 | 0 | 0.06 | 14.79 | 12.00 | 26.00 | 26.00 |
| locust_u50 | /ask_json | 380 | 0 | 1.27 | 33.38 | 26.00 | 71.00 | 146.00 |
| locust_stream_u20 | /ask | 39 | 0 | 0.67 | 3.23 | 1.90 | 6.90 | 7.10 |
| locust_stream_u20 | TTFT | 39 | 0 | 0.67 | 3.23 | 1.90 | 6.90 | 7.10 |
