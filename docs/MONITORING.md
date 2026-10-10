# Monitoring setup

## 1. Langfuse (self-hosted, port 3000)
```bash
git clone https://github.com/langfuse/langfuse.git && cd langfuse && docker compose up -d   # official compose
```
Open http://localhost:3000 → create an account, an organization and a project → *Settings → API keys*.
Put the keys in `.env`:
```
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```
(From inside the API container `localhost` is the container itself: compose already maps
`host.docker.internal:3000`.) Restart the API; every `/ask` now appears under *Traces* with spans
`retrieval` and `generation`, token usage and cost. The `X-Trace-Id` response header is the trace id.

## 2. Prometheus + Grafana + Alertmanager
```bash
docker compose -f docker-compose.monitoring.yml up -d
```
Grafana http://localhost:3001 (admin / admin) → dashboard **Legal RAG** (RAGAS gauges with the 0.80 threshold line,
p50/p95/p99 latency, drift, token cost, TTFT). Prometheus alerts: http://localhost:9090/alerts
(`LowFaithfulness` < 0.80, `QueryDriftHigh`, `HighP95Latency`). Set a real receiver in
`monitoring/prometheus/alertmanager.yml` (Slack/Discord webhook) for notifications.

## 3. RAGAS monitoring session (≥ 50 questions)
```bash
export LLM_PROVIDER=GROQ GROQ_API_KEY=...          # what the API uses to answer
export JUDGE_API_KEY=$GROQ_API_KEY                  # RAGAS judge (any OpenAI-compatible endpoint)
export ALERT_WEBHOOK_URL=https://hooks.slack.com/... # optional
python -m src.monitoring.ragas_monitor --questions data/evaluation/rag_eval_gemini_50.json --label session-6a --pushgateway http://localhost:9091
```
It logs a run per session to MLflow experiment `civil-code-rag-monitoring` (run it in several sessions: the trend
is the MLflow chart view), attaches the 4 RAGAS scores to each question's Langfuse trace, pushes the means to
Prometheus and alerts if faithfulness < 0.80. **Prove the alert once**: `--threshold 0.99` forces it
(see `reports/alerts.log` and the webhook message), then screenshot it.

Sampled live scoring: `ONLINE_FAITHFULNESS_SAMPLE_RATE=0.2` scores 20 % of real `/ask` calls in the background.

## 4. Drift
```bash
python -m src.monitoring.drift_demo --questions data/evaluation/rag_eval.json     # writes reports/drift_reference.npz + drift_demo.json
```
Commit `reports/drift_reference.npz` (the API image copies it). Then `GET /drift` and the Grafana panel show drift;
the demo report shows in-domain ≈ 0 vs family-law queries clearly above the 0.15 threshold.
