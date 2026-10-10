"""Locust load test for the RAG API (FastAPI ``/ask`` or the BentoML service).

    # rubric run: 50 concurrent users, 5 minutes, report into /reports
    locust -f load_test/locustfile.py --host http://localhost:8000 --headless \
        -u 50 -r 5 -t 5m --csv reports/locust_u50 --html reports/locust_u50.html

Environment:
    ENDPOINT   path to hit                       (default /ask; Bento streaming is also /ask)
    STREAM     1 = read the response as a stream and report time-to-first-token as "TTFT"
    QUESTIONS  JSON file with the eval questions (default data/evaluation/rag_eval.json)
    THINK_MIN / THINK_MAX   seconds a simulated user waits between questions (default 0.5 / 2)
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from locust import HttpUser, between, events, task

FALLBACK = [
    "ما هي شروط صحة العقد؟",
    "ما حكم الغلط في العقد؟",
    "متى يكون العقد باطلا؟",
    "What is the effect of a contract between the parties?",
    "What are the rules on abuse of rights?",
    "هل المادة 400 ما زالت سارية؟",
    "What does Article 147 provide?",
    "ما هو التعويض عن الفعل الضار؟",
]


def load_questions() -> list[str]:
    path = Path(os.getenv("QUESTIONS", "data/evaluation/rag_eval.json"))
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        questions = [row.get("question") or row.get("user_input") for row in data]
        questions = [q for q in questions if q]
        if questions:
            return questions
    return FALLBACK


QUESTIONS = load_questions()
ENDPOINT = os.getenv("ENDPOINT", "/ask")
STREAM = os.getenv("STREAM", "0") == "1"


class AskUser(HttpUser):
    wait_time = between(float(os.getenv("THINK_MIN", "0.5")), float(os.getenv("THINK_MAX", "2")))

    @task
    def ask(self) -> None:
        payload = {"question": random.choice(QUESTIONS)}
        if STREAM:
            self._ask_stream(payload)
            return
        with self.client.post(ENDPOINT, json=payload, name=ENDPOINT, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                body = resp.json()
            except ValueError:
                resp.failure("not JSON")
                return
            if not body.get("answer") or "sources" not in body:
                resp.failure("empty answer or no sources field")

    def _ask_stream(self, payload: dict) -> None:
        start = time.perf_counter()
        first = None
        size = 0
        with self.client.post(
            ENDPOINT, json=payload, name=ENDPOINT, stream=True, catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            for chunk in resp.iter_content(chunk_size=None):
                if chunk and first is None:
                    first = time.perf_counter() - start
                size += len(chunk)
            if first is None:
                resp.failure("empty stream")
                return
            resp.success()
        events.request.fire(
            request_type="STREAM",
            name="TTFT",
            response_time=first * 1000,
            response_length=size,
            exception=None,
            context={},
        )
