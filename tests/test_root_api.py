from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import src.routes.legal as legal
import src.routes.root as root
from main import app

client = TestClient(app)

SRC = [
    {
        "article_number": 5,
        "citation": "Egyptian Civil Code, Article 5",
        "text_ar": "ar",
        "text_en": "en",
        "is_repealed": False,
    }
]


class FakeRetriever:
    collection_name = "x"
    client = None

    def retrieve(self, query, top_k):
        return SRC


class FakeLLM:
    def invoke(self, prompt):
        return SimpleNamespace(content="وفقًا للمادة 5", usage_metadata=None)

    async def astream(self, prompt):
        for d in ["وفقًا ", "للمادة ", "5"]:
            yield SimpleNamespace(content=d, usage_metadata=None)


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(legal, "retriever", FakeRetriever())
    monkeypatch.setattr(legal, "llm", FakeLLM())
    root._health_cache.update(at=0.0, articles=None, chunks=None)


def test_ask_contract():
    r = client.post("/ask", json={"question": "ما هي شروط العقد؟"})
    assert r.status_code == 200
    assert r.json() == {"answer": "وفقًا للمادة 5", "sources": ["Egyptian Civil Code, Article 5"]}


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "   "}, {}, {"question": "a"}])
def test_ask_rejects_empty_question_with_422(body):
    assert client.post("/ask", json=body).status_code == 422


def test_ask_stream_is_sse_with_tokens_then_sources():
    r = client.post("/ask/stream", json={"question": "ما هي شروط العقد؟"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert body.count("event: token") == 3
    assert body.index("event: sources") > body.index("event: token")
    assert "Egyptian Civil Code, Article 5" in body


def test_health_reports_article_and_chunk_counts(monkeypatch):
    monkeypatch.setattr(root, "_count_index", lambda: (1149, 2298))
    body = client.get("/health").json()
    assert body == {
        "status": "healthy",
        "documents_indexed": 1149,  # alias of articles_indexed, kept for older clients
        "articles_indexed": 1149,
        "chunks_indexed": 2298,
    }


def test_health_is_503_when_vector_store_is_down(monkeypatch):
    def boom():
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(root, "_count_index", boom)
    r = client.get("/health")
    body = r.json()
    assert r.status_code == 503 and body["status"] == "unhealthy"
    assert body["articles_indexed"] == 0 and body["chunks_indexed"] == 0


def test_root_redirects_to_docs():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"


def test_metrics_endpoint_exposes_rag_metrics():
    client.post("/ask", json={"question": "ما هي شروط العقد؟"})
    text = client.get("/metrics").text
    assert "rag_requests_total" in text and "rag_tokens_total" in text