from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/legal/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["documents_indexed"] == 1149


def test_empty_question_returns_422():
    response = client.post(
        "/api/v1/legal/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_whitespace_question_returns_422():
    response = client.post(
        "/api/v1/legal/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_too_short_question_returns_422():
    response = client.post(
        "/api/v1/legal/ask",
        json={"question": "a"},
    )

    assert response.status_code == 422


def test_missing_question_returns_422():
    response = client.post(
        "/api/v1/legal/ask",
        json={},
    )

    assert response.status_code == 422

def test_question_whitespace_is_trimmed():
    request = {
        "question": "  ما هي شروط العقد؟  ",
    }

    from src.schemas.legal_query import LegalQueryRequest

    parsed = LegalQueryRequest(**request)

    assert parsed.question == "ما هي شروط العقد؟"
    
def test_ask_returns_answer_and_article_sources(monkeypatch):
    fake_sources = [
        {
            "article_number": 5,
            "book": "Persons",
            "chapter": None,
            "section": None,
            "topic": "Abuse of Rights",
            "text_ar": "Arabic legal text for Article 5.",
            "text_en": "English legal text for Article 5.",
            "is_repealed": False,
            "source_page": 10,
            "citation": "Egyptian Civil Code, Article 5",
            "rerank_score": 0.95,
        }
    ]

    class FakeResponse:
        content = "وفقًا للمادة 5 من القانون المدني المصري..."

    class FakeRetriever:
        def retrieve(self, query, top_k):
            return fake_sources

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse()

    import src.routes.legal as legal

    monkeypatch.setattr(legal, "retriever", FakeRetriever())
    monkeypatch.setattr(legal, "llm", FakeLLM())

    response = client.post(
        "/api/v1/legal/ask",
        json={"question": "ما هي شروط استعمال الحق؟"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "وفقًا للمادة 5 من القانون المدني المصري..."
    )
    assert data["sources"] == [
        "Egyptian Civil Code, Article 5"
    ]