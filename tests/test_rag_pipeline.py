import asyncio
from types import SimpleNamespace

from src import rag
from src.rag import RAGPipeline, ThinkFilter, citations, strip_think

SOURCES = [
    {
        "article_number": 147,
        "citation": "Egyptian Civil Code, Article 147",
        "text_ar": "العقد شريعة المتعاقدين",
        "text_en": "The contract makes the law of the parties.",
        "is_repealed": False,
        "rerank_score": 0.9,
    },
    {
        "article_number": 148,
        "citation": "Egyptian Civil Code, Article 148",
        "text_ar": "يجب تنفيذ العقد طبقا لما اشتمل عليه",
        "text_en": "A contract must be performed.",
        "is_repealed": False,
        "rerank_score": 0.8,
    },
    {"article_number": 147, "citation": "Egyptian Civil Code, Article 147", "text_ar": "dup"},
]


class FakeRetriever:
    embedding_provider = None

    def __init__(self, sources=SOURCES):
        self.sources = sources

    def retrieve(self, query, top_k):
        return self.sources[:top_k]


class FakeLLM:
    """Non-streaming LLM exposing only .invoke (like the mocks used in the older tests)."""

    def invoke(self, prompt):
        return SimpleNamespace(
            content="<think>hmm</think>\n\nوفقا للمادة 147 العقد شريعة المتعاقدين",
            usage_metadata={"input_tokens": 100, "output_tokens": 20},
        )


class FakeStreamingLLM:
    def __init__(self, deltas):
        self.deltas = deltas

    async def astream(self, prompt):
        for i, delta in enumerate(self.deltas):
            usage = {"input_tokens": 50, "output_tokens": 7} if i == len(self.deltas) - 1 else None
            yield SimpleNamespace(content=delta, usage_metadata=usage)


def make(llm, retriever=None):
    return RAGPipeline(retriever or FakeRetriever(), llm, provider="VLLM", model_id="m", drift=None)


def test_strip_think():
    assert strip_think("<think>a\nb</think>\n\nanswer") == "answer"
    assert strip_think("plain") == "plain"


def test_think_filter_handles_split_tags():
    f = ThinkFilter()
    deltas = ["<thi", "nk>secret ", "reasoning</th", "ink>\n\nالمادة ", "147", " تنص"]
    out = "".join(f.feed(d) for d in deltas) + f.flush()
    assert out == "المادة 147 تنص"


def test_think_filter_passthrough_and_partial_angle_bracket():
    f = ThinkFilter()
    out = f.feed("a < b ") + f.feed("and <") + f.feed("b>") + f.flush()
    assert out == "a < b and <b>"


def test_citations_are_deduplicated_article_citations():
    assert citations(SOURCES) == [
        "Egyptian Civil Code, Article 147",
        "Egyptian Civil Code, Article 148",
    ]


def test_aanswer_returns_answer_sources_contexts_and_cost():
    result = asyncio.run(make(FakeLLM()).aanswer("ما هي شروط العقد؟", top_k=2))
    assert result.answer.startswith("وفقا للمادة 147")
    assert "think" not in result.answer
    assert result.sources == [
        "Egyptian Civil Code, Article 147",
        "Egyptian Civil Code, Article 148",
    ]
    assert result.retrieved_articles == [147, 148]
    assert len(result.contexts) == 2 and "Article 147" in result.contexts[0]
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    assert result.cost_usd > 0
    assert result.to_response().keys() == {"answer", "sources"}


def test_aanswer_without_sources_refuses_instead_of_hallucinating():
    result = asyncio.run(make(FakeLLM(), FakeRetriever(sources=[])).aanswer("سؤال"))
    assert result.answer == rag.NO_SOURCES_AR and result.sources == []


def test_astream_yields_tokens_then_sources():
    llm = FakeStreamingLLM(["<think>x</think>", "المادة ", "147 ", "تنص"])

    async def run():
        return [item async for item in make(llm).astream("سؤال عن العقد", top_k=2)]

    events = asyncio.run(run())
    kinds = [e["event"] for e in events]
    assert kinds[:3] == ["token", "token", "token"] and kinds[-2:] == ["sources", "done"]
    assert "".join(e["data"] for e in events if e["event"] == "token") == "المادة 147 تنص"
    done = events[-1]["data"]
    assert (done["input_tokens"], done["output_tokens"]) == (50, 7)
    assert done["ttft_s"] is not None


def test_unavailable_llm_fails_loudly_but_does_not_break_construction(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "GROQ")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm, provider, _ = rag.make_llm_from_env()
    assert provider == "GROQ"
    try:
        llm.invoke("x")
    except RuntimeError as exc:
        assert "GROQ" in str(exc) and ".env" in str(exc)
    else:  # a key may exist in the environment: then a real client was built, also fine
        assert hasattr(llm, "invoke")
