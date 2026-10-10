import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

load_dotenv()

MODEL = os.getenv("GROQ_MODEL_ID", "qwen/qwen3.8-27b")
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing")

dataset = json.loads(
    Path("data/evaluation/ragas_dataset.json").read_text(encoding="utf-8")
)
sample = next(x for x in dataset if x["id"] == "q11_en")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Use the OpenAI-compatible adapter and JSON response mode.
judge = llm_factory(
    model=MODEL,
    client=client,
    temperature=0,
    max_tokens=900,
    adapter="instructor",
)

metric = Faithfulness(llm=judge)

async def main():
    try:
        result = await metric.ascore(
            user_input=sample["user_input"],
            response=sample["response"],
            retrieved_contexts=sample["retrieved_contexts"],
        )
        print("Faithfulness:", result.value)
    except Exception as exc:
        print("Exception:", type(exc).__name__, repr(exc))
        completion = getattr(exc, "last_completion", None)
        if completion is not None:
            print("Finish reason:", getattr(completion.choices[0], "finish_reason", None))
            print("Raw response:", completion.choices[0].message.content)
        raise

asyncio.run(main())
