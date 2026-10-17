import os
import sys

# Tests never talk to Langfuse / a real LLM.
for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.pop(var, None)
os.environ.setdefault("LLM_PROVIDER", "OLLAMA")  # constructing ChatOllama needs no credentials

# Never run the background RAGAS judge during tests (it would call a real LLM and still be
# running when pytest exits). Set before the app is imported; load_dotenv() does not override it.
os.environ["ONLINE_FAITHFULNESS_SAMPLE_RATE"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))