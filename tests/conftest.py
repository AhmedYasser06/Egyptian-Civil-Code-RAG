import os
import sys

# Tests never talk to Langfuse / a real LLM.
for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ.pop(var, None)
os.environ.setdefault("LLM_PROVIDER", "OLLAMA")  # constructing ChatOllama needs no credentials
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
