import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .Enums import LLMEnums


class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str, model_id: str = None, model_temperature: float = 0.2):
        
        if provider == LLMEnums.OLLAMA.value:
            return ChatOllama(model=model_id, temperature=model_temperature)
        
        elif provider == LLMEnums.GOOGLE_GENAI.value:
            return ChatGoogleGenerativeAI(model=model_id, temperature=model_temperature)
        
        elif provider == LLMEnums.GROQ.value:
            return ChatGroq(model=model_id, temperature=model_temperature)

        elif provider == LLMEnums.VLLM.value:
            # vLLM exposes an OpenAI-compatible API (`vllm serve <model>`).
            # Qwen3 "thinking" is switched off: it would burn tokens and break citations.
            return ChatOpenAI(
                model=model_id,
                temperature=model_temperature,
                base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
                stream_usage=True,
                max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "700")),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        return None