import os

# Provider SDKs are imported lazily: each environment (Kaggle, CI, laptop) only
# installs the ones it uses, so a missing package must not break the others.
from .Enums import LLMEnums


class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str, model_id: str = None, model_temperature: float = 0.2):
        
        if provider == LLMEnums.OLLAMA.value:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model_id, temperature=model_temperature)
        
        elif provider == LLMEnums.GOOGLE_GENAI.value:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_id, temperature=model_temperature)
        
        elif provider == LLMEnums.GROQ.value:
            from langchain_groq import ChatGroq
            return ChatGroq(model=model_id, temperature=model_temperature)

        elif provider == LLMEnums.VLLM.value:
            # vLLM exposes an OpenAI-compatible API (`vllm serve <model>`).
            # Qwen3 "thinking" is switched off: it would burn tokens and break citations.
            from langchain_openai import ChatOpenAI

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