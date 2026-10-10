"""Token cost accounting.

Prices are USD per 1M tokens and are *assumptions you should edit*: hosted APIs change prices, and
for a self-hosted vLLM model the "price" is your amortised GPU cost. Override any of them with
environment variables, e.g. ``PRICE_VLLM_INPUT_PER_1M=0.04``.
"""

from __future__ import annotations

import os

# provider -> (input $/1M tokens, output $/1M tokens)
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "GROQ": (0.29, 0.59),
    "GOOGLE_GENAI": (0.30, 2.50),
    "OLLAMA": (0.0, 0.0),
    # ~ a T4 at $0.35/h serving ~1.5M tokens/h of AWQ Qwen3-8B => roughly $0.23/1M. Rounded.
    "VLLM": (0.10, 0.25),
}


def price_per_million(provider: str) -> tuple[float, float]:
    provider = (provider or "").upper()
    default_in, default_out = DEFAULT_PRICES.get(provider, (0.0, 0.0))
    price_in = float(os.getenv(f"PRICE_{provider}_INPUT_PER_1M", default_in))
    price_out = float(os.getenv(f"PRICE_{provider}_OUTPUT_PER_1M", default_out))
    return price_in, price_out


def compute_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD of one LLM call."""
    price_in, price_out = price_per_million(provider)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Fallback when the provider returns no usage (~3 chars/token for Arabic-heavy text)."""
    return max(1, round(len(text or "") / 3))
