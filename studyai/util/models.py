"""
Chat model via OpenAI API (billig standard: gpt-4o-mini).

Miljö: OPENAI_API_KEY (krävs), OPENAI_MODEL, valfritt OPENAI_BASE_URL.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def _api_key() -> str:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY saknas. Lägg till den i .env (se .env.example)."
        )
    return key


def default_chat_model() -> str:
    return (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()


def get_model(
    *,
    model: str | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    ChatOpenAI. Vanliga kwargs: temperature, max_tokens, top_p, ...

    Ollama-specifika argument (num_predict, repeat_penalty) ignoreras tyst
    så äldre anrop inte kraschar.
    """
    kwargs = {k: v for k, v in kwargs.items() if k not in ("num_predict", "repeat_penalty")}

    params: dict[str, Any] = {
        "model": model or default_chat_model(),
        "api_key": _api_key(),
    }
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if base_url:
        params["base_url"] = base_url
    params.update(kwargs)
    return ChatOpenAI(**params)
