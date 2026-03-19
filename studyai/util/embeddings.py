"""
Embeddings via OpenAI (standard: text-embedding-3-small — billig och bra för FAISS).

Miljö: OPENAI_API_KEY, valfritt OPENAI_EMBEDDING_MODEL.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def default_embedding_model() -> str:
    return (os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small").strip()


def get_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY saknas. Lägg till den i .env (se .env.example)."
        )
    kwargs: dict = {
        "model": model or default_embedding_model(),
        "api_key": api_key,
    }
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)
