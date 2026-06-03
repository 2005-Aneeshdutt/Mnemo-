from __future__ import annotations

from typing import Any

from mnemo import config


def create_client() -> Any:
    """
    Return a chat client. Uses Gemini if GEMINI_API_KEY is set, otherwise Groq.
    Both clients expose identical chat.completions.create() and embeddings.create() interfaces.
    """
    if config.GEMINI_API_KEY:
        import openai
        return openai.OpenAI(
            base_url=config.GEMINI_BASE_URL,
            api_key=config.GEMINI_API_KEY,
        )

    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY or GROQ_API_KEY in your .env file."
        )

    import os
    os.environ.setdefault("GROQ_API_KEY", config.GROQ_API_KEY)
    from groq import Groq
    return Groq()


def active_provider() -> str:
    return "gemini" if config.GEMINI_API_KEY else "groq"
