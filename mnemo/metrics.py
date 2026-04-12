"""Rough context-size estimates (for eval / README; not billed token counts)."""


def approx_tokens_from_text(*parts: str) -> int:
    """~4 chars per token (common heuristic for Latin text)."""
    total = sum(len(p) for p in parts if p)
    return max(0, total // 4)


def approx_tokens_chat_messages(messages: list[dict[str, str]]) -> int:
    return approx_tokens_from_text(*(m.get("content") or "" for m in messages))
