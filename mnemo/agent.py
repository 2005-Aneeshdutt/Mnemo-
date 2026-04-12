from __future__ import annotations

from dataclasses import dataclass, field
from sqlite3 import Connection

from groq import Groq

from mnemo import config
from mnemo.extractor import extract_memory
from mnemo.pipeline import persist_augmentation
from mnemo.retriever import retrieve_hybrid, retrieve_top_k
from mnemo import store
from mnemo.metrics import approx_tokens_from_text


BASELINE_LAST_N_SYSTEM = """You are a helpful assistant. Use only the recent conversation turns in this chat.
If something was said earlier but is not visible in these turns, say you are not sure rather than guessing."""


MEMORY_INSTRUCTION = """You have access to LONG_TERM_MEMORY (structured facts, semantic triples, and turn summaries).
Use them for recall and consistency. If memory conflicts with the current message, prefer the current message.
If unsure, say you are unsure rather than inventing past details."""


@dataclass
class ChatState:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)


def _format_memory_block(chunks: list) -> str:
    if not chunks:
        return "(no stored memories yet)"

    triples: list[str] = []
    summaries: list[str] = []
    facts: list[str] = []

    for row in chunks:
        kind = row["kind"] or "fact"
        if kind == "triple" and row["subj"] and row["pred"] and row["obj"]:
            triples.append(f"- ({row['subj']}) —[{row['pred']}]→ ({row['obj']})")
        elif kind == "summary":
            c = (row["content"] or "").replace("[summary] ", "", 1).strip()
            if c:
                summaries.append(f"- {c}")
        else:
            facts.append(f"- {row['content']}")

    parts: list[str] = []
    if triples:
        parts.append("SEMANTIC TRIPLES:\n" + "\n".join(triples))
    if facts:
        parts.append("FACTS:\n" + "\n".join(facts))
    if summaries:
        parts.append("TURN SUMMARIES:\n" + "\n".join(summaries))
    return "\n\n".join(parts) if parts else "(empty memory slice)"


def build_system_prompt(memory_block: str) -> str:
    return f"{MEMORY_INSTRUCTION}\n\nLONG_TERM_MEMORY:\n{memory_block}"


def estimate_context_tokens(
    client: Groq,
    conn: Connection,
    state: ChatState,
    user_text: str,
) -> dict[str, int]:
    """
    Rough token estimates for the request that answered `user_text`.
    Expects `state` **after** chat_turn (last two messages = user + assistant for this turn).
    """
    hist = state.messages[:-2] if len(state.messages) >= 2 else []

    rows = store.list_chunks_for_session(conn, state.session_id)
    if config.EMBEDDINGS_DISABLED:
        picked = retrieve_top_k(rows, user_text, config.TOP_K)
    else:
        try:
            picked = retrieve_hybrid(client, rows, user_text, config.TOP_K)
        except Exception:
            picked = retrieve_top_k(rows, user_text, config.TOP_K)
    memory_block = _format_memory_block(picked)
    system_mem = build_system_prompt(memory_block)
    tail = hist[-config.RECENT_MESSAGES :]
    mem_tok = approx_tokens_from_text(
        system_mem,
        *[m.get("content", "") for m in tail],
        user_text,
    )

    n = config.RECENT_MESSAGES
    base_tok = approx_tokens_from_text(
        BASELINE_LAST_N_SYSTEM,
        *[m.get("content", "") for m in hist[-n:]],
        user_text,
    )

    full_tok = approx_tokens_from_text(
        BASELINE_LAST_N_SYSTEM,
        *[m.get("content", "") for m in hist],
        user_text,
    )
    return {"memory_prompt": mem_tok, "baseline_last_n_prompt": base_tok, "full_history_prompt": full_tok}


def chat_turn(
    client: Groq,
    conn: Connection,
    state: ChatState,
    user_text: str,
) -> str:
    rows = store.list_chunks_for_session(conn, state.session_id)
    if config.EMBEDDINGS_DISABLED:
        picked = retrieve_top_k(rows, user_text, config.TOP_K)
    else:
        try:
            picked = retrieve_hybrid(client, rows, user_text, config.TOP_K)
        except Exception:
            picked = retrieve_top_k(rows, user_text, config.TOP_K)

    memory_block = _format_memory_block(picked)
    system_content = build_system_prompt(memory_block)

    api_messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    tail = state.messages[-config.RECENT_MESSAGES :]
    api_messages.extend(tail)
    api_messages.append({"role": "user", "content": user_text})

    try:
        resp = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=api_messages,
            temperature=0.6,
        )
    except Exception as e:
        raise RuntimeError(
            f"Chat completion failed ({config.CHAT_MODEL}). Check GROQ_API_KEY, quotas, and model name. {e}"
        ) from e
    assistant_text = (resp.choices[0].message.content or "").strip()

    state.messages.append({"role": "user", "content": user_text})
    state.messages.append({"role": "assistant", "content": assistant_text})

    try:
        extracted = extract_memory(client, user_text, assistant_text)
        persist_augmentation(client, conn, state.session_id, extracted)
    except Exception as e:
        # Chat already succeeded; memory write failure should not drop the reply
        import logging

        logging.getLogger(__name__).warning("Memory extraction/persist failed (reply still returned): %s", e)

    return assistant_text


def chat_turn_last_n_only(
    client: Groq,
    state: ChatState,
    user_text: str,
    n_recent: int | None = None,
) -> str:
    """
    Baseline: no DB read/write. Only system + last N user/assistant pairs + current user message.
    Used for eval comparison against full Mnemo memory.
    """
    n = n_recent if n_recent is not None else config.RECENT_MESSAGES
    api_messages: list[dict[str, str]] = [{"role": "system", "content": BASELINE_LAST_N_SYSTEM}]
    tail = state.messages[-n:]
    api_messages.extend(tail)
    api_messages.append({"role": "user", "content": user_text})

    try:
        resp = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=api_messages,
            temperature=0.6,
        )
    except Exception as e:
        raise RuntimeError(
            f"Chat completion failed ({config.CHAT_MODEL}). Check GROQ_API_KEY and model availability. {e}"
        ) from e

    assistant_text = (resp.choices[0].message.content or "").strip()
    state.messages.append({"role": "user", "content": user_text})
    state.messages.append({"role": "assistant", "content": assistant_text})
    return assistant_text

