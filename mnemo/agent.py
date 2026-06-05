from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from sqlite3 import Connection
from typing import Callable, Generator

from groq import Groq

from mnemo import config
from mnemo.extractor import extract_memory
from mnemo.pipeline import persist_augmentation
from mnemo.retriever import retrieve_hybrid, retrieve_top_k
from mnemo import store
from mnemo import metrics as metrics_mod
from mnemo.metrics import approx_tokens_from_text
from mnemo import tools as memory_tools
from mnemo import compactor

logger = logging.getLogger(__name__)


def _extract_and_persist(
    client: Groq,
    conn: Connection,
    session_id: str,
    user_text: str,
    assistant_text: str,
    max_retries: int = 3,
) -> None:
    """Extract memories and persist, retrying on 429 rate-limit errors."""
    for attempt in range(max_retries):
        try:
            extracted = extract_memory(client, user_text, assistant_text)
            persist_augmentation(client, conn, session_id, extracted)
            return
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 15 * (2 ** attempt)
                logger.warning("Extraction rate-limited, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                logger.warning("Memory extraction/persist failed (reply still returned): %s", e)
                return

BASELINE_LAST_N_SYSTEM = """You are a helpful assistant. Use only the recent conversation turns in this chat.
If something was said earlier but is not visible in these turns, say you are not sure rather than guessing."""

AGENTIC_SYSTEM = """You are a helpful assistant with access to a persistent long-term memory system.

You have four memory tools:
- recall(query): search memory for relevant facts, triples, or summaries
- remember(content): explicitly store something important the user tells you
- forget(chunk_id): delete a memory that is no longer true
- update_fact(chunk_id, new_content): update a memory that has changed

How to use them well:
- Call recall BEFORE answering questions about the user's past, preferences, or context.
- Call remember when the user states something durable (name, preference, fact about themselves).
- Call forget or update_fact when the user corrects or revokes something previously stored.
- You may call multiple tools in sequence if needed.
- After using tools, give your final reply directly — do not narrate what you just did."""

# Maximum tool-call rounds per turn to prevent runaway loops.
_MAX_TOOL_ROUNDS = 6


@dataclass
class ChatState:
    session_id: str
    tenant_id: str = "default"
    messages: list[dict] = field(default_factory=list)
    turn_count: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


def _format_profile_block(conn: Connection, tenant_id: str) -> str:
    rows = store.list_profile_chunks(conn, tenant_id)
    if not rows:
        return ""
    lines: list[str] = []
    for r in rows:
        kind = r["kind"] or "fact"
        if kind == "triple" and r["subj"]:
            lines.append(f"- ({r['subj']}) —[{r['pred']}]→ ({r['obj']})")
        else:
            lines.append(f"- {r['content']}")
    return "\n".join(lines)


def _build_system_with_profile(conn: Connection, tenant_id: str) -> str:
    profile = _format_profile_block(conn, tenant_id)
    if not profile:
        return AGENTIC_SYSTEM
    return f"{AGENTIC_SYSTEM}\n\nUSER PROFILE (persists across all sessions):\n{profile}"


# ---------------------------------------------------------------------------
# Token estimation (unchanged, kept for eval harness)
# ---------------------------------------------------------------------------

def estimate_context_tokens(
    client: Groq,
    conn: Connection,
    state: ChatState,
    user_text: str,
) -> dict[str, int]:
    hist = state.messages[:-2] if len(state.messages) >= 2 else []

    rows = store.list_chunks_for_session(conn, state.session_id, limit=config.MEMORY_MAX_ROWS)
    if config.EMBEDDINGS_DISABLED:
        picked = retrieve_top_k(rows, user_text, config.TOP_K)
    else:
        try:
            picked = retrieve_hybrid(client, rows, user_text, config.TOP_K)
        except Exception:
            picked = retrieve_top_k(rows, user_text, config.TOP_K)
    memory_block = _format_memory_block(picked)
    system_mem = AGENTIC_SYSTEM + "\n\n" + memory_block
    tail = hist[-config.RECENT_MESSAGES:]
    mem_tok = approx_tokens_from_text(
        system_mem, *[m.get("content", "") for m in tail], user_text
    )

    n = config.RECENT_MESSAGES
    base_tok = approx_tokens_from_text(
        BASELINE_LAST_N_SYSTEM, *[m.get("content", "") for m in hist[-n:]], user_text
    )
    full_tok = approx_tokens_from_text(
        BASELINE_LAST_N_SYSTEM, *[m.get("content", "") for m in hist], user_text
    )
    return {"memory_prompt": mem_tok, "baseline_last_n_prompt": base_tok, "full_history_prompt": full_tok}


# ---------------------------------------------------------------------------
# Agentic chat turn
# ---------------------------------------------------------------------------

def _record_token_savings(state: ChatState, picked: list) -> None:
    """Record tokens saved by retrieval vs sending full history."""
    full_chars = sum(len(m.get("content") or "") for m in state.messages)
    sent_chars = sum(len(m.get("content") or "") for m in state.messages[-config.RECENT_MESSAGES:])
    saved = max(0, (full_chars - sent_chars) // 4)
    metrics_mod.record_tokens_saved(saved)
    metrics_mod.record_memory_retrieval(hits=len(picked), misses=1 if not picked else 0)


def _trim_messages(messages: list[dict], max_chars: int = 600) -> list[dict]:
    """Truncate individual message content to keep request sizes bounded."""
    out = []
    for m in messages:
        c = m.get("content") or ""
        if len(c) > max_chars:
            c = c[:max_chars] + "…"
        out.append({**m, "content": c})
    return out


def _passive_fallback(
    client: Groq,
    conn: Connection,
    state: ChatState,
    user_text: str,
) -> str:
    """Plain completion with memory pre-injected — used when model doesn't support tool calls."""
    session_rows = store.list_chunks_for_session(conn, state.session_id, limit=config.MEMORY_MAX_ROWS)
    profile_rows = store.list_profile_chunks(conn, state.tenant_id)
    all_rows = list(session_rows) + list(profile_rows)

    if config.EMBEDDINGS_DISABLED:
        picked = retrieve_top_k(all_rows, user_text, config.TOP_K) if all_rows else []
    else:
        try:
            picked = retrieve_hybrid(client, all_rows, user_text, config.TOP_K) if all_rows else []
        except Exception:
            picked = retrieve_top_k(all_rows, user_text, config.TOP_K) if all_rows else []

    _record_token_savings(state, picked)
    memory_block = _format_memory_block(picked)
    # Hard cap to keep request size bounded on low-TPM free tiers
    if len(memory_block) > 1200:
        memory_block = memory_block[:1200] + "…"

    profile_block = _format_profile_block(conn, state.tenant_id)
    system_content = "You are a helpful assistant with access to long-term memory.\n\n"
    if profile_block:
        system_content += f"USER PROFILE (persists across sessions):\n{profile_block}\n\n"
    system_content += f"SESSION MEMORY:\n{memory_block}"
    api_messages: list[dict] = [{"role": "system", "content": system_content}]
    api_messages.extend(_trim_messages(state.messages[-config.RECENT_MESSAGES:]))
    api_messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=api_messages,
        temperature=0.6,
    )
    assistant_text = (resp.choices[0].message.content or "").strip()
    state.messages.append({"role": "user", "content": user_text})
    state.messages.append({"role": "assistant", "content": assistant_text})
    state.turn_count += 1

    _extract_and_persist(client, conn, state.session_id, user_text, assistant_text)

    if compactor.should_compact(state.turn_count):
        compactor.compact_session_async(client, conn, state.session_id)

    return assistant_text


def chat_turn(
    client: Groq,
    conn: Connection,
    state: ChatState,
    user_text: str,
    on_tool_call: Callable[[str, dict, str], None] | None = None,
) -> str:
    """
    Tool-use loop: model can call recall/remember/forget/update_fact any number
    of times (up to _MAX_TOOL_ROUNDS) before producing a final text reply.

    on_tool_call(tool_name, args, result) — optional callback for live display.
    """
    if not config.TOOL_USE_ENABLED:
        return _passive_fallback(client, conn, state, user_text)

    # Record token savings from history truncation before tool-use loop
    full_chars = sum(len(m.get("content") or "") for m in state.messages)
    sent_chars = sum(len(m.get("content") or "") for m in state.messages[-config.RECENT_MESSAGES:])
    metrics_mod.record_tokens_saved(max(0, (full_chars - sent_chars) // 4))

    system_prompt = _build_system_with_profile(conn, state.tenant_id)
    api_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    tail = state.messages[-config.RECENT_MESSAGES:]
    api_messages.extend(tail)
    api_messages.append({"role": "user", "content": user_text})

    assistant_text = ""

    for _ in range(_MAX_TOOL_ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=api_messages,
                tools=memory_tools.TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.6,
            )
        except Exception as e:
            err_str = str(e)
            # Some models (e.g. llama-3.3-70b-versatile) emit XML-style tool calls
            # that Groq rejects with a 400 tool_use_failed. Fall back to passive injection.
            if "tool_use_failed" in err_str or ("400" in err_str and "tool" in err_str.lower()):
                logger.warning("Tool-use not supported by model %s; falling back to passive memory. %s", config.CHAT_MODEL, e)
                return _passive_fallback(client, conn, state, user_text)
            raise RuntimeError(
                f"Chat completion failed ({config.CHAT_MODEL}). "
                f"Check GROQ_API_KEY, quotas, and model name. {e}"
            ) from e

        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        if finish == "tool_calls" and msg.tool_calls:
            # Append the assistant's tool-call message (no content field needed)
            api_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool and feed results back
            for tc in msg.tool_calls:
                name, args = memory_tools.parse_tool_call(tc)
                result = memory_tools.execute_tool(name, args, client, conn, state.session_id, tenant_id=state.tenant_id)
                if on_tool_call:
                    on_tool_call(name, args, result)
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # Model gave a plain text reply — we're done
            assistant_text = (msg.content or "").strip()
            break
    else:
        # Exceeded max rounds — force a final answer without tools
        api_messages.append({"role": "user", "content": "Please give your final answer now."})
        try:
            resp = client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=api_messages,
                temperature=0.6,
            )
            assistant_text = (resp.choices[0].message.content or "").strip()
        except Exception:
            assistant_text = "(Could not generate a response after tool use.)"

    state.messages.append({"role": "user", "content": user_text})
    state.messages.append({"role": "assistant", "content": assistant_text})
    state.turn_count += 1

    _extract_and_persist(client, conn, state.session_id, user_text, assistant_text)

    if compactor.should_compact(state.turn_count):
        compactor.compact_session_async(client, conn, state.session_id)

    return assistant_text


# ---------------------------------------------------------------------------
# Streaming variant
# ---------------------------------------------------------------------------

def chat_turn_stream(
    client: Groq,
    conn: Connection,
    state: ChatState,
    user_text: str,
    on_tool_call: Callable[[str, dict, str], None] | None = None,
) -> Generator[str, None, None]:
    """
    Streaming version of chat_turn. Yields text chunks as they arrive.
    Tool calls (if any) run synchronously first; then the final reply streams.
    Callers must exhaust the generator — memory extraction runs after the last chunk.
    """
    if not config.TOOL_USE_ENABLED:
        yield from _passive_fallback_stream(client, conn, state, user_text)
        return

    # Run tool-use rounds synchronously to completion, then stream the final reply.
    system_prompt = _build_system_with_profile(conn, state.tenant_id)
    api_messages: list[dict] = [{"role": "system", "content": system_prompt}]
    api_messages.extend(state.messages[-config.RECENT_MESSAGES:])
    api_messages.append({"role": "user", "content": user_text})

    # Non-streaming tool-use rounds
    for _ in range(_MAX_TOOL_ROUNDS):
        try:
            resp = client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=api_messages,
                tools=memory_tools.TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.6,
            )
        except Exception as e:
            err_str = str(e)
            if "tool_use_failed" in err_str or ("400" in err_str and "tool" in err_str.lower()):
                logger.warning("Tool-use fallback triggered in stream mode: %s", e)
                yield from _passive_fallback_stream(client, conn, state, user_text)
                return
            raise RuntimeError(
                f"Chat completion failed ({config.CHAT_MODEL}). {e}"
            ) from e

        msg = resp.choices[0].message
        if resp.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
            api_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                name, args = memory_tools.parse_tool_call(tc)
                result = memory_tools.execute_tool(name, args, client, conn, state.session_id, tenant_id=state.tenant_id)
                if on_tool_call:
                    on_tool_call(name, args, result)
                api_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            # Model gave a plain reply without using tools — stream it
            api_messages.append({"role": "user", "content": user_text})
            break
    else:
        api_messages.append({"role": "user", "content": "Please give your final answer now."})

    # Stream the final reply
    assistant_text = yield from _stream_final(client, api_messages)
    _finish_turn(client, conn, state, user_text, assistant_text)


def _stream_final(client: Groq, api_messages: list[dict]) -> Generator[str, None, str]:
    """Stream a completion and return the full assembled text."""
    full = ""
    try:
        stream = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=api_messages,
            temperature=0.6,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full += delta
                yield delta
    except Exception as e:
        raise RuntimeError(f"Streaming completion failed ({config.CHAT_MODEL}). {e}") from e
    return full


def _passive_fallback_stream(
    client: Groq,
    conn: Connection,
    state: ChatState,
    user_text: str,
) -> Generator[str, None, None]:
    session_rows = store.list_chunks_for_session(conn, state.session_id, limit=config.MEMORY_MAX_ROWS)
    profile_rows = store.list_profile_chunks(conn, state.tenant_id)
    all_rows = list(session_rows) + list(profile_rows)

    if config.EMBEDDINGS_DISABLED:
        picked = retrieve_top_k(all_rows, user_text, config.TOP_K) if all_rows else []
    else:
        try:
            picked = retrieve_hybrid(client, all_rows, user_text, config.TOP_K) if all_rows else []
        except Exception:
            picked = retrieve_top_k(all_rows, user_text, config.TOP_K) if all_rows else []

    memory_block = _format_memory_block(picked)
    if len(memory_block) > 1200:
        memory_block = memory_block[:1200] + "…"

    profile_block = _format_profile_block(conn, state.tenant_id)
    system_content = "You are a helpful assistant with access to long-term memory.\n\n"
    if profile_block:
        system_content += f"USER PROFILE (persists across sessions):\n{profile_block}\n\n"
    system_content += f"SESSION MEMORY:\n{memory_block}"
    api_messages: list[dict] = [{"role": "system", "content": system_content}]
    api_messages.extend(_trim_messages(state.messages[-config.RECENT_MESSAGES:]))
    api_messages.append({"role": "user", "content": user_text})

    assistant_text = yield from _stream_final(client, api_messages)
    _finish_turn(client, conn, state, user_text, assistant_text)


def _finish_turn(
    client: Groq,
    conn: Connection,
    state: ChatState,
    user_text: str,
    assistant_text: str,
) -> None:
    state.messages.append({"role": "user", "content": user_text})
    state.messages.append({"role": "assistant", "content": assistant_text})
    state.turn_count += 1
    _extract_and_persist(client, conn, state.session_id, user_text, assistant_text)
    if compactor.should_compact(state.turn_count):
        compactor.compact_session_async(client, conn, state.session_id)


# ---------------------------------------------------------------------------
# Baseline (eval only — no tool use, no DB)
# ---------------------------------------------------------------------------

def chat_turn_last_n_only(
    client: Groq,
    state: ChatState,
    user_text: str,
    n_recent: int | None = None,
) -> str:
    n = n_recent if n_recent is not None else config.RECENT_MESSAGES
    api_messages: list[dict] = [{"role": "system", "content": BASELINE_LAST_N_SYSTEM}]
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
            f"Chat completion failed ({config.CHAT_MODEL}). "
            f"Check GROQ_API_KEY and model availability. {e}"
        ) from e

    assistant_text = (resp.choices[0].message.content or "").strip()
    state.messages.append({"role": "user", "content": user_text})
    state.messages.append({"role": "assistant", "content": assistant_text})
    return assistant_text
