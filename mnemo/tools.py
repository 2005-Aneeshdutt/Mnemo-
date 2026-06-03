from __future__ import annotations

import json
import logging
from sqlite3 import Connection
from typing import Any

import numpy as np
from groq import Groq

from mnemo import config
from mnemo import embeddings as emb
from mnemo import store
from mnemo.retriever import retrieve_hybrid, retrieve_top_k

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas (Groq function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search both session memory and the persistent user profile for relevant facts. "
                "Call this before answering questions about the user's past, preferences, or context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Describe what you want to remember.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Store a fact in session memory (lasts this conversation only). "
                "Use for temporary or session-specific information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact to store. Be specific and self-contained.",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_profile",
            "description": (
                "Store a fact permanently in the user's profile — persists across ALL sessions. "
                "Use for durable facts: name, location, job, preferences, relationships, allergies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The durable fact to store in the user profile.",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": (
                "Delete a specific session memory entry by its ID. "
                "Use when the user asks to forget something or a fact is no longer true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "integer",
                        "description": "The numeric ID from a prior recall result.",
                    }
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_fact",
            "description": (
                "Overwrite the content of an existing session memory entry. "
                "Use when a previously stored fact has changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "integer",
                        "description": "The numeric ID of the memory entry to update.",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The replacement content.",
                    },
                },
                "required": ["chunk_id", "new_content"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr if arr.size else None


def _format_recall_results(rows: list, label: str = "") -> str:
    if not rows:
        return "No matching memories found."
    lines: list[str] = []
    prefix = f"[{label}] " if label else ""
    for r in rows:
        kind = r["kind"] or "fact"
        if kind == "triple" and r["subj"]:
            lines.append(f"{prefix}[{r['id']}] triple: ({r['subj']}) —[{r['pred']}]→ ({r['obj']})")
        else:
            lines.append(f"{prefix}[{r['id']}] {kind}: {r['content']}")
    return "\n".join(lines)


def execute_tool(
    name: str,
    args: dict[str, Any],
    client: Groq,
    conn: Connection,
    session_id: str,
    tenant_id: str = "default",
) -> str:
    """Dispatch a tool call and return a string result to feed back to the model."""

    if name == "recall":
        query = str(args.get("query", "")).strip()
        if not query:
            return "Error: query is required."

        # Search session memory
        session_rows = store.list_chunks_for_session(conn, session_id, limit=config.MEMORY_MAX_ROWS)
        profile_rows = store.list_profile_chunks(conn, tenant_id)
        all_rows = list(session_rows) + list(profile_rows)

        if not all_rows:
            return "Memory is empty."

        if config.EMBEDDINGS_DISABLED:
            picked = retrieve_top_k(all_rows, query, config.TOP_K)
        else:
            try:
                picked = retrieve_hybrid(client, all_rows, query, config.TOP_K)
            except Exception:
                picked = retrieve_top_k(all_rows, query, config.TOP_K)

        # Label profile rows so model knows they're cross-session
        profile_ids = {r["id"] for r in profile_rows}
        lines: list[str] = []
        for r in picked:
            kind = r["kind"] or "fact"
            is_profile = r["id"] in profile_ids
            tag = "[profile]" if is_profile else "[session]"
            if kind == "triple" and r["subj"]:
                lines.append(f"{tag} [{r['id']}] triple: ({r['subj']}) —[{r['pred']}]→ ({r['obj']})")
            else:
                lines.append(f"{tag} [{r['id']}] {kind}: {r['content']}")
        return "\n".join(lines) if lines else "No matching memories found."

    if name == "remember":
        content = str(args.get("content", "")).strip()
        if not content:
            return "Error: content is required."
        blob: bytes | None = None
        if not config.EMBEDDINGS_DISABLED:
            vec, _ = emb.try_embed_query(client, content)
            if vec is not None:
                blob = emb.vec_to_blob(vec)
        chunk_id = store.add_memory_unit(conn, session_id, "fact", content, embedding=blob)
        return f"Stored in session memory [{chunk_id}]."

    if name == "remember_profile":
        content = str(args.get("content", "")).strip()
        if not content:
            return "Error: content is required."
        blob = None
        if not config.EMBEDDINGS_DISABLED:
            vec, _ = emb.try_embed_query(client, content)
            if vec is not None:
                blob = emb.vec_to_blob(vec)
        chunk_id = store.add_profile_unit(conn, tenant_id, "fact", content, embedding=blob)
        return f"Stored in user profile [{chunk_id}] — will persist across all sessions."

    if name == "forget":
        chunk_id = args.get("chunk_id")
        if chunk_id is None:
            return "Error: chunk_id is required."
        deleted = store.delete_chunk(conn, int(chunk_id), session_id)
        if deleted:
            return f"Session memory [{chunk_id}] deleted."
        return f"Memory [{chunk_id}] not found in this session."

    if name == "update_fact":
        chunk_id = args.get("chunk_id")
        new_content = str(args.get("new_content", "")).strip()
        if chunk_id is None or not new_content:
            return "Error: chunk_id and new_content are required."
        blob = None
        if not config.EMBEDDINGS_DISABLED:
            vec, _ = emb.try_embed_query(client, new_content)
            if vec is not None:
                blob = emb.vec_to_blob(vec)
        updated = store.update_chunk_content(conn, int(chunk_id), session_id, new_content, embedding=blob)
        if updated:
            return f"Session memory [{chunk_id}] updated."
        return f"Memory [{chunk_id}] not found in this session."

    return f"Unknown tool: {name}"


def parse_tool_call(tool_call: Any) -> tuple[str, dict[str, Any]]:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    return name, args
