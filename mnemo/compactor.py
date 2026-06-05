from __future__ import annotations

import json
import logging
import threading
import time
from sqlite3 import Connection
from typing import Any

from groq import Groq

from mnemo import config
from mnemo import embeddings as emb
from mnemo import metrics as metrics_mod
from mnemo import store

logger = logging.getLogger(__name__)

COMPACTION_SYSTEM = """You are a memory compaction agent for a long-term memory system.

You will receive all stored memories for a conversation session: facts, semantic triples, and summaries.
Your job is to produce a clean, compact, canonical version by:
- Merging near-duplicate facts into one
- Resolving contradictions (keep the more recent or more specific version)
- Dropping trivial or redundant entries
- Preserving all durable facts about the user (name, preferences, relationships, locations)

Output JSON only. Schema:
{
  "facts": string[],
  "triples": { "subject": string, "predicate": string, "object": string }[]
}

Rules:
- Be aggressive about merging. 10 clean memories beat 40 noisy ones.
- Resolve contradictions: if two facts conflict, keep the one that is more specific or more recent.
- facts: standalone sentences, self-contained (resolve all pronouns).
- triples: stable relations only. Use concise predicates (lives_in, prefers, works_at, etc).
- Do NOT add new information that wasn't in the input."""


def _format_memories_for_compaction(rows: list) -> str:
    lines: list[str] = []
    for r in rows:
        kind = r["kind"] or "fact"
        ts = float(r["created_at"]) if r["created_at"] is not None else 0.0
        if kind == "triple" and r["subj"]:
            lines.append(f"[triple] ({r['subj']}) —[{r['pred']}]→ ({r['obj']})  (t={ts:.0f})")
        elif kind == "summary":
            c = (r["content"] or "").replace("[summary] ", "", 1).strip()
            lines.append(f"[summary] {c}  (t={ts:.0f})")
        else:
            lines.append(f"[fact] {r['content']}  (t={ts:.0f})")
    return "\n".join(lines)


def _run_compaction(
    client: Groq,
    conn: Connection,
    session_id: str,
) -> dict[str, Any]:
    rows = store.list_chunks_for_session(conn, session_id, limit=config.MEMORY_MAX_ROWS)
    if len(rows) < config.COMPACT_MIN_ROWS:
        return {"skipped": True, "reason": "below min rows", "rows": len(rows)}

    memory_text = _format_memories_for_compaction(rows)
    try:
        resp = client.chat.completions.create(
            model=config.EXTRACT_MODEL,
            messages=[
                {"role": "system", "content": COMPACTION_SYSTEM},
                {"role": "user", "content": f"CURRENT MEMORIES:\n{memory_text}"},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning("Compaction LLM call failed: %s", e)
        return {"skipped": True, "reason": str(e)}

    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Compaction returned invalid JSON")
        return {"skipped": True, "reason": "invalid json"}

    facts = [str(f).strip() for f in (data.get("facts") or []) if str(f).strip()]
    triples_raw = data.get("triples") or []
    triples: list[dict[str, str]] = []
    if isinstance(triples_raw, list):
        for t in triples_raw:
            if not isinstance(t, dict):
                continue
            s = str(t.get("subject", "")).strip()
            p = str(t.get("predicate", "")).strip()
            o = str(t.get("object", "")).strip()
            if s and p and o:
                triples.append({"subject": s, "predicate": p, "object": o})

    if not facts and not triples:
        logger.warning("Compaction produced empty output — keeping existing memories")
        return {"skipped": True, "reason": "empty output"}

    # Embed compacted content in one batch
    texts: list[str] = facts + [f"({t['subject']}) —[{t['predicate']}]→ ({t['object']})" for t in triples]
    vec_list = None
    if not config.EMBEDDINGS_DISABLED:
        vec_list, _ = emb.try_embed_texts(client, texts)

    # Atomically replace old memories with compacted set
    store.clear_session(conn, session_id)
    now = time.time()

    for i, fact in enumerate(facts):
        blob = emb.vec_to_blob(vec_list[i]) if vec_list and i < len(vec_list) else None
        store.add_memory_unit(conn, session_id, "fact", fact, embedding=blob)

    for j, t in enumerate(triples):
        idx = len(facts) + j
        blob = emb.vec_to_blob(vec_list[idx]) if vec_list and idx < len(vec_list) else None
        content = f"({t['subject']}) —[{t['predicate']}]→ ({t['object']})"
        store.add_memory_unit(
            conn, session_id, "triple", content,
            subj=t["subject"], pred=t["predicate"], obj=t["object"],
            embedding=blob,
        )

    new_rows = len(facts) + len(triples)
    metrics_mod.record_compaction(original_rows=len(rows), new_rows=new_rows)
    result = {
        "compacted": True,
        "original_rows": len(rows),
        "new_facts": len(facts),
        "new_triples": len(triples),
    }
    logger.info("Compaction complete for %s: %d → %d rows", session_id, len(rows), new_rows)
    return result


def compact_session_async(client: Groq, conn: Connection, session_id: str) -> None:
    """Fire-and-forget compaction in a background thread."""
    def _run() -> None:
        try:
            _run_compaction(client, conn, session_id)
        except Exception as e:
            logger.warning("Background compaction failed for %s: %s", session_id, e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def should_compact(turn_count: int) -> bool:
    every = config.COMPACT_EVERY_N
    return every > 0 and turn_count > 0 and turn_count % every == 0
