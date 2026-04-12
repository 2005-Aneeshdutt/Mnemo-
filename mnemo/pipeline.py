from __future__ import annotations

from sqlite3 import Connection
from typing import Any

import numpy as np
from groq import Groq

from mnemo import config
from mnemo import embeddings as emb
from mnemo import store


def _triple_line(subj: str, pred: str, obj: str) -> str:
    return f"({subj}) —[{pred}]→ ({obj})"


def persist_augmentation(
    client: Groq,
    conn: Connection,
    session_id: str,
    extracted: dict[str, Any],
) -> dict[str, Any]:
    """
    Write summaries, triples, and facts; attach embeddings when enabled.
    """
    counts: dict[str, Any] = {"summaries": 0, "triples": 0, "facts": 0}
    # Each entry: (kind, content, subj, pred, obj)
    rows_meta: list[tuple[str, str, str | None, str | None, str | None]] = []

    summary = (extracted.get("summary") or "").strip()
    if summary:
        content = f"[summary] {summary}"
        rows_meta.append(("summary", content, None, None, None))
        counts["summaries"] = 1

    triples = extracted.get("triples") or []
    if isinstance(triples, list):
        for t in triples:
            if not isinstance(t, dict):
                continue
            s = str(t.get("subject", "")).strip()
            p = str(t.get("predicate", "")).strip()
            o = str(t.get("object", "")).strip()
            if not (s and p and o):
                continue
            line = _triple_line(s, p, o)
            rows_meta.append(("triple", line, s, p, o))
            counts["triples"] += 1

    facts = extracted.get("facts") or []
    if isinstance(facts, list):
        for f in facts:
            text = str(f).strip()
            if not text:
                continue
            rows_meta.append(("fact", text, None, None, None))
            counts["facts"] += 1

    if not rows_meta:
        return counts

    texts_to_embed = [m[1] for m in rows_meta]

    vec_list: list[np.ndarray] | None = None
    err: str | None = None
    if not config.EMBEDDINGS_DISABLED:
        vec_list, err = emb.try_embed_texts(client, texts_to_embed)

    for i, meta in enumerate(rows_meta):
        kind, content, subj, pred, obj = meta
        blob = None
        if vec_list is not None and i < len(vec_list):
            blob = emb.vec_to_blob(vec_list[i])
        store.add_memory_unit(
            conn,
            session_id,
            kind,
            content,
            subj=subj,
            pred=pred,
            obj=obj,
            embedding=blob,
        )

    if err:
        counts["_embed_error"] = err
    return counts

