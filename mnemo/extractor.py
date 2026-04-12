from __future__ import annotations

import json
from typing import Any

from groq import Groq

from mnemo import config


EXTRACTION_SYSTEM = """You are the Advanced Augmentation stage of a persistent memory system (long-context agent memory).
From the latest user message and assistant reply, produce STRUCTURED memory for later retrieval.

Output JSON only (no markdown). Schema:
{
  "facts": string[],
  "triples": { "subject": string, "predicate": string, "object": string }[],
  "summary": string
}

Rules:
- "triples": semantic triples for stable relations (who/what/where/when/preferences). Use concise English predicates
  (e.g. "lives_in", "works_at", "prefers", "allergic_to", "birthday_is"). Empty array if none.
- "facts": additional standalone sentences not easily expressed as triples; max 12 items; self-contained
  (resolve pronouns to names or "the user").
- "summary": one short sentence describing the turn; may be "" if nothing important happened.
- Omit small talk unless it encodes a durable user-specific preference or fact."""


def extract_memory(client: Groq, user_text: str, assistant_text: str, model: str | None = None) -> dict[str, Any]:
    model = model or config.EXTRACT_MODEL
    payload = json.dumps({"user": user_text, "assistant": assistant_text}, ensure_ascii=False)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": payload},
        ],
        temperature=0.15,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)

    facts = data.get("facts") or []
    if not isinstance(facts, list):
        facts = []
    facts = [str(f).strip() for f in facts if str(f).strip()]

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

    summary = (data.get("summary") or "").strip()
    return {"facts": facts, "triples": triples, "summary": summary}

