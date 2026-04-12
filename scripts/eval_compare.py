"""
Offline-style comparison: measures how many memory rows exist after scripted turns
and optionally runs keyword retrieval overlap (no LLM judge).

Usage (requires GROQ_API_KEY):
  python scripts/eval_compare.py
"""
from __future__ import annotations

import os
import sys

# Project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from groq import Groq

from mnemo import config
from mnemo.agent import ChatState, chat_turn
from mnemo import store


def main() -> int:
    if not config.GROQ_API_KEY:
        print("Set GROQ_API_KEY", file=sys.stderr)
        return 1
    os.environ.setdefault("GROQ_API_KEY", config.GROQ_API_KEY)

    session = "eval_session"
    db_path = config.DB_PATH
    client = Groq()
    conn = store.connect(db_path)
    store.init_schema(conn)
    store.clear_session(conn, session)

    state = ChatState(session_id=session)
    script = [
        "My name is Alex and I work as a data engineer in Seattle.",
        "I prefer concise answers and I am allergic to peanuts.",
        "What city do I work in and what should you avoid mentioning in food suggestions?",
    ]

    for turn in script:
        reply = chat_turn(client, conn, state, turn)
        print("Q:", turn[:80], "..." if len(turn) > 80 else "")
        print("A:", reply[:200], "...\n" if len(reply) > 200 else "\n")

    rows = store.list_chunks_for_session(conn, session)
    triples = sum(1 for r in rows if r["kind"] == "triple")
    facts = sum(1 for r in rows if r["kind"] == "fact")
    sums = sum(1 for r in rows if r["kind"] == "summary")
    emb = sum(1 for r in rows if r["embedding"] is not None)

    print("---")
    print(f"Rows: {len(rows)} | triples={triples} facts={facts} summaries={sums} with_embedding={emb}")
    print(f"DB: {db_path}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
