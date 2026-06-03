from __future__ import annotations

import argparse
import os
import sys

from mnemo import config
from mnemo.client import create_client, active_provider
from mnemo.agent import ChatState, chat_turn_stream
from mnemo import store
from mnemo.tenancy import scope_session


def _banner() -> None:
    print(
        "Mnemo — persistent memory (triples + facts + summaries + hybrid retrieval)\n"
        "Commands: /quit, /memory, /triples, /clear, /help\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mnemo CLI")
    parser.add_argument("--session", default="default", help="Session id (isolates memory)")
    parser.add_argument("--tenant", default="default", help="Tenant id (multi-tenant isolation; matches API X-Tenant-ID)")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Lexical-only retrieval (no Groq embeddings API)",
    )
    args = parser.parse_args(argv)

    if not config.GROQ_API_KEY and not config.GEMINI_API_KEY:
        print("Set GROQ_API_KEY or GEMINI_API_KEY in environment or .env file.", file=sys.stderr)
        return 1

    if args.no_embeddings:
        config.EMBEDDINGS_DISABLED = True

    client = create_client()
    conn = store.connect(config.DB_PATH)
    store.init_schema(conn)

    try:
        scoped = scope_session(args.tenant, args.session)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    state = ChatState(session_id=scoped, tenant_id=args.tenant)
    _banner()
    mode = "lexical-only" if config.EMBEDDINGS_DISABLED else "hybrid + ANN (if enough vectors)"
    print(
        f"Tenant: {args.tenant} | Session: {args.session}\n"
        f"Scoped key: {scoped} | DB: {config.DB_PATH}\n"
        f"Provider: {active_provider()} | Model: {config.CHAT_MODEL} | Retrieval: {mode}\n"
    )

    while True:
        try:
            line = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line == "/quit":
            break
        if line == "/help":
            _banner()
            continue
        if line == "/memory":
            rows = store.list_chunks_for_session(conn, state.session_id)
            if not rows:
                print("(no memories stored)\n")
                continue
            for r in rows:
                emb = "yes" if r["embedding"] else "no"
                extra = ""
                if r["kind"] == "triple" and r["subj"]:
                    extra = f" | ({r['subj']})[{r['pred']}]({r['obj']})"
                print(f"  [{r['id']}] {r['kind']:<8} emb={emb}{extra}\n      {r['content']}")
            print()
            continue
        if line == "/triples":
            rows = store.list_chunks_for_session(conn, state.session_id)
            ts = [r for r in rows if r["kind"] == "triple"]
            if not ts:
                print("(no triples yet)\n")
                continue
            for r in ts:
                print(f"  ({r['subj']}) —[{r['pred']}]→ ({r['obj']})")
            print()
            continue
        if line == "/clear":
            n = store.clear_session(conn, state.session_id)
            state.messages.clear()
            print(f"Cleared {n} memory row(s).\n")
            continue

        def _on_tool(name: str, args: dict, result: str) -> None:
            label = {
                "recall": "searching memory",
                "remember": "storing memory",
                "forget": "deleting memory",
                "update_fact": "updating memory",
            }.get(name, name)
            detail = args.get("query") or args.get("content") or args.get("chunk_id") or ""
            print(f"  [memory] {label}: {detail!r}")
            preview = result.split("\n")[0][:120]
            print(f"           → {preview}")

        try:
            print("AI> ", end="", flush=True)
            for chunk in chat_turn_stream(client, conn, state, line, on_tool_call=_on_tool):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            msg = str(e)
            print(f"\nError: {msg}\n", file=sys.stderr)
            if "embed" in msg.lower() or "embedding" in msg.lower():
                print(
                    "Tip: retry with --no-embeddings or set MNEMO_NO_EMBEDDINGS=1 in .env.\n",
                    file=sys.stderr,
                )
            if "401" in msg or ("invalid" in msg.lower() and "key" in msg.lower()):
                print("Tip: check GROQ_API_KEY in .env.\n", file=sys.stderr)
            continue

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

