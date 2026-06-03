"""
LoCoMo-style harness: JSON sessions, substring checks, memory vs last-N baseline, metrics.

Usage:
  python eval/run_locomo.py eval/data/sample_locomo.json
  python eval/run_locomo.py eval/data/sample_locomo.json --report eval/results/report.json
  python eval/run_locomo.py eval/data/sample_locomo.json --mode memory
  python eval/run_locomo.py eval/data/sample_locomo.json --mode baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mnemo import config
from mnemo.client import create_client
from mnemo.agent import ChatState, chat_turn, chat_turn_last_n_only, estimate_context_tokens
from mnemo import store
from mnemo.tenancy import scope_session


def _check_reply(reply: str, must: list[str], must_not: list[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    low = reply.lower()
    for s in must:
        if s.lower() not in low:
            reasons.append(f"missing: {s!r}")
    for s in must_not:
        if s and s.lower() in low:
            reasons.append(f"forbidden present: {s!r}")
    return (len(reasons) == 0, reasons)


def _run_session(
    client: Groq,
    conn: Any,
    sess: dict[str, Any],
    *,
    use_memory: bool,
    results: dict[str, Any],
    token_samples: list[dict[str, Any]],
    turn_sleep: float = 0.0,
) -> dict[str, Any]:
    sid = sess.get("id", "anon")
    tenant = (sess.get("tenant_id") or "eval").strip() or "eval"
    turns: list[str] = sess.get("turns") or []
    checks = sess.get("checks") or []

    suffix = "-mem" if use_memory else "-base"
    scoped = scope_session(tenant, f"locomo-{sid}{suffix}")
    if use_memory:
        store.clear_session(conn, scoped)
    state = ChatState(session_id=scoped)

    sess_out: dict[str, Any] = {"id": sid, "mode": "memory" if use_memory else "baseline_last_n", "checks": []}
    last_reply = ""

    for i, user_line in enumerate(turns, start=1):
        for attempt in range(4):
            try:
                if use_memory:
                    last_reply = chat_turn(client, conn, state, user_line)
                else:
                    last_reply = chat_turn_last_n_only(client, state, user_line)
                break
            except Exception as e:
                if ("429" in str(e) or "503" in str(e)) and attempt < 3:
                    wait = 30 * (2 ** attempt)
                    print(f"  Rate limited — waiting {wait}s before retry {attempt + 1}/3...", flush=True)
                    time.sleep(wait)
                else:
                    raise

        if turn_sleep > 0:
            time.sleep(turn_sleep)

        for chk in checks:
            if int(chk.get("after_user_turn", -1)) != i:
                continue
            must = chk.get("answer_must_contain") or []
            must_not = chk.get("answer_must_not_contain") or []
            ok, reasons = _check_reply(last_reply, must, must_not)
            entry = {
                "after_user_turn": i,
                "question": chk.get("question") or user_line,
                "ok": ok,
                "reasons": reasons,
                "reply_preview": last_reply[:500],
            }
            sess_out["checks"].append(entry)
            if ok:
                results["passed"] += 1
            else:
                results["failed"] += 1

            if use_memory:
                try:
                    est = estimate_context_tokens(client, conn, state, user_line)
                    token_samples.append(
                        {
                            "session": sid,
                            "after_turn": i,
                            **est,
                        }
                    )
                except Exception as e:
                    token_samples.append({"session": sid, "after_turn": i, "error": str(e)})

    return sess_out


def run_file(path: Path, out_report: Path | None, mode: str, turn_sleep: float = 0.0) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    sessions = data.get("sessions") or []
    if not config.GROQ_API_KEY and not config.GEMINI_API_KEY:
        raise SystemExit("Set GROQ_API_KEY or GEMINI_API_KEY in .env in the project root.")

    client = create_client()
    conn = store.connect(config.DB_PATH)
    store.init_schema(conn)

    out: dict[str, Any] = {
        "file": str(path),
        "mode": mode,
        "mnemo_config": {
            "top_k": config.TOP_K,
            "recent_messages": config.RECENT_MESSAGES,
            "embeddings_disabled": config.EMBEDDINGS_DISABLED,
            "chat_model": config.CHAT_MODEL,
        },
        "memory": None,
        "baseline_last_n": None,
        "comparison": None,
    }

    token_samples: list[dict[str, Any]] = []

    if mode in ("memory", "both"):
        res_mem = {"passed": 0, "failed": 0, "sessions": []}
        for sess in sessions:
            res_mem["sessions"].append(_run_session(client, conn, sess, use_memory=True, results=res_mem, token_samples=token_samples, turn_sleep=turn_sleep))
        out["memory"] = res_mem
        out["memory"]["context_token_samples"] = token_samples

    if mode in ("baseline", "both"):
        token_samples_b: list[dict[str, Any]] = []
        res_base = {"passed": 0, "failed": 0, "sessions": []}
        for sess in sessions:
            res_base["sessions"].append(
                _run_session(client, conn, sess, use_memory=False, results=res_base, token_samples=token_samples_b, turn_sleep=turn_sleep)
            )
        out["baseline_last_n"] = res_base

    if mode == "both" and out["memory"] and out["baseline_last_n"]:
        mp = out["memory"]["passed"]
        mf = out["memory"]["failed"]
        bp = out["baseline_last_n"]["passed"]
        bf = out["baseline_last_n"]["failed"]
        mt = mp + mf
        bt = bp + bf
        out["comparison"] = {
            "memory_pass_rate": round(mp / mt, 4) if mt else None,
            "baseline_pass_rate": round(bp / bt, 4) if bt else None,
            "memory_passed": mp,
            "memory_failed": mf,
            "baseline_passed": bp,
            "baseline_failed": bf,
        }
        samples = out["memory"].get("context_token_samples") or []
        if samples and not samples[0].get("error"):
            mem_vals = [s["memory_prompt"] for s in samples if "memory_prompt" in s]
            base_vals = [s["baseline_last_n_prompt"] for s in samples if "baseline_last_n_prompt" in s]
            full_vals = [s["full_history_prompt"] for s in samples if "full_history_prompt" in s]
            if mem_vals and base_vals and full_vals:
                out["comparison"]["approx_prompt_tokens_avg"] = {
                    "memory_retrieval": round(sum(mem_vals) / len(mem_vals)),
                    "baseline_last_n_only": round(sum(base_vals) / len(base_vals)),
                    "hypothetical_full_history": round(sum(full_vals) / len(full_vals)),
                }

    conn.close()

    if out_report:
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {out_report}")

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="LoCoMo-style JSON eval (memory vs baseline)")
    parser.add_argument("json_path", type=Path, help="Path to JSON (see eval/data/sample_locomo.json)")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write JSON report (default: eval/results/report.json when omitted and --mode both)",
    )
    parser.add_argument(
        "--mode",
        choices=("memory", "baseline", "both"),
        default="both",
        help="Run Mnemo memory, last-N baseline, or both for comparison",
    )
    parser.add_argument(
        "--turn-sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between turns (use ~8 on Groq free tier to avoid TPM limits)",
    )
    args = parser.parse_args()

    if not args.json_path.is_file():
        print("File not found:", args.json_path, file=sys.stderr)
        return 1

    report_path = args.report
    if report_path is None and args.mode == "both":
        report_path = ROOT / "eval" / "results" / "report.json"

    r = run_file(args.json_path, report_path, args.mode, turn_sleep=args.turn_sleep)

    if args.mode == "both" and r.get("comparison"):
        print(json.dumps(r["comparison"], indent=2))
    elif r.get("memory"):
        print(json.dumps({"passed": r["memory"]["passed"], "failed": r["memory"]["failed"]}, indent=2))
    elif r.get("baseline_last_n"):
        print(json.dumps({"passed": r["baseline_last_n"]["passed"], "failed": r["baseline_last_n"]["failed"]}, indent=2))

    any_fail = 0
    if r.get("memory"):
        any_fail += r["memory"]["failed"]
    if r.get("baseline_last_n"):
        any_fail += r["baseline_last_n"]["failed"]
    return 0 if any_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
