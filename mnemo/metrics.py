"""Runtime observability: thread-safe counters, latency histograms, and exporters."""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Token estimation helpers (used by eval harness and agent)
# ---------------------------------------------------------------------------

def approx_tokens_from_text(*parts: str) -> int:
    """~4 chars per token heuristic for Latin text."""
    total = sum(len(p) for p in parts if p)
    return max(0, total // 4)


def approx_tokens_chat_messages(messages: list[dict[str, str]]) -> int:
    return approx_tokens_from_text(*(m.get("content") or "" for m in messages))


# ---------------------------------------------------------------------------
# Thread-safe metrics registry
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


class _Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    # -- writes --

    def inc(self, name: str, by: float = 1.0, **labels: str) -> None:
        key = _label_key(name, labels)
        with self._lock:
            self._counters[key] += by

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = _label_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)

    # -- reads --

    def counter_value(self, name: str, **labels: str) -> float:
        key = _label_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            histograms = {k: _summarise(list(v)) for k, v in self._histograms.items()}
        return {"counters": counters, "histograms": histograms}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


def _label_key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    pairs = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{pairs}}}"


def _summarise(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "sum": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(values)
    return {
        "count": len(s),
        "sum": round(sum(s), 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
        "p50": round(_percentile(s, 0.50), 3),
        "p95": round(_percentile(s, 0.95), 3),
        "p99": round(_percentile(s, 0.99), 3),
    }


# Module-level singleton — shared across all server workers in-process.
REGISTRY = _Registry()


# ---------------------------------------------------------------------------
# Named metric helpers — call these throughout the codebase
# ---------------------------------------------------------------------------

def record_request(endpoint: str, status: int, latency_ms: float) -> None:
    """Track HTTP request count and latency."""
    REGISTRY.inc("mnemo_requests_total", endpoint=endpoint, status=str(status))
    REGISTRY.observe("mnemo_request_duration_ms", latency_ms, endpoint=endpoint)


def record_tokens_saved(saved: int) -> None:
    """Tokens saved by memory retrieval vs sending full conversation history."""
    if saved > 0:
        REGISTRY.inc("mnemo_tokens_saved_total", by=float(saved))


def record_memory_retrieval(hits: int, misses: int = 0) -> None:
    """Chunks returned (hits) vs queries with zero results (misses)."""
    if hits > 0:
        REGISTRY.inc("mnemo_memory_hits_total", by=float(hits))
    if misses > 0:
        REGISTRY.inc("mnemo_memory_misses_total", by=float(misses))


def record_memory_write(kind: str, count: int = 1) -> None:
    """Memory units persisted, labeled by kind (fact / triple / summary)."""
    if count > 0:
        REGISTRY.inc("mnemo_memories_written_total", by=float(count), kind=kind)


def record_compaction(original_rows: int = 0, new_rows: int = 0) -> None:
    """Background compaction completed."""
    REGISTRY.inc("mnemo_compactions_total")
    if original_rows > 0:
        saved = max(0, original_rows - new_rows)
        REGISTRY.inc("mnemo_compaction_rows_removed_total", by=float(saved))


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def metrics_json() -> dict[str, Any]:
    """Return all metrics as a JSON-serialisable dict."""
    snap = REGISTRY.snapshot()
    return {
        "counters": snap["counters"],
        "histograms": snap["histograms"],
    }


def metrics_prometheus() -> str:
    """Return metrics in Prometheus text exposition format."""
    snap = REGISTRY.snapshot()
    lines: list[str] = []

    _COUNTER_HELP = {
        "mnemo_requests_total": "Total HTTP requests handled",
        "mnemo_tokens_saved_total": "Approximate tokens saved by memory retrieval vs full history",
        "mnemo_memory_hits_total": "Memory chunks returned by retrieval queries",
        "mnemo_memory_misses_total": "Retrieval queries that returned zero results",
        "mnemo_memories_written_total": "Memory units written to the store",
        "mnemo_compactions_total": "Background compaction runs completed",
        "mnemo_compaction_rows_removed_total": "Memory rows removed by compaction",
    }

    _HISTOGRAM_HELP = {
        "mnemo_request_duration_ms": "HTTP request latency in milliseconds",
    }

    emitted_counter_names: set[str] = set()
    for key, val in sorted(snap["counters"].items()):
        name = key.split("{")[0]
        if name not in emitted_counter_names:
            help_text = _COUNTER_HELP.get(name, name)
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            emitted_counter_names.add(name)
        lines.append(f"{key} {int(val)}")

    emitted_hist_names: set[str] = set()
    for key, summary in sorted(snap["histograms"].items()):
        name = key.split("{")[0]
        label_part = key[len(name):]  # e.g. '{endpoint="/v1/chat"}'
        base = label_part.rstrip("}")  # '{endpoint="/v1/chat"'

        if name not in emitted_hist_names:
            help_text = _HISTOGRAM_HELP.get(name, name)
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} summary")
            emitted_hist_names.add(name)

        for quantile, pval in [("0.5", summary["p50"]), ("0.95", summary["p95"]), ("0.99", summary["p99"])]:
            if base:
                q_key = f"{name}{base},quantile=\"{quantile}\"}}"
            else:
                q_key = f'{name}{{quantile="{quantile}"}}'
            lines.append(f"{q_key} {pval}")

        if base:
            lines.append(f"{name}_count{label_part} {summary['count']}")
            lines.append(f"{name}_sum{label_part} {summary['sum']}")
        else:
            lines.append(f"{name}_count {summary['count']}")
            lines.append(f"{name}_sum {summary['sum']}")

    return "\n".join(lines) + ("\n" if lines else "")
