# Observability

Mnemo ships with a built-in metrics system — no Prometheus client library or external agent required. Every running instance exposes a standard scrape endpoint and a JSON alternative out of the box.

## Endpoints

| Endpoint | Auth | Format | Use |
|----------|------|--------|-----|
| `GET /metrics` | None | Prometheus text | Prometheus, Grafana Agent, VictoriaMetrics |
| `GET /v1/metrics` | API key | JSON | Dashboards, alerting webhooks, debugging |

---

## Metrics reference

### Counters

#### `mnemo_requests_total`
Labels: `endpoint`, `status`

Total HTTP requests handled, broken down by path and HTTP status code.

```
mnemo_requests_total{endpoint="/v1/chat",status="200"} 1284
mnemo_requests_total{endpoint="/v1/chat",status="429"} 12
mnemo_requests_total{endpoint="/health",status="200"} 3600
```

**What to watch:** a sustained rise in `status="500"` on `/v1/chat` usually means the LLM provider is returning errors — check API key validity and quota.

---

#### `mnemo_tokens_saved_total`

Cumulative approximate tokens saved by memory retrieval versus injecting the full conversation history. Computed each turn as:

```
saved = max(0, (len(full_history_chars) - len(sent_history_chars)) / 4)
```

Savings are zero (or slightly negative) for short sessions where full history fits in the context window. They compound strongly from turn ~14 onward. See the [benchmark](../README.md#benchmark) for per-turn numbers.

---

#### `mnemo_memory_hits_total`

Number of memory chunks returned by retrieval queries across all turns. A consistently low value (near zero) with many turns suggests embeddings are disabled or the store is empty.

---

#### `mnemo_memory_misses_total`

Number of retrieval queries that returned zero results. A high ratio of misses to hits early in a session is normal (the store is building up). A high ratio in established sessions may indicate retrieval weights need tuning.

---

#### `mnemo_memories_written_total`
Labels: `kind` (`fact`, `triple`, `summary`)

Memory units persisted after each turn, broken down by type. Useful for understanding what the extraction model is producing.

```
mnemo_memories_written_total{kind="fact"} 312
mnemo_memories_written_total{kind="triple"} 198
mnemo_memories_written_total{kind="summary"} 84
```

A healthy session typically produces more triples and facts than summaries.

---

#### `mnemo_compactions_total`

Number of background compaction runs completed. Compaction fires every `MNEMO_COMPACT_EVERY_N` turns when the session has at least `MNEMO_COMPACT_MIN_ROWS` rows.

---

#### `mnemo_compaction_rows_removed_total`

Cumulative rows deleted by compaction (original row count minus compacted row count). A consistently low or zero value means sessions are short enough that compaction rarely fires, or the store is already clean.

---

### Summaries (histograms)

#### `mnemo_request_duration_ms`
Labels: `endpoint`

Wall-clock request latency in milliseconds, with quantiles computed over all observations since server start.

```
mnemo_request_duration_ms{endpoint="/v1/chat",quantile="0.5"}  38.4
mnemo_request_duration_ms{endpoint="/v1/chat",quantile="0.95"} 112.7
mnemo_request_duration_ms{endpoint="/v1/chat",quantile="0.99"} 198.3
mnemo_request_duration_ms_count{endpoint="/v1/chat"} 1284
mnemo_request_duration_ms_sum{endpoint="/v1/chat"} 68210.4
```

`/v1/chat` latency is dominated by the LLM provider round-trip. `/health` and `/metrics` should stay under 5 ms.

---

## Prometheus setup

Add a scrape job to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: mnemo
    static_configs:
      - targets: ["localhost:8765"]
    metrics_path: /metrics
    scrape_interval: 15s
```

No authentication is required for `/metrics` — it is intentionally unauthenticated so Prometheus can scrape without credential management.

---

## Grafana dashboard

Once Prometheus is scraping Mnemo, useful panels to create:

**Requests/sec**
```promql
rate(mnemo_requests_total{endpoint="/v1/chat"}[1m])
```

**p95 chat latency**
```promql
mnemo_request_duration_ms{endpoint="/v1/chat", quantile="0.95"}
```

**Error rate**
```promql
rate(mnemo_requests_total{status=~"5.."}[5m])
/ rate(mnemo_requests_total[5m])
```

**Tokens saved (cumulative)**
```promql
mnemo_tokens_saved_total
```

**Memory write rate by kind**
```promql
rate(mnemo_memories_written_total[5m])
```

**Compaction activity**
```promql
increase(mnemo_compactions_total[1h])
```

---

## JSON metrics endpoint

`GET /v1/metrics` returns a JSON object that is easier to consume programmatically than the Prometheus text format. It is protected by the API key.

```bash
curl -s http://127.0.0.1:8765/v1/metrics \
  -H "Authorization: Bearer my-secret-key" | python3 -m json.tool
```

Example response:

```json
{
  "counters": {
    "mnemo_requests_total{endpoint=\"/v1/chat\",status=\"200\"}": 42,
    "mnemo_tokens_saved_total": 8320,
    "mnemo_memory_hits_total": 176,
    "mnemo_memories_written_total{kind=\"fact\"}": 89,
    "mnemo_memories_written_total{kind=\"triple\"}": 54,
    "mnemo_memories_written_total{kind=\"summary\"}": 23,
    "mnemo_compactions_total": 2,
    "mnemo_compaction_rows_removed_total": 56
  },
  "histograms": {
    "mnemo_request_duration_ms{endpoint=\"/v1/chat\"}": {
      "count": 42,
      "sum": 2104.8,
      "min": 18.4,
      "max": 310.7,
      "p50": 38.2,
      "p95": 114.6,
      "p99": 201.3
    }
  }
}
```

---

## Implementation notes

- **No external dependencies.** The registry (`mnemo/metrics.py`) uses only Python stdlib — `threading`, `collections.defaultdict`.
- **Thread-safe.** All counter increments and histogram observations acquire a single `threading.Lock`. Multiple uvicorn workers in the same process share the same registry instance.
- **In-memory only.** Metrics reset on server restart. For persistence across restarts, scrape with Prometheus and store in its TSDB.
- **Quantiles are computed on read**, not stored as fixed buckets, so they reflect the true distribution of all observations since startup.
