# API Reference

Base URL: `http://127.0.0.1:8765` (default; configure with `MNEMO_API_HOST` / `MNEMO_API_PORT`)

Interactive docs (Swagger UI): `http://127.0.0.1:8765/docs`

## Authentication

Protected endpoints require a bearer token when `MNEMO_API_KEY` is set:

```
Authorization: Bearer <your-api-key>
```

If `MNEMO_API_KEY` is empty (default), all endpoints are unauthenticated. Set it in `.env` before exposing the server externally.

---

## Endpoints

### `GET /health`

Liveness check. No authentication required.

**Response `200`**
```json
{ "status": "ok" }
```

---

### `GET /metrics`

Prometheus text-format scrape endpoint. No authentication required.

**Response `200`** — `text/plain`
```
# HELP mnemo_requests_total Total HTTP requests handled
# TYPE mnemo_requests_total counter
mnemo_requests_total{endpoint="/v1/chat",status="200"} 42

# HELP mnemo_request_duration_ms HTTP request latency in milliseconds
# TYPE mnemo_request_duration_ms summary
mnemo_request_duration_ms{endpoint="/v1/chat",quantile="0.5"} 38.2
mnemo_request_duration_ms{endpoint="/v1/chat",quantile="0.95"} 114.6
mnemo_request_duration_ms{endpoint="/v1/chat",quantile="0.99"} 201.3
mnemo_request_duration_ms_count{endpoint="/v1/chat"} 42
mnemo_request_duration_ms_sum{endpoint="/v1/chat"} 2104.8

# HELP mnemo_tokens_saved_total Approximate tokens saved vs full history
# TYPE mnemo_tokens_saved_total counter
mnemo_tokens_saved_total 8320
```

---

### `GET /v1/metrics`

JSON metrics snapshot. **Requires auth.**

**Response `200`**
```json
{
  "counters": {
    "mnemo_requests_total{endpoint=\"/v1/chat\",status=\"200\"}": 42,
    "mnemo_tokens_saved_total": 8320,
    "mnemo_memory_hits_total": 176,
    "mnemo_memories_written_total{kind=\"fact\"}": 89,
    "mnemo_memories_written_total{kind=\"triple\"}": 54,
    "mnemo_compactions_total": 2
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

### `POST /v1/chat`

Send a chat message. Runs the full agentic tool-use loop and returns the complete reply. **Requires auth.**

**Rate limit:** `MNEMO_RATE_LIMIT_CHAT` (default `60/minute`) per tenant+IP.

**Request body**
```json
{
  "tenant_id": "acme",
  "session_id": "support-42",
  "message": "What was the project codename I mentioned earlier?"
}
```

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `tenant_id` | string | `"default"` | 1–64 chars |
| `session_id` | string | `"default"` | 1–256 chars |
| `message` | string | — | 1–32 000 chars, required |

**Headers (optional)**
```
X-Tenant-ID: acme
```
If present, `X-Tenant-ID` overrides `tenant_id` in the body.

**Response `200`**
```json
{
  "reply": "The project codename you mentioned was Nightingale."
}
```

**Response `429`** — rate limit exceeded

**Response `500`** — LLM call failed (check API key and model availability)

**Example**
```bash
curl -s http://127.0.0.1:8765/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-key" \
  -d '{"tenant_id":"acme","session_id":"s1","message":"My name is Aneesh."}'
```

---

### `POST /v1/chat/stream`

Streaming chat via Server-Sent Events (SSE). Tool calls run synchronously first; the final reply streams token by token. **Requires auth.**

**Request body** — identical to `POST /v1/chat`

**Response `200`** — `text/event-stream`
```
data: The

data:  project

data:  codename

data:  was

data:  Nightingale.

data: [DONE]
```

Each `data:` line is one text chunk. `[DONE]` signals the end of the stream. On error, a `data: [ERROR] <message>` line is emitted before closing.

**Example — stream with curl**
```bash
curl -s http://127.0.0.1:8765/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"acme","session_id":"s1","message":"What is my name?"}'
```

**Example — consume in Python**
```python
import httpx

with httpx.stream("POST", "http://127.0.0.1:8765/v1/chat/stream",
                  json={"message": "What is my name?"},
                  headers={"Authorization": "Bearer my-key"}) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            print(line[6:], end="", flush=True)
```

---

### `GET /v1/sessions/{session_id}/memory`

List all memory chunks stored for a session. **Requires auth.**

**Path parameters**

| Parameter | Description |
|-----------|-------------|
| `session_id` | Session identifier |

**Headers (optional)**
```
X-Tenant-ID: acme
```

**Response `200`**
```json
{
  "tenant_id": "acme",
  "session_id": "s1",
  "scoped_session": "acme:s1",
  "count": 3,
  "items": [
    {
      "id": "a1b2c3d4-...",
      "kind": "fact",
      "content": "The user's name is Aneesh.",
      "subject": null,
      "predicate": null,
      "object": null,
      "has_embedding": true,
      "created_at": 1748123456.789
    },
    {
      "id": "e5f6g7h8-...",
      "kind": "triple",
      "content": "(user) —[name]→ (Aneesh)",
      "subject": "user",
      "predicate": "name",
      "object": "Aneesh",
      "has_embedding": true,
      "created_at": 1748123456.790
    }
  ]
}
```

**Example**
```bash
curl -s http://127.0.0.1:8765/v1/sessions/s1/memory \
  -H "X-Tenant-ID: acme" \
  -H "Authorization: Bearer my-secret-key"
```

---

### `DELETE /v1/sessions/{session_id}/memory`

Clear all memory for a session. Also clears the in-memory `ChatState` message history for that session. **Requires auth.**

**Response `200`**
```json
{
  "tenant_id": "acme",
  "session_id": "s1",
  "deleted": 14
}
```

`deleted` is the number of rows removed from the database.

---

### `GET /v1/users/{tenant_id}/profile`

List all cross-session profile facts for a tenant. **Requires auth.**

**Response `200`**
```json
{
  "tenant_id": "acme",
  "count": 2,
  "items": [
    {
      "id": "...",
      "kind": "fact",
      "content": "The user prefers dark mode UIs.",
      "subject": null,
      "predicate": null,
      "object": null,
      "created_at": 1748100000.0
    }
  ]
}
```

Profile items appear in the system prompt of **every** session for this tenant.

---

### `DELETE /v1/users/{tenant_id}/profile`

Clear all profile facts for a tenant. **Requires auth.**

**Response `200`**
```json
{
  "tenant_id": "acme",
  "deleted": 5
}
```

---

## Error responses

All error responses follow FastAPI's default format:

```json
{ "detail": "Human-readable error message" }
```

| Status | Meaning |
|--------|---------|
| `400` | Invalid tenant/session ID (contains illegal characters or exceeds length limit) |
| `401` | Missing or invalid `Authorization` header |
| `422` | Request body validation failed (missing required field, value out of range) |
| `429` | Rate limit exceeded |
| `500` | LLM call failed — check API key, model name, and provider quota |

---

## Rate limiting

Rate limits are applied per `tenant_id:client_IP` pair, using a sliding-window algorithm provided by `slowapi`.

| Endpoint group | Default limit | Env var |
|----------------|--------------|---------|
| `/v1/chat`, `/v1/chat/stream` | `60/minute` | `MNEMO_RATE_LIMIT_CHAT` |
| All other endpoints | `120/minute` | `MNEMO_RATE_LIMIT` |

When a limit is exceeded, the server returns `429 Too Many Requests` with a `Retry-After` header.
