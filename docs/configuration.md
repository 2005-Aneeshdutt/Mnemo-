# Configuration Reference

All configuration is via environment variables. Copy `.env.example` to `.env` and edit it — the file is loaded automatically on startup.

## LLM Provider

Mnemo supports Groq and Google Gemini. Set exactly one API key; if both are set, Gemini takes priority for chat completions.

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key. Required unless `GEMINI_API_KEY` is set. |
| `GEMINI_API_KEY` | — | Google Gemini API key. Takes priority over Groq when set. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Chat model. For Gemini use e.g. `gemini-2.0-flash`. |
| `GROQ_EXTRACT_MODEL` | `llama-3.1-8b-instant` | Model used for per-turn memory extraction. Always uses Groq. |
| `GROQ_EMBED_MODEL` | `nomic-embed-text-v1_5` | Embedding model. Set `MNEMO_NO_EMBEDDINGS=1` to skip. |
| `GROQ_EMBED_MODEL_FALLBACKS` | `nomic-embed-text-v1.5` | Comma-separated fallback embedding models tried in order. |

**Recommended chat models:**

| Provider | Model | Notes |
|----------|-------|-------|
| Groq | `llama-3.3-70b-versatile` | Default; strong tool use |
| Groq | `llama-3.1-70b-versatile` | Good fallback |
| Gemini | `gemini-2.0-flash` | Fast, low cost |
| Gemini | `gemini-1.5-pro` | Stronger reasoning |

## Memory and Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMO_TOP_K` | `12` | Number of memory chunks retrieved per turn and injected into context. Higher values improve recall but increase prompt size. |
| `MNEMO_RECENT_MSG` | `12` | Recent conversation turns kept in the short-term context window alongside retrieved memories. |
| `MNEMO_MEMORY_MAX_ROWS` | `500` | Maximum rows loaded from SQLite before retrieval scoring. Capped to avoid slow full-table scoring at large scales. |
| `MNEMO_NO_EMBEDDINGS` | `0` | Set `1` to use lexical-only retrieval (no API calls for embeddings). Useful for free-tier keys or offline use. |
| `MNEMO_TOOL_USE` | `1` | Set `0` to disable agentic tool-calling. Falls back to passive memory injection (retrieved chunks prepended to system prompt). |

### Hybrid retrieval weights

The three scoring components are combined as a weighted sum. Weights must not sum to zero; they don't need to sum to 1.

| Variable | Default | Component |
|----------|---------|-----------|
| `MNEMO_W_DENSE` | `0.5` | Cosine similarity between query embedding and chunk embedding |
| `MNEMO_W_LEX` | `0.35` | Lexical overlap (token intersection over union) |
| `MNEMO_W_REC` | `0.15` | Recency decay — more recent chunks score higher |

**Tuning guidance:**
- For factual Q&A with long sessions: increase `MNEMO_W_DENSE`, decrease `MNEMO_W_REC`
- For real-time assistants where recent context matters most: increase `MNEMO_W_REC`
- When embeddings are disabled (`MNEMO_NO_EMBEDDINGS=1`): only `MNEMO_W_LEX` and `MNEMO_W_REC` have effect

### ANN prefilter (FAISS)

When embeddings are enabled and the store has ≥ `MNEMO_ANN_MIN_ROWS` rows, FAISS is used to pre-select candidates before full hybrid scoring.

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMO_ANN_ENABLED` | `1` | Set `0` to disable FAISS and always do full linear scan |
| `MNEMO_ANN_MIN_ROWS` | `32` | Minimum rows in store before ANN prefilter activates |
| `MNEMO_ANN_CANDIDATE_MULT` | `8` | Candidates = `TOP_K × ANN_CANDIDATE_MULT` |
| `MNEMO_ANN_MIN_CANDIDATES` | `64` | Floor on candidate count regardless of `TOP_K` |
| `MNEMO_ANN_MAX_CANDIDATES` | `512` | Ceiling on candidate count |

## Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMO_DB_PATH` | `data/memory.db` | Path to the SQLite database file. The `data/` directory is created automatically if it does not exist. |

SQLite runs in WAL (Write-Ahead Logging) mode for concurrent read safety.

## Compaction

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMO_COMPACT_EVERY_N` | `20` | Run compaction after every N turns. Set `0` to disable. |
| `MNEMO_COMPACT_MIN_ROWS` | `15` | Minimum rows in the session store before compaction fires. Prevents wasted LLM calls on very short sessions. |

Compaction runs in a background daemon thread and does not block chat responses.

## API Server

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMO_API_HOST` | `127.0.0.1` | Host to bind. Set `0.0.0.0` to accept external connections. |
| `MNEMO_API_PORT` | `8765` | Port to listen on. |
| `MNEMO_API_KEY` | — | If set, all endpoints except `/health` and `/metrics` require `Authorization: Bearer <key>`. Leave empty to disable auth. |
| `MNEMO_RATE_LIMIT` | `120/minute` | Rate limit for all non-chat endpoints, per tenant+IP. |
| `MNEMO_RATE_LIMIT_CHAT` | `60/minute` | Rate limit for `/v1/chat` and `/v1/chat/stream`. |

Rate limit format: `<count>/<period>` where period is `second`, `minute`, or `hour`.

## Example `.env` files

### Minimal (Groq, no embeddings, no auth)

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
MNEMO_NO_EMBEDDINGS=1
```

### Production (Gemini, embeddings, API key, custom limits)

```env
GEMINI_API_KEY=AIzaSy...
GROQ_API_KEY=gsk_...           # still needed for extraction model
GROQ_MODEL=gemini-2.0-flash
MNEMO_API_KEY=my-strong-secret
MNEMO_API_HOST=0.0.0.0
MNEMO_API_PORT=8765
MNEMO_RATE_LIMIT_CHAT=30/minute
MNEMO_COMPACT_EVERY_N=20
MNEMO_DB_PATH=/var/data/mnemo.db
```

### Eval / testing (no embeddings, no tool use, no compaction)

```env
GROQ_API_KEY=gsk_...
MNEMO_NO_EMBEDDINGS=1
MNEMO_TOOL_USE=0
MNEMO_COMPACT_EVERY_N=0
```
