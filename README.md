# Mnemo

[![CI](https://github.com/2005-Aneeshdutt/Mnemo-/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/2005-Aneeshdutt/Mnemo-/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Mnemo is a persistent memory layer for LLM agents. Rather than passively injecting retrieved context, Mnemo exposes memory as a set of tools the model actively calls — deciding when to recall, store, update, or delete facts. A background compaction agent periodically consolidates memories to maintain quality across long conversations.

## Table of Contents

- [Why Mnemo](#why-mnemo)
- [Architecture](#architecture)
- [Features](#features)
- [Benchmark](#benchmark)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [HTTP API](#http-api)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Development](#development)
- [License](#license)

## Why Mnemo

Standard approaches to LLM memory either send the full conversation history (expensive, noisy at scale) or truncate it (poor recall). Mnemo provides a structured alternative:

- **Active memory tools** — the model calls `recall`, `remember`, `forget`, and `update_fact` explicitly rather than receiving a passive memory dump.
- **Cross-session user profiles** — durable facts (name, preferences, location) persist across all sessions under the same tenant, injected into every system prompt.
- **Hybrid retrieval** — dense embedding similarity, lexical overlap, and recency are combined into a single score to surface the most relevant memories.
- **Background compaction** — a second LLM pass runs every N turns, merging duplicates and resolving contradictions so memory quality stays stable as sessions grow.
- **Streaming** — token-by-token output for both CLI and HTTP, with tool calls resolved synchronously before the reply streams.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    After each turn                       │
│  User + Assistant → Extractor → Pipeline → SQLite DB    │
│                                          → User Profile  │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│               Every N turns (background)                 │
│  SQLite DB → Compaction Agent → Condensed Memory DB     │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                    Each question                         │
│  User → Tool-use loop → recall/remember/forget/update   │
│                       → Final reply streams             │
└─────────────────────────────────────────────────────────┘
```

## Features

| Area | Details |
|------|---------|
| **Agentic tools** | `recall`, `remember`, `remember_profile`, `forget`, `update_fact` via LLM function-calling |
| **User profile** | Cross-session facts scoped by tenant; always visible in system prompt |
| **Memory extraction** | Per-turn extraction of facts, semantic triples, and turn summaries |
| **Persistence** | SQLite with WAL mode; triple upsert for contradiction resolution; cosine-similarity deduplication |
| **Retrieval** | Hybrid dense + lexical + recency scoring; optional FAISS ANN prefilter |
| **Compaction** | Background LLM agent consolidates memory every N turns |
| **Streaming** | SSE streaming for CLI and `/v1/chat/stream` endpoint |
| **Multi-provider** | Supports Groq and Google Gemini; swap via environment variable |
| **Interfaces** | Interactive CLI and FastAPI service with rate limiting and multi-tenant auth |

## Benchmark

Evaluated on a custom long-memory QA dataset using lexical-only retrieval mode:

| Mode | Pass rate | Checks passed |
|------|-----------|---------------|
| **Mnemo memory** | **95.5%** | 21 / 22 |
| Baseline (last-N only) | 100% | 22 / 22 |

**Note:** These results are on short sessions (≤ 25 turns) where the baseline context window covers the full conversation. The benchmark dataset `eval/data/locomo_bench.json` contains 30-turn sessions specifically designed to stress-test long-range recall — where facts stated in turns 1–5 fall outside the 12-turn baseline window. Run it yourself:

```bash
python eval/run_locomo.py eval/data/locomo_bench.json --mode both
```

## Quick Start

**Prerequisites:** Python 3.11+, a [Groq](https://console.groq.com) or [Gemini](https://aistudio.google.com/app/apikey) API key.

```bash
git clone https://github.com/2005-Aneeshdutt/Mnemo-.git
cd Mnemo-
python -m venv .venv
```

Activate:
- **Windows:** `.\.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
cp .env.example .env
# Set GROQ_API_KEY or GEMINI_API_KEY in .env
```

## CLI Usage

```bash
python main.py
python main.py --tenant acme --session support-42
python main.py --no-embeddings
```

Tool calls and memory activity are printed in real time:

```
You> what's my favorite color?
  [memory] searching memory: 'favorite color'
           → [3] fact: user's favorite color is blue
AI> Your favorite color is blue.
```

Built-in commands: `/memory`, `/triples`, `/clear`, `/help`, `/quit`

## HTTP API

```bash
python main.py serve
# Docs at http://127.0.0.1:8765/docs
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/v1/chat` | Chat turn with retrieval and memory write |
| `POST` | `/v1/chat/stream` | Streaming chat (SSE) |
| `GET` | `/v1/sessions/{session_id}/memory` | List session memory |
| `DELETE` | `/v1/sessions/{session_id}/memory` | Clear session memory |
| `GET` | `/v1/users/{tenant_id}/profile` | List user profile facts |
| `DELETE` | `/v1/users/{tenant_id}/profile` | Clear user profile |

### Examples

```bash
# Chat
curl -s http://127.0.0.1:8765/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"demo","session_id":"s1","message":"My name is Aneesh."}'

# Streaming chat
curl -s http://127.0.0.1:8765/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"demo","session_id":"s1","message":"What is my name?"}'

# View user profile
curl -s http://127.0.0.1:8765/v1/users/demo/profile
```

## Configuration

Copy `.env.example` to `.env`.

### Provider

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq API key (used if `GEMINI_API_KEY` is not set) |
| `GEMINI_API_KEY` | Google Gemini API key (takes priority over Groq) |
| `GROQ_MODEL` | Chat model name (e.g. `gemini-2.0-flash`, `llama-3.3-70b-versatile`) |
| `GROQ_EXTRACT_MODEL` | Extraction model (default: `llama-3.1-8b-instant`) |
| `GROQ_EMBED_MODEL` | Embedding model (default: `nomic-embed-text-v1_5`) |

### Memory and Retrieval

| Variable | Default | Purpose |
|----------|---------|---------|
| `MNEMO_TOP_K` | `12` | Retrieved memories per turn |
| `MNEMO_RECENT_MSG` | `12` | Recent turns in short-term context |
| `MNEMO_MEMORY_MAX_ROWS` | `500` | Max rows loaded for retrieval |
| `MNEMO_NO_EMBEDDINGS` | `0` | Set `1` for lexical-only retrieval |
| `MNEMO_TOOL_USE` | `1` | Set `0` to disable agentic tool use |
| `MNEMO_W_DENSE` | `0.5` | Dense similarity weight |
| `MNEMO_W_LEX` | `0.35` | Lexical match weight |
| `MNEMO_W_REC` | `0.15` | Recency weight |
| `MNEMO_DB_PATH` | `data/memory.db` | SQLite database path |

### Compaction

| Variable | Default | Purpose |
|----------|---------|---------|
| `MNEMO_COMPACT_EVERY_N` | `20` | Compact after every N turns (0 = disabled) |
| `MNEMO_COMPACT_MIN_ROWS` | `15` | Minimum stored rows before compaction fires |

### API Server

| Variable | Default | Purpose |
|----------|---------|---------|
| `MNEMO_API_HOST` | `127.0.0.1` | Bind host |
| `MNEMO_API_PORT` | `8765` | Bind port |
| `MNEMO_API_KEY` | — | Optional bearer token for protected endpoints |
| `MNEMO_RATE_LIMIT` | `120/minute` | Default route rate limit |
| `MNEMO_RATE_LIMIT_CHAT` | `60/minute` | Chat endpoint rate limit |

## Evaluation

```bash
# Full comparison (memory vs last-N baseline)
python eval/run_locomo.py eval/data/sample_locomo.json --mode both

# Long-session stress test (30-turn sessions)
python eval/run_locomo.py eval/data/locomo_bench.json --mode both

# Recommended flags for free-tier API limits
MNEMO_NO_EMBEDDINGS=1 MNEMO_TOOL_USE=0 MNEMO_COMPACT_EVERY_N=0 \
  python eval/run_locomo.py eval/data/locomo_bench.json --mode both --turn-sleep 8
```

## Development

```bash
pytest -q          # 29 tests
```

CI runs on every push and pull request to `main`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No API key error | Set `GROQ_API_KEY` or `GEMINI_API_KEY` in `.env` |
| Embedding 404 | Set `MNEMO_NO_EMBEDDINGS=1` or update `GROQ_EMBED_MODEL` |
| Tool-use 400 errors | Set `MNEMO_TOOL_USE=0` if the model does not support function calling |
| Rate limit 429/413 | Use `--turn-sleep 8` in eval; reduce `MNEMO_RECENT_MSG` |

## License

[Apache 2.0](LICENSE)

## Author

[Aneesh Dutt](https://github.com/2005-Aneeshdutt)
