# Architecture

This document describes how Mnemo works internally — the data flow for each turn, the retrieval pipeline, and the design decisions behind the key components.

## Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Each chat turn                              │
│                                                                      │
│  User message                                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐    recall / remember     ┌──────────────────────┐  │
│  │  Tool-use   │ ◄──────────────────────► │   Memory Store       │  │
│  │   Loop      │    forget / update_fact  │   (SQLite WAL)       │  │
│  └─────────────┘                          └──────────────────────┘  │
│       │                                            ▲                 │
│       │  Final reply (streamed)                    │                 │
│       ▼                                            │                 │
│  Assistant text ──── Extractor ──── Pipeline ──────┘                │
│                      (LLM call)    (embed + dedup + write)          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    Every N turns (background thread)                 │
│                                                                      │
│  SQLite DB ──► Compaction Agent (LLM) ──► Condensed SQLite DB       │
│                merge duplicates                                      │
│                resolve contradictions                                │
│                drop trivial entries                                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Tool-use loop (`mnemo/agent.py`)

The core chat loop gives the model four memory tools and runs up to `_MAX_TOOL_ROUNDS = 6` rounds before forcing a final answer:

| Tool | What it does |
|------|-------------|
| `recall(query)` | Vector + lexical search over stored memories; returns the top-K chunks |
| `remember(content)` | Explicitly stores a fact or statement in the session store |
| `remember_profile(content)` | Stores a durable cross-session fact in the tenant user profile |
| `forget(chunk_id)` | Deletes a specific memory by ID |
| `update_fact(chunk_id, new_content)` | Replaces an existing memory's content |

The model decides autonomously when to call tools. If the underlying model rejects tool-call payloads (some models emit XML-style tool calls that the API rejects), the loop falls back to **passive memory injection** — the same retrieval result is prepended to the system prompt without tool-calling.

### 2. Retrieval pipeline (`mnemo/retriever.py`)

Every query runs a three-component hybrid score:

```
score = W_dense · cosine(q_vec, chunk_vec)
      + W_lex   · lexical_overlap(query_tokens, chunk_tokens)
      + W_rec   · recency_decay(chunk_timestamp)
```

Default weights: `W_dense = 0.5`, `W_lex = 0.35`, `W_rec = 0.15` — tunable via env vars.

**FAISS ANN prefilter** (when embeddings are enabled and row count ≥ `MNEMO_ANN_MIN_ROWS`): instead of scoring every row, FAISS returns `ANN_CANDIDATE_MULT × TOP_K` approximate nearest neighbours first, then the full hybrid scorer re-ranks that candidate set. This keeps retrieval sub-linear as the store grows.

### 3. Memory extraction (`mnemo/extractor.py` + `mnemo/pipeline.py`)

After each turn, a lightweight extraction LLM call (`GROQ_EXTRACT_MODEL`, default `llama-3.1-8b-instant`) reads the user+assistant exchange and returns structured JSON:

```json
{
  "summary": "User is planning to move to Denver in June.",
  "triples": [
    { "subject": "user", "predicate": "plans_to_move_to", "object": "Denver" },
    { "subject": "move", "predicate": "scheduled_for", "object": "June" }
  ],
  "facts": [
    "The user's emergency contact is Maya, who lives in Boulder."
  ]
}
```

`pipeline.py` then:
1. Embeds all extracted texts in a single batch
2. Deduplicates against existing embeddings (cosine similarity ≥ 0.92 → skip)
3. Upserts triples (same subject+predicate → overwrite object, resolving contradictions)
4. Appends new facts and summaries

### 4. Compaction agent (`mnemo/compactor.py`)

Fires in a background daemon thread every `MNEMO_COMPACT_EVERY_N` turns (default 20), provided at least `MNEMO_COMPACT_MIN_ROWS` rows exist.

The compaction LLM call receives **all** stored memories and is asked to produce a clean, minimal set — merging duplicates, resolving contradictions, and dropping trivial entries. The original rows are atomically replaced with the compacted set. This keeps memory quality stable in very long sessions.

Typical compaction ratio observed in testing: **40 rows → 14 rows** (~66% reduction).

### 5. Storage (`mnemo/store.py`)

SQLite with WAL mode for concurrent read safety. Two tables:

**`memory_chunks`** — per-session memories:

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT (UUID) | Primary key |
| `session_id` | TEXT | Scoped session key (`tenant:session`) |
| `kind` | TEXT | `fact`, `triple`, or `summary` |
| `content` | TEXT | Human-readable text |
| `subj`, `pred`, `obj` | TEXT | Triple fields (NULL for facts/summaries) |
| `embedding` | BLOB | float32 array, NULL when embeddings disabled |
| `created_at` | REAL | Unix timestamp (used for recency scoring) |

**`user_profile`** — cross-session durable facts:

Same schema as `memory_chunks`, scoped by `tenant_id` instead of `session_id`. Profile rows are injected into every system prompt for the tenant, across all sessions.

### 6. Multi-tenancy (`mnemo/tenancy.py`)

Session IDs are namespaced as `{tenant_id}:{session_id}` internally. The HTTP API accepts `tenant_id` in the request body or the `X-Tenant-ID` header. Both IDs are validated (alphanumeric + safe punctuation, max length) before use as storage keys.

### 7. Observability (`mnemo/metrics.py`)

A zero-dependency thread-safe registry collects counters and latency histograms in-process. An ASGI middleware on the FastAPI app records every request's endpoint, status code, and wall-clock latency. The `/metrics` endpoint exports in Prometheus text format; `/v1/metrics` exports JSON.

See [observability.md](observability.md) for full details.

---

## Data flow: a single turn

```
1.  POST /v1/chat  { tenant_id, session_id, message }
        │
2.  scope_session(tenant, session) → scoped_id = "tenant:session"
        │
3.  Retrieve ChatState (or create new one)
        │
4.  Build system prompt:
        ├── AGENTIC_SYSTEM (tool instructions)
        └── User profile block (from user_profile table)
        │
5.  Tool-use loop:
        ├── LLM call with tools=[recall, remember, forget, update_fact]
        ├── If finish_reason == "tool_calls":
        │       execute tool → append tool result to api_messages → repeat
        └── If finish_reason == "stop":
                assistant_text = response content → break
        │
6.  Append user + assistant messages to ChatState.messages
        │
7.  Background: extract_memory(user_text, assistant_text)
        │             ↓
        │         persist_augmentation(extracted)
        │             ↓
        │         embed + dedup + write to SQLite
        │
8.  If turn_count % COMPACT_EVERY_N == 0:
        └── compact_session_async() → daemon thread
        │
9.  Return ChatResponse { reply: assistant_text }
```

---

## Design decisions

**Why SQLite and not a vector database?**
SQLite is zero-infrastructure, embeds in-process, and is fast enough for the memory sizes Mnemo targets (hundreds to low thousands of rows per session). FAISS provides ANN speed at larger scales without a separate service. A vector DB like Qdrant or Pinecone would add operational complexity with no benefit until row counts reach the tens of thousands.

**Why active tool-calling instead of passive RAG?**
Passive RAG always injects the top-K retrieved chunks whether they're relevant or not. The model has no signal about retrieval quality. With tools, the model can choose not to call `recall` if the answer is obvious from context, call it multiple times with different queries, or call `forget` to clean up stale data. This reduces noise and gives the model agency over its own memory.

**Why a separate extraction LLM call?**
Extracting structured facts inline (as part of the chat completion) would pollute the conversation context and force a slower model. Using a small fast model (`llama-3.1-8b-instant`) for extraction keeps latency low and the chat model focused on the conversation.

**Why background compaction instead of inline?**
Compaction is expensive (one full LLM call over all memories) and not latency-sensitive. Running it in a daemon thread means the user's chat turn returns immediately.
