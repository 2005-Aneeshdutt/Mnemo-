# Development Guide

## Prerequisites

- Python 3.11+
- A [Groq](https://console.groq.com) or [Google Gemini](https://aistudio.google.com/app/apikey) API key

## Setup

```bash
git clone https://github.com/2005-Aneeshdutt/Mnemo-.git
cd Mnemo-
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the example environment file and fill in at least one API key:

```bash
cp .env.example .env
```

## Running tests

```bash
pytest -q                   # all 56 tests
pytest tests/test_metrics.py -v   # metrics module only
pytest -k "retriever"       # filter by name
```

Tests run without a real API key — `conftest.py` injects a dummy `GROQ_API_KEY` and a temporary SQLite database automatically.

## Project structure

```
mnemo/
├── agent.py        # Tool-use loop, chat_turn, streaming variant
├── ann.py          # FAISS ANN index builder and searcher
├── auth.py         # Bearer token authentication dependency
├── cli.py          # Interactive terminal client
├── client.py       # LLM client factory (Groq / Gemini)
├── compactor.py    # Background memory compaction agent
├── config.py       # All configuration via environment variables
├── embeddings.py   # Embedding calls + cosine similarity helpers
├── extractor.py    # Per-turn memory extraction (LLM call → structured JSON)
├── metrics.py      # Thread-safe metrics registry + Prometheus exporter
├── pipeline.py     # Embed, dedup, and write extracted memories to SQLite
├── retriever.py    # Hybrid dense + lexical + recency scoring
├── server.py       # FastAPI app, routes, middleware
├── store.py        # SQLite schema and CRUD helpers
├── tenancy.py      # Session scoping and ID validation
└── tools.py        # Tool schemas and execution dispatch

tests/              # pytest test suite (56 tests, 9 modules)
eval/               # Benchmark harness and datasets
docs/               # This documentation
```

## Environment variables

All variables are optional except for one API key. See the full reference in [configuration.md](configuration.md).

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | One of these | Groq API key |
| `GEMINI_API_KEY` | One of these | Google Gemini API key |

## Adding a new memory tool

Tools are defined in `mnemo/tools.py`. To add one:

1. Add a JSON schema entry to `TOOL_SCHEMAS` following the existing pattern.
2. Add a handler branch in `execute_tool()`.
3. Update the `AGENTIC_SYSTEM` prompt in `agent.py` to mention the new tool.
4. Add tests in `tests/test_tools.py`.

## Switching LLM providers

Set `GEMINI_API_KEY` in `.env` to switch to Google Gemini — the Groq client is automatically bypassed. To use a specific model:

```env
# Gemini
GEMINI_API_KEY=your-key
GROQ_MODEL=gemini-2.0-flash

# Groq
GROQ_API_KEY=your-key
GROQ_MODEL=llama-3.3-70b-versatile
```

The extraction model (`GROQ_EXTRACT_MODEL`) always uses Groq's API regardless of which chat provider is active, so keep `GROQ_API_KEY` set even when using Gemini for chat.

## Disabling features for development

```bash
# Fastest iteration — no embeddings, no tool use, no compaction
MNEMO_NO_EMBEDDINGS=1 MNEMO_TOOL_USE=0 MNEMO_COMPACT_EVERY_N=0 python main.py

# HTTP server with verbose uvicorn logging
python main.py serve
```

## CI

GitHub Actions runs `pytest -q` on every push and pull request to `main`. The workflow is defined in `.github/workflows/ci.yml`. No API keys are required — the test suite uses a dummy key and mocks all LLM calls.

## Evaluation

The eval harness at `eval/run_locomo.py` runs full LLM calls and **does** require a real API key.

```bash
# Quick smoke test (sample dataset, both modes)
python eval/run_locomo.py eval/data/sample_locomo.json --mode both

# Full benchmark (30-turn sessions, stress test long recall)
python eval/run_locomo.py eval/data/locomo_bench.json --mode both

# Free-tier friendly flags (adds sleep between turns to avoid rate limits)
MNEMO_NO_EMBEDDINGS=1 MNEMO_TOOL_USE=0 MNEMO_COMPACT_EVERY_N=0 \
  python eval/run_locomo.py eval/data/locomo_bench.json --mode both --turn-sleep 8
```

Results are written to `eval/results/report.json`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `RuntimeError: Set GROQ_API_KEY or GEMINI_API_KEY` | Add at least one API key to `.env` |
| `404` on embedding calls | Set `MNEMO_NO_EMBEDDINGS=1` or update `GROQ_EMBED_MODEL` to a model your account has access to |
| `400 tool_use_failed` during chat | The active chat model doesn't support function calling — set `MNEMO_TOOL_USE=0` |
| `429` / `413` rate limit errors in eval | Add `--turn-sleep 8`; reduce `MNEMO_RECENT_MSG` to shorten prompts |
| Tests fail with import errors | Ensure the virtual environment is activated and `pip install -r requirements.txt` has been run |
