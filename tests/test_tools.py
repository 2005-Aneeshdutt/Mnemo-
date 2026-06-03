from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np

from mnemo import store
from mnemo.tools import execute_tool, parse_tool_call, TOOL_SCHEMAS


def _in_memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_schema(conn)
    return conn


def _mock_client() -> MagicMock:
    client = MagicMock()
    return client


SESSION = "test::tools"


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

def test_tool_schemas_valid() -> None:
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"recall", "remember", "remember_profile", "forget", "update_fact"}


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------

def test_remember_stores_fact() -> None:
    conn = _in_memory_conn()
    with patch("mnemo.tools.config.EMBEDDINGS_DISABLED", True):
        result = execute_tool("remember", {"content": "user likes jazz"}, _mock_client(), conn, SESSION)
    assert "Stored" in result
    rows = store.list_chunks_for_session(conn, SESSION)
    assert any("jazz" in r["content"] for r in rows)


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

def test_recall_empty_session() -> None:
    conn = _in_memory_conn()
    with patch("mnemo.tools.config.EMBEDDINGS_DISABLED", True):
        result = execute_tool("recall", {"query": "anything"}, _mock_client(), conn, SESSION)
    assert "empty" in result.lower() or "no match" in result.lower()


def test_recall_finds_stored_fact() -> None:
    conn = _in_memory_conn()
    with patch("mnemo.tools.config.EMBEDDINGS_DISABLED", True):
        execute_tool("remember", {"content": "user's favorite food is sushi"}, _mock_client(), conn, SESSION)
        result = execute_tool("recall", {"query": "favorite food"}, _mock_client(), conn, SESSION)
    assert "sushi" in result


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------

def test_forget_removes_fact() -> None:
    conn = _in_memory_conn()
    chunk_id = store.add_memory_unit(conn, SESSION, "fact", "a temporary fact")
    with patch("mnemo.tools.config.EMBEDDINGS_DISABLED", True):
        result = execute_tool("forget", {"chunk_id": chunk_id}, _mock_client(), conn, SESSION)
    assert "deleted" in result.lower()
    rows = store.list_chunks_for_session(conn, SESSION)
    assert not any(r["id"] == chunk_id for r in rows)


def test_forget_wrong_session_fails() -> None:
    conn = _in_memory_conn()
    chunk_id = store.add_memory_unit(conn, "other::session", "fact", "belongs elsewhere")
    with patch("mnemo.tools.config.EMBEDDINGS_DISABLED", True):
        result = execute_tool("forget", {"chunk_id": chunk_id}, _mock_client(), conn, SESSION)
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# update_fact
# ---------------------------------------------------------------------------

def test_update_fact_changes_content() -> None:
    conn = _in_memory_conn()
    chunk_id = store.add_memory_unit(conn, SESSION, "fact", "user lives in NYC")
    with patch("mnemo.tools.config.EMBEDDINGS_DISABLED", True):
        result = execute_tool(
            "update_fact",
            {"chunk_id": chunk_id, "new_content": "user lives in LA"},
            _mock_client(), conn, SESSION,
        )
    assert "updated" in result.lower()
    rows = store.list_chunks_for_session(conn, SESSION)
    updated = next(r for r in rows if r["id"] == chunk_id)
    assert "LA" in updated["content"]


# ---------------------------------------------------------------------------
# parse_tool_call
# ---------------------------------------------------------------------------

def test_parse_tool_call() -> None:
    tc = MagicMock()
    tc.function.name = "recall"
    tc.function.arguments = '{"query": "favorite color"}'
    name, args = parse_tool_call(tc)
    assert name == "recall"
    assert args["query"] == "favorite color"


def test_parse_tool_call_bad_json() -> None:
    tc = MagicMock()
    tc.function.name = "remember"
    tc.function.arguments = "not json"
    name, args = parse_tool_call(tc)
    assert name == "remember"
    assert args == {}
