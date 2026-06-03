from __future__ import annotations

import sqlite3
from unittest.mock import patch

from mnemo import store
from mnemo.tools import execute_tool, TOOL_SCHEMAS


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store.init_schema(conn)
    return conn


TENANT = "acme"
SESSION = f"{TENANT}::session-1"
SESSION_B = f"{TENANT}::session-2"


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------

def test_profile_schema_created() -> None:
    conn = _conn()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "user_profile_chunks" in tables


def test_add_and_list_profile() -> None:
    conn = _conn()
    store.add_profile_unit(conn, TENANT, "fact", "user's name is Aneesh")
    rows = store.list_profile_chunks(conn, TENANT)
    assert len(rows) == 1
    assert "Aneesh" in rows[0]["content"]


def test_profile_isolated_from_session() -> None:
    conn = _conn()
    store.add_profile_unit(conn, TENANT, "fact", "profile fact")
    store.add_memory_unit(conn, SESSION, "fact", "session fact")
    profile_rows = store.list_profile_chunks(conn, TENANT)
    session_rows = store.list_chunks_for_session(conn, SESSION)
    assert len(profile_rows) == 1
    assert len(session_rows) == 1
    assert profile_rows[0]["content"] != session_rows[0]["content"]


def test_profile_shared_across_sessions() -> None:
    conn = _conn()
    store.add_profile_unit(conn, TENANT, "fact", "user prefers dark mode")
    rows_a = store.list_profile_chunks(conn, TENANT)
    rows_b = store.list_profile_chunks(conn, TENANT)
    assert len(rows_a) == 1
    assert len(rows_b) == 1


def test_clear_profile() -> None:
    conn = _conn()
    store.add_profile_unit(conn, TENANT, "fact", "to be cleared")
    store.clear_profile(conn, TENANT)
    assert store.list_profile_chunks(conn, TENANT) == []


def test_upsert_profile_triple() -> None:
    conn = _conn()
    store.upsert_profile_triple(conn, TENANT, "user", "lives_in", "NYC", "(user) —[lives_in]→ (NYC)")
    store.upsert_profile_triple(conn, TENANT, "user", "lives_in", "LA", "(user) —[lives_in]→ (LA)")
    rows = store.list_profile_chunks(conn, TENANT)
    assert len(rows) == 1
    assert rows[0]["obj"] == "LA"


# ---------------------------------------------------------------------------
# Tool layer
# ---------------------------------------------------------------------------

def _mock_client():
    from unittest.mock import MagicMock
    return MagicMock()


def test_remember_profile_tool() -> None:
    conn = _conn()
    with patch("mnemo.tools.config.EMBEDDINGS_DISABLED", True):
        result = execute_tool(
            "remember_profile",
            {"content": "user's favorite language is Python"},
            _mock_client(), conn, SESSION, tenant_id=TENANT,
        )
    assert "profile" in result.lower()
    rows = store.list_profile_chunks(conn, TENANT)
    assert any("Python" in r["content"] for r in rows)


def test_recall_searches_profile_and_session() -> None:
    conn = _conn()
    store.add_profile_unit(conn, TENANT, "fact", "user's name is Aneesh")
    store.add_memory_unit(conn, SESSION, "fact", "user asked about Python today")
    with patch("mnemo.tools.config.EMBEDDINGS_DISABLED", True):
        result = execute_tool("recall", {"query": "name"}, _mock_client(), conn, SESSION, tenant_id=TENANT)
    assert "Aneesh" in result


def test_recall_labels_profile_vs_session() -> None:
    conn = _conn()
    store.add_profile_unit(conn, TENANT, "fact", "user is an engineer")
    store.add_memory_unit(conn, SESSION, "fact", "user discussed career today")
    with patch("mnemo.tools.config.EMBEDDINGS_DISABLED", True):
        result = execute_tool("recall", {"query": "engineer career"}, _mock_client(), conn, SESSION, tenant_id=TENANT)
    assert "[profile]" in result or "[session]" in result


def test_profile_not_visible_to_other_tenant() -> None:
    conn = _conn()
    store.add_profile_unit(conn, "tenant-a", "fact", "secret fact for tenant A")
    rows = store.list_profile_chunks(conn, "tenant-b")
    assert rows == []
