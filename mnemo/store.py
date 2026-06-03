from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # FastAPI runs sync handlers in a thread pool; allow use from worker threads.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL mode allows concurrent readers alongside writers (no full-table locks).
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r[1]) for r in rows}


def _migrate(conn: sqlite3.Connection) -> None:
    """Upgrade legacy single-column memory_chunks to full structured rows."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_chunks'"
    ).fetchone()
    if not cur:
        return
    cols = _table_columns(conn, "memory_chunks")
    if "kind" not in cols:
        conn.execute("ALTER TABLE memory_chunks ADD COLUMN kind TEXT NOT NULL DEFAULT 'fact'")
    if "subj" not in cols:
        conn.execute("ALTER TABLE memory_chunks ADD COLUMN subj TEXT")
    if "pred" not in cols:
        conn.execute("ALTER TABLE memory_chunks ADD COLUMN pred TEXT")
    if "obj" not in cols:
        conn.execute("ALTER TABLE memory_chunks ADD COLUMN obj TEXT")
    if "embedding" not in cols:
        conn.execute("ALTER TABLE memory_chunks ADD COLUMN embedding BLOB")
    conn.commit()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memory_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'fact',
            content TEXT NOT NULL,
            subj TEXT,
            pred TEXT,
            obj TEXT,
            embedding BLOB,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_chunks(session_id);
        CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_chunks(created_at);
        CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory_chunks(session_id, kind);

        CREATE TABLE IF NOT EXISTS user_profile_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'fact',
            content TEXT NOT NULL,
            subj TEXT,
            pred TEXT,
            obj TEXT,
            embedding BLOB,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_profile_tenant ON user_profile_chunks(tenant_id);
        """
    )
    conn.commit()
    _migrate(conn)


# ---------------------------------------------------------------------------
# User profile (cross-session, scoped by tenant_id)
# ---------------------------------------------------------------------------

def add_profile_unit(
    conn: sqlite3.Connection,
    tenant_id: str,
    kind: str,
    content: str,
    *,
    subj: str | None = None,
    pred: str | None = None,
    obj: str | None = None,
    embedding: bytes | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO user_profile_chunks (tenant_id, kind, content, subj, pred, obj, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            tenant_id,
            kind,
            content.strip(),
            subj.strip() if subj else None,
            pred.strip() if pred else None,
            obj.strip() if obj else None,
            embedding,
            time.time(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def upsert_profile_triple(
    conn: sqlite3.Connection,
    tenant_id: str,
    subj: str,
    pred: str,
    obj: str,
    content: str,
    embedding: bytes | None = None,
) -> int:
    existing = conn.execute(
        "SELECT id FROM user_profile_chunks WHERE tenant_id = ? AND kind = 'triple' AND subj = ? AND pred = ?",
        (tenant_id, subj.strip(), pred.strip()),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE user_profile_chunks SET obj = ?, content = ?, embedding = ?, created_at = ? WHERE id = ?",
            (obj.strip(), content.strip(), embedding, time.time(), existing["id"]),
        )
        conn.commit()
        return int(existing["id"])
    return add_profile_unit(conn, tenant_id, "triple", content, subj=subj, pred=pred, obj=obj, embedding=embedding)


def list_profile_chunks(
    conn: sqlite3.Connection, tenant_id: str, limit: int = 200
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT id, tenant_id, kind, content, subj, pred, obj, embedding, created_at
            FROM user_profile_chunks WHERE tenant_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (tenant_id, limit),
        )
    )


def clear_profile(conn: sqlite3.Connection, tenant_id: str) -> int:
    cur = conn.execute("DELETE FROM user_profile_chunks WHERE tenant_id = ?", (tenant_id,))
    conn.commit()
    return cur.rowcount


def delete_profile_chunk(conn: sqlite3.Connection, chunk_id: int, tenant_id: str) -> bool:
    cur = conn.execute(
        "DELETE FROM user_profile_chunks WHERE id = ? AND tenant_id = ?", (chunk_id, tenant_id)
    )
    conn.commit()
    return cur.rowcount > 0


def update_profile_chunk(
    conn: sqlite3.Connection,
    chunk_id: int,
    tenant_id: str,
    new_content: str,
    embedding: bytes | None = None,
) -> bool:
    cur = conn.execute(
        "UPDATE user_profile_chunks SET content = ?, embedding = ?, created_at = ? WHERE id = ? AND tenant_id = ?",
        (new_content.strip(), embedding, time.time(), chunk_id, tenant_id),
    )
    conn.commit()
    return cur.rowcount > 0


def add_memory_unit(
    conn: sqlite3.Connection,
    session_id: str,
    kind: str,
    content: str,
    *,
    subj: str | None = None,
    pred: str | None = None,
    obj: str | None = None,
    embedding: bytes | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO memory_chunks (session_id, kind, content, subj, pred, obj, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            kind,
            content.strip(),
            subj.strip() if subj else None,
            pred.strip() if pred else None,
            obj.strip() if obj else None,
            embedding,
            time.time(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def upsert_triple(
    conn: sqlite3.Connection,
    session_id: str,
    subj: str,
    pred: str,
    obj: str,
    content: str,
    embedding: bytes | None = None,
) -> int:
    """Insert triple; if (session_id, subj, pred) already exists, update it instead."""
    existing = conn.execute(
        "SELECT id FROM memory_chunks WHERE session_id = ? AND kind = 'triple' AND subj = ? AND pred = ?",
        (session_id, subj.strip(), pred.strip()),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE memory_chunks SET obj = ?, content = ?, embedding = ?, created_at = ? WHERE id = ?",
            (obj.strip(), content.strip(), embedding, time.time(), existing["id"]),
        )
        conn.commit()
        return int(existing["id"])
    return add_memory_unit(conn, session_id, "triple", content, subj=subj, pred=pred, obj=obj, embedding=embedding)


def update_embedding(conn: sqlite3.Connection, chunk_id: int, embedding: bytes) -> None:
    conn.execute("UPDATE memory_chunks SET embedding = ? WHERE id = ?", (embedding, chunk_id))
    conn.commit()


def list_chunks_for_session(
    conn: sqlite3.Connection, session_id: str, limit: int = 500
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT id, session_id, kind, content, subj, pred, obj, embedding, created_at
            FROM memory_chunks WHERE session_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (session_id, limit),
        )
    )


def clear_session(conn: sqlite3.Connection, session_id: str) -> int:
    cur = conn.execute("DELETE FROM memory_chunks WHERE session_id = ?", (session_id,))
    conn.commit()
    return cur.rowcount


def delete_chunk(conn: sqlite3.Connection, chunk_id: int, session_id: str) -> bool:
    """Delete a single chunk; returns True if a row was actually removed."""
    cur = conn.execute(
        "DELETE FROM memory_chunks WHERE id = ? AND session_id = ?", (chunk_id, session_id)
    )
    conn.commit()
    return cur.rowcount > 0


def update_chunk_content(
    conn: sqlite3.Connection,
    chunk_id: int,
    session_id: str,
    new_content: str,
    embedding: bytes | None = None,
) -> bool:
    """Overwrite content (and optionally embedding) of a chunk; returns True on success."""
    cur = conn.execute(
        "UPDATE memory_chunks SET content = ?, embedding = ?, created_at = ? WHERE id = ? AND session_id = ?",
        (new_content.strip(), embedding, time.time(), chunk_id, session_id),
    )
    conn.commit()
    return cur.rowcount > 0


# Backwards-compatible name used by older code paths
def add_chunk(conn: sqlite3.Connection, session_id: str, content: str) -> int:
    return add_memory_unit(conn, session_id, "fact", content)

