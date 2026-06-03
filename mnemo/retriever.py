from __future__ import annotations

import math
import re
from sqlite3 import Row

import numpy as np
from groq import Groq

from mnemo import ann
from mnemo import config
from mnemo import embeddings as emb

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def keyword_score(query: str, doc: str) -> float:
    q = _tokens(query)
    d = _tokens(doc)
    if not q or not d:
        return 0.0
    inter = len(q & d)
    if inter == 0:
        return 0.0
    return inter / math.sqrt(len(q))


def _blob_to_vec(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr if arr.size else None


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _score_rows(
    rows: list[Row],
    query: str,
    q_vec: np.ndarray | None,
) -> list[tuple[float, Row]]:
    """Hybrid dense + lexical + recency for an arbitrary row list."""
    if not rows:
        return []

    dense_raw: list[float] = []
    for r in rows:
        v = _blob_to_vec(r["embedding"])
        if q_vec is not None and v is not None and q_vec.shape == v.shape:
            dense_raw.append(emb.cosine_similarity(q_vec, v))
        else:
            dense_raw.append(0.0)

    lex_raw = [keyword_score(query, r["content"] or "") for r in rows]

    timestamps = [float(r["created_at"]) for r in rows]
    min_ts, max_ts = min(timestamps), max(timestamps)
    span = max_ts - min_ts
    rec_raw = [(t - min_ts) / span if span > 1e-9 else 0.0 for t in timestamps]

    nd = _normalize_scores(dense_raw)
    nl = _normalize_scores(lex_raw)
    nr = _normalize_scores(rec_raw)

    w_d = config.HYBRID_W_DENSE
    w_l = config.HYBRID_W_LEX
    w_r = config.HYBRID_W_REC
    if q_vec is None or all(x == 0.0 for x in dense_raw):
        s = w_l + w_r
        w_d, w_l, w_r = 0.0, w_l / s, w_r / s

    scored: list[tuple[float, Row]] = []
    for i, r in enumerate(rows):
        score = w_d * nd[i] + w_l * nl[i] + w_r * nr[i]
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _ann_candidate_indices(rows: list[Row], query_vec: np.ndarray, k: int) -> set[int]:
    """FAISS top indices into `rows` (subset that have embeddings)."""
    mat, row_idx_map = ann.row_embedding_matrix(rows)
    if mat.shape[0] < config.ANN_MIN_ROWS:
        return set(range(len(rows)))

    want = min(
        config.ANN_MAX_CANDIDATES,
        max(config.ANN_MIN_CANDIDATES, k * config.ANN_CANDIDATE_MULT),
    )
    want = min(want, mat.shape[0])

    try:
        ann.l2_normalize_rows(mat)
        q = query_vec.astype(np.float32).reshape(-1)
        if q.shape[0] != mat.shape[1]:
            return set(range(len(rows)))
        top_local = ann.ann_top_indices(q, mat, want)
    except Exception:
        return set(range(len(rows)))

    return {row_idx_map[int(i)] for i in top_local if 0 <= int(i) < len(row_idx_map)}


def retrieve_hybrid(
    client: Groq,
    rows: list[Row],
    query: str,
    k: int,
) -> list[Row]:
    """
    Dense + lexical + recency. Optionally prefilters dense candidates with FAISS (ANN)
    when there are enough embedded rows; rows without embeddings are always included.
    """
    if not rows or k <= 0:
        return []

    q_vec: np.ndarray | None = None
    if not config.EMBEDDINGS_DISABLED:
        q_vec, _err = emb.try_embed_query(client, query)

    use_ann = (
        config.ANN_ENABLED
        and q_vec is not None
        and not config.EMBEDDINGS_DISABLED
        and sum(1 for r in rows if r["embedding"]) >= config.ANN_MIN_ROWS
    )

    if use_ann:
        ann_idx = _ann_candidate_indices(rows, q_vec, k)
        no_emb_idx = {i for i, r in enumerate(rows) if not r["embedding"]}
        pick_idx = sorted(ann_idx | no_emb_idx)
        subset = [rows[i] for i in pick_idx]
        scored = _score_rows(subset, query, q_vec)
    else:
        scored = _score_rows(rows, query, q_vec)

    return [r for _, r in scored[:k]]


def retrieve_top_k(rows: list[Row], query: str, k: int, recency_weight: float = 0.15) -> list[Row]:
    """Lexical-only top-k (no client)."""
    if not rows or k <= 0:
        return []
    timestamps = [float(r["created_at"]) for r in rows]
    min_ts, max_ts = min(timestamps), max(timestamps)
    span = max_ts - min_ts
    scored: list[tuple[float, Row]] = []
    for i, r in enumerate(rows):
        base = keyword_score(query, r["content"])
        recency = (timestamps[i] - min_ts) / span if span > 1e-9 else 0.0
        score = base + recency_weight * recency
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:k]]

