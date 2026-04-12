from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sqlite3 import Row

logger = logging.getLogger(__name__)


def _faiss():
    import faiss

    return faiss


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """In-place row L2 normalize (for cosine via inner product)."""
    x = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x /= norms
    return x


def ann_top_indices(
    query_vec: np.ndarray,
    doc_matrix: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Return indices of top-k rows in doc_matrix by cosine similarity.
    doc_matrix and query must be same embedding dim; rows should be L2-normalized.
    """
    faiss = _faiss()
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    x = np.asarray(doc_matrix, dtype=np.float32)
    if x.shape[0] == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    d = x.shape[1]
    if q.shape[1] != d:
        raise ValueError(f"Query dim {q.shape[1]} != doc dim {d}")
    faiss.normalize_L2(q)
    faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(d)
    index.add(x)
    k = min(k, x.shape[0])
    _, indices = index.search(q, k)
    return indices[0]


def row_embedding_matrix(rows: list) -> tuple[np.ndarray, list[int]]:
    """
    Build (n, d) matrix from rows with non-null embedding blobs.
    Returns matrix and original row indices into `rows` list.
    """
    blobs: list[bytes] = []
    idx_map: list[int] = []
    for i, r in enumerate(rows):
        b = r["embedding"]
        if b:
            blobs.append(b)
            idx_map.append(i)
    if not blobs:
        return np.zeros((0, 1), dtype=np.float32), []
    vecs = [np.frombuffer(b, dtype=np.float32) for b in blobs]
    dim = vecs[0].shape[0]
    mat = np.stack(vecs, axis=0).astype(np.float32)
    if mat.shape[1] != dim:
        raise ValueError("Inconsistent embedding dimensions in corpus")
    return mat, idx_map

