import numpy as np

from mnemo import ann


def test_ann_top_indices() -> None:
    x = np.eye(8, dtype=np.float32)
    q = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    idx = ann.ann_top_indices(q, x, 3)
    assert len(idx) == 3
    assert int(idx[0]) == 0


def test_row_matrix_empty() -> None:
    mat, idx = ann.row_embedding_matrix([])
    assert mat.shape[0] == 0
    assert idx == []
