from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from groq import Groq

from mnemo import config

logger = logging.getLogger(__name__)


def vec_to_blob(vec: np.ndarray) -> bytes:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return v.tobytes()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def embed_texts(
    client: Groq,
    texts: Sequence[str],
    models: list[str] | None = None,
) -> tuple[list[np.ndarray], str]:
    """
    Batch embed; tries each model in `models` until one succeeds.
    Returns (vectors, model_used).
    """
    candidates = models if models is not None else config.embedding_model_candidates()
    clean = [t.strip() if t else "" for t in texts]
    if not any(clean):
        return [np.zeros(1, dtype=np.float32) for _ in clean], candidates[0]

    last_err: Exception | None = None
    for model in candidates:
        try:
            resp = client.embeddings.create(model=model, input=list(clean))
            by_index = {item.index: np.array(item.embedding, dtype=np.float32) for item in resp.data}
            vecs = [by_index[i] for i in range(len(clean))]
            if model != candidates[0]:
                logger.info("Using embedding fallback model: %s", model)
            return vecs, model
        except Exception as e:
            last_err = e
            logger.debug("Embedding model %s failed: %s", model, e)
            continue

    assert last_err is not None
    raise last_err


def embed_query(client: Groq, query: str, models: list[str] | None = None) -> tuple[np.ndarray, str]:
    vecs, model = embed_texts(client, [query], models=models)
    return vecs[0], model


def try_embed_texts(
    client: Groq, texts: Sequence[str]
) -> tuple[list[np.ndarray] | None, str | None]:
    """Returns (vectors, error_message). On failure vectors is None."""
    try:
        vecs, _model = embed_texts(client, texts)
        return vecs, None
    except Exception as e:
        logger.warning("All embedding models failed (%s); stored rows may lack vectors.", e)
        hint = (
            " Set MNEMO_NO_EMBEDDINGS=1 (or MEMORI_NO_EMBEDDINGS=1) for lexical-only retrieval, "
            "or fix GROQ_EMBED_MODEL / account embedding access."
        )
        return None, str(e) + hint


def try_embed_query(client: Groq, query: str) -> tuple[np.ndarray | None, str | None]:
    try:
        v, _ = embed_query(client, query)
        return v, None
    except Exception as e:
        return None, str(e) + (
            " (Set MNEMO_NO_EMBEDDINGS=1 to skip embeddings for retrieval.)"
        )

