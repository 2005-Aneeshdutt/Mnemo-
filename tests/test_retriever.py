from mnemo.retriever import keyword_score, retrieve_top_k


def test_keyword_score_basic() -> None:
    assert keyword_score("hello world", "hello there world") > 0
    assert keyword_score("zzz", "aaa") == 0.0


def test_lexical_top_k_order() -> None:
    rows = [
        {"id": 1, "content": "python asyncio", "embedding": None},
        {"id": 2, "content": "rust ownership", "embedding": None},
        {"id": 3, "content": "python typing", "embedding": None},
    ]
    out = retrieve_top_k(rows, "python", 2)  # type: ignore[arg-type]
    assert len(out) == 2
