from mnemo import store


def test_memory_roundtrip(tmp_path) -> None:
    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    store.add_memory_unit(conn, "dev::s1", "fact", "alpha", embedding=None)
    store.add_memory_unit(conn, "dev::s1", "triple", "(a)—[p]→(b)", subj="a", pred="p", obj="b", embedding=None)
    rows = store.list_chunks_for_session(conn, "dev::s1")
    assert len(rows) == 2
    n = store.clear_session(conn, "dev::s1")
    assert n == 2
