from mnemo.tenancy import parse_scoped_session, scope_session


def test_scope_roundtrip() -> None:
    s = scope_session("acme", "chat-1")
    assert s == "acme::chat-1"
    t, sess = parse_scoped_session(s)
    assert t == "acme"
    assert sess == "chat-1"


def test_scope_default() -> None:
    s = scope_session("default", "default")
    assert s == "default::default"
