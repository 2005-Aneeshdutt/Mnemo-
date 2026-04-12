import mnemo.config as mnemo_config
from starlette.requests import Request

from mnemo import auth


def test_auth_skips_when_no_server_key(monkeypatch) -> None:
    monkeypatch.setattr(mnemo_config, "MNEMO_API_KEY", "")
    scope = {"type": "http", "headers": []}
    request = Request(scope)
    auth.require_api_key(request)  # no-op


def test_auth_rejects_bad_key(monkeypatch) -> None:
    monkeypatch.setattr(mnemo_config, "MNEMO_API_KEY", "secret")
    scope = {"type": "http", "headers": []}
    request = Request(scope)
    try:
        from fastapi import HTTPException

        auth.require_api_key(request)
    except HTTPException as e:
        assert e.status_code == 401
    else:
        raise AssertionError("expected 401")
