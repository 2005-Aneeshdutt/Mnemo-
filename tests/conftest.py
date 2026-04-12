"""Ensure test env before `mnemo` is imported (see first import in each test file)."""
from __future__ import annotations

import os
import tempfile


def pytest_configure() -> None:
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = "pytest_dummy_gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    if not os.environ.get("MNEMO_DB_PATH"):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.environ["MNEMO_DB_PATH"] = path
    os.environ.setdefault("MNEMO_RATE_LIMIT", "100000/second")
    os.environ.setdefault("MNEMO_RATE_LIMIT_CHAT", "100000/second")
    os.environ.setdefault("MNEMO_API_KEY", "")
