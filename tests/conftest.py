import os

import pytest


@pytest.fixture(autouse=True)
def _env_dedupe_off(monkeypatch):
    # Disable signal dedupe across tests to avoid interference when publisher is exercised
    monkeypatch.setenv("SIGNALS_DEDUPE_WINDOW_SEC", "0")
    # Unit tests use deterministic provider fixtures; integration tests can
    # explicitly re-enable network collection when required.
    monkeypatch.setenv("PUBLIC_CONTEXT_ENABLED", "false")
    yield
