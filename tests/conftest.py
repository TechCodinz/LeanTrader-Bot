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


@pytest.fixture(autouse=True)
def _v1608_fake_bybit_free_balance_compat(monkeypatch):
    """Keep the legacy deterministic Bybit fixture realistic for sell sizing.

    v1.60.8 intentionally requires a freshly reconciled *free* base-asset
    balance before any sell. The historical FakeBybit fixture only returned
    totals, so its sell tests would model an impossible exchange response.
    Supply deterministic free balances in the test double only; production
    continues to fail closed when a real balance response omits free quantity.
    """

    try:
        from tests import test_testnet_execution as testnet_tests
    except ImportError:
        yield
        return

    fake_class = getattr(testnet_tests, "FakeBybit", None)
    if fake_class is None:
        yield
        return

    original_fetch_balance = fake_class.fetch_balance

    def fetch_balance_with_free(self):
        payload = original_fetch_balance(self)
        if not isinstance(payload, dict) or "free" in payload:
            return payload
        return {
            **payload,
            "free": {
                "USDT": 10_000.0,
                "BTC": 1.0,
                "ETH": 1.0,
            },
        }

    monkeypatch.setattr(fake_class, "fetch_balance", fetch_balance_with_free)
    yield
