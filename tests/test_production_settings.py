from __future__ import annotations

import pytest

from leantrader.production.settings import SafetyError, Settings


def test_safe_defaults_are_paper_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in ("TRADING_MODE", "ENABLE_LIVE", "ALLOW_LIVE", "LIVE_CONFIRM"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env()
    assert settings.starting_cash == 50.0
    assert settings.order_usd == 2.0


@pytest.mark.parametrize(
    ("name", "value"),
    [("TRADING_MODE", "live"), ("ENABLE_LIVE", "true"), ("ALLOW_LIVE", "true"), ("LIVE_CONFIRM", "YES")],
)
def test_any_live_flag_is_rejected(monkeypatch, tmp_path, name, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("ENABLE_LIVE", "false")
    monkeypatch.setenv("ALLOW_LIVE", "false")
    monkeypatch.setenv("LIVE_CONFIRM", "NO")
    monkeypatch.setenv(name, value)
    with pytest.raises(SafetyError):
        Settings.from_env()
