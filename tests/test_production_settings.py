from __future__ import annotations

import pytest
from pathlib import Path

from leantrader.production.settings import SafetyError, Settings


def test_safe_defaults_are_paper_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in ("TRADING_MODE", "ENABLE_LIVE", "ALLOW_LIVE", "LIVE_CONFIRM", "BYBIT_TESTNET_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings.from_env()
    assert settings.starting_cash == 50.0
    assert settings.order_usd == 2.0
    assert settings.testnet_enabled is False
    assert settings.memory_max_episodes == 5000
    assert settings.brain_min_strategy_samples == 50
    assert settings.brain_quarantine_min_samples == 100
    assert settings.capital_principal_floor_fraction == 0.70
    assert settings.capital_profit_reinvest_fraction == 0.50


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


def test_testnet_requires_explicit_confirmation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BYBIT_TESTNET_ENABLED", "true")
    monkeypatch.delenv("BYBIT_TESTNET_CONFIRM", raising=False)
    with pytest.raises(SafetyError, match="TESTNET_CONFIRM"):
        Settings.from_env()


def test_testnet_configuration_cannot_grant_live_authority(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BYBIT_TESTNET_ENABLED", "true")
    monkeypatch.setenv("BYBIT_TESTNET_CONFIRM", "I_UNDERSTAND_TESTNET_ONLY")
    settings = Settings.from_env()
    assert settings.testnet_enabled is True
    assert settings.exchange == "bybit"
    assert settings.testnet_max_order_usd == 10.0


def test_auto_symbols_enable_dynamic_all_eligible_market_rotation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPER_SYMBOLS", "AUTO")
    settings = Settings.from_env()
    assert settings.market_universe_mode == "dynamic"
    assert settings.symbols == ()
    assert settings.market_quote == "USDT"
    assert settings.market_scan_batch_size == 18
    assert settings.confirm_timeframes == ()


def test_non_bybit_exchange_uses_dynamic_capability_attestation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_EXCHANGE", "okx")
    monkeypatch.setenv("PAPER_SYMBOLS", "AUTO")
    monkeypatch.setenv("CONFIRM_TIMEFRAMES", "AUTO")
    settings = Settings.from_env()
    assert settings.exchange == "okx"
    assert settings.market_universe_mode == "dynamic"
    assert settings.confirm_timeframes == ()


def test_non_bybit_cannot_reuse_bybit_testnet_execution(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_EXCHANGE", "okx")
    monkeypatch.setenv("BYBIT_TESTNET_ENABLED", "true")
    monkeypatch.setenv("BYBIT_TESTNET_CONFIRM", "I_UNDERSTAND_TESTNET_ONLY")
    with pytest.raises(SafetyError, match="only for Bybit"):
        Settings.from_env()


def test_model_research_requires_known_provider_model_and_https(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("MODEL_RESEARCH_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="MODEL_RESEARCH_PROVIDER"):
        Settings.from_env()
    monkeypatch.setenv("MODEL_RESEARCH_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_RESEARCH_MODEL", "research-model")
    monkeypatch.setenv("MODEL_RESEARCH_ENDPOINT", "http://unsafe.example")
    with pytest.raises(ValueError, match="HTTPS"):
        Settings.from_env()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MARKET_SCAN_BATCH_SIZE", "0", "MARKET_SCAN_BATCH_SIZE"),
        ("MARKET_REFRESH_SECONDS", "30", "MARKET_REFRESH_SECONDS"),
        ("MARKET_MIN_QUOTE_VOLUME_USD", "-1", "MARKET_MIN_QUOTE_VOLUME_USD"),
        ("MARKET_MAX_SPREAD_BPS", "0", "MARKET_MAX_SPREAD_BPS"),
    ],
)
def test_dynamic_market_safety_bounds(monkeypatch, tmp_path, name, value, message):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPER_SYMBOLS", "AUTO")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        Settings.from_env()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MARKET_EVIDENCE_MIN_SAMPLES", "2", "MARKET_EVIDENCE_MIN_SAMPLES"),
        ("MARKET_EVIDENCE_WINDOW", "4", "MARKET_EVIDENCE_WINDOW"),
        ("ROUTER_MIN_ADVANCED_CONFIDENCE", "1.1", "ROUTER_MIN_ADVANCED_CONFIDENCE"),
        ("ROUTER_MIN_COMBINED_SCORE", "1.1", "ROUTER_MIN_COMBINED_SCORE"),
        ("ROUTER_NEGATIVE_CONSENSUS_VETO", "0.1", "ROUTER_NEGATIVE_CONSENSUS_VETO"),
    ],
)
def test_decision_router_safety_bounds(monkeypatch, tmp_path, name, value, message):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(name, value)
    if name == "MARKET_EVIDENCE_WINDOW":
        monkeypatch.setenv("MARKET_EVIDENCE_MIN_SAMPLES", "5")
    with pytest.raises(ValueError, match=message):
        Settings.from_env()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MEMORY_MAX_EPISODES", "99", "MEMORY_MAX_EPISODES"),
        ("MEMORY_HALF_LIFE_HOURS", "0", "MEMORY_HALF_LIFE_HOURS"),
        ("BRAIN_MIN_STRATEGY_SAMPLES", "2", "BRAIN_MIN_STRATEGY_SAMPLES"),
        ("CAPITAL_PRINCIPAL_FLOOR_FRACTION", "1.1", "CAPITAL_PRINCIPAL_FLOOR_FRACTION"),
        ("CAPITAL_PROFIT_REINVEST_FRACTION", "-0.1", "CAPITAL_PROFIT_REINVEST_FRACTION"),
    ],
)
def test_cns_brain_memory_capital_safety_bounds(monkeypatch, tmp_path, name, value, message):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        Settings.from_env()


def test_compose_passes_cns_brain_memory_and_capital_controls_to_container():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    for name in (
        "CNS_STATE_PATH",
        "BRAIN_STATE_PATH",
        "MEMORY_RETENTION_STATE_PATH",
        "CAPITAL_GROWTH_STATE_PATH",
        "MEMORY_MAX_EPISODES",
        "MEMORY_HALF_LIFE_HOURS",
        "BRAIN_MIN_STRATEGY_SAMPLES",
        "BRAIN_NEGATIVE_EXPECTANCY_FLOOR",
        "BRAIN_QUARANTINE_MIN_SAMPLES",
        "BRAIN_QUARANTINE_EXPECTANCY_FLOOR",
        "BRAIN_RECOVERY_EXPECTANCY_FLOOR",
        "CAPITAL_PRINCIPAL_FLOOR_FRACTION",
        "CAPITAL_PROFIT_REINVEST_FRACTION",
    ):
        assert f"{name}:" in compose
