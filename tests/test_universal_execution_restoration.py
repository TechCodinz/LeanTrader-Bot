import os

from src.leantrader.execution.broker_ccxt import (
    BrokerCCXT,
    _PROBE_CACHE,
)
from src.leantrader.execution.router import (
    execution_status,
    route_order,
)


def clear_env():
    for key in (
        "EXECUTION_MODE",
        "CCXT_TESTNET",
        "BYBIT_TESTNET",
        "ENABLE_LIVE",
        "ALLOW_LIVE",
        "LIVE_CONFIRM",
        "API_ENVIRONMENT",
        "EXCHANGE_ENVIRONMENT",
        "API_KEY",
        "API_SECRET",
        "CCXT_API_KEY",
        "CCXT_API_SECRET",
        "BYBIT_API_KEY",
        "BYBIT_API_SECRET",
    ):
        os.environ.pop(
            key,
            None,
        )

    _PROBE_CACHE.clear()


def test_paper_mode():
    clear_env()

    os.environ[
        "EXECUTION_MODE"
    ] = "paper"

    status = execution_status()

    assert (
        status[
            "execution_mode"
        ]
        == "paper"
    )

    result = route_order(
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "qty": 0.001,
            "price": 50000.0,
        }
    )

    assert result["ok"] is True
    assert (
        result["simulated"]
        is True
    )
    assert (
        result["authority"]
        == "paper"
    )


def test_auto_without_api_is_paper():
    clear_env()

    os.environ[
        "EXECUTION_MODE"
    ] = "auto"

    broker = BrokerCCXT()

    assert (
        broker.resolve_mode()
        == "paper"
    )


def test_explicit_testnet_without_api_fails_closed():
    clear_env()

    os.environ[
        "EXECUTION_MODE"
    ] = "testnet"

    result = route_order(
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "qty": 0.001,
            "price": 50000.0,
        }
    )

    assert result["ok"] is False
    assert (
        result["executed"]
        is False
    )
    assert (
        result["simulated"]
        is False
    )


def test_explicit_live_without_api_fails_closed():
    clear_env()

    os.environ[
        "EXECUTION_MODE"
    ] = "live"

    result = route_order(
        {
            "symbol": "BTC/USDT",
            "side": "buy",
            "qty": 0.001,
            "price": 50000.0,
        }
    )

    assert result["ok"] is False
    assert (
        result["executed"]
        is False
    )
    assert (
        result["simulated"]
        is False
    )


def test_auto_can_detect_testnet_without_order():
    clear_env()

    os.environ[
        "EXECUTION_MODE"
    ] = "auto"
    os.environ[
        "API_KEY"
    ] = "placeholder"
    os.environ[
        "API_SECRET"
    ] = "placeholder"

    broker = BrokerCCXT()

    broker._probe_environment = (
        lambda env:
        env == "testnet"
    )

    assert (
        broker.resolve_mode()
        == "testnet"
    )


def test_auto_can_detect_live_without_order():
    clear_env()

    os.environ[
        "EXECUTION_MODE"
    ] = "auto"
    os.environ[
        "API_KEY"
    ] = "placeholder-live"
    os.environ[
        "API_SECRET"
    ] = "placeholder-live"

    broker = BrokerCCXT()

    broker._probe_environment = (
        lambda env:
        env == "live"
    )

    assert (
        broker.resolve_mode()
        == "live"
    )
