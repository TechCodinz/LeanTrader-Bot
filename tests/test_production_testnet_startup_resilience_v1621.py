from __future__ import annotations

from pathlib import Path

import pytest

from leantrader.production.testnet_execution import (
    BybitTestnetExecutionEngine,
    TestnetSafetyError,
)


class StartupBybit:
    def __init__(
        self,
        shared,
        *,
        unsafe=False,
    ):
        self.id = "bybit"
        self.shared = shared
        self.unsafe = unsafe
        self.calls = []
        self.created = []

        self.has = {
            "fetchBalance": True,
            "createOrder": True,
            "createMarketOrder": True,
            "fetchOrder": True,
            "fetchOpenOrder": True,
            "fetchClosedOrder": True,
            "fetchOpenOrders": True,
            "fetchClosedOrders": True,
            "cancelOrder": True,
            "fetchMyTrades": True,
        }

        self.markets = {
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "spot": True,
                "active": True,
                "quote": "USDT",
            },
        }

        self.urls = {
            "api": {
                "public": "https://api.bybit.com",
                "private": "https://api.bybit.com",
            },
        }

    def set_sandbox_mode(
        self,
        enabled,
    ):
        self.calls.append(
            ("sandbox", enabled)
        )

        if not self.unsafe:
            self.urls = {
                "api": {
                    "public": "https://api-testnet.bybit.com",
                    "private": "https://api-testnet.bybit.com",
                },
            }

    def load_markets(
        self,
    ):
        self.calls.append(
            ("load_markets",)
        )

        self.shared[
            "load_attempts"
        ] += 1

        if (
            self.shared[
                "load_attempts"
            ]
            <= self.shared.get(
                "failures_before_success",
                0,
            )
        ):
            raise RuntimeError(
                'bybit {"retCode":10016,'
                '"retMsg":"Internal System Error."}'
            )

    def private_get_v5_user_query_api(
        self,
    ):
        return {
            "result": {
                "readOnly": 0,
                "permissions": {
                    "Spot": [
                        "SpotTrade"
                    ],
                    "Wallet": [],
                },
                "ips": [
                    "127.0.0.1"
                ],
                "type": 1,
            },
        }

    def fetch_balance(
        self,
    ):
        return {
            "total": {
                "USDT": 10_000.0,
            },
        }

    def close(
        self,
    ):
        self.calls.append(
            ("close",)
        )


def build_engine(
    tmp_path: Path,
    factory,
):
    key = (
        tmp_path
        / "key"
    )

    secret = (
        tmp_path
        / "secret"
    )

    key.write_text(
        "testnet-key-v1621",
        encoding="utf-8",
    )

    secret.write_text(
        "testnet-secret-v1621",
        encoding="utf-8",
    )

    return BybitTestnetExecutionEngine(
        api_key_path=key,
        api_secret_path=secret,
        state_path=(
            tmp_path
            / "state.json"
        ),
        confirmation=(
            "I_UNDERSTAND_TESTNET_ONLY"
        ),
        exchange_factory=factory,
    )


def test_transient_10016_degrades_without_crashing_and_recovers(
    tmp_path,
):
    shared = {
        "load_attempts": 0,
        "failures_before_success": 1,
    }

    adapters = []

    def factory(
        _config,
    ):
        adapter = StartupBybit(
            shared
        )

        adapters.append(
            adapter
        )

        return adapter

    engine = build_engine(
        tmp_path,
        factory,
    )

    # The transient provider failure must
    # NOT escape start() and kill the process.
    engine.start()

    first = engine.health()

    recovery = first[
        "startup_recovery"
    ]

    assert recovery[
        "degraded"
    ] is True

    assert recovery[
        "ready"
    ] is False

    assert recovery[
        "startup_failures"
    ] == 1

    assert first[
        "authenticated"
    ] is False

    # Fail closed while degraded.
    assert engine.eligible_symbols(
        "USDT"
    ) == set()

    assert len(
        adapters
    ) == 1

    # Force the bounded retry to be due.
    engine._v1621_next_retry_monotonic = (
        0.0
    )

    recovered = (
        engine.safe_snapshot()
    )

    assert recovered[
        "startup_recovery"
    ][
        "ready"
    ] is True

    assert recovered[
        "startup_recovery"
    ][
        "degraded"
    ] is False

    assert recovered[
        "startup_recovery"
    ][
        "recovery_attempts"
    ] == 1

    assert recovered[
        "startup_recovery"
    ][
        "recovery_successes"
    ] == 1

    assert recovered[
        "authenticated"
    ] is True

    assert recovered[
        "sandbox_endpoint_verified"
    ] is True

    assert engine.eligible_symbols(
        "USDT"
    ) == {
        "BTC/USDT",
    }

    # Recovery used a fresh adapter.
    assert len(
        adapters
    ) == 2


def test_degraded_executor_cannot_place_orders(
    tmp_path,
):
    shared = {
        "load_attempts": 0,
        "failures_before_success": 10,
    }

    adapters = []

    def factory(
        _config,
    ):
        adapter = StartupBybit(
            shared
        )

        adapters.append(
            adapter
        )

        return adapter

    engine = build_engine(
        tmp_path,
        factory,
    )

    engine.start()

    with pytest.raises(
        TestnetSafetyError,
        match="temporarily degraded",
    ):
        engine.mirror_events(
            [
                {
                    "timestamp": (
                        "2026-08-27T00:00:00+00:00"
                    ),
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "price": 100.0,
                    "quantity": 0.01,
                    "reason": "test",
                },
            ]
        )

    assert all(
        adapter.created == []
        for adapter in adapters
    )

    snapshot = (
        engine.health()
    )

    assert snapshot[
        "startup_recovery"
    ][
        "orders_allowed_while_degraded"
    ] is False

    assert snapshot[
        "live_authority"
    ] is False


def test_unsafe_endpoint_is_not_treated_as_recoverable(
    tmp_path,
):
    shared = {
        "load_attempts": 0,
        "failures_before_success": 0,
    }

    def factory(
        _config,
    ):
        return StartupBybit(
            shared,
            unsafe=True,
        )

    engine = build_engine(
        tmp_path,
        factory,
    )

    with pytest.raises(
        TestnetSafetyError,
        match="not exclusively",
    ):
        engine.start()

    assert engine.authenticated is False
