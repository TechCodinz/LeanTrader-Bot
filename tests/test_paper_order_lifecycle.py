"""Deterministic paper order-lifecycle verification.

These exercise the real execution fabric -- src/leantrader/execution/router.py
and broker_emulator.py -- not a mock of it. Nothing here reaches the network:
every market order is given an explicit reference price, so the router's ticker
fallback is never taken.

What the paper backend actually implements is narrower than a full exchange
lifecycle, and the tests are written to record that rather than paper over it.
BrokerEmulator holds orders in an instance dict, and route_order constructs a
fresh BrokerEmulator per call, so no order survives the call that created it.
That makes fetch-after-create, cancel, cancel_all, position and ledger
unreachable through the router today. test_emulator_state_is_per_call proves
that directly instead of leaving it as an assumption, and the NOT_IMPLEMENTED
tests assert the current contract so a future implementation trips them.
"""

import os

import pytest

from src.leantrader.execution.broker_emulator import BrokerEmulator
from src.leantrader.execution.router import (
    resolve_execution_context,
    route_order,
)


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch):
    """Run every test without inherited exchange credentials or mode overrides.

    Other modules in this suite set credential and EXCHANGE_ID environment
    variables that persist for the session. Without this the profile tests see
    apparent credentials and the router reports authority="testnet" instead of
    refusing, so the tests would pass alone and fail in a full run.
    """
    for name in list(os.environ):
        upper = name.upper()
        if any(
            hint in upper
            for hint in ("API_KEY", "APIKEY", "SECRET", "TOKEN", "PASSPHRASE", "PASSWORD")
        ):
            monkeypatch.delenv(name, raising=False)
    for name in (
        "EXCHANGE_ID",
        "EXECUTION_MODE",
        "BROKER_MODE",
        "TRADING_MODE",
        "LIVE_CONFIRM",
        "TRADING_ENABLED",
        "ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)

SYMBOL = "BTC/USDT"
REF = 50_000.0
QTY = 0.01
SLIP_BPS = 2.0  # BrokerEmulator default


def market_payload(side="buy", price=REF, qty=QTY):
    return {
        "symbol": SYMBOL,
        "side": side,
        "quantity": qty,
        "order_type": "market",
        "price": price,
    }


# ---------------------------------------------------------------- market order


def test_market_buy_create_submit_acknowledge_fill():
    receipt = route_order(market_payload("buy"), "paper")

    # create -> submit -> acknowledgement
    assert receipt["ok"] is True
    assert receipt["submitted"] is True
    assert receipt["executed"] is True
    assert receipt["simulated"] is True
    assert receipt["authority"] == "paper"
    assert receipt["execution_mode"] == "paper"
    assert receipt["backend"] == "emu"

    order = receipt["order"]
    assert order["id"], "acknowledgement must carry an order id"
    assert order["status"] == "filled"
    assert order["symbol"] == SYMBOL
    assert order["side"] == "buy"
    assert order["filled"] == pytest.approx(QTY)

    # a buy pays the spread: ref * (1 + 2bps)
    assert order["avg_px"] == pytest.approx(REF * (1 + SLIP_BPS / 10_000.0))


def test_market_sell_fills_on_the_other_side_of_the_reference():
    receipt = route_order(market_payload("sell"), "paper")
    order = receipt["order"]

    assert order["side"] == "sell"
    assert order["status"] == "filled"
    assert order["filled"] == pytest.approx(QTY)
    # a sell receives the other side: ref * (1 - 2bps)
    assert order["avg_px"] == pytest.approx(REF * (1 - SLIP_BPS / 10_000.0))


def test_buy_and_sell_fills_straddle_the_reference_price():
    buy = route_order(market_payload("buy"), "paper")["order"]["avg_px"]
    sell = route_order(market_payload("sell"), "paper")["order"]["avg_px"]

    assert sell < REF < buy
    # symmetric about the reference
    assert (buy - REF) == pytest.approx(REF - sell)


def test_partial_fills_sum_to_the_requested_quantity():
    """Partial sizes are drawn randomly; their sum is the invariant."""
    order = route_order(market_payload("buy"), "paper")["order"]

    partials = order["partials"]
    assert len(partials) == 2
    assert sum(partials) == pytest.approx(QTY)
    assert all(p > 0 for p in partials)
    assert order["filled"] == pytest.approx(sum(partials))


def test_market_order_without_a_reference_price_is_rejected_not_invented():
    """No usable price must mean rejection, never a fabricated fill."""
    emulator = BrokerEmulator()
    result = emulator.market(SYMBOL, "buy", QTY, 0.0)

    assert result["status"] == "rejected"
    assert result["error"] == "paper_reference_price_unavailable"
    assert result["filled"] == 0.0
    assert result["avg_px"] == 0.0
    assert result["id"] is None


# ------------------------------------------------------- limit / pending order


def test_limit_order_away_from_market_stays_open():
    receipt = route_order(
        {
            "symbol": SYMBOL,
            "side": "buy",
            "quantity": QTY,
            "order_type": "limit",
            "price": REF * 0.5,  # far below market: must not fill
        },
        "paper",
    )

    assert receipt["ok"] is True
    assert receipt["submitted"] is True
    assert receipt["executed"] is False, "a resting limit order has not executed"

    order = receipt["order"]
    assert order["status"] == "open"
    assert order["pending"] is True
    assert order["filled"] == 0.0
    assert order["price"] == pytest.approx(REF * 0.5)
    assert order["type"] == "limit"


def test_pending_order_is_recorded_on_its_emulator_instance():
    emulator = BrokerEmulator()
    submitted = emulator.submit_pending(SYMBOL, "buy", QTY, "limit", REF * 0.5)

    stored = emulator.orders[submitted["id"]]
    assert stored.status == "open"
    assert stored.filled == 0.0
    assert stored.qty == pytest.approx(QTY)
    assert stored.symbol == SYMBOL


# --------------------------------------------------- why fetch/cancel are gaps


def test_emulator_state_is_per_call_so_orders_do_not_survive_routing():
    """route_order builds a new BrokerEmulator each call.

    This is the concrete reason fetch-after-create, cancel, cancel_all,
    position and ledger cannot work through the router today: the object that
    held the order is discarded when route_order returns.
    """
    first = route_order(
        {
            "symbol": SYMBOL,
            "side": "buy",
            "quantity": QTY,
            "order_type": "limit",
            "price": REF * 0.5,
        },
        "paper",
    )
    second = route_order(
        {
            "symbol": SYMBOL,
            "side": "buy",
            "quantity": QTY,
            "order_type": "limit",
            "price": REF * 0.5,
        },
        "paper",
    )

    assert first["order"]["id"] != second["order"]["id"]
    # No router-level lookup exists to find either one again.
    import src.leantrader.execution.router as router_module

    for name in ("fetch_order", "cancel_order", "cancel_all_orders", "route_cancel"):
        assert not hasattr(router_module, name), (
            f"{name} now exists; give it a real lifecycle test and update "
            "the NOT_IMPLEMENTED entries in this module's docstring"
        )


@pytest.mark.parametrize(
    "operation",
    ["cancel", "cancel_all", "stop_loss", "take_profit", "trailing", "position_fetch"],
)
def test_lifecycle_operations_not_implemented_by_paper_backend(operation):
    """Records the current contract. These fail if the surface grows."""
    emulator = BrokerEmulator()
    assert not hasattr(emulator, operation)


def test_paper_backend_does_not_persist_across_construction():
    """restart_reconciliation is NOT_IMPLEMENTED_BY_PAPER_BACKEND."""
    first = BrokerEmulator()
    first.submit_pending(SYMBOL, "buy", QTY, "limit", REF * 0.5)
    assert len(first.orders) == 1

    rebuilt = BrokerEmulator()
    assert rebuilt.orders == {}, "no state is written, so none can be reconciled"


# --------------------------------------------------------- execution profiles


def test_paper_profile_routes_to_the_emulator():
    receipt = route_order(market_payload(), "paper")
    assert receipt["backend"] == "emu"
    assert receipt["authority"] == "paper"
    assert receipt["simulated"] is True


@pytest.mark.parametrize("mode", ["testnet", "live"])
def test_authenticated_profiles_without_credentials_refuse_rather_than_fall_back(mode):
    """A profile the operator selected must not be quietly downgraded."""
    receipt = route_order(market_payload(), mode)

    assert receipt["ok"] is False
    assert receipt["executed"] is not True
    assert receipt["execution_mode"] == mode, "operator-selected profile is preserved"
    assert receipt.get("authority") in (None, "none")
    assert receipt.get("simulated") is not True, "must not silently become paper"

    reason = str(receipt.get("error") or receipt.get("reason") or "")
    assert "authority" in reason or "credential" in reason


@pytest.mark.parametrize("mode", ["paper", "testnet", "live"])
def test_execution_context_preserves_the_requested_mode(mode):
    context = resolve_execution_context(market_payload(), mode)
    assert context["execution_mode"] == mode


def test_unspecified_mode_defaults_to_paper_not_to_an_authenticated_profile():
    context = resolve_execution_context(market_payload(), None)
    assert context["execution_mode"] == "paper"
    assert context["authority"] == "paper"
