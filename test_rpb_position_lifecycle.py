"""The owned-position lifecycle added to REAL_PROFIT_BOT.

Two things are under test. First, that the lifecycle is correct: a position is
created only from an authenticated non-zero fill, realized PnL is computed from
actual fills and actual fees, and the exit taxonomy fires where the historical
fast lane fired. Second, that nothing existing was removed -- Telegram, its
channels, the legacy counters and the owned-inventory SELL guards must all
still be there.

No network, no credentials, no orders.
"""

import json
import time

import pytest

from rpb_position_lifecycle import (
    DYNAMIC_TIMEOUT,
    MOMENTUM_DECAY,
    STOP_LOSS,
    TAKE_PROFIT,
    TRAILING_PROFIT,
    ExitEvaluator,
    PositionLedger,
    reconcile_fill,
)


class FakeExchange:
    """Returns whatever the venue would return, including an ack with no fill."""

    def __init__(self, order=None, refetch=None):
        self._order = order or {}
        self._refetch = refetch
        self.fetch_calls = 0

    def fetch_order(self, order_id, symbol):
        self.fetch_calls += 1
        return self._refetch if self._refetch is not None else self._order


@pytest.fixture
def ledger(tmp_path):
    return PositionLedger(tmp_path / "positions.json")


# ------------------------------------------------------- fill reconciliation


def test_an_acknowledgement_with_no_fill_is_not_a_fill():
    ack = {"id": "o-1", "status": "open", "filled": 0.0}
    ex = FakeExchange(ack, refetch=ack)

    fill = reconcile_fill(ex, "DOT/USDT", ack, attempts=2, delay_seconds=0)

    assert fill["filled"] == 0.0
    assert fill["order_id"] == "o-1"
    assert ex.fetch_calls >= 1, "an open order must be re-read, not assumed"


def test_a_zero_fill_never_creates_a_position(ledger):
    fill = {"filled": 0.0, "average": 0.0, "order_id": "o-1"}
    assert ledger.open_position("DOT/USDT", fill) is None
    assert ledger.open_symbols() == []


def test_a_real_fill_creates_an_owned_position(ledger):
    fill = {
        "order_id": "o-2",
        "client_order_id": "c-2",
        "filled": 10.0,
        "average": 0.9422,
        "cost": 9.422,
        "fee_quote": 0.0094,
        "fee_base": 0.0,
    }
    rec = ledger.open_position("DOT/USDT", fill, strategy="momentum", confidence=95)

    assert rec["sellable_quantity"] == 10.0
    assert rec["average_entry"] == pytest.approx(0.9422)
    assert rec["entry_cost"] == pytest.approx(9.422)
    assert rec["order_id"] == "o-2"
    assert rec["client_order_id"] == "c-2"
    assert ledger.owns("DOT/USDT")


def test_a_base_denominated_buy_fee_reduces_sellable_quantity(ledger):
    """Bybit spot takes the BUY fee in base. Selling `filled` would overshoot."""
    fill = {"filled": 10.0, "average": 1.0, "cost": 10.0, "fee_base": 0.01}
    rec = ledger.open_position("DOT/USDT", fill)

    assert rec["filled_quantity"] == 10.0
    assert rec["sellable_quantity"] == pytest.approx(9.99)


def test_an_order_that_fills_on_refetch_is_picked_up():
    ack = {"id": "o-3", "status": "open", "filled": 0.0}
    filled = {
        "id": "o-3",
        "status": "closed",
        "filled": 5.0,
        "average": 2.0,
        "cost": 10.0,
        "fee": {"cost": 0.01, "currency": "USDT"},
    }
    ex = FakeExchange(ack, refetch=filled)

    fill = reconcile_fill(ex, "SOL/USDT", ack, attempts=3, delay_seconds=0)

    assert fill["filled"] == 5.0
    assert fill["fee_quote"] == pytest.approx(0.01)
    assert fill["cost"] == pytest.approx(10.0)


# ------------------------------------------------------------- realized PnL


def test_realized_net_pnl_comes_from_fills_and_fees(ledger):
    ledger.open_position(
        "DOT/USDT",
        {"filled": 10.0, "average": 1.0, "cost": 10.0, "fee_quote": 0.01},
    )
    settled = ledger.close_position(
        "DOT/USDT",
        {"filled": 10.0, "average": 1.05, "cost": 10.5, "fee_quote": 0.0105, "order_id": "x-1"},
        TAKE_PROFIT,
    )

    assert settled["gross_pnl"] == pytest.approx(0.5)
    assert settled["total_fees"] == pytest.approx(0.0205)
    assert settled["realized_net_pnl"] == pytest.approx(0.4795)
    assert settled["exit_reason"] == TAKE_PROFIT
    assert not ledger.owns("DOT/USDT")


def test_a_losing_close_is_reported_as_a_loss(ledger):
    ledger.open_position("X/USDT", {"filled": 10.0, "average": 1.0, "cost": 10.0})
    settled = ledger.close_position(
        "X/USDT", {"filled": 10.0, "average": 0.98, "cost": 9.8}, STOP_LOSS
    )

    assert settled["realized_net_pnl"] < 0
    assert ledger.stats()["authentic_wins"] == 0
    assert ledger.stats()["authentic_win_rate"] == 0.0


def test_fees_can_turn_a_gross_win_into_a_net_loss(ledger):
    """The number that matters is net after actual costs."""
    ledger.open_position(
        "X/USDT", {"filled": 10.0, "average": 1.0, "cost": 10.0, "fee_quote": 0.02}
    )
    settled = ledger.close_position(
        "X/USDT", {"filled": 10.0, "average": 1.001, "cost": 10.01, "fee_quote": 0.02}, TAKE_PROFIT
    )

    assert settled["gross_pnl"] > 0
    assert settled["realized_net_pnl"] < 0


def test_a_partial_exit_leaves_the_remainder_owned(ledger):
    ledger.open_position("X/USDT", {"filled": 10.0, "average": 1.0, "cost": 10.0})
    ledger.close_position("X/USDT", {"filled": 4.0, "average": 1.1, "cost": 4.4}, TAKE_PROFIT)

    assert ledger.owns("X/USDT")
    assert ledger.positions["X/USDT"]["remaining_quantity"] == pytest.approx(6.0)


def test_a_zero_fill_sell_does_not_close_the_position(ledger):
    ledger.open_position("X/USDT", {"filled": 10.0, "average": 1.0, "cost": 10.0})
    assert ledger.close_position("X/USDT", {"filled": 0.0}, STOP_LOSS) is None
    assert ledger.owns("X/USDT")


# ----------------------------------------------------------------- exits


def test_take_profit_fires_at_the_historical_threshold():
    ev = ExitEvaluator()
    rec = {"average_entry": 100.0, "entry_timestamp": time.time(), "peak_price": 100.0}
    exit_now, reason, _ = ev.evaluate(rec, 100.0 * (1 + ev.take_profit_bps / 10_000.0))
    assert exit_now and reason == TAKE_PROFIT


def test_stop_loss_fires_at_the_historical_threshold():
    ev = ExitEvaluator()
    rec = {"average_entry": 100.0, "entry_timestamp": time.time(), "peak_price": 100.0}
    exit_now, reason, _ = ev.evaluate(rec, 100.0 * (1 - ev.stop_loss_bps / 10_000.0))
    assert exit_now and reason == STOP_LOSS


def test_trailing_arms_only_after_real_profit():
    ev = ExitEvaluator()
    now = time.time()
    rec = {"average_entry": 100.0, "entry_timestamp": now, "peak_price": 100.0}

    # Never profitable: a small dip must not be read as giving back gains.
    assert ev.evaluate(rec, 99.9, now=now)[0] is False

    ev.evaluate(rec, 100.4, now=now)          # peak arms the trail
    exit_now, reason, _ = ev.evaluate(rec, 100.22, now=now)
    assert exit_now and reason == TRAILING_PROFIT


def test_a_dead_position_times_out():
    ev = ExitEvaluator()
    now = time.time()
    rec = {
        "average_entry": 100.0,
        "entry_timestamp": now - ev.max_hold_seconds - 1,
        "peak_price": 100.0,
    }
    exit_now, reason, _ = ev.evaluate(rec, 100.0, now=now)
    assert exit_now and reason == DYNAMIC_TIMEOUT


def test_reversed_momentum_exits_before_the_stop():
    ev = ExitEvaluator()
    now = time.time()
    rec = {"average_entry": 100.0, "entry_timestamp": now, "peak_price": 100.0}
    price = 100.0 * (1 - (ev.momentum_decay_bps + 1) / 10_000.0)
    exit_now, reason, _ = ev.evaluate(rec, price, now=now, change_pct=-4.0)
    assert exit_now and reason == MOMENTUM_DECAY


def test_no_usable_price_yields_no_exit():
    """A missing mark must never be guessed into an exit."""
    ev = ExitEvaluator()
    rec = {"average_entry": 100.0, "entry_timestamp": time.time(), "peak_price": 100.0}
    assert ev.evaluate(rec, 0.0)[0] is False


def test_the_evaluator_can_only_exit_never_block_an_entry():
    """Intelligence must not starve execution."""
    import inspect

    src = inspect.getsource(ExitEvaluator)
    for forbidden in ("should_enter", "allow_entry", "can_buy", "qualify"):
        assert forbidden not in src


# ------------------------------------------------- persistence + preservation


def test_ownership_survives_a_restart(tmp_path):
    path = tmp_path / "p.json"
    a = PositionLedger(path)
    a.open_position("DOT/USDT", {"filled": 10.0, "average": 0.94, "cost": 9.4})

    b = PositionLedger(path)
    assert b.owns("DOT/USDT")
    assert b.positions["DOT/USDT"]["average_entry"] == pytest.approx(0.94)


def test_telegram_and_its_channels_are_untouched():
    """The premium/free distribution is product behaviour, not scaffolding."""
    src = open("REAL_PROFIT_BOT.py").read()
    assert "self.telegram_bot_token" in src
    assert "self.vip_chat_id" in src
    assert "self.free_chat_id" in src
    assert "self.admin_chat_id" in src
    assert "def send_telegram" in src


def test_the_legacy_counters_still_exist():
    """Existing messages read these; they are kept, just no longer claimed."""
    src = open("REAL_PROFIT_BOT.py").read()
    assert "self.total_profit" in src
    assert "self.total_trades" in src
    assert "self.winning_trades" in src


def test_the_working_sell_guards_are_preserved():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "Non-executable dust" in src
    assert "Owned before sell" in src
    assert "amount_to_precision" in src
    assert "QUARANTINED PAIR" in src


def test_both_execution_modes_remain():
    """Testnet and live capability both stay; selection stays runtime."""
    src = open("REAL_PROFIT_BOT.py").read()
    assert "ccxt.bybit" in src
    assert "ccxt.gate" in src
    assert 'def __init__(self, mode: str = "live")' in src


def test_the_win_counter_is_no_longer_pinned_at_one_hundred_percent():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "if profit > 0:\n                                self.winning_trades += 1" not in src
    assert "authentic_wins" in src


def test_owned_symbols_are_skipped_by_the_entry_scan():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "self.ledger.owns(symbol)" in src


def test_positions_are_supervised_before_each_scan():
    src = open("REAL_PROFIT_BOT.py").read()
    assert "self.manage_owned_positions()" in src


def test_the_lifecycle_places_no_orders_itself():
    """REAL_PROFIT_BOT stays the execution owner."""
    import ast
    import inspect

    import rpb_position_lifecycle as mod

    tree = ast.parse(inspect.getsource(mod))
    called = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    for forbidden in (
        "create_order",
        "create_market_buy_order",
        "create_market_sell_order",
    ):
        assert forbidden not in called
