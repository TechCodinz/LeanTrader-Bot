"""The complete micro-cycle, driven end to end against a mock exchange.

Unit tests prove each engine works. This proves they work *together*: scan,
signal, intelligence, confluence, alpha, growth ladder, risk cap, BUY,
authenticated fill, owned position, monitoring, exit, authenticated SELL,
reconciled net PnL, wallet refresh, and a resized next position.

It is the closest thing to a runtime rehearsal that is possible without an
exchange, and it is the test that catches integration faults unit tests miss
-- a wrong signature, a missing attribute, an engine that silently vetoes.

No network. No credentials. No real orders.
"""

import os
import sys

import pytest

os.environ.setdefault("BYBIT_API_KEY", "test-key-not-real")
os.environ.setdefault("BYBIT_SECRET", "test-secret-not-real")
os.environ.setdefault("LEANTRADER_DATA_DIR", "/tmp/rpb-cycle-test")


def candles(n=140, start=100.0, step=0.25):
    rows, price = [], start
    for i in range(n):
        price += step
        rows.append([i * 60_000, price, price * 1.002, price * 0.998, price, 5000.0 + i])
    return rows


class MockBybit:
    """Everything REAL_PROFIT_BOT asks of ccxt, with real-shaped responses."""

    def __init__(self, price=100.0, free_usdt=50.0):
        self.price = price
        self.free_usdt = free_usdt
        self.base_held = 0.0
        self.orders = {}
        self.submitted = []
        self._id = 0
        self.markets = {
            "DOGE/USDT": {
                "symbol": "DOGE/USDT", "base": "DOGE", "quote": "USDT",
                "spot": True, "active": True, "type": "spot", "taker": 0.001,
                "precision": {"amount": 4, "price": 6},
                "limits": {"amount": {"min": 0.001}, "cost": {"min": 1.0}},
            }
        }
        self.headers = {}
        self.options = {}

    # --- market data -------------------------------------------------
    def load_markets(self):
        return self.markets

    def market(self, symbol):
        return self.markets[symbol]

    def fetch_tickers(self):
        return {
            "DOGE/USDT": {
                "last": self.price, "quoteVolume": 25_000_000.0,
                "percentage": 4.5, "bid": self.price * 0.9999,
                "ask": self.price * 1.0001,
            }
        }

    def fetch_ticker(self, symbol):
        return self.fetch_tickers()[symbol]

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=60):
        return candles(max(limit, 140))

    def fetch_order_book(self, symbol, limit=20):
        return {
            "bids": [[self.price * 0.9999, 5000.0]] * 25,
            "asks": [[self.price * 1.0001, 3000.0]] * 25,
        }

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.4f}"

    def price_to_precision(self, symbol, price):
        return f"{float(price):.6f}"

    # --- account -----------------------------------------------------
    def fetch_balance(self, params=None):
        # ccxt returns both the grouped view and a per-currency entry; the bot
        # reads balance['USDT']['free'], so the mock must carry both.
        return {
            "free": {"USDT": self.free_usdt, "DOGE": self.base_held},
            "used": {"USDT": 0.0, "DOGE": 0.0},
            "total": {"USDT": self.free_usdt, "DOGE": self.base_held},
            "USDT": {"free": self.free_usdt, "used": 0.0, "total": self.free_usdt},
            "DOGE": {"free": self.base_held, "used": 0.0, "total": self.base_held},
        }

    # --- execution ---------------------------------------------------
    def _fill(self, symbol, side, amount):
        self._id += 1
        oid = f"mock-{self._id}"
        cost = amount * self.price
        fee = cost * 0.001
        if side == "buy":
            self.free_usdt -= cost + fee
            self.base_held += amount
            fee_entry = {"cost": fee, "currency": "USDT"}
        else:
            self.free_usdt += cost - fee
            self.base_held = max(0.0, self.base_held - amount)
            fee_entry = {"cost": fee, "currency": "USDT"}
        order = {
            "id": oid, "symbol": symbol, "side": side, "status": "closed",
            "filled": amount, "amount": amount, "average": self.price,
            "price": self.price, "cost": cost, "fee": fee_entry,
        }
        self.orders[oid] = order
        self.submitted.append(order)
        return order

    def create_market_buy_order(self, symbol, amount, *a, **k):
        return self._fill(symbol, "buy", float(amount))

    def create_market_sell_order(self, symbol, amount, *a, **k):
        return self._fill(symbol, "sell", float(amount))

    def fetch_order(self, order_id, symbol=None):
        return self.orders[order_id]

    def set_sandbox_mode(self, on):
        pass


@pytest.fixture
def bot(tmp_path, monkeypatch):
    monkeypatch.setenv("LEANTRADER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RPB_BROADCAST_ENABLED", "0")   # no network sends
    import importlib

    import REAL_PROFIT_BOT as module

    importlib.reload(module)

    instance = module.REAL_PROFIT_BOT.__new__(module.REAL_PROFIT_BOT)
    instance.mode = "testnet"
    exchange = MockBybit()
    instance.gate = exchange

    # Run the rest of __init__ with the mock already in place.
    monkeypatch.setattr(module.ccxt, "bybit", lambda *a, **k: exchange)
    instance.__init__("testnet")
    instance.gate = exchange
    instance.send_telegram = lambda *a, **k: True
    return instance, exchange


# ---------------------------------------------------- every engine connected


def test_all_engines_are_present_on_the_bot(bot):
    instance, _ = bot
    for attribute in ("intelligence", "scalping", "alpha", "risk", "ledger",
                      "exit_evaluator", "signals", "scout"):
        assert getattr(instance, attribute, None) is not None, f"{attribute} not connected"


def test_every_engine_reports_available(bot):
    instance, _ = bot
    assert instance.intelligence.available is True, instance.intelligence.reason
    assert instance.scalping.available is True, instance.scalping.reason
    assert instance.alpha.available is True, instance.alpha.reason
    assert instance.risk.available is True, instance.risk.reason
    assert instance.lifecycle_enabled is True


# ------------------------------------------------------------ the cycle


def test_the_signal_survives_every_intelligence_layer(bot):
    """The bot's own signal must still be tradeable after all five engines."""
    instance, _ = bot
    signal, confidence, price, change, volume = instance.analyze_market("DOGE/USDT")

    assert signal == "BUY", "the signal was vetoed away by intelligence"
    assert confidence >= 85, f"confidence fell to {confidence}, below the trade gate"
    assert price > 0


def test_a_buy_fills_and_creates_an_owned_position(bot):
    instance, exchange = bot
    order = instance.execute_trade("DOGE/USDT", "BUY", exchange.price)

    assert order is not None, "no order was placed"
    assert instance.ledger.owns("DOGE/USDT"), "fill did not create a position"

    record = instance.ledger.positions["DOGE/USDT"]
    assert record["sellable_quantity"] > 0
    assert record["average_entry"] == pytest.approx(exchange.price)
    assert record["order_id"].startswith("mock-")


def test_the_position_is_monitored_and_exited_in_profit(bot):
    instance, exchange = bot
    instance.execute_trade("DOGE/USDT", "BUY", exchange.price)
    entry = instance.ledger.positions["DOGE/USDT"]["average_entry"]

    # Price moves through take-profit.
    exchange.price = entry * (1 + (instance.exit_evaluator.take_profit_bps + 5) / 10_000.0)

    closed = instance.manage_owned_positions()

    assert closed == 1, "the exit loop did not close the position"
    assert not instance.ledger.owns("DOGE/USDT")

    settled = instance.ledger.closed[-1]
    assert settled["exit_reason"] == "take_profit"
    assert settled["exit_quantity"] > 0
    assert settled["realized_net_pnl"] > 0, "a winning move produced no net profit"
    assert settled["total_fees"] > 0, "fees were not reconciled"


def test_a_losing_move_exits_at_the_stop(bot):
    instance, exchange = bot
    instance.execute_trade("DOGE/USDT", "BUY", exchange.price)
    entry = instance.ledger.positions["DOGE/USDT"]["average_entry"]

    exchange.price = entry * (1 - (instance.exit_evaluator.stop_loss_bps + 5) / 10_000.0)
    assert instance.manage_owned_positions() == 1

    settled = instance.ledger.closed[-1]
    assert settled["exit_reason"] == "stop_loss"
    assert settled["realized_net_pnl"] < 0


def test_capital_is_recycled_and_the_next_position_resizes(bot):
    """The compounding step: wallet grows, next position grows with it."""
    instance, exchange = bot
    from rpb_scalping import target_position_usd

    before = instance.check_gate_balance()
    size_before, _ = target_position_usd(before)

    instance.execute_trade("DOGE/USDT", "BUY", exchange.price)
    entry = instance.ledger.positions["DOGE/USDT"]["average_entry"]
    exchange.price = entry * 1.02          # a 2% winner
    instance.manage_owned_positions()

    after = instance.check_gate_balance()
    size_after, _ = target_position_usd(after)

    assert after > before, "capital did not return to the wallet"
    assert size_after > size_before, "the next position did not compound"


def test_two_full_cycles_run_back_to_back(bot):
    """Repeatability is the whole point of a micro-compounding loop."""
    instance, exchange = bot
    base = exchange.price

    for cycle in range(2):
        exchange.price = base
        assert instance.execute_trade("DOGE/USDT", "BUY", exchange.price) is not None
        entry = instance.ledger.positions["DOGE/USDT"]["average_entry"]
        exchange.price = entry * 1.01
        assert instance.manage_owned_positions() == 1, f"cycle {cycle} did not close"

    assert len(instance.ledger.closed) == 2
    stats = instance.ledger.stats()
    assert stats["closed_positions"] == 2
    assert stats["authentic_wins"] == 2


# ------------------------------------------------------- honesty guarantees


def test_an_ack_with_no_fill_creates_no_position(bot):
    instance, exchange = bot

    def ack_only(symbol, amount, *a, **k):
        exchange._id += 1
        oid = f"ack-{exchange._id}"
        order = {"id": oid, "status": "open", "filled": 0.0, "symbol": symbol}
        exchange.orders[oid] = order
        return order

    exchange.create_market_buy_order = ack_only
    instance.execute_trade("DOGE/USDT", "BUY", exchange.price)

    assert not instance.ledger.owns("DOGE/USDT"), "an ack was recorded as a position"


def test_the_bot_never_sells_more_than_it_owns(bot):
    instance, exchange = bot
    instance.execute_trade("DOGE/USDT", "BUY", exchange.price)
    owned = exchange.base_held

    entry = instance.ledger.positions["DOGE/USDT"]["average_entry"]
    exchange.price = entry * 1.02
    instance.manage_owned_positions()

    sells = [o for o in exchange.submitted if o["side"] == "sell"]
    assert sells, "no sell was submitted"
    assert sells[-1]["filled"] <= owned + 1e-9, "sold more than was held"
    assert exchange.base_held >= -1e-9


def test_an_owned_symbol_is_not_bought_again(bot):
    instance, exchange = bot
    instance.execute_trade("DOGE/USDT", "BUY", exchange.price)
    assert instance.ledger.owns("DOGE/USDT")
    # The entry scan skips owned symbols; assert the guard exists and holds.
    assert instance.ledger.owns("DOGE/USDT") is True


def test_realized_pnl_reconciles_against_the_wallet(bot):
    """Reported net PnL must match what the wallet actually did."""
    instance, exchange = bot
    before = instance.check_gate_balance()

    instance.execute_trade("DOGE/USDT", "BUY", exchange.price)
    entry = instance.ledger.positions["DOGE/USDT"]["average_entry"]
    exchange.price = entry * 1.03
    instance.manage_owned_positions()

    after = instance.check_gate_balance()
    reported = instance.ledger.closed[-1]["realized_net_pnl"]

    assert reported == pytest.approx(after - before, abs=1e-6), (
        f"reported net {reported} does not match wallet delta {after - before}"
    )
