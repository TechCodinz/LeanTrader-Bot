"""Deterministic tests for the signal -> authenticated order pipeline.

Nothing here reaches the network or reads real credentials. A stub broker
stands in for BrokerCCXT with the same surface preflight uses, so every
blocker class can be produced on demand and asserted on.

The regression these protect against is specific: before this, a router
refusal was a truthy dict, so ExecutionOrchestrator recorded a position,
published a trade record and logged "TRADE EXECUTED" for orders that were
never placed. Those trade records then fed the learning path.
"""

import asyncio
import json
import os

import pytest

from src.leantrader.execution import preflight
from src.leantrader.execution.preflight import (
    Blocked,
    PreparedOrder,
    SizingPolicy,
    classify_receipt,
    normalize_symbol,
    prepare_order,
    size_order,
)

SYMBOL = "BTC/USDT"
PRICE = 64_000.0
FREE_USDT = 13.956  # the account this was built against


def spot_market(**overrides):
    market = {
        "symbol": SYMBOL,
        "base": "BTC",
        "quote": "USDT",
        "spot": True,
        "active": True,
        "taker": 0.001,
        "limits": {
            "amount": {"min": 0.000048},
            "cost": {"min": 5.0},
        },
    }
    market.update(overrides)
    return market


class StubExchange:
    """Just the two precision helpers preflight calls on a ccxt client."""

    def __init__(self, amount_step=8, price_step=2):
        self.amount_step = amount_step
        self.price_step = price_step

    def amount_to_precision(self, symbol, amount):
        factor = 10 ** self.amount_step
        return str(int(float(amount) * factor) / factor)

    def price_to_precision(self, symbol, price):
        factor = 10 ** self.price_step
        return str(int(float(price) * factor) / factor)


class StubBroker:
    def __init__(
        self,
        authority="testnet",
        markets=None,
        balance=None,
        ticker=None,
        exchange=None,
        market_mode="spot",
        markets_error=None,
        balance_error=None,
        ticker_error=None,
    ):
        self._authority = authority
        self.exchange_id = "bybit"
        self.market_mode = market_mode
        self._markets = {SYMBOL: spot_market()} if markets is None else markets
        self._balance = (
            {"free": {"USDT": FREE_USDT, "BTC": 0.0}} if balance is None else balance
        )
        self._ticker = {"last": PRICE} if ticker is None else ticker
        self._exchange = StubExchange() if exchange is None else exchange
        self._markets_error = markets_error
        self._balance_error = balance_error
        self._ticker_error = ticker_error

    @property
    def authority(self):
        return self._authority

    def resolve_mode(self):
        return "testnet" if self._authority == "testnet" else self._authority

    def load_markets(self):
        if self._markets_error:
            raise self._markets_error
        return self._markets

    def fetch_balance(self):
        if self._balance_error:
            raise self._balance_error
        return self._balance

    def fetch_ticker(self, symbol):
        if self._ticker_error:
            raise self._ticker_error
        return self._ticker

    def _make_exchange(self, environment, authenticated):
        return self._exchange


@pytest.fixture(autouse=True)
def _isolated_telemetry(tmp_path, monkeypatch):
    """Counters go to a scratch file; caches start empty for every test."""
    monkeypatch.setenv(
        "EXECUTION_TELEMETRY_PATH", str(tmp_path / "execution_telemetry.json")
    )
    for name in (
        "EXECUTION_ALLOCATION_FRACTION",
        "EXECUTION_MAX_FRACTION",
        "EXECUTION_FEE_RESERVE_MULTIPLE",
        "EXECUTION_RESERVE_QUOTE",
        "EXECUTION_EXCHANGE_OVERRIDE",
    ):
        monkeypatch.delenv(name, raising=False)
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    yield
    preflight.reset_caches()
    preflight.reset_shared_brokers()


def intent(**overrides):
    payload = {
        "symbol": SYMBOL,
        "side": "buy",
        "price": PRICE,
        "confidence": 0.85,
        "order_type": "market",
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------ normalization


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BTC/USDT", "BTC/USDT"),
        ("btcusdt", "BTC/USDT"),
        ("BTC-USDT", "BTC/USDT"),
        ("btc_usdt", "BTC/USDT"),
        ("BTC/USDT:USDT", "BTC/USDT"),
        (" eth/usdt ", "ETH/USDT"),
    ],
)
def test_symbol_spellings_the_engines_emit_all_normalize(raw, expected):
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "GARBAGE", "A/B/C", "/USDT", "BTC/"])
def test_unparseable_symbols_are_rejected_not_guessed(raw):
    assert normalize_symbol(raw) == ""


# ------------------------------------------------------------------ sizing


def test_small_balance_is_raised_to_the_venue_minimum_when_it_can_fund_it():
    notional, reason = size_order(
        free_quote=FREE_USDT,
        price=PRICE,
        min_notional=5.0,
        min_amount=0.000048,
        fee_rate=0.001,
        confidence=0.0,
    )
    assert notional == pytest.approx(5.0)
    assert "minimum" in reason
    assert notional <= FREE_USDT


def test_confidence_moves_size_inside_the_configured_band():
    low, _ = size_order(FREE_USDT, PRICE, 5.0, 0.000048, 0.001, confidence=0.10)
    high, _ = size_order(FREE_USDT, PRICE, 5.0, 0.000048, 0.001, confidence=0.95)

    assert low < high
    policy = SizingPolicy()
    assert high <= FREE_USDT * policy.max_fraction


def test_size_never_exceeds_the_balance_after_the_fee_reserve():
    policy = SizingPolicy(allocation_fraction=1.0, max_fraction=1.0)
    notional, reason = size_order(
        FREE_USDT, PRICE, 0.0, 0.0, 0.001, confidence=1.0, policy=policy
    )
    fee_reserve = FREE_USDT * 0.001 * policy.fee_reserve_multiple
    assert notional == pytest.approx(FREE_USDT - fee_reserve)
    assert notional < FREE_USDT


def test_balance_below_the_venue_minimum_refuses_rather_than_undersizing():
    notional, reason = size_order(
        free_quote=3.0,
        price=PRICE,
        min_notional=5.0,
        min_amount=0.000048,
        fee_rate=0.001,
        confidence=0.95,
    )
    assert notional == 0.0
    assert "minimum" in reason


def test_a_min_amount_floor_is_honoured_when_it_exceeds_min_notional():
    # 1.0 base unit at $20 is a $20 floor even though min cost says $5.
    notional, reason = size_order(
        free_quote=100.0,
        price=20.0,
        min_notional=5.0,
        min_amount=1.0,
        fee_rate=0.001,
        confidence=0.0,
    )
    assert notional >= 20.0


def test_zero_balance_or_price_sizes_nothing():
    assert size_order(0.0, PRICE, 5.0, 0.0, 0.001)[0] == 0.0
    assert size_order(FREE_USDT, 0.0, 5.0, 0.0, 0.001)[0] == 0.0


# --------------------------------------------------------------- preflight


def test_a_fundable_buy_becomes_a_prepared_order():
    prepared, blocked = prepare_order(intent(), broker=StubBroker())

    assert blocked is None
    assert isinstance(prepared, PreparedOrder)
    assert prepared.symbol == SYMBOL
    assert prepared.side == "buy"
    assert prepared.amount > 0
    assert prepared.notional >= prepared.min_notional
    assert prepared.notional <= FREE_USDT
    assert prepared.execution_mode == "testnet"
    assert prepared.exchange_id == "bybit"

    payload = prepared.to_payload()
    assert payload["backend"] == "ccxt"
    assert payload["qty"] == prepared.amount
    assert payload["execution_mode"] == "testnet"
    # A market order carries a reference price, never a limit price.
    assert "price" not in payload
    assert payload["reference_price"] == prepared.price


@pytest.mark.parametrize("side", ["", "hold", "close", None])
def test_a_non_directional_intent_blocks_as_invalid(side):
    prepared, blocked = prepare_order(intent(side=side), broker=StubBroker())
    assert prepared is None
    assert blocked.blocker == preflight.INVALID_INTENT


def test_an_unparseable_symbol_blocks_before_any_venue_call():
    prepared, blocked = prepare_order(intent(symbol="???"), broker=StubBroker())
    assert prepared is None
    assert blocked.blocker == preflight.SYMBOL_NOT_NORMALIZED


def test_without_any_execution_authority_nothing_is_prepared():
    prepared, blocked = prepare_order(intent(), broker=StubBroker(authority="none"))
    assert prepared is None
    assert blocked.blocker == preflight.NO_EXECUTION_AUTHORITY


def test_paper_sizes_against_declared_equity_not_a_discovered_balance(monkeypatch):
    """Paper has no account to read, so its equity is configuration.

    It still goes through the same market, minimum-notional and precision
    checks, so a paper run exercises the same path an authenticated one does.
    """
    monkeypatch.setenv("PAPER_EQUITY_QUOTE", "250")

    broker = StubBroker(authority="paper", balance_error=AssertionError("no fetch"))
    prepared, blocked = prepare_order(intent(), broker=broker)

    assert blocked is None
    assert prepared.free_quote == pytest.approx(250.0)
    assert prepared.notional <= 250.0
    assert prepared.meta["authority"] == "paper"


def test_a_paper_equity_of_zero_blocks_rather_than_sizing_from_nothing(monkeypatch):
    monkeypatch.setenv("PAPER_EQUITY_QUOTE", "0")
    prepared, blocked = prepare_order(
        intent(), broker=StubBroker(authority="paper")
    )
    assert prepared is None
    assert blocked.blocker == preflight.INSUFFICIENT_FREE_BALANCE


def test_an_authenticated_run_never_falls_back_to_paper_equity(monkeypatch):
    """The declared paper number must not stand in for a real balance."""
    monkeypatch.setenv("PAPER_EQUITY_QUOTE", "100000")
    broker = StubBroker(authority="testnet", balance={"free": {"USDT": 0.0}})
    prepared, blocked = prepare_order(intent(), broker=broker)

    assert prepared is None
    assert blocked.blocker == preflight.INSUFFICIENT_FREE_BALANCE


def test_a_symbol_the_venue_does_not_list_blocks():
    prepared, blocked = prepare_order(
        intent(symbol="NOPE/USDT"), broker=StubBroker()
    )
    assert prepared is None
    assert blocked.blocker == preflight.MARKET_NOT_LISTED


def test_a_derivative_market_blocks_when_the_runtime_trades_spot():
    broker = StubBroker(markets={SYMBOL: spot_market(spot=False, swap=True)})
    prepared, blocked = prepare_order(intent(), broker=broker)
    assert prepared is None
    assert blocked.blocker == preflight.MARKET_NOT_SPOT


def test_a_delisted_market_blocks():
    broker = StubBroker(markets={SYMBOL: spot_market(active=False)})
    prepared, blocked = prepare_order(intent(), broker=broker)
    assert prepared is None
    assert blocked.blocker == preflight.MARKET_INACTIVE


def test_unreadable_market_metadata_blocks_rather_than_assuming_defaults():
    broker = StubBroker(markets_error=RuntimeError("boom"))
    prepared, blocked = prepare_order(intent(), broker=broker)
    assert prepared is None
    assert blocked.blocker == preflight.MARKET_METADATA_UNAVAILABLE


def test_no_price_anywhere_blocks_rather_than_inventing_one():
    broker = StubBroker(ticker={})
    prepared, blocked = prepare_order(
        intent(price=None), broker=broker
    )
    assert prepared is None
    assert blocked.blocker == preflight.REFERENCE_PRICE_UNAVAILABLE


def test_price_is_taken_from_the_ticker_when_the_signal_carries_none():
    broker = StubBroker(ticker={"last": 61_000.0})
    prepared, blocked = prepare_order(intent(price=None), broker=broker)
    assert blocked is None
    assert prepared.price == pytest.approx(61_000.0)


def test_an_unreadable_balance_blocks():
    broker = StubBroker(balance_error=RuntimeError("auth"))
    prepared, blocked = prepare_order(intent(), broker=broker)
    assert prepared is None
    assert blocked.blocker == preflight.BALANCE_UNAVAILABLE


def test_an_empty_free_balance_blocks():
    broker = StubBroker(balance={"free": {"USDT": 0.0}})
    prepared, blocked = prepare_order(intent(), broker=broker)
    assert prepared is None
    assert blocked.blocker == preflight.INSUFFICIENT_FREE_BALANCE


def test_a_balance_under_the_venue_minimum_blocks_on_min_notional():
    broker = StubBroker(balance={"free": {"USDT": 2.0}})
    prepared, blocked = prepare_order(intent(), broker=broker)
    assert prepared is None
    assert blocked.blocker == preflight.BELOW_MIN_NOTIONAL


def test_a_coarse_amount_step_that_rounds_the_order_away_blocks():
    # Whole-unit amount precision on a $64k asset: 0.0001 BTC rounds to 0.
    broker = StubBroker(exchange=StubExchange(amount_step=0))
    prepared, blocked = prepare_order(intent(), broker=broker)
    assert prepared is None
    assert blocked.blocker == preflight.PRECISION_COLLAPSED_TO_ZERO


def test_rounding_below_the_venue_min_amount_blocks():
    # Step of 4 decimals rounds 0.0000978 BTC down to 0.0000, caught as a
    # collapse; a step of 5 leaves 0.00009 which is above min. Use a market
    # whose min_amount sits above what the step can express.
    broker = StubBroker(
        markets={SYMBOL: spot_market(limits={
            "amount": {"min": 0.01},
            "cost": {"min": 5.0},
        })},
    )
    prepared, blocked = prepare_order(intent(), broker=broker)
    assert prepared is None
    # $13.95 cannot buy 0.01 BTC, so this is refused on funding, not rounding.
    assert blocked.blocker in {
        preflight.BELOW_MIN_NOTIONAL,
        preflight.BELOW_MIN_AMOUNT,
    }


def test_a_sell_is_checked_against_the_base_balance_not_the_quote():
    broker = StubBroker(balance={"free": {"USDT": 500.0, "BTC": 0.0}})
    prepared, blocked = prepare_order(intent(side="sell"), broker=broker)
    assert prepared is None
    assert blocked.blocker == preflight.INSUFFICIENT_FREE_BALANCE


def test_a_sell_with_base_inventory_prepares():
    broker = StubBroker(balance={"free": {"USDT": 0.0, "BTC": 0.01}})
    prepared, blocked = prepare_order(intent(side="sell"), broker=broker)
    assert blocked is None
    assert prepared.side == "sell"
    assert prepared.amount > 0


# --------------------------------------------------------- receipt reading


def test_an_acknowledged_order_with_an_id_is_not_a_blocker():
    receipt = {
        "ok": True,
        "executed": True,
        "order": {"id": "1234", "status": "closed"},
    }
    assert classify_receipt(receipt) is None


def test_a_refusal_is_classified_even_though_the_dict_is_truthy():
    receipt = {"ok": False, "error": "no_authenticated_execution_authority"}
    assert bool(receipt) is True, "the bug was reading this dict for truth"
    assert classify_receipt(receipt) == preflight.NO_EXECUTION_AUTHORITY


def test_an_exchange_error_is_classified_as_an_exchange_reject():
    receipt = {"ok": False, "error": "bybit InsufficientFunds"}
    assert classify_receipt(receipt) == preflight.EXCHANGE_REJECT


def test_an_ok_receipt_without_an_order_id_is_not_an_acknowledgement():
    receipt = {"ok": True, "executed": True, "order": {"status": "open"}}
    assert classify_receipt(receipt) == preflight.NO_ORDER_ID


def test_an_ok_receipt_that_did_not_execute_is_a_refusal():
    receipt = {"ok": True, "executed": False, "order": {"id": "1"}}
    assert classify_receipt(receipt) == preflight.ROUTER_REFUSED


@pytest.mark.parametrize("receipt", [None, "", [], 0])
def test_a_non_receipt_is_a_refusal(receipt):
    assert classify_receipt(receipt) == preflight.ROUTER_REFUSED


# ---------------------------------------------------------------- counters


def test_blockers_are_counted_and_survive_a_reread():
    prepare_order(intent(side="hold"), broker=StubBroker())
    prepare_order(intent(side="hold"), broker=StubBroker())
    prepare_order(intent(symbol="???"), broker=StubBroker())

    snapshot = preflight.telemetry_snapshot()
    assert snapshot["blockers"][preflight.INVALID_INTENT]["count"] == 2
    assert snapshot["blockers"][preflight.SYMBOL_NOT_NORMALIZED]["count"] == 1
    assert snapshot["attempts"] == 3
    assert snapshot["prepared"] == 0

    on_disk = json.loads(
        open(os.environ["EXECUTION_TELEMETRY_PATH"], encoding="utf-8").read()
    )
    assert on_disk["blockers"][preflight.INVALID_INTENT]["count"] == 2


def test_a_prepared_order_is_counted_as_prepared():
    prepare_order(intent(), broker=StubBroker())
    snapshot = preflight.telemetry_snapshot()
    assert snapshot["attempts"] == 1
    assert snapshot["prepared"] == 1


def test_blocker_details_carry_no_credential_material():
    prepare_order(intent(), broker=StubBroker(balance_error=RuntimeError("apikey=abc123 secret=xyz")))
    snapshot = preflight.telemetry_snapshot()
    detail = snapshot["blockers"][preflight.BALANCE_UNAVAILABLE].get("last_detail", "")
    # Only the exception class name is recorded, never its message.
    assert detail == "RuntimeError"


def test_every_declared_blocker_class_is_a_unique_upper_snake_name():
    assert len(set(preflight.BLOCKER_CLASSES)) == len(preflight.BLOCKER_CLASSES)
    for name in preflight.BLOCKER_CLASSES:
        assert name == name.upper()
        assert " " not in name


# --------------------------------------------- orchestrator regression tests


class RecordingHub:
    def __init__(self):
        self.trades = []

    async def publish_trade(self, trade):
        self.trades.append(trade)


def build_orchestrator():
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator

    hub = RecordingHub()
    orchestrator = ExecutionOrchestrator(
        data_hub=hub,
        trading_engines={},
        risk_engine=None,
        ledger=None,
        mode="testnet",
    )
    return orchestrator, hub


def test_a_refused_order_publishes_no_trade_and_records_no_position(monkeypatch):
    """The core regression: refusals used to be recorded as fills."""
    orchestrator, hub = build_orchestrator()

    monkeypatch.setattr(
        preflight,
        "prepare_order",
        lambda i, **kw: (
            PreparedOrder(
                symbol=SYMBOL,
                side="buy",
                amount=0.0001,
                price=PRICE,
                order_type="market",
                exchange_id="bybit",
                execution_mode="testnet",
                notional=6.4,
                quote_currency="USDT",
                free_quote=FREE_USDT,
                min_notional=5.0,
                min_amount=0.000048,
                fee_rate=0.001,
            ),
            None,
        ),
    )

    import EXECUTION_ORCHESTRATOR as module

    monkeypatch.setattr(
        module,
        "route_order",
        lambda payload: {
            "ok": False,
            "error": "no_authenticated_execution_authority",
        },
    )

    result = asyncio.run(
        orchestrator.execute_trade(SYMBOL, "buy", 0.9, {"symbol": SYMBOL})
    )

    assert result["ok"] is False
    assert result["blocker"] == preflight.NO_EXECUTION_AUTHORITY
    assert hub.trades == [], "a refusal must not become a trade record"
    assert orchestrator.risk_manager.open_positions == {}
    assert orchestrator.total_trades == 0


def test_an_acknowledged_order_publishes_a_trade_carrying_the_order_id(monkeypatch):
    orchestrator, hub = build_orchestrator()

    monkeypatch.setattr(
        preflight,
        "prepare_order",
        lambda i, **kw: (
            PreparedOrder(
                symbol=SYMBOL,
                side="buy",
                amount=0.0001,
                price=PRICE,
                order_type="market",
                exchange_id="bybit",
                execution_mode="testnet",
                notional=6.4,
                quote_currency="USDT",
                free_quote=FREE_USDT,
                min_notional=5.0,
                min_amount=0.000048,
                fee_rate=0.001,
            ),
            None,
        ),
    )

    import EXECUTION_ORCHESTRATOR as module

    monkeypatch.setattr(
        module,
        "route_order",
        lambda payload: {
            "ok": True,
            "executed": True,
            "exchange": "bybit",
            "execution_mode": "testnet",
            "authority": "testnet",
            "order": {
                "id": "ORD-1",
                "status": "closed",
                "filled": 0.0001,
                "average": 64_010.0,
            },
        },
    )

    result = asyncio.run(
        orchestrator.execute_trade(SYMBOL, "buy", 0.9, {"symbol": SYMBOL})
    )

    assert result["ok"] is True
    assert len(hub.trades) == 1

    trade = hub.trades[0]
    assert trade["order_id"] == "ORD-1"
    assert trade["entry_price"] == 64_010.0
    assert trade["execution_mode"] == "testnet"
    assert trade["realized_pnl"] is None, "nothing is realized until it closes"
    assert SYMBOL in orchestrator.risk_manager.open_positions


def test_a_blocked_preflight_never_reaches_the_router(monkeypatch):
    orchestrator, hub = build_orchestrator()

    monkeypatch.setattr(
        preflight,
        "prepare_order",
        lambda i, **kw: (
            None,
            Blocked(preflight.BELOW_MIN_NOTIONAL, "too small", "sizing"),
        ),
    )

    import EXECUTION_ORCHESTRATOR as module

    def _must_not_run(payload):
        raise AssertionError("route_order called for a blocked intent")

    monkeypatch.setattr(module, "route_order", _must_not_run)

    result = asyncio.run(
        orchestrator.execute_trade(SYMBOL, "buy", 0.9, {"symbol": SYMBOL})
    )

    assert result["blocker"] == preflight.BELOW_MIN_NOTIONAL
    assert hub.trades == []


def test_a_refused_close_leaves_the_position_open_and_books_no_pnl(monkeypatch):
    orchestrator, hub = build_orchestrator()
    orchestrator.risk_manager.record_position(SYMBOL, "buy", 0.0001, 64_000.0)

    import EXECUTION_ORCHESTRATOR as module

    monkeypatch.setattr(
        module,
        "route_order",
        lambda payload: {"ok": False, "error": "bybit InsufficientFunds"},
    )

    result = asyncio.run(
        orchestrator.close_position(SYMBOL, 65_000.0, "take_profit")
    )

    assert result is None
    assert SYMBOL in orchestrator.risk_manager.open_positions
    assert hub.trades == []
    assert orchestrator.total_profit == 0.0


def test_a_filled_close_books_pnl_net_of_fees(monkeypatch):
    orchestrator, hub = build_orchestrator()
    orchestrator.risk_manager.record_position(SYMBOL, "buy", 0.001, 64_000.0)

    import EXECUTION_ORCHESTRATOR as module

    monkeypatch.setattr(
        module,
        "route_order",
        lambda payload: {
            "ok": True,
            "executed": True,
            "exchange": "bybit",
            "execution_mode": "testnet",
            "order": {
                "id": "CLOSE-1",
                "status": "closed",
                "average": 65_000.0,
                "fee": {"cost": 0.065, "currency": "USDT"},
            },
        },
    )
    monkeypatch.setattr(preflight, "shared_broker", lambda *a, **k: StubBroker())
    monkeypatch.setattr(
        preflight, "load_markets_cached", lambda broker: {SYMBOL: spot_market()}
    )

    record = asyncio.run(
        orchestrator.close_position(SYMBOL, 65_000.0, "take_profit")
    )

    assert record is not None
    assert record["order_id"] == "CLOSE-1"
    assert record["exit_price"] == 65_000.0
    # 0.001 BTC from 64000 to 65000 is $1.00 gross.
    assert record["gross_pnl"] == pytest.approx(1.0)
    assert record["fees"] > 0
    assert record["realized_pnl"] == pytest.approx(
        record["gross_pnl"] - record["fees"]
    )
    assert record["realized_pnl"] < record["gross_pnl"]
    assert SYMBOL not in orchestrator.risk_manager.open_positions


def test_an_acknowledged_close_without_a_fill_price_books_nothing(monkeypatch):
    orchestrator, hub = build_orchestrator()
    orchestrator.risk_manager.record_position(SYMBOL, "buy", 0.001, 64_000.0)

    import EXECUTION_ORCHESTRATOR as module

    monkeypatch.setattr(
        module,
        "route_order",
        lambda payload: {
            "ok": True,
            "executed": True,
            "order": {"id": "CLOSE-2", "status": "open"},
        },
    )

    record = asyncio.run(
        orchestrator.close_position(SYMBOL, 65_000.0, "stop_loss")
    )

    assert record is None
    assert SYMBOL in orchestrator.risk_manager.open_positions
    assert hub.trades == []
