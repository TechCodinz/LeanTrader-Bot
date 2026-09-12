"""Timeframes, data venues, minimum tickets, inventory and intent identity.

Five things the runtime was getting wrong, each with a different shape:

  internal M1/M5/M15/M30 reaching Bybit unchanged -> "Invalid period!"
  forex symbols priced off a crypto venue because it happens to execute
  a 0.919 USDT ticket against a 1.00 minimum, rediscovered every cycle
  ETH and SOL held on Testnet with no owner and no exit path
  attempts=42 / prepared=1 / submitted=0 / acknowledged=1

No venue is contacted and no order is submitted anywhere in this module.
"""

import time

import pytest

from src.leantrader.execution import intent as execution_intent
from src.leantrader.execution import inventory as inventory_reconciler
from src.leantrader.execution import preflight
from src.leantrader.universe import instruments, routing
from src.leantrader.universe.timeframes import (
    TIMEFRAME_NOT_SUPPORTED,
    aggregate_ohlcv,
    canonical_timeframe,
    plan_timeframe,
    venue_timeframes,
)
from src.leantrader.universe.venues import CapabilityRegistry, capabilities

BYBIT_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"]
ON_BYBIT = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT")


def market(symbol, active=True, min_cost=5.0, min_amount=0.0001):
    return {
        "symbol": symbol,
        "spot": True,
        "active": active,
        "taker": 0.001,
        "limits": {"amount": {"min": min_amount}, "cost": {"min": min_cost}},
        "precision": {"amount": 8, "price": 2},
    }


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("EXECUTION_TELEMETRY_PATH", str(tmp_path / "t.json"))
    monkeypatch.setenv("UNIVERSE_SNAPSHOT_PATH", str(tmp_path / "snap.json"))
    capabilities.reset()
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    execution_intent.reset_outcomes()
    yield
    capabilities.reset()
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    execution_intent.reset_outcomes()


class RecordingVenue:
    """Records every request, so "did not call" is provable."""

    def __init__(self, listed=ON_BYBIT, timeframes=None):
        self.listed = set(listed)
        self.timeframes = {t: t for t in (timeframes or BYBIT_TIMEFRAMES)}
        self.ohlcv_requests = []
        self.ticker_requests = []
        self.orders = []

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=100):
        self.ohlcv_requests.append((symbol, timeframe))
        if symbol not in self.listed:
            raise ValueError(f"{symbol} is not a listed market")
        if timeframe not in self.timeframes:
            raise ValueError("Invalid period!")
        return [[i, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(limit)]

    def fetch_ticker(self, symbol):
        self.ticker_requests.append(symbol)
        if symbol not in self.listed:
            raise ValueError(f"{symbol} is not a listed market")
        return {"last": 100.0}

    def create_order(self, *args, **kwargs):
        self.orders.append(args)
        raise AssertionError("an order reached a venue during the test suite")


# --------------------------------------------- 1-3. timeframe normalization


@pytest.mark.parametrize(
    "internal,expected",
    [
        ("M1", "1m"), ("M5", "5m"), ("M15", "15m"), ("M30", "30m"),
        ("H1", "1h"), ("H4", "4h"), ("D1", "1d"), ("W1", "1w"),
        ("m1", "1m"), ("h4", "4h"), ("1min", "1m"), ("60", "1h"),
        ("1m", "1m"), ("4h", "4h"), ("1d", "1d"),
    ],
)
def test_internal_labels_normalize_to_canonical_intervals(internal, expected):
    assert canonical_timeframe(internal) == expected


def test_one_month_is_not_folded_into_one_minute():
    """ccxt distinguishes 1M from 1m by case, and so must this."""
    assert canonical_timeframe("1M") == "1M"
    assert canonical_timeframe("1m") == "1m"


@pytest.mark.parametrize("raw", ["", None, "bogus", "M", "xyz42"])
def test_an_unrecognised_label_is_refused_not_guessed(raw):
    assert canonical_timeframe(raw) == ""
    plan = plan_timeframe(raw, BYBIT_TIMEFRAMES)
    assert plan.supported is False
    assert plan.classification == TIMEFRAME_NOT_SUPPORTED


@pytest.mark.parametrize("internal", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
def test_no_raw_internal_label_reaches_the_adapter(internal):
    """The observed defect: M1..M30 passed through to Bybit unchanged."""
    plan = plan_timeframe(internal, BYBIT_TIMEFRAMES)

    assert plan.supported is True
    assert plan.venue_timeframe in BYBIT_TIMEFRAMES
    assert plan.venue_timeframe != internal


def test_an_unsupported_interval_is_classified_before_the_call():
    venue = RecordingVenue(timeframes=["1m", "5m", "1h"])
    plan = plan_timeframe("7m", list(venue.timeframes))

    assert plan.supported is False
    assert plan.classification == TIMEFRAME_NOT_SUPPORTED
    assert venue.ohlcv_requests == [], "the venue was called for a known-bad interval"


def test_an_interval_is_aggregated_only_when_that_is_exact():
    exact = plan_timeframe("30m", ["1m", "5m", "15m", "1h"])
    assert exact.supported and exact.aggregate_from == "15m"
    assert exact.aggregate_factor == 2

    inexact = plan_timeframe("4h", ["90m"])
    assert inexact.supported is False


def test_a_materially_different_interval_is_never_substituted():
    plan = plan_timeframe("4h", ["1d", "1w"])
    assert plan.supported is False
    assert plan.venue_timeframe == ""


def test_aggregation_preserves_ohlcv_semantics():
    rows = [
        [0, 10.0, 12.0, 9.0, 11.0, 100.0],
        [1, 11.0, 15.0, 8.0, 14.0, 200.0],
    ]
    merged = aggregate_ohlcv(rows, 2)

    assert len(merged) == 1
    _ts, open_, high, low, close, volume = merged[0]
    assert open_ == 10.0 and close == 14.0
    assert high == 15.0 and low == 8.0
    assert volume == 300.0


def test_a_trailing_partial_group_is_dropped_not_emitted():
    rows = [[i, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(5)]
    assert len(aggregate_ohlcv(rows, 2)) == 2


def test_venue_timeframes_are_read_from_the_adapter():
    assert set(venue_timeframes(RecordingVenue())) == set(BYBIT_TIMEFRAMES)

    class NoDeclaration:
        pass

    assert venue_timeframes(NoDeclaration()) == ()


# ----------------------------------------------- 4-7. data venue separation


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("BTC/USDT", instruments.CRYPTO),
        ("ETH/USDT", instruments.CRYPTO),
        ("BTC/USD", instruments.CRYPTO),
        ("EUR/USD", instruments.FX),
        ("GBP/USD", instruments.FX),
        ("USD/JPY", instruments.FX),
        ("EUR/GBP", instruments.FX),
        ("AUD/USD", instruments.FX),
        ("XAU/USD", instruments.METAL),
    ],
)
def test_instruments_are_classified_by_asset_class(symbol, expected):
    assert instruments.classify_asset(symbol) == expected


@pytest.mark.parametrize(
    "symbol", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "EUR/GBP"]
)
def test_forex_is_never_routed_to_a_crypto_venue(symbol):
    """The observed defect: FX pairs queried against Bybit."""
    allowed, classification, detail = instruments.may_use_venue_for_data(
        "bybit", symbol
    )

    assert allowed is False
    assert classification == instruments.DATA_VENUE_INCOMPATIBLE
    assert "fx" in detail


def test_a_crypto_venue_still_serves_crypto():
    allowed, classification, _detail = instruments.may_use_venue_for_data(
        "bybit", "BTC/USDT"
    )
    assert allowed is True
    assert classification == instruments.CRYPTO


def test_market_data_routing_names_a_compatible_provider():
    route = instruments.resolve_data_route("EUR/USD", execution_venue="bybit")

    assert route.available is True
    assert route.venue != "bybit"
    assert route.asset_class == instruments.FX


def test_an_instrument_with_no_provider_is_classified_not_fabricated():
    route = instruments.resolve_data_route("???", execution_venue="bybit")
    assert route.available is False
    assert route.classification == instruments.DATA_PROVIDER_UNAVAILABLE


def test_a_listed_pair_receives_ohlcv_and_a_non_listed_one_does_not(monkeypatch):
    registry = CapabilityRegistry()
    registry.record_venue_markets(
        "bybit", {s: market(s) for s in ON_BYBIT}, environment="testnet"
    )
    monkeypatch.setattr(routing, "capabilities", registry)

    venue = RecordingVenue()
    for symbol in ON_BYBIT + ("BCH/USDT", "FTM/USDT", "EUR/USD"):
        compatible, _c, _d = instruments.may_use_venue_for_data("bybit", symbol)
        if not compatible:
            continue
        allowed, _c, _d = routing.may_call_venue(
            "bybit", symbol, environment="testnet"
        )
        if allowed:
            venue.fetch_ohlcv(symbol, timeframe="1m")

    requested = {symbol for symbol, _tf in venue.ohlcv_requests}
    assert requested == set(ON_BYBIT)
    assert "EUR/USD" not in requested
    assert "BCH/USDT" not in requested


# -------------------------------------------------- 8-10. persisted snapshot


def test_the_snapshot_is_readable_by_another_process(tmp_path, monkeypatch):
    from src.leantrader.universe import venues

    monkeypatch.setenv("UNIVERSE_SNAPSHOT_PATH", str(tmp_path / "cross.json"))
    capabilities.record_venue_markets(
        "bybit", {s: market(s) for s in ON_BYBIT}, environment="testnet"
    )

    assert venues.write_snapshot() is True

    payload = venues.read_snapshot()
    assert payload["schema_version"] == 2
    assert len(payload["markets"]) == len(ON_BYBIT)

    fresh = CapabilityRegistry()
    assert fresh.load_snapshot(payload) >= len(ON_BYBIT)
    assert fresh.resolve("bybit", "DOGE/USDT", environment="testnet").callable


def test_the_snapshot_shows_a_listed_pair_as_listed(tmp_path, monkeypatch):
    """DOGE/USDT was being reported unavailable while Bybit listed it."""
    from src.leantrader.universe import venues

    monkeypatch.setenv("UNIVERSE_SNAPSHOT_PATH", str(tmp_path / "doge.json"))
    capabilities.record_venue_markets(
        "bybit", {"DOGE/USDT": market("DOGE/USDT")}, environment="testnet"
    )
    venues.write_snapshot()

    rows = venues.read_snapshot()["markets"]
    doge = next(r for r in rows if r["canonical_symbol"] == "DOGE/USDT")

    assert doge["listed"] is True
    assert doge["testnet_capable"] is True
    assert doge["venue"] == "bybit"
    assert doge["asset_class"] == instruments.CRYPTO
    assert doge["min_notional"] == 5.0


def test_a_symbol_follows_live_metadata_and_changes_after_revalidation():
    """FTM/USDT is absent or present according to the venue, not a guess."""
    registry = CapabilityRegistry()
    registry.record_venue_markets(
        "bybit", {s: market(s) for s in ON_BYBIT}, environment="testnet"
    )

    assert not registry.resolve("bybit", "FTM/USDT", environment="testnet").callable

    record = registry.record_for("bybit", "FTM/USDT", environment="testnet")
    record.next_refresh_at = time.time() - 1
    assert registry.resolve("bybit", "FTM/USDT", environment="testnet").callable

    listed = {s: market(s) for s in ON_BYBIT + ("FTM/USDT",)}
    registry.record_venue_markets("bybit", listed, environment="testnet")
    assert registry.resolve("bybit", "FTM/USDT", environment="testnet").callable


def test_snapshot_age_is_reported_so_stale_is_not_shown_as_fresh():
    from src.leantrader.universe import venues

    assert venues.snapshot_age_seconds(None) is None
    assert venues.snapshot_age_seconds({"written_at": time.time() - 120}) >= 119


def test_the_snapshot_carries_no_credentials(tmp_path, monkeypatch):
    from src.leantrader.universe import venues

    monkeypatch.setenv("UNIVERSE_SNAPSHOT_PATH", str(tmp_path / "s.json"))
    capabilities.record_venue_markets(
        "bybit", {"BTC/USDT": market("BTC/USDT")}, environment="testnet"
    )
    venues.write_snapshot()

    text = (tmp_path / "s.json").read_text().lower()
    for token in ("apikey", "api_key", "secret", "passphrase", "password"):
        assert token not in text


# ----------------------------------- 11-14. minimum executable ticket


def _broker(free_usdt, price=140.0, min_cost=1.0, min_amount=0.0):
    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"
        authority = "testnet"

        def resolve_mode(self):
            return "testnet"

        def load_markets(self):
            return {
                "SOL/USDT": market(
                    "SOL/USDT", min_cost=min_cost, min_amount=min_amount
                )
            }

        def fetch_balance(self):
            return {"free": {"USDT": free_usdt}}

        def fetch_ticker(self, symbol):
            return {"last": price}

        def _make_exchange(self, environment, authenticated):
            return None

    return Broker()


def test_the_minimum_executable_ticket_is_computed_not_discovered():
    ticket = preflight.minimum_executable_ticket(
        symbol="SOL/USDT",
        venue="bybit",
        price=140.0,
        min_notional=1.0,
        min_amount=0.0,
        fee_rate=0.001,
    )

    assert ticket.notional == pytest.approx(1.0)
    assert ticket.total_cost > ticket.notional, "fees and buffer must be included"
    assert ticket.fee_allowance > 0
    assert ticket.slippage_allowance > 0


def test_a_sub_minimum_candidate_is_not_retried_every_cycle():
    """0.919 against a 1.00 minimum: a known answer, not a near miss."""
    broker = _broker(free_usdt=0.92, min_cost=1.0)

    first = preflight.prepare_order(
        {"symbol": "SOL/USDT", "side": "buy", "price": 140.0}, broker=broker
    )
    assert first[0] is None
    assert first[1].blocker == preflight.CAPITAL_BELOW_EXECUTABLE_MINIMUM
    assert first[1].stage == "minimum_ticket"

    remembered = preflight.economically_infeasible(
        "bybit", "SOL/USDT", 0.92, price=140.0, min_notional=1.0
    )
    assert remembered, "the answer must be remembered, not recomputed"

    second = preflight.prepare_order(
        {"symbol": "SOL/USDT", "side": "buy", "price": 140.0}, broker=broker
    )
    assert second[1].blocker == preflight.CAPITAL_BELOW_EXECUTABLE_MINIMUM


def test_the_cooldown_lifts_when_the_balance_changes():
    """Remembering must not outlive the reason for it."""
    preflight.note_economically_infeasible(
        "bybit", "SOL/USDT", 0.92, "too small", price=140.0, min_notional=1.0
    )

    assert preflight.economically_infeasible(
        "bybit", "SOL/USDT", 0.92, price=140.0, min_notional=1.0
    )
    assert not preflight.economically_infeasible(
        "bybit", "SOL/USDT", 25.0, price=140.0, min_notional=1.0
    )


def test_the_cooldown_lifts_when_the_venue_minimum_changes():
    preflight.note_economically_infeasible(
        "bybit", "SOL/USDT", 0.92, "too small", price=140.0, min_notional=5.0
    )
    assert not preflight.economically_infeasible(
        "bybit", "SOL/USDT", 0.92, price=140.0, min_notional=1.0
    )


def test_sizing_produces_the_smallest_executable_ticket_when_capital_allows():
    prepared, blocked = preflight.prepare_order(
        {"symbol": "SOL/USDT", "side": "buy", "price": 140.0},
        broker=_broker(free_usdt=12.0, min_cost=1.0),
    )

    assert blocked is None
    assert prepared.notional >= 1.0
    assert prepared.notional <= 12.0


def test_a_risk_budget_below_the_minimum_refuses_rather_than_raising_risk():
    """Never increase risk merely to make a trade happen."""
    prepared, blocked = preflight.prepare_order(
        {
            "symbol": "SOL/USDT",
            "side": "buy",
            "price": 140.0,
            "risk_budget": 0.50,
        },
        broker=_broker(free_usdt=50.0, min_cost=1.0),
    )

    assert prepared is None
    assert blocked.blocker == preflight.CAPITAL_BELOW_EXECUTABLE_MINIMUM
    assert "risk budget" in blocked.detail


def test_a_sufficient_risk_budget_permits_the_minimum_ticket():
    prepared, blocked = preflight.prepare_order(
        {
            "symbol": "SOL/USDT",
            "side": "buy",
            "price": 140.0,
            "risk_budget": 8.0,
        },
        broker=_broker(free_usdt=50.0, min_cost=1.0),
    )

    assert blocked is None
    assert prepared.notional <= 8.0


# ------------------------------------------ 14-16. inventory reconciliation


def _holdings():
    balance = {
        "free": {"USDT": 1.2, "ETH": 0.0021, "SOL": 0.055, "SHIB": 12.0},
        "used": {},
        "total": {},
    }
    markets = {
        "ETH/USDT": market("ETH/USDT", min_cost=5.0, min_amount=0.0001),
        "SOL/USDT": market("SOL/USDT", min_cost=1.0, min_amount=0.01),
        "SHIB/USDT": market("SHIB/USDT", min_cost=5.0, min_amount=100000),
    }
    tickers = {
        "ETH/USDT": {"last": 3100.0},
        "SOL/USDT": {"last": 140.0},
        "SHIB/USDT": {"last": 0.000008},
    }
    return balance, markets, tickers


def test_existing_inventory_reduces_what_a_new_buy_can_spend():
    balance, markets, tickers = _holdings()
    items = inventory_reconciler.reconcile(balance, markets, tickers)
    capital = inventory_reconciler.spendable_capital(1.2, items)

    assert capital["spendable_quote"] == pytest.approx(1.2)
    assert capital["inventory_value"] > capital["free_quote"]
    assert capital["portfolio_value"] > capital["free_quote"]
    assert capital["cash_fraction"] < 0.2


def test_meaningful_inventory_gets_an_owner_and_an_exit_classification():
    balance, markets, tickers = _holdings()
    items = inventory_reconciler.reconcile(
        balance,
        markets,
        tickers,
        known_positions={
            "ETH/USDT": {
                "owner": "execution_orchestrator",
                "owner_alive": True,
                "entry_price": 3000.0,
            }
        },
    )
    by_asset = {item.asset: item for item in items}

    eth = by_asset["ETH"]
    assert eth.classification == inventory_reconciler.ACTIVE_MANAGED_POSITION
    assert eth.exit_state == inventory_reconciler.EXIT_ELIGIBLE
    assert eth.can_close_now is True
    assert eth.unrealized_pnl is not None

    sol = by_asset["SOL"]
    assert sol.classification == inventory_reconciler.ORPHANED_POSITION
    assert sol.exit_state == inventory_reconciler.EXIT_ELIGIBLE


def test_dust_is_classified_separately_and_not_counted_as_a_position():
    balance, markets, tickers = _holdings()
    items = inventory_reconciler.reconcile(balance, markets, tickers)
    summary = inventory_reconciler.summarize(items)

    shib = next(item for item in items if item.asset == "SHIB")
    assert shib.classification == inventory_reconciler.DUST
    assert shib.can_close_now is False
    assert shib.exit_blocked_reason == inventory_reconciler.DUST_BELOW_SELL_MINIMUM

    assert summary["dust_assets"] == 1
    assert summary["managed_positions"] + summary["orphaned_positions"] == 2
    assert summary["reclaimable_capital"] > 0


def test_inventory_in_a_delisted_market_is_not_counted_as_reclaimable():
    balance, markets, tickers = _holdings()
    markets["SOL/USDT"] = market("SOL/USDT", active=False)

    items = inventory_reconciler.reconcile(balance, markets, tickers)
    sol = next(item for item in items if item.asset == "SOL")

    assert sol.classification == inventory_reconciler.NON_EXECUTABLE_INVENTORY
    assert sol.can_close_now is False


def test_reconciliation_places_no_orders():
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "src/leantrader/execution/inventory.py").read_text())
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    for forbidden in ("create_order", "route_order", "order", "market"):
        assert forbidden not in called


# ------------------------------------ 17-19. intent identity and rejections


def test_a_decision_below_threshold_records_a_structured_rejection():
    intent = execution_intent.intent_from_decision(
        {
            "action": "buy",
            "confidence": 0.42,
            "signal": {"symbol": "SOL/USDT", "strategy": "scalp", "source": "scalper"},
        },
        environment="testnet",
    )
    intent.stop(
        execution_intent.THRESHOLD,
        execution_intent.DECISION_REJECTED_CONFIDENCE,
        "confidence 0.42 < threshold 0.80",
    )

    summary = execution_intent.outcome_summary()
    assert summary["stopped_by_outcome"][
        execution_intent.DECISION_REJECTED_CONFIDENCE
    ] == 1
    assert summary["by_source_engine"]["scalper"] == 1
    assert summary["stopped_by_stage"][execution_intent.THRESHOLD] == 1


def test_a_rejection_names_the_stage_it_stopped_at():
    intent = execution_intent.ExecutionIntent(symbol="SOL/USDT", side="buy")
    intent.advance(execution_intent.CANDIDATE)
    intent.stop(execution_intent.PREFLIGHT, "BELOW_MIN_NOTIONAL", "too small")

    assert intent.terminal is True
    assert intent.stage == execution_intent.PREFLIGHT
    assert [step["stage"] for step in intent.trail] == [
        execution_intent.CANDIDATE,
        execution_intent.PREFLIGHT,
    ]
    assert intent.trail[-1]["ok"] is False


def test_intent_identity_survives_into_the_order_payload():
    intent = execution_intent.intent_from_decision(
        {
            "action": "buy",
            "confidence": 0.9,
            "signal": {"symbol": "SOL/USDT", "strategy": "scalp", "source": "scalper"},
        },
        environment="testnet",
        venue="bybit",
    )
    intent.candidate_id = execution_intent.new_candidate_id()
    payload = intent.to_payload()

    for key in (
        "intent_id",
        "correlation_id",
        "source_engine",
        "source_strategy",
        "decision_id",
        "candidate_id",
    ):
        assert payload[key], f"{key} missing from the order payload"

    assert payload["correlation_id"] == intent.correlation_id


def test_a_qualified_decision_reaches_preflight_with_its_identity(monkeypatch):
    from EXECUTION_ORCHESTRATOR import ExecutionOrchestrator

    seen = []

    class Hub:
        async def publish_trade(self, trade):
            pass

    def fake_prepare(order_intent, **kwargs):
        seen.append(order_intent)
        return None, preflight.Blocked(
            preflight.CAPITAL_BELOW_EXECUTABLE_MINIMUM, "too small", "minimum_ticket"
        )

    monkeypatch.setattr(preflight, "prepare_order", fake_prepare)

    import asyncio

    orchestrator = ExecutionOrchestrator(Hub(), {}, None, None, mode="testnet")
    result = asyncio.run(
        orchestrator.process_decision(
            {
                "action": "buy",
                "confidence": 0.95,
                "signal": {"symbol": "SOL/USDT", "source": "scalper", "data": {}},
            }
        )
    )

    assert seen, "a qualified decision never reached preflight"
    assert seen[0]["symbol"] == "SOL/USDT"
    assert "risk_budget" in seen[0], "capital awareness must reach preflight"

    summary = execution_intent.outcome_summary()
    assert summary["traced_intents"] >= 1


# ---------------------------------------- 20-21. telemetry semantics


def test_an_acknowledgement_can_never_exist_without_a_submission():
    """The observed anomaly: submitted=0, acknowledged=1."""
    violations = preflight.lifecycle_violations(
        {"attempts": 42, "prepared": 1, "submitted": 0, "acknowledged": 1}
    )
    assert violations
    assert "acknowledged=1 exceeds submitted=0" in violations[0]


def test_a_well_formed_funnel_has_no_violations():
    assert preflight.lifecycle_violations(
        {
            "attempts": 42,
            "prepared": 6,
            "submitted": 4,
            "acknowledged": 3,
            "fills": 2,
            "closes": 1,
        }
    ) == []


def test_the_legacy_bots_count_submissions_where_they_count_acknowledgements():
    """Their missing submitted increment is what produced the anomaly."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("REAL_PROFIT_BOT.py", "MICRO_TRADING_BOT.py"):
        tree = ast.parse((root / name).read_text())
        events = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "record_event"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        assert "acknowledged" in events
        assert "submitted" in events, f"{name} acknowledges without submitting"


def test_recovered_orders_are_counted_apart_from_this_runs_submissions():
    # A realistic funnel: the invariant is about lifecycle shape, and a
    # submission with no preparation would be its own violation.
    preflight.record_event("attempts", 5)
    preflight.record_event("prepared", 3)
    preflight.record_event("submitted", 2)
    preflight.record_recovered_order()
    preflight.record_recovered_order(external=True)

    snapshot = preflight.telemetry_snapshot()
    assert snapshot["submitted"] == 2
    assert snapshot["recovered_orders"] == 1
    assert snapshot["reconciled_external_orders"] == 1
    assert preflight.lifecycle_violations(snapshot) == []


def test_this_runs_counters_are_separable_from_the_persisted_totals():
    preflight.record_event("submitted", 3)
    current = preflight.current_run_counters()

    assert current["submitted"] == 3
    assert preflight.telemetry_snapshot()["run_id"] == preflight.RUN_ID


def test_a_refusal_is_never_counted_as_an_acknowledgement_or_fill():
    assert preflight.classify_receipt({"ok": False, "error": "rejected"})
    assert preflight.classify_receipt({"ok": True, "executed": False, "order": {"id": "1"}})
    assert preflight.classify_receipt({"ok": True, "executed": True, "order": {}})
    assert preflight.classify_receipt(
        {"ok": True, "executed": True, "order": {"id": "1"}}
    ) is None


# ------------------------------------- 22-24. authority, modes, no orders


def test_route_order_remains_the_sole_order_authority():
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for name in (
        "ultra_core.py",
        "EXECUTION_ORCHESTRATOR.py",
        "src/leantrader/execution/inventory.py",
        "src/leantrader/execution/intent.py",
        "src/leantrader/universe/instruments.py",
        "src/leantrader/universe/timeframes.py",
    ):
        tree = ast.parse((root / name).read_text())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "create_order" not in called, f"{name} calls create_order directly"


@pytest.mark.parametrize("environment", ["paper", "testnet", "live"])
def test_all_three_environments_remain_intact(environment, monkeypatch):
    registry = CapabilityRegistry()
    registry.record_venue_markets(
        "bybit", {s: market(s) for s in ON_BYBIT}, environment=environment
    )
    monkeypatch.setattr(routing, "capabilities", registry)
    monkeypatch.setenv("BYBIT_API_KEY", "fake" + "0" * 14)  # secret-scan: allow

    decision = routing.select_venue("BTC/USDT", environment, preferred="bybit")
    assert decision.executable is True


def test_no_order_is_submitted_anywhere_in_this_module():
    venue = RecordingVenue()
    assert venue.orders == []
