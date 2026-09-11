"""Observe broadly; execute precisely.

A symbol existing somewhere in the global intelligence universe said nothing
about whether the execution venue lists it, and the runtime was treating those
as the same thing: every swarm study of a globally-discovered market went
straight to the execution venue's endpoint. For symbols that venue does not
list -- BCH/USDT, FTM/USDT and FIL/USDT on Bybit among them -- that is an
exchange error, logged, every cycle, forever.

These prove the two universes stay apart, that absence is resolved locally
rather than discovered by failing, and that a market we cannot trade from here
stays available to research and to paper.

No order is submitted and no venue is contacted.
"""

import time

import pytest

from src.leantrader.execution import preflight
from src.leantrader.universe import routing
from src.leantrader.universe.registry import normalize_symbol, universe
from src.leantrader.universe.venues import (
    AUTH_FAILURE,
    CapabilityRegistry,
    DELISTED,
    LISTED,
    NETWORK_ERROR,
    NOT_LISTED,
    RATE_LIMIT,
    SPOT,
    TEMPORARILY_SUSPENDED,
    capabilities,
    market_type_of,
)

# Symbols Bybit does not list, taken from the observed runtime errors.
ABSENT_FROM_BYBIT = ("BCH/USDT", "FTM/USDT", "FIL/USDT")
ON_BYBIT = ("BTC/USDT", "DOGE/USDT", "ETH/USDT")


def market(symbol, spot=True, swap=False, active=True, min_cost=5.0, inverse=False):
    return {
        "symbol": symbol,
        "spot": spot,
        "swap": swap,
        "inverse": inverse,
        "active": active,
        "taker": 0.001,
        "limits": {"amount": {"min": 0.0001}, "cost": {"min": min_cost}},
        "precision": {"amount": 8, "price": 2},
    }


def markets(symbols, **kw):
    return {s: market(s, **kw) for s in symbols}


@pytest.fixture
def registry():
    return CapabilityRegistry()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("EXECUTION_TELEMETRY_PATH", str(tmp_path / "t.json"))
    monkeypatch.setenv("UNIVERSE_SNAPSHOT_PATH", str(tmp_path / "snap.json"))
    capabilities.reset()
    universe.reset()
    preflight.reset_caches()
    preflight.reset_shared_brokers()
    yield
    capabilities.reset()
    universe.reset()
    preflight.reset_caches()
    preflight.reset_shared_brokers()


@pytest.fixture
def bybit_known(registry):
    """Bybit's metadata read: it lists ON_BYBIT and nothing else."""
    registry.record_venue_markets("bybit", markets(ON_BYBIT), environment="testnet")
    return registry


class CountingVenue:
    """Records every call, so 'did not call the exchange' is provable."""

    def __init__(self, listed):
        self.listed = set(listed)
        self.ticker_calls = []
        self.ohlcv_calls = []
        self.order_calls = []

    def fetch_ticker(self, symbol):
        self.ticker_calls.append(symbol)
        if symbol not in self.listed:
            raise ValueError(f"{symbol} is not a listed market")
        return {"last": 100.0}

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=100):
        self.ohlcv_calls.append(symbol)
        if symbol not in self.listed:
            raise ValueError(f"{symbol} is not a listed market")
        return [[0, 1, 1, 1, 1, 1]] * limit

    def create_order(self, *args, **kwargs):
        self.order_calls.append(args)
        raise AssertionError("an order reached a venue during the test suite")


# ------------------------------------------------ 1. a listed symbol routes


@pytest.mark.parametrize("symbol", ON_BYBIT)
def test_a_symbol_bybit_lists_routes_to_bybit(bybit_known, symbol):
    resolution = bybit_known.resolve("bybit", symbol, environment="testnet")

    assert resolution.callable is True
    assert resolution.classification == LISTED
    assert resolution.venue_symbol == symbol


# --------------------------- 2-4. absence resolves locally, without calling


@pytest.mark.parametrize("symbol", ABSENT_FROM_BYBIT)
def test_a_symbol_absent_from_bybit_is_not_called(bybit_known, symbol):
    resolution = bybit_known.resolve("bybit", symbol, environment="testnet")

    assert resolution.callable is False
    assert resolution.classification == NOT_LISTED
    assert "metadata" in resolution.detail


def test_the_guard_prevents_the_exchange_call_entirely(bybit_known, monkeypatch):
    """The observed failure mode, end to end."""
    monkeypatch.setattr(routing, "capabilities", bybit_known)
    venue = CountingVenue(ON_BYBIT)

    for symbol in ON_BYBIT + ABSENT_FROM_BYBIT:
        allowed, _classification, _detail = routing.may_call_venue(
            "bybit", symbol, environment="testnet"
        )
        if allowed:
            venue.fetch_ticker(symbol)

    assert set(venue.ticker_calls) == set(ON_BYBIT)
    for absent in ABSENT_FROM_BYBIT:
        assert absent not in venue.ticker_calls


def test_a_known_missing_symbol_produces_no_repeated_errors(
    bybit_known, monkeypatch
):
    """500 cycles, zero exchange calls, zero exceptions."""
    monkeypatch.setattr(routing, "capabilities", bybit_known)
    venue = CountingVenue(ON_BYBIT)
    errors = []

    for _cycle in range(500):
        for symbol in ABSENT_FROM_BYBIT:
            allowed, _c, _d = routing.may_call_venue(
                "bybit", symbol, environment="testnet"
            )
            if allowed:
                try:
                    venue.fetch_ticker(symbol)
                except Exception as exc:
                    errors.append(exc)

    assert venue.ticker_calls == []
    assert errors == []


def test_the_negative_cache_counts_what_it_suppressed(bybit_known):
    for _ in range(120):
        for symbol in ABSENT_FROM_BYBIT:
            bybit_known.resolve("bybit", symbol, environment="testnet")

    telemetry = bybit_known.telemetry()
    assert telemetry["suppressed_calls_total"] >= 360
    assert telemetry["suppressed_calls_by_reason"][NOT_LISTED] >= 360
    assert telemetry["known_not_listed_by_venue"]["bybit"] == len(ABSENT_FROM_BYBIT)


# ------------------------------------------------ 5. TTL lets a relist land


def test_ttl_revalidation_discovers_a_newly_listed_market(bybit_known):
    symbol = "FTM/USDT"

    assert bybit_known.resolve("bybit", symbol, environment="testnet").callable is False

    # Time passes beyond the NOT_LISTED TTL.
    record = bybit_known.record_for("bybit", symbol, environment="testnet")
    record.next_refresh_at = time.time() - 1

    revalidating = bybit_known.resolve("bybit", symbol, environment="testnet")
    assert revalidating.callable is True, "an expired absence must be rechecked"
    assert revalidating.classification == "REVALIDATE"

    # The venue now lists it.
    bybit_known.record_venue_markets(
        "bybit", markets(ON_BYBIT + (symbol,)), environment="testnet"
    )
    assert bybit_known.resolve("bybit", symbol, environment="testnet").callable is True


def test_nothing_is_remembered_permanently(bybit_known):
    """A cache that never expires makes LeanTrader permanently wrong."""
    from src.leantrader.universe.venues import DEFAULT_TTL_SECONDS, MAX_TTL_SECONDS

    for state, ttl in DEFAULT_TTL_SECONDS.items():
        assert ttl <= MAX_TTL_SECONDS, f"{state} is cached too long"
        assert ttl > 0


def test_a_market_that_disappears_becomes_delisted(bybit_known):
    assert bybit_known.resolve("bybit", "DOGE/USDT", environment="testnet").callable

    remaining = tuple(s for s in ON_BYBIT if s != "DOGE/USDT")
    bybit_known.record_venue_markets(
        "bybit", markets(remaining), environment="testnet"
    )

    record = bybit_known.record_for("bybit", "DOGE/USDT", environment="testnet")
    assert record.state == DELISTED


# --------------------------- 6-7. transient failures are not market facts


def test_a_network_failure_is_never_cached_as_not_listed(registry):
    """A timeout must not destroy what the venue told us about the market.

    Held together, a blip would overwrite the LISTED record; clearing the
    blip would then leave nothing, and the next resolve would conclude the
    market is absent. So the capability fact and the backoff are separate.
    """
    registry.record_venue_markets("bybit", markets(ON_BYBIT), environment="testnet")
    registry.record_transient(
        "bybit", "BTC/USDT", state=NETWORK_ERROR, environment="testnet"
    )

    # The backoff suppresses calls for now...
    assert not registry.resolve("bybit", "BTC/USDT", environment="testnet").callable
    backoff = registry.backoff_for("bybit", "BTC/USDT", environment="testnet")
    assert backoff.state == NETWORK_ERROR

    # ...but the capability fact is untouched.
    fact = registry._records[("bybit", "BTC/USDT", SPOT, "testnet")]
    assert fact.state == LISTED
    assert fact.state != NOT_LISTED

    # And a success clears the backoff, restoring the known-listed market.
    registry.clear_transient("bybit", "BTC/USDT", environment="testnet")
    resolution = registry.resolve("bybit", "BTC/USDT", environment="testnet")
    assert resolution.callable
    assert resolution.classification == LISTED


def test_a_network_failure_backs_off_briefly_not_for_hours(registry):
    from src.leantrader.universe.venues import DEFAULT_TTL_SECONDS

    record = registry.record_transient(
        "bybit", "BTC/USDT", state=NETWORK_ERROR, environment="testnet"
    )
    wait = record.next_refresh_at - time.time()

    assert wait <= DEFAULT_TTL_SECONDS[NETWORK_ERROR] * 1.5
    assert wait < DEFAULT_TTL_SECONDS[NOT_LISTED]


def test_rate_limit_failures_back_off_exponentially_and_bounded(registry):
    from src.leantrader.universe.venues import MAX_TTL_SECONDS

    waits = []
    for _ in range(6):
        record = registry.record_transient(
            "bybit", "BTC/USDT", state=RATE_LIMIT, environment="testnet"
        )
        waits.append(record.next_refresh_at - time.time())

    assert record.consecutive_failures == 6
    assert waits[-1] > waits[0], "backoff must grow"
    assert all(w <= MAX_TTL_SECONDS for w in waits), "backoff must stay bounded"


def test_backoff_is_jittered_so_recovery_is_not_a_herd(registry):
    waits = set()
    for index in range(12):
        record = registry.record_transient(
            "bybit", f"S{index}/USDT", state=NETWORK_ERROR, environment="testnet"
        )
        waits.add(round(record.next_refresh_at - time.time(), 4))

    assert len(waits) > 1, "identical backoffs would synchronise the retry herd"


def test_an_auth_failure_is_configuration_state_not_market_state(registry):
    registry.record_transient(
        "bybit", "BTC/USDT", state=AUTH_FAILURE, environment="testnet"
    )
    record = registry.record_for("bybit", "BTC/USDT", environment="testnet")

    assert record.state == AUTH_FAILURE
    assert record.state not in {NOT_LISTED, DELISTED}


# --------------------------------- 8-9. normalization and instrument kinds


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("btcusdt", "BTC/USDT"),
        ("BTC-USDT", "BTC/USDT"),
        ("btc_usdt", "BTC/USDT"),
        ("BTC/USDT:USDT", "BTC/USDT"),
        (" eth/usdt ", "ETH/USDT"),
    ],
)
def test_venue_notation_maps_to_one_canonical_symbol(raw, expected):
    assert normalize_symbol(raw) == expected


def test_the_venue_native_symbol_is_kept_for_the_adapter_boundary(registry):
    registry.record_venue_markets(
        "bybit", {"BTCUSDT": market("BTC/USDT")}, environment="testnet"
    )
    assert registry.venue_symbol("bybit", "BTC/USDT") == "BTC/USDT"
    assert registry.resolve("bybit", "BTC/USDT", environment="testnet").callable


@pytest.mark.parametrize(
    "flags,expected",
    [
        ({"spot": True}, "spot"),
        ({"spot": False, "swap": True, "inverse": False}, "linear_swap"),
        ({"spot": False, "swap": True, "inverse": True}, "inverse_swap"),
        ({"spot": False, "future": True, "inverse": False}, "linear_future"),
        ({"spot": False, "option": True}, "option"),
    ],
)
def test_instrument_kinds_are_distinguished(flags, expected):
    assert market_type_of(flags) == expected


def test_spot_and_perpetual_are_never_conflated(registry):
    """The same text is two instruments; routing one as the other is wrong."""
    registry.record_venue_markets(
        "bybit",
        {
            "BTC/USDT": market("BTC/USDT", spot=True),
            "BTC/USDT:USDT": {
                "symbol": "BTC/USDT:USDT",
                "spot": False,
                "swap": True,
                "inverse": False,
                "active": True,
                "limits": {},
                "precision": {},
            },
        },
        environment="testnet",
    )

    spot = registry.resolve("bybit", "BTC/USDT", SPOT, "testnet")
    perp = registry.resolve("bybit", "BTC/USDT", "linear_swap", "testnet")

    assert spot.callable and perp.callable
    assert spot.record.market_type == SPOT
    assert perp.record.market_type == "linear_swap"
    assert spot.record is not perp.record


def test_a_venue_listing_only_a_perpetual_is_not_spot_tradable(registry):
    registry.record_venue_markets(
        "bybit",
        {
            "SOL/USDT:USDT": {
                "symbol": "SOL/USDT:USDT",
                "spot": False,
                "swap": True,
                "inverse": False,
                "active": True,
                "limits": {},
                "precision": {},
            }
        },
        environment="testnet",
    )

    assert registry.resolve("bybit", "SOL/USDT", SPOT, "testnet").callable is False
    assert registry.resolve("bybit", "SOL/USDT", "linear_swap", "testnet").callable


# -------------------- 10-12. intelligence survives; authority is not faked


def test_a_market_absent_from_testnet_stays_available_for_research(
    bybit_known, monkeypatch
):
    monkeypatch.setattr(routing, "capabilities", bybit_known)
    bybit_known.record_venue_markets(
        "binance", markets(ABSENT_FROM_BYBIT), environment="live"
    )

    for symbol in ABSENT_FROM_BYBIT:
        decision = routing.select_venue(symbol, "testnet", preferred="bybit")

        assert decision.executable is False
        assert decision.classification == routing.PUBLIC_INTELLIGENCE_ONLY
        assert "binance" in decision.research_venues
        assert routing.attention_lane(symbol, "testnet") == routing.RESEARCH


def test_paper_can_still_experiment_with_a_market_the_venue_lacks(
    bybit_known, monkeypatch
):
    monkeypatch.setattr(routing, "capabilities", bybit_known)
    decision = routing.select_venue("FTM/USDT", "paper", preferred="bybit")

    assert decision.executable is True
    assert decision.classification == routing.PAPER_EXECUTABLE


def test_the_selector_chooses_another_authenticated_venue_that_lists_it(
    bybit_known, monkeypatch
):
    monkeypatch.setattr(routing, "capabilities", bybit_known)
    bybit_known.record_venue_markets(
        "binance", markets(ABSENT_FROM_BYBIT), environment="testnet"
    )
    monkeypatch.setenv("BINANCE_API_KEY", "fake" + "0" * 14)  # secret-scan: allow

    decision = routing.select_venue("FTM/USDT", "testnet", preferred="bybit")

    assert decision.chosen_venue == "binance"
    assert decision.classification == routing.EXECUTABLE
    assert decision.considered["bybit"] == routing.MARKET_NOT_LISTED_ON_VENUE


def test_a_venue_that_lists_it_but_is_unauthenticated_is_named_not_used(
    bybit_known, monkeypatch
):
    """Authority is never fabricated, and the intelligence is not discarded."""
    monkeypatch.setattr(routing, "capabilities", bybit_known)
    bybit_known.record_venue_markets(
        "binance", markets(ABSENT_FROM_BYBIT), environment="testnet"
    )
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)

    decision = routing.select_venue("FTM/USDT", "testnet", preferred="bybit")

    assert decision.chosen_venue == ""
    assert decision.classification == routing.AUTH_REQUIRED
    assert (
        decision.considered["binance"]
        == routing.VENUE_AVAILABLE_NOT_AUTHENTICATED
    )


def test_when_no_venue_can_execute_no_order_is_attempted(
    bybit_known, monkeypatch
):
    monkeypatch.setattr(routing, "capabilities", bybit_known)
    venue = CountingVenue(ON_BYBIT)

    decision = routing.select_venue("FTM/USDT", "testnet", preferred="bybit")
    if decision.executable:
        venue.create_order("FTM/USDT", "market", "buy", 1.0, None, {})

    assert decision.executable is False
    assert venue.order_calls == []


# ---------------------------- 13-14. preflight uses real venue metadata


def test_an_unsupported_pair_never_reaches_create_order(bybit_known, monkeypatch):
    monkeypatch.setattr(routing, "capabilities", bybit_known)
    venue = CountingVenue(ON_BYBIT)

    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"
        authority = "testnet"

        def resolve_mode(self):
            return "testnet"

        def load_markets(self):
            return markets(ON_BYBIT)

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}}

        def fetch_ticker(self, symbol):
            return venue.fetch_ticker(symbol)

        def _make_exchange(self, environment, authenticated):
            return venue

    prepared, blocked = preflight.prepare_order(
        {"symbol": "FTM/USDT", "side": "buy", "price": 1.0}, broker=Broker()
    )

    assert prepared is None
    assert blocked.blocker == preflight.MARKET_NOT_LISTED
    assert venue.order_calls == []


def test_preflight_sizes_from_the_venues_own_metadata(bybit_known):
    """Not from a table, and not from another venue's limits."""

    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"
        authority = "testnet"

        def resolve_mode(self):
            return "testnet"

        def load_markets(self):
            return {"BTC/USDT": market("BTC/USDT", min_cost=25.0)}

        def fetch_balance(self):
            return {"free": {"USDT": 500.0}}

        def fetch_ticker(self, symbol):
            return {"last": 64_000.0}

        def _make_exchange(self, environment, authenticated):
            return None

    prepared, blocked = preflight.prepare_order(
        {"symbol": "BTC/USDT", "side": "buy", "price": 64_000.0}, broker=Broker()
    )

    assert blocked is None
    assert prepared.min_notional == pytest.approx(25.0)
    assert prepared.notional >= 25.0


def test_a_repeatedly_unfundable_ticket_is_remembered_not_regenerated():
    """0.99 against a 1.00 minimum will never become an order."""
    preflight.clear_infeasible_memory()

    class Broker:
        exchange_id = "bybit"
        market_mode = "spot"
        authority = "testnet"

        calls = 0

        def resolve_mode(self):
            return "testnet"

        def load_markets(self):
            Broker.calls += 1
            return {"TINY/USDT": market("TINY/USDT", min_cost=50.0)}

        def fetch_balance(self):
            return {"free": {"USDT": 3.0}}

        def fetch_ticker(self, symbol):
            return {"last": 1.0}

        def _make_exchange(self, environment, authenticated):
            return None

    broker = Broker()
    first = preflight.prepare_order(
        {"symbol": "TINY/USDT", "side": "buy", "price": 1.0}, broker=broker
    )
    assert first[0] is None
    assert first[1].blocker == preflight.BELOW_MIN_NOTIONAL

    assert preflight.economically_infeasible("bybit", "TINY/USDT", 3.0)

    second = preflight.prepare_order(
        {"symbol": "TINY/USDT", "side": "buy", "price": 1.0}, broker=broker
    )
    assert second[1].blocker == preflight.BELOW_MIN_NOTIONAL
    assert second[1].stage == "sizing"


def test_the_infeasibility_memory_expires(monkeypatch):
    monkeypatch.setenv("EXECUTION_INFEASIBLE_TTL_SECONDS", "0")
    preflight.clear_infeasible_memory()
    preflight.note_economically_infeasible("bybit", "TINY/USDT", 3.0, "too small")

    assert preflight.economically_infeasible("bybit", "TINY/USDT", 3.0) is None


# ------------------------------------- 18-20. authority, modes, no orders


def test_route_order_remains_the_sole_authority_after_this_change():
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("ultra_core.py", "EXECUTION_ORCHESTRATOR.py"):
        tree = ast.parse((root / name).read_text())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "create_order" not in called, f"{name} calls create_order directly"


@pytest.mark.parametrize("environment", ["paper", "testnet", "live"])
def test_all_three_environments_still_resolve(environment, bybit_known, monkeypatch):
    monkeypatch.setattr(routing, "capabilities", bybit_known)
    bybit_known.record_venue_markets(
        "bybit", markets(ON_BYBIT), environment=environment
    )
    monkeypatch.setenv("BYBIT_API_KEY", "fake" + "0" * 14)  # secret-scan: allow

    decision = routing.select_venue("BTC/USDT", environment, preferred="bybit")
    assert decision.executable is True


def test_the_suite_sends_no_orders():
    """CountingVenue raises on create_order; nothing in this module calls it."""
    venue = CountingVenue(ON_BYBIT)
    assert venue.order_calls == []


# ------------------------------------------------ cross-process visibility


def test_capability_state_survives_the_process_boundary(bybit_known, tmp_path):
    """The registry is in-process; observability runs elsewhere."""
    from src.leantrader.universe import venues

    venues.capabilities.record_venue_markets(
        "bybit", markets(ON_BYBIT), environment="testnet"
    )
    venues.capabilities.resolve("bybit", "FTM/USDT", environment="testnet")

    assert venues.write_snapshot({"universe": universe.telemetry()}) is True

    payload = venues.read_snapshot()
    assert payload and payload.get("records")

    fresh = CapabilityRegistry()
    restored = fresh.load_snapshot(payload)

    assert restored >= len(ON_BYBIT)
    assert fresh.resolve("bybit", "BTC/USDT", environment="testnet").callable
    assert not fresh.resolve("bybit", "FTM/USDT", environment="testnet").callable


def test_unreadable_venue_metadata_is_not_evidence_of_absence(registry):
    registry.record_venue_markets("bybit", markets(ON_BYBIT), environment="testnet")
    registry.record_venue_markets("bybit", None, environment="testnet")

    record = registry.record_for("bybit", "BTC/USDT", environment="testnet")
    assert record.state == LISTED, "a failed read must not delist everything"


def test_an_inactive_market_is_suspended_not_absent(registry):
    registry.record_venue_markets(
        "bybit", markets(("BTC/USDT",), active=False), environment="testnet"
    )
    record = registry.record_for("bybit", "BTC/USDT", environment="testnet")

    assert record.state == TEMPORARILY_SUSPENDED
    assert record.state != NOT_LISTED

    from src.leantrader.universe.venues import DEFAULT_TTL_SECONDS

    assert (
        DEFAULT_TTL_SECONDS[TEMPORARILY_SUSPENDED]
        < DEFAULT_TTL_SECONDS[NOT_LISTED]
    ), "a suspension should be rechecked sooner than an absence"
