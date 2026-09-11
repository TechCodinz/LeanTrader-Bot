"""The universe must stay broad, and breadth must not become authority.

Two defects motivate these. LeanTrader discovers thousands of markets, but
the engines were reading hardcoded lists -- three majors in the swarm, six in
the continuous trader, an empty list in the micro bot -- because nothing
connected discovery to them. And a market seen on one venue was at risk of
being treated as tradable on another.

Nothing here touches the network.
"""

import time

import pytest

from src.leantrader.universe.registry import (
    INACTIVE,
    MarketUniverse,
    NOT_ON_EXECUTION_VENUE,
    NOT_SPOT,
    NO_METADATA,
    normalize_symbol,
)


def ticker(last=1.0, quote_volume=5_000_000.0, change=3.0, bid=None, ask=None):
    return {
        "last": last,
        "quoteVolume": quote_volume,
        "percentage": change,
        "bid": bid if bid is not None else last * 0.999,
        "ask": ask if ask is not None else last * 1.001,
    }


def market_meta(spot=True, active=True, min_cost=5.0, min_amount=0.0001, taker=0.001):
    return {
        "spot": spot,
        "active": active,
        "taker": taker,
        "limits": {"amount": {"min": min_amount}, "cost": {"min": min_cost}},
        "precision": {"amount": 8, "price": 2},
    }


# A universe shaped like the real one: a couple of majors, a mid cap, and a
# long tail of micro-priced markets.
BROAD_TICKERS = {
    "BTC/USDT": ticker(last=64_000.0, quote_volume=900_000_000.0, change=1.5),
    "ETH/USDT": ticker(last=3_100.0, quote_volume=400_000_000.0, change=2.0),
    "SOL/USDT": ticker(last=140.0, quote_volume=80_000_000.0, change=6.0),
    "DOGE/USDT": ticker(last=0.14, quote_volume=30_000_000.0, change=4.0),
    "PEPE/USDT": ticker(last=0.00000812, quote_volume=12_000_000.0, change=9.0),
    "BONK/USDT": ticker(last=0.00002140, quote_volume=6_000_000.0, change=7.0),
    "FLOKI/USDT": ticker(last=0.00014, quote_volume=3_000_000.0, change=5.0),
    "TINY/USDT": ticker(last=0.00000004, quote_volume=90_000.0, change=12.0),
}

BROAD_MARKETS = {symbol: market_meta() for symbol in BROAD_TICKERS}


@pytest.fixture
def registry():
    return MarketUniverse()


@pytest.fixture
def broad(registry):
    registry.ingest_venue("bybit", BROAD_TICKERS, markets=BROAD_MARKETS)
    registry.apply_execution_venue("bybit", BROAD_MARKETS)
    return registry


# ------------------------------------------------------------- normalization


@pytest.mark.parametrize(
    "raw,expected",
    [("btcusdt", "BTC/USDT"), ("PEPE-USDT", "PEPE/USDT"), ("bonk_usdt", "BONK/USDT")],
)
def test_venue_spellings_normalize_to_one_symbol(raw, expected):
    assert normalize_symbol(raw) == expected


def test_the_same_symbol_on_two_venues_is_one_normalized_symbol(registry):
    registry.ingest_venue("bybit", {"DOGE/USDT": ticker()}, markets=BROAD_MARKETS)
    registry.ingest_venue("binance", {"dogeusdt": ticker()}, markets={})

    assert registry.symbols() == ["DOGE/USDT"]
    assert len(registry.all_markets()) == 2, "one record per venue"


def test_derivative_listings_are_not_ingested_as_spot(registry):
    recorded = registry.ingest_venue(
        "bybit",
        {"BTC/USDT": ticker(), "BTC/USDT:USDT": ticker()},
        markets=BROAD_MARKETS,
    )
    assert recorded == 1
    assert registry.symbols() == ["BTC/USDT"]


def test_quote_filter_keeps_the_configured_quotes(registry):
    registry.ingest_venue(
        "bybit",
        {"BTC/USDT": ticker(), "BTC/EUR": ticker()},
        markets={},
        quote_filter=("USDT",),
    )
    assert registry.symbols() == ["BTC/USDT"]


# ------------------------------------------------- 3. breadth of the universe


def test_the_universe_includes_micro_priced_non_major_symbols(broad):
    symbols = broad.symbols(executable_only=True)

    assert len(symbols) == len(BROAD_TICKERS)
    for micro in ("PEPE/USDT", "BONK/USDT", "FLOKI/USDT", "TINY/USDT"):
        assert micro in symbols, f"{micro} was dropped from the universe"


def test_ranking_is_not_dominated_by_the_majors(broad):
    ranked = [m.symbol for m in broad.rank(capital_quote=13.95)]

    assert ranked, "nothing ranked"
    head = set(ranked[:5])
    assert head - {"BTC/USDT", "ETH/USDT", "SOL/USDT"}, (
        "the top of the ranking is only majors; this is the defect"
    )


def test_nominal_unit_price_alone_does_not_win(registry):
    """Many zeros is not evidence of a better opportunity."""
    registry.ingest_venue(
        "bybit",
        {
            # Vanishing unit price, but almost no volume and a wide spread.
            "DUST/USDT": ticker(
                last=0.00000001, quote_volume=800.0, change=2.0,
                bid=0.000000009, ask=0.000000011,
            ),
            # A real market at a normal price.
            "SOL/USDT": ticker(last=140.0, quote_volume=80_000_000.0, change=5.0),
        },
        markets={"DUST/USDT": market_meta(), "SOL/USDT": market_meta()},
    )
    registry.apply_execution_venue(
        "bybit", {"DUST/USDT": market_meta(), "SOL/USDT": market_meta()}
    )

    ranked = [m.symbol for m in registry.rank(capital_quote=13.95)]
    assert ranked.index("SOL/USDT") < ranked.index("DUST/USDT")


# ----------------------------------------------------- 4. exclusion of unusable


def test_an_inactive_market_is_not_executable(registry):
    registry.ingest_venue("bybit", {"OLD/USDT": ticker()}, markets={})
    registry.apply_execution_venue(
        "bybit", {"OLD/USDT": market_meta(active=False)}
    )

    market = registry.get("OLD/USDT")
    assert market.execution_eligible is False
    assert market.execution_blocker == INACTIVE
    assert "OLD/USDT" not in registry.symbols(executable_only=True)


def test_a_derivative_only_market_is_not_executable_as_spot(registry):
    registry.ingest_venue("bybit", {"PERP/USDT": ticker()}, markets={})
    registry.apply_execution_venue("bybit", {"PERP/USDT": market_meta(spot=False)})

    market = registry.get("PERP/USDT")
    assert market.execution_eligible is False
    assert market.execution_blocker == NOT_SPOT


def test_inactive_markets_are_excluded_from_ranking(broad):
    broad.apply_execution_venue(
        "bybit",
        {
            symbol: market_meta(active=(symbol != "TINY/USDT"))
            for symbol in BROAD_TICKERS
        },
    )
    ranked = [m.symbol for m in broad.rank(capital_quote=13.95)]
    assert "TINY/USDT" not in ranked


# ------------------------------ 5 & 11. discovery is not execution authority


def test_a_symbol_seen_elsewhere_is_not_executable_here(registry):
    """Discovery on Binance says nothing about Bybit."""
    registry.ingest_venue("binance", {"ONLYBIN/USDT": ticker()}, markets={})
    registry.ingest_venue("bybit", {"BTC/USDT": ticker(last=64_000.0)}, markets={})

    registry.apply_execution_venue("bybit", {"BTC/USDT": market_meta()})

    foreign = registry.get("ONLYBIN/USDT")
    assert foreign is not None, "it is still studied"
    assert foreign.execution_eligible is False
    assert foreign.execution_blocker == NOT_ON_EXECUTION_VENUE
    assert not registry.execution_venue_lists("ONLYBIN/USDT")

    assert registry.get("BTC/USDT").execution_eligible is True
    assert registry.execution_venue_lists("BTC/USDT")


def test_public_discovery_on_many_venues_grants_no_execution_anywhere(registry):
    for venue in ("binance", "okx", "kucoin", "gateio", "mexc", "bitget"):
        registry.ingest_venue(venue, {"WIDE/USDT": ticker()}, markets={})

    # The authenticated venue does not list it.
    registry.apply_execution_venue("bybit", {"BTC/USDT": market_meta()})

    studied = [m for m in registry.all_markets() if m.symbol == "WIDE/USDT"]
    assert len(studied) == 6, "it is studied on every venue that lists it"
    assert all(m.execution_eligible is False for m in studied)
    assert registry.symbols(executable_only=True) == []


def test_unreadable_execution_metadata_is_unknown_not_ineligible(registry):
    registry.ingest_venue("bybit", BROAD_TICKERS, markets=BROAD_MARKETS)
    registry.apply_execution_venue("bybit", None)

    markets = registry.all_markets()
    assert all(m.execution_eligible is None for m in markets)
    assert all(m.execution_blocker == NO_METADATA for m in markets)
    assert registry.telemetry()["execution_eligibility_unknown"] == len(markets)


def test_the_execution_venues_own_limits_win(registry):
    """A minimum learned elsewhere must not be used to size an order here."""
    registry.ingest_venue(
        "binance",
        {"DOGE/USDT": ticker(last=0.14)},
        markets={"DOGE/USDT": market_meta(min_cost=1.0)},
    )
    registry.ingest_venue(
        "bybit",
        {"DOGE/USDT": ticker(last=0.14)},
        markets={"DOGE/USDT": market_meta(min_cost=1.0)},
    )
    registry.apply_execution_venue(
        "bybit", {"DOGE/USDT": market_meta(min_cost=5.0)}
    )

    for market in registry.all_markets():
        assert market.min_notional == pytest.approx(5.0)


# ------------------------------------------ 6. small-account execution economics


def test_a_market_whose_minimum_exceeds_the_balance_is_not_a_candidate(registry):
    registry.ingest_venue("bybit", {"BIG/USDT": ticker(last=100.0)}, markets={})
    registry.apply_execution_venue("bybit", {"BIG/USDT": market_meta(min_cost=50.0)})

    assert registry.rank(capital_quote=13.95) == []
    assert registry.micro_candidates(capital_quote=13.95) == []
    # With more capital the same market is fine.
    assert registry.rank(capital_quote=500.0)


def test_the_fee_reserve_is_counted_against_the_balance(registry):
    """A minimum the balance can only just cover leaves nothing for the exit."""
    registry.ingest_venue("bybit", {"EDGE/USDT": ticker(last=1.0)}, markets={})
    # 10 USDT minimum, 1% taker: 10 + 0.2 round trip = 10.2 required.
    registry.apply_execution_venue(
        "bybit", {"EDGE/USDT": market_meta(min_cost=10.0, taker=0.01)}
    )

    assert registry.rank(capital_quote=10.1) == [], "no room for the round trip"
    assert registry.rank(capital_quote=11.0), "with the reserve covered it is placeable"


def test_micro_candidates_favour_what_a_small_balance_can_actually_fund(broad):
    broad.apply_execution_venue(
        "bybit",
        {
            "BTC/USDT": market_meta(min_cost=100.0),   # unaffordable here
            "ETH/USDT": market_meta(min_cost=50.0),    # unaffordable here
            "DOGE/USDT": market_meta(min_cost=5.0),
            "PEPE/USDT": market_meta(min_cost=5.0),
            "BONK/USDT": market_meta(min_cost=5.0),
            "FLOKI/USDT": market_meta(min_cost=5.0),
            "SOL/USDT": market_meta(min_cost=5.0),
            "TINY/USDT": market_meta(min_cost=5.0),
        },
    )

    candidates = [m.symbol for m in broad.micro_candidates(capital_quote=13.95644171)]

    assert candidates, "a 13.95 USDT wallet has candidates"
    assert "BTC/USDT" not in candidates
    assert "ETH/USDT" not in candidates
    assert any(
        s in candidates for s in ("PEPE/USDT", "BONK/USDT", "FLOKI/USDT", "DOGE/USDT")
    )


def test_with_no_declared_capital_feasibility_does_not_filter(broad):
    """Studying the universe must not require having money."""
    assert len(broad.rank(capital_quote=0.0)) == len(BROAD_TICKERS)


def test_a_wide_spread_is_penalised(registry):
    registry.ingest_venue(
        "bybit",
        {
            "TIGHT/USDT": ticker(last=1.0, bid=0.9999, ask=1.0001),
            "WIDE/USDT": ticker(last=1.0, bid=0.95, ask=1.05),
        },
        markets={},
    )
    registry.apply_execution_venue(
        "bybit", {"TIGHT/USDT": market_meta(), "WIDE/USDT": market_meta()}
    )

    ranked = [m.symbol for m in registry.rank(capital_quote=100.0)]
    assert ranked.index("TIGHT/USDT") < ranked.index("WIDE/USDT")


def test_an_implausible_daily_move_is_not_rewarded_without_limit(registry):
    registry.ingest_venue(
        "bybit",
        {
            "STEADY/USDT": ticker(last=1.0, change=8.0),
            "SPIKE/USDT": ticker(last=1.0, change=400.0),
        },
        markets={},
    )
    registry.apply_execution_venue(
        "bybit", {"STEADY/USDT": market_meta(), "SPIKE/USDT": market_meta()}
    )

    ranked = [m.symbol for m in registry.rank(capital_quote=100.0)]
    assert ranked.index("STEADY/USDT") < ranked.index("SPIKE/USDT")


# ------------------------------------------------- 12. bounded swarm coverage


def test_shards_cover_the_universe_and_are_stable(broad):
    assigned = broad.assign_shards(100)
    assert assigned == len(broad.all_markets())

    shards = {m.symbol: m.shard for m in broad.all_markets()}
    assert all(0 <= s < 100 for s in shards.values())

    # Growing the universe must not reshuffle what an agent already owns.
    broad.ingest_venue("bybit", {"NEW/USDT": ticker()}, markets={})
    broad.assign_shards(100)
    for market in broad.all_markets():
        if market.symbol in shards:
            assert market.shard == shards[market.symbol]


def test_a_cycle_hands_out_one_market_per_agent_not_one_market_to_all(broad):
    items = broad.next_work_items(count=5)

    assert len(items) == 5
    assert len({item.symbol for item in items}) == 5, (
        "every agent got the same market; this is the swarm defect"
    )


def test_work_never_exceeds_the_agent_count(broad):
    assert len(broad.next_work_items(count=3)) == 3
    # More agents than markets: bounded by what exists, not padded.
    assert len(broad.next_work_items(count=1000)) == len(BROAD_TICKERS)


def test_coverage_rotates_so_the_tail_is_not_starved(broad):
    """Repeated cycles must reach markets the first cycle skipped."""
    seen = set()
    for _ in range(6):
        for item in broad.next_work_items(count=3):
            seen.add(item.symbol)
            broad.touch_analysis(item.symbol)

    assert seen == set(BROAD_TICKERS), f"never studied: {set(BROAD_TICKERS) - seen}"


def test_scheduling_spawns_nothing(broad):
    """The scheduler hands out work items; it does not create tasks."""
    items = broad.next_work_items(count=1000)
    assert all(hasattr(item, "symbol") and hasattr(item, "timeframe") for item in items)
    assert len(items) <= len(broad.all_markets())


def test_study_is_not_restricted_to_executable_markets(registry):
    registry.ingest_venue("binance", {"FOREIGN/USDT": ticker()}, markets={})
    registry.apply_execution_venue("bybit", {})

    items = registry.next_work_items(count=10)
    assert [i.symbol for i in items] == ["FOREIGN/USDT"], (
        "a market we cannot trade is still worth learning from"
    )

    executable_only = registry.next_work_items(count=10, executable_only=True)
    assert executable_only == []


def test_timeframes_rotate_across_cycles(broad):
    seen = set()
    for _ in range(8):
        for item in broad.next_work_items(count=2, timeframes=("1m", "5m", "1h")):
            seen.add(item.timeframe)
    assert len(seen) > 1, "the same timeframe every cycle is not coverage"


def test_analysis_is_recorded_and_reported(broad):
    broad.touch_analysis("PEPE/USDT")
    market = broad.get("PEPE/USDT")

    assert market.analysis_count == 1
    assert market.staleness_seconds < 5

    coverage = broad.coverage(window_seconds=60)
    assert coverage["analyzed_in_window"] == 1
    assert coverage["markets_total"] == len(BROAD_TICKERS)


def test_a_never_analysed_market_is_maximally_stale(broad):
    assert broad.get("BTC/USDT").staleness_seconds == float("inf")


# ------------------------------------------------------------- telemetry


def test_telemetry_answers_how_much_of_the_universe_is_in_use(broad):
    broad.assign_shards(100)
    broad.touch_analysis("DOGE/USDT")

    telemetry = broad.telemetry(capital_quote=13.95)

    assert telemetry["markets_discovered_by_venue"] == {"bybit": len(BROAD_TICKERS)}
    assert telemetry["normalized_unique_symbols"] == len(BROAD_TICKERS)
    assert telemetry["active_spot_usdt_markets"] == len(BROAD_TICKERS)
    assert telemetry["markets_assigned_to_swarm"] == len(BROAD_TICKERS)
    assert telemetry["markets_analyzed_last_1m"] == 1
    assert telemetry["markets_analyzed_ever"] == 1
    assert telemetry["execution_venue"] == "bybit"
    assert telemetry["execution_eligible"] == len(BROAD_TICKERS)
    assert telemetry["micro_candidates"] >= 1


def test_telemetry_ranks_why_markets_are_not_executable(registry):
    registry.ingest_venue(
        "binance", {"A/USDT": ticker(), "B/USDT": ticker()}, markets={}
    )
    registry.ingest_venue("bybit", {"C/USDT": ticker()}, markets={})
    registry.apply_execution_venue("bybit", {"C/USDT": market_meta(spot=False)})

    reasons = registry.telemetry()["ineligibility_reasons"]
    assert reasons[NOT_ON_EXECUTION_VENUE] == 2
    assert reasons[NOT_SPOT] == 1


def test_explain_says_why_one_market_never_reached_execution(registry):
    registry.ingest_venue("binance", {"NOPE/USDT": ticker()}, markets={})
    registry.apply_execution_venue("bybit", {})

    explanation = registry.explain("NOPE/USDT")
    assert explanation["known"] is True
    assert explanation["execution_eligible"] is False
    assert explanation["execution_blocker"] == NOT_ON_EXECUTION_VENUE


def test_explain_is_honest_about_a_market_it_never_saw(registry):
    explanation = registry.explain("GHOST/USDT")
    assert explanation["known"] is False
    assert "never discovered" in explanation["reason"]


def test_signal_state_is_tracked_per_market(broad):
    broad.set_signal_state("PEPE/USDT", "buy")
    assert broad.get("PEPE/USDT").signal_state == "buy"
    assert broad.telemetry()["signals_by_state"] == {"buy": 1}


def test_refreshing_a_market_keeps_its_history(broad):
    first = broad.get("DOGE/USDT")
    first_seen = first.first_seen
    broad.touch_analysis("DOGE/USDT")

    time.sleep(0.01)
    broad.ingest_venue("bybit", {"DOGE/USDT": ticker(last=0.15)}, markets={})

    refreshed = broad.get("DOGE/USDT")
    assert refreshed.first_seen == first_seen, "rediscovery is not a new market"
    assert refreshed.analysis_count == 1, "learned context survives a refresh"
    assert refreshed.price == pytest.approx(0.15)
