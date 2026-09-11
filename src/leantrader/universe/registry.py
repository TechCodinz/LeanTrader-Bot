"""One authoritative normalized representation of every market seen.

LeanTrader was built to discover and learn across a broad dynamic universe --
thousands of markets across several venues, including micro-priced and
micro-cap ones. The discovery half of that still worked: DynamicMarketScanner
pulls bulk tickers from every connected venue. The half that was missing was
everything downstream. Nothing consumed the scanner's output, so
``trading_universe`` was read in four places and assigned in none, and the
engines fell back to whatever was hardcoded in them: three majors in the
swarm, six in the continuous trader, an empty list in the micro bot.

This is the missing middle. It does not discover; it normalizes, ranks and
schedules what discovery found:

    discovery (native scanner)
      -> ingest_venue()        normalized record per market
      -> apply_execution_venue()  eligibility, from the execution venue only
      -> rank()                what is worth trading with the capital we have
      -> next_work_items()     what the swarm studies this cycle
      -> telemetry()           how much of the universe we are actually using

Two rules hold throughout.

Discovery is not authority. A market seen on Binance is a market we may
study; it is executable only if the venue we are authenticated against lists
it right now. ``execution_eligible`` is set from the execution venue's own
market metadata and from nothing else.

Nominal price is not edge. A token quoted at 0.00000012 is not a better
opportunity than one at 60000 because it has more zeros. Ranking uses
execution economics -- what a real balance can fund at the venue's own
minimums, after fees and spread -- and treats micro-cap status as one feature
among many rather than as a thesis.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Why a market cannot be executed on the authenticated venue. These sit
# alongside the execution blocker classes in execution/preflight.py; these are
# decided from metadata, before an intent exists.
NOT_ON_EXECUTION_VENUE = "NOT_LISTED_ON_EXECUTION_VENUE"
NOT_SPOT = "NOT_SPOT_ON_EXECUTION_VENUE"
INACTIVE = "INACTIVE_ON_EXECUTION_VENUE"
NO_METADATA = "EXECUTION_VENUE_METADATA_UNAVAILABLE"
MIN_NOTIONAL_ABOVE_CAPITAL = "MIN_NOTIONAL_ABOVE_AVAILABLE_CAPITAL"

DEFAULT_TIMEFRAMES: Tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def normalize_symbol(value: Any) -> str:
    """Return a ccxt-style BASE/QUOTE symbol, or "" if it is not one.

    Shared with execution preflight deliberately: a symbol has to mean the
    same thing to the scheduler and to the router, or a market studied under
    one spelling would be looked up for execution under another.
    """
    from ..execution.preflight import normalize_symbol as _normalize

    return _normalize(value)


@dataclass
class Market:
    """What is known about one market on one venue."""

    venue: str
    symbol: str
    base: str = ""
    quote: str = ""
    market_type: str = "spot"
    active: bool = True

    price: float = 0.0
    quote_volume_24h: float = 0.0
    change_pct_24h: float = 0.0
    bid: float = 0.0
    ask: float = 0.0

    amount_precision: Optional[float] = None
    price_precision: Optional[float] = None
    min_amount: float = 0.0
    min_notional: float = 0.0
    taker_fee: float = 0.001

    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_analysis_at: float = 0.0
    analysis_count: int = 0

    supported_timeframes: Tuple[str, ...] = DEFAULT_TIMEFRAMES
    execution_eligible: Optional[bool] = None
    execution_blocker: str = ""
    signal_state: str = "none"
    shard: Optional[int] = None
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.venue}:{self.symbol}"

    @property
    def spread(self) -> Optional[float]:
        """Relative spread, or None when there is no usable book quote."""
        if self.bid > 0.0 and self.ask > self.bid:
            mid = (self.bid + self.ask) / 2.0
            if mid > 0.0:
                return (self.ask - self.bid) / mid
        return None

    @property
    def liquidity_proxy(self) -> float:
        """Stand-in for depth: 24h quote volume is what tickers actually give.

        Named a proxy because it is one. Real depth needs an order book, which
        is not fetched for thousands of markets; a market that ranks well here
        still has its book checked before anything is sent.
        """
        return self.quote_volume_24h

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.first_seen)

    @property
    def staleness_seconds(self) -> float:
        """How long since anything studied this market."""
        if not self.last_analysis_at:
            return float("inf")
        return max(0.0, time.time() - self.last_analysis_at)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["spread"] = self.spread
        data["liquidity_proxy"] = self.liquidity_proxy
        data["supported_timeframes"] = list(self.supported_timeframes)
        return data


@dataclass
class WorkItem:
    """One market/timeframe assignment for one swarm agent, this cycle."""

    symbol: str
    timeframe: str
    venue: str
    priority: float
    reason: str = ""


class MarketUniverse:
    """The registry. One instance owns the normalized view; see ``universe``.

    Thread-safe because discovery runs in worker threads (the scanner's ccxt
    calls go through asyncio.to_thread) while the scheduler is read from the
    event loop.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._markets: Dict[str, Market] = {}
        self._execution_venue: str = ""
        self._execution_symbols: Dict[str, Dict[str, Any]] = {}
        self._execution_metadata_at: float = 0.0
        self._venue_counts: Dict[str, int] = {}
        self._cycle: int = 0
        self._analyzed_at: List[float] = []

    # ------------------------------------------------------------- ingestion

    def ingest_venue(
        self,
        venue: str,
        tickers: Dict[str, Any],
        markets: Optional[Dict[str, Any]] = None,
        quote_filter: Optional[Sequence[str]] = None,
    ) -> int:
        """Fold one venue's bulk tickers into the registry.

        ``tickers`` is what ccxt's fetch_tickers returns; ``markets`` is
        load_markets output when the caller has it, which is where precision
        and the venue minimums come from. Returns how many markets were
        recorded or refreshed.
        """
        venue = str(venue or "").strip().lower()
        if not venue or not tickers:
            return 0

        quotes = tuple(q.upper() for q in (quote_filter or ("USDT",)))
        markets = markets or {}
        seen = 0
        now = time.time()

        with self._lock:
            for raw_symbol, ticker in tickers.items():
                symbol = normalize_symbol(raw_symbol)
                if not symbol:
                    continue
                base, quote = symbol.split("/")
                if quotes and quote not in quotes:
                    continue
                # ccxt uses BASE/QUOTE:SETTLE for derivatives; normalize_symbol
                # drops the settle leg, so the raw spelling is what says
                # whether this was a derivative listing.
                if ":" in str(raw_symbol):
                    continue

                ticker = ticker if isinstance(ticker, dict) else {}
                meta = markets.get(raw_symbol) or markets.get(symbol) or {}
                meta = meta if isinstance(meta, dict) else {}

                key = f"{venue}:{symbol}"
                market = self._markets.get(key)
                if market is None:
                    market = Market(venue=venue, symbol=symbol, base=base, quote=quote)
                    self._markets[key] = market

                market.last_seen = now
                market.price = _f(ticker.get("last") or ticker.get("close"), market.price)
                market.quote_volume_24h = _f(
                    ticker.get("quoteVolume"), market.quote_volume_24h
                )
                market.change_pct_24h = _f(ticker.get("percentage"), market.change_pct_24h)
                market.bid = _f(ticker.get("bid"), market.bid)
                market.ask = _f(ticker.get("ask"), market.ask)

                if meta:
                    market.active = meta.get("active") is not False
                    market.market_type = (
                        "spot" if meta.get("spot") else str(meta.get("type") or "spot")
                    )
                    market.taker_fee = _f(meta.get("taker"), market.taker_fee)
                    limits = meta.get("limits") or {}
                    amount = limits.get("amount") or {}
                    cost = limits.get("cost") or {}
                    market.min_amount = _f(amount.get("min"), market.min_amount)
                    market.min_notional = _f(cost.get("min"), market.min_notional)
                    precision = meta.get("precision") or {}
                    if precision.get("amount") is not None:
                        market.amount_precision = _f(precision.get("amount"), 0.0)
                    if precision.get("price") is not None:
                        market.price_precision = _f(precision.get("price"), 0.0)

                seen += 1

            self._venue_counts[venue] = sum(
                1 for m in self._markets.values() if m.venue == venue
            )

        return seen

    # -------------------------------------------------------- execution view

    def apply_execution_venue(
        self,
        venue: str,
        markets: Optional[Dict[str, Any]],
    ) -> int:
        """Record which markets the authenticated venue actually lists.

        This is the only thing that can make a market executable. A symbol
        discovered on another venue stays study-only until it appears here,
        under the same normalized spelling, as an active spot market.

        Passing ``None`` means the metadata could not be read, which is not
        the same as "nothing is eligible": every market is marked with
        EXECUTION_VENUE_METADATA_UNAVAILABLE rather than a false negative.
        """
        venue = str(venue or "").strip().lower()

        with self._lock:
            self._execution_venue = venue

            if markets is None:
                self._execution_symbols = {}
                self._execution_metadata_at = 0.0
                for market in self._markets.values():
                    market.execution_eligible = None
                    market.execution_blocker = NO_METADATA
                return 0

            normalized: Dict[str, Dict[str, Any]] = {}
            for raw_symbol, meta in markets.items():
                if ":" in str(raw_symbol):
                    continue
                symbol = normalize_symbol(raw_symbol)
                if symbol and isinstance(meta, dict):
                    normalized[symbol] = meta

            self._execution_symbols = normalized
            self._execution_metadata_at = time.time()

            eligible = 0
            for market in self._markets.values():
                meta = normalized.get(market.symbol)
                if meta is None:
                    market.execution_eligible = False
                    market.execution_blocker = NOT_ON_EXECUTION_VENUE
                    continue
                if meta.get("spot") is not True:
                    market.execution_eligible = False
                    market.execution_blocker = NOT_SPOT
                    continue
                if meta.get("active") is False:
                    market.execution_eligible = False
                    market.execution_blocker = INACTIVE
                    continue

                market.execution_eligible = True
                market.execution_blocker = ""
                eligible += 1

                # The execution venue's own limits win: they are the ones an
                # order is actually checked against.
                limits = meta.get("limits") or {}
                market.min_amount = _f(
                    (limits.get("amount") or {}).get("min"), market.min_amount
                )
                market.min_notional = _f(
                    (limits.get("cost") or {}).get("min"), market.min_notional
                )
                market.taker_fee = _f(meta.get("taker"), market.taker_fee)

            return eligible

    def execution_venue_lists(self, symbol: str) -> bool:
        """Does the authenticated venue currently list this symbol as spot?"""
        normalized = normalize_symbol(symbol)
        with self._lock:
            meta = self._execution_symbols.get(normalized)
        if not isinstance(meta, dict):
            return False
        return meta.get("spot") is True and meta.get("active") is not False

    # ---------------------------------------------------------------- access

    def all_markets(self) -> List[Market]:
        with self._lock:
            return list(self._markets.values())

    def get(self, symbol: str, venue: Optional[str] = None) -> Optional[Market]:
        normalized = normalize_symbol(symbol)
        with self._lock:
            if venue:
                return self._markets.get(f"{str(venue).lower()}:{normalized}")
            for market in self._markets.values():
                if market.symbol == normalized:
                    return market
        return None

    def symbols(self, executable_only: bool = False) -> List[str]:
        """Distinct normalized symbols, deduplicated across venues."""
        with self._lock:
            found = {
                m.symbol
                for m in self._markets.values()
                if not executable_only or m.execution_eligible
            }
        return sorted(found)

    def touch_analysis(self, symbol: str, venue: Optional[str] = None) -> None:
        """Record that something studied this market just now."""
        market = self.get(symbol, venue)
        if market is None:
            return
        with self._lock:
            market.last_analysis_at = time.time()
            market.analysis_count += 1
            self._analyzed_at.append(market.last_analysis_at)
            if len(self._analyzed_at) > 20000:
                del self._analyzed_at[:10000]

    def set_signal_state(self, symbol: str, state: str) -> None:
        market = self.get(symbol)
        if market is not None:
            with self._lock:
                market.signal_state = str(state)

    # --------------------------------------------------------------- ranking

    def rank(
        self,
        capital_quote: float = 0.0,
        limit: int = 0,
        executable_only: bool = True,
        min_quote_volume: float = 0.0,
    ) -> List[Market]:
        """Markets worth trading, best first, for the capital we actually have.

        Scoring is multiplicative over factors that each answer a question a
        trader would ask: can this order be placed at all, is there enough
        volume behind it, is it moving, is the spread survivable, and has it
        been ignored long enough to be worth another look. Nominal unit price
        is not a factor.
        """
        markets = self.all_markets()
        scored: List[Tuple[float, Market]] = []

        for market in markets:
            if executable_only and not market.execution_eligible:
                continue
            if not market.active:
                continue
            if market.price <= 0.0:
                continue
            if min_quote_volume and market.quote_volume_24h < min_quote_volume:
                continue

            feasibility = self._feasibility_score(market, capital_quote)
            if feasibility <= 0.0:
                market.scores["feasibility"] = 0.0
                continue

            liquidity = self._liquidity_score(market)
            movement = self._movement_score(market)
            spread = self._spread_score(market)
            freshness = self._freshness_score(market)

            score = feasibility * liquidity * movement * spread * freshness
            market.scores = {
                "feasibility": round(feasibility, 6),
                "liquidity": round(liquidity, 6),
                "movement": round(movement, 6),
                "spread": round(spread, 6),
                "freshness": round(freshness, 6),
                "total": round(score, 8),
            }
            scored.append((score, market))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        ordered = [market for _, market in scored]
        return ordered[:limit] if limit else ordered

    def _feasibility_score(self, market: Market, capital_quote: float) -> float:
        """Can this capital place, and later exit, a position in this market?

        Zero means the order could not be placed at all -- the venue's own
        minimum is more than the balance can fund once fees and an exit are
        reserved. Between zero and one, higher means the minimum leaves more
        headroom, because a position sized at exactly the minimum has no room
        to be scaled or partially closed.
        """
        if capital_quote <= 0.0:
            # No capital declared: rank on market quality alone rather than
            # pretending every market is equally affordable.
            return 1.0

        floor = max(market.min_notional, market.min_amount * market.price)
        if floor <= 0.0:
            # No published minimum. Treat as placeable but do not reward it
            # over a market whose limits we actually know.
            return 0.8

        # Both legs of the round trip, plus a reserve so a fill does not
        # leave the account unable to pay the exit.
        round_trip_fees = floor * market.taker_fee * 2.0
        required = floor + round_trip_fees
        if required > capital_quote:
            return 0.0

        headroom = capital_quote / required
        # 1x required is barely placeable; 5x or more is comfortable.
        return min(1.0, 0.2 + 0.2 * min(headroom, 4.0))

    def _liquidity_score(self, market: Market) -> float:
        """Volume, compressed. A 100x volume difference is not 100x the edge."""
        volume = market.liquidity_proxy
        if volume <= 0.0:
            return 0.05
        return min(1.0, math.log10(1.0 + volume) / 8.0)

    def _movement_score(self, market: Market) -> float:
        """Something has to be moving, but a 60% daily move is a warning."""
        change = abs(market.change_pct_24h)
        if change <= 0.0:
            return 0.1
        if change <= 15.0:
            return min(1.0, 0.2 + change / 15.0 * 0.8)
        # Beyond 15% the move is more likely to be a listing event, a
        # squeeze, or something we cannot exit cleanly.
        return max(0.15, 1.0 - (change - 15.0) / 85.0)

    def _spread_score(self, market: Market) -> float:
        """A spread wider than the expected move is a loss on entry."""
        spread = market.spread
        if spread is None:
            return 0.6
        if spread <= 0.0:
            return 1.0
        if spread >= 0.02:
            return 0.05
        return max(0.05, 1.0 - spread / 0.02)

    def _freshness_score(self, market: Market) -> float:
        """Reward markets nothing has looked at recently, so coverage rotates."""
        staleness = market.staleness_seconds
        if staleness == float("inf"):
            return 1.0
        return min(1.0, 0.3 + staleness / 600.0 * 0.7)

    # --------------------------------------------------- micro-capital view

    def micro_candidates(
        self,
        capital_quote: float,
        limit: int = 50,
        max_unit_price: float = 0.0,
    ) -> List[Market]:
        """Ranked markets a small account can actually trade.

        "Micro" here is about the account, not the token. A wallet holding a
        few tens of USDT needs markets whose minimum notional it can fund with
        room left for fees and an exit -- which is a property of the venue's
        limits, not of how many zeros the price has. ``max_unit_price`` is
        offered because low unit price genuinely helps precision granularity,
        but it is off by default and is never the ranking itself.
        """
        ranked = self.rank(capital_quote=capital_quote, executable_only=True)
        if max_unit_price > 0.0:
            ranked = [m for m in ranked if 0.0 < m.price <= max_unit_price]
        return ranked[:limit] if limit else ranked

    def major_candidates(self, limit: int = 20) -> List[Market]:
        """The high-volume end of the universe, for comparison and reporting."""
        ranked = sorted(
            (m for m in self.all_markets() if m.execution_eligible),
            key=lambda m: m.quote_volume_24h,
            reverse=True,
        )
        return ranked[:limit] if limit else ranked

    # ------------------------------------------------------------ scheduling

    def assign_shards(self, shard_count: int) -> int:
        """Spread every known market across ``shard_count`` shards.

        Assignment is by symbol hash, so a market keeps its shard as the
        universe grows and whatever state an agent built for it stays with the
        same agent instead of being reshuffled every scan.
        """
        if shard_count <= 0:
            return 0
        with self._lock:
            for market in self._markets.values():
                market.shard = hash(market.symbol) % shard_count
            return len(self._markets)

    def next_work_items(
        self,
        count: int,
        timeframes: Optional[Sequence[str]] = None,
        capital_quote: float = 0.0,
        executable_only: bool = False,
    ) -> List[WorkItem]:
        """What the swarm should study this cycle.

        The swarm has a fixed number of agents and the universe has thousands
        of markets, so coverage is a rotation rather than a sweep: each cycle
        takes the highest-priority markets that are not currently fresh, one
        work item per agent. Priority already includes staleness, so a market
        that keeps losing rises until it is picked, and nothing is starved.

        Study is deliberately not restricted to executable markets by default
        -- learning from a venue we cannot trade is the point of watching it.
        """
        if count <= 0:
            return []

        timeframes = tuple(timeframes or DEFAULT_TIMEFRAMES)
        ranked = self.rank(
            capital_quote=capital_quote,
            executable_only=executable_only,
        )
        if not ranked:
            return []

        with self._lock:
            self._cycle += 1
            cycle = self._cycle

        items: List[WorkItem] = []
        for index, market in enumerate(ranked):
            if len(items) >= count:
                break
            # Rotate the timeframe with the cycle so a market is not always
            # studied on the same one.
            supported = market.supported_timeframes or timeframes
            eligible = [tf for tf in timeframes if tf in supported] or list(timeframes)
            timeframe = eligible[(cycle + index) % len(eligible)]
            items.append(
                WorkItem(
                    symbol=market.symbol,
                    timeframe=timeframe,
                    venue=market.venue,
                    priority=market.scores.get("total", 0.0),
                    reason=(
                        "stale" if market.staleness_seconds > 600 else "ranked"
                    ),
                )
            )
        return items

    def coverage(self, window_seconds: float = 300.0) -> Dict[str, Any]:
        """How much of the universe has actually been studied recently."""
        cutoff = time.time() - window_seconds
        markets = self.all_markets()
        analyzed = [m for m in markets if m.last_analysis_at >= cutoff]
        ever = [m for m in markets if m.analysis_count > 0]
        return {
            "markets_total": len(markets),
            "analyzed_in_window": len(analyzed),
            "analyzed_ever": len(ever),
            "window_seconds": window_seconds,
            "coverage_fraction": (len(analyzed) / len(markets)) if markets else 0.0,
        }

    # ------------------------------------------------------------- telemetry

    def telemetry(self, capital_quote: float = 0.0) -> Dict[str, Any]:
        """Counts that answer how much of what we found we are actually using."""
        markets = self.all_markets()
        now = time.time()

        by_venue: Dict[str, int] = {}
        for market in markets:
            by_venue[market.venue] = by_venue.get(market.venue, 0) + 1

        eligible = [m for m in markets if m.execution_eligible]
        ineligible = [m for m in markets if m.execution_eligible is False]

        blockers: Dict[str, int] = {}
        for market in ineligible:
            reason = market.execution_blocker or "UNCLASSIFIED"
            blockers[reason] = blockers.get(reason, 0) + 1

        signals: Dict[str, int] = {}
        for market in markets:
            if market.signal_state and market.signal_state != "none":
                signals[market.signal_state] = signals.get(market.signal_state, 0) + 1

        def analyzed_since(seconds: float) -> int:
            cutoff = now - seconds
            return sum(1 for m in markets if m.last_analysis_at >= cutoff)

        micro = self.micro_candidates(capital_quote, limit=0) if capital_quote else []

        return {
            "markets_discovered_by_venue": by_venue,
            "normalized_unique_symbols": len({m.symbol for m in markets}),
            "active_spot_usdt_markets": sum(
                1
                for m in markets
                if m.active and m.market_type == "spot" and m.quote == "USDT"
            ),
            "markets_assigned_to_swarm": sum(1 for m in markets if m.shard is not None),
            "markets_analyzed_last_1m": analyzed_since(60),
            "markets_analyzed_last_5m": analyzed_since(300),
            "markets_analyzed_ever": sum(1 for m in markets if m.analysis_count > 0),
            "micro_candidates": len(micro),
            "major_candidates": len(self.major_candidates(limit=0)),
            "execution_eligible": len(eligible),
            "execution_ineligible": len(ineligible),
            "execution_eligibility_unknown": sum(
                1 for m in markets if m.execution_eligible is None
            ),
            "execution_venue": self._execution_venue,
            "execution_metadata_age_seconds": (
                now - self._execution_metadata_at if self._execution_metadata_at else None
            ),
            "signals_by_state": signals,
            "ineligibility_reasons": dict(
                sorted(blockers.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "cycles": self._cycle,
        }

    def explain(self, symbol: str) -> Dict[str, Any]:
        """Why this particular market has or has not reached execution."""
        market = self.get(symbol)
        if market is None:
            return {
                "symbol": normalize_symbol(symbol) or str(symbol),
                "known": False,
                "reason": "never discovered on any connected venue",
            }

        return {
            "symbol": market.symbol,
            "known": True,
            "venue": market.venue,
            "active": market.active,
            "market_type": market.market_type,
            "price": market.price,
            "quote_volume_24h": market.quote_volume_24h,
            "spread": market.spread,
            "min_notional": market.min_notional,
            "min_amount": market.min_amount,
            "execution_eligible": market.execution_eligible,
            "execution_blocker": market.execution_blocker,
            "analysis_count": market.analysis_count,
            "staleness_seconds": (
                None if market.staleness_seconds == float("inf")
                else round(market.staleness_seconds, 1)
            ),
            "signal_state": market.signal_state,
            "shard": market.shard,
            "scores": dict(market.scores),
        }

    def reset(self) -> None:
        """Drop everything. For tests."""
        with self._lock:
            self._markets.clear()
            self._execution_symbols.clear()
            self._execution_venue = ""
            self._execution_metadata_at = 0.0
            self._venue_counts.clear()
            self._cycle = 0
            self._analyzed_at.clear()


# One registry per process. The engines import this rather than building
# their own, so there is a single normalized view rather than one per engine.
universe = MarketUniverse()


def configured_quotes() -> Tuple[str, ...]:
    """Quote currencies to keep during discovery."""
    raw = os.getenv("UNIVERSE_QUOTES", "USDT")
    quotes = tuple(q.strip().upper() for q in raw.split(",") if q.strip())
    return quotes or ("USDT",)
