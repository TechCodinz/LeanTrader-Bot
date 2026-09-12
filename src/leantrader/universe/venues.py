"""What each venue can actually trade, and what LeanTrader has learned trying.

A symbol existing somewhere in the global intelligence universe says nothing
about whether a particular exchange lists it. The runtime was treating those
as the same thing: the swarm studies markets discovered across every connected
venue, and each study went straight to the execution venue's OHLCV endpoint.
For a symbol that venue does not list -- BCH/USDT, FTM/USDT and FIL/USDT on
Bybit among them -- that is an exchange error, logged, every cycle, forever.

Two universes, held apart:

    GLOBAL INTELLIGENCE UNIVERSE
        what can be observed, modelled, compared, paper-traded, learned from.
        Deliberately enormous.

    VENUE EXECUTION UNIVERSE
        what one authenticated destination can trade right now. Necessarily
        small, and knowable before any call is made.

A market in the first but not the second is the normal case, not a fault. It
resolves locally, as a classification, without touching the network.

The negative memory is what makes that cheap. Once a venue's own market
metadata says it does not list a symbol, that is authoritative and worth
remembering -- but not forever, because exchanges list and relist. Every
recorded absence carries its evidence, how often it has been confirmed, and
when it is due to be checked again. Transient failures are recorded
differently: a timeout is not evidence about a market, and must never harden
into "not listed".
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------- states
#
# Expected routing decisions. None of these is an operational error; each is
# a fact about a venue that the router should know before calling it.

LISTED = "LISTED"
NOT_LISTED = "NOT_LISTED"
NOT_SUPPORTED_IN_TESTNET = "NOT_SUPPORTED_IN_TESTNET"
MARKET_TYPE_UNSUPPORTED = "MARKET_TYPE_UNSUPPORTED"
DELISTED = "DELISTED"
TEMPORARILY_SUSPENDED = "TEMPORARILY_SUSPENDED"
AUTH_REQUIRED = "AUTH_REQUIRED"
PRECISION_METADATA_MISSING = "PRECISION_METADATA_MISSING"
VENUE_METADATA_UNAVAILABLE = "VENUE_METADATA_UNAVAILABLE"

# Nothing has been learned about this venue yet -- no metadata read, no
# snapshot loaded. Absence of evidence, which must never be reported as
# evidence of absence.
UNKNOWN_NO_RUNTIME_SNAPSHOT = "UNKNOWN_NO_RUNTIME_SNAPSHOT"

# Transient conditions. These describe the connection, not the market, and
# are held separately so they can never harden into an absence.
NETWORK_ERROR = "NETWORK_ERROR"
RATE_LIMIT = "RATE_LIMIT"
AUTH_FAILURE = "AUTH_FAILURE"

# How long each state is trusted before it is checked again. Chosen by how
# fast the underlying fact actually changes: a delisting is stable, a
# suspension is not, and a timeout says nothing at all beyond "try later".
DEFAULT_TTL_SECONDS: Dict[str, float] = {
    NOT_LISTED: 6 * 3600.0,
    NOT_SUPPORTED_IN_TESTNET: 6 * 3600.0,
    MARKET_TYPE_UNSUPPORTED: 24 * 3600.0,
    DELISTED: 24 * 3600.0,
    TEMPORARILY_SUSPENDED: 900.0,
    PRECISION_METADATA_MISSING: 3600.0,
    VENUE_METADATA_UNAVAILABLE: 300.0,
    AUTH_REQUIRED: 300.0,
    AUTH_FAILURE: 300.0,
    NETWORK_ERROR: 30.0,
    RATE_LIMIT: 60.0,
}

# Nothing is remembered permanently. Exchanges list new markets and relist old
# ones, and a cache that never expires would make LeanTrader permanently wrong
# about a market that came back.
MAX_TTL_SECONDS = 24 * 3600.0

# States that mean "do not call the venue for this symbol".
BLOCKING_STATES = frozenset(
    {
        NOT_LISTED,
        NOT_SUPPORTED_IN_TESTNET,
        MARKET_TYPE_UNSUPPORTED,
        DELISTED,
        TEMPORARILY_SUSPENDED,
        VENUE_METADATA_UNAVAILABLE,
    }
)

# States that describe the connection rather than the market. They suppress
# calls only for as long as their backoff lasts.
TRANSIENT_STATES = frozenset({NETWORK_ERROR, RATE_LIMIT, AUTH_FAILURE})

SPOT = "spot"


def _now() -> float:
    return time.time()


def _jitter(seconds: float) -> float:
    """Spread retries so a venue outage does not produce a synchronised herd."""
    return seconds * (0.75 + random.random() * 0.5)


@dataclass
class CapabilityRecord:
    """One venue's answer about one market, and how we came to believe it."""

    venue: str
    canonical_symbol: str
    market_type: str = SPOT
    environment: str = "live"

    state: str = LISTED
    venue_symbol: str = ""
    evidence: str = ""
    first_observed: float = field(default_factory=_now)
    last_verified: float = field(default_factory=_now)
    verification_count: int = 1
    next_refresh_at: float = 0.0

    # Execution metadata, present only when the venue lists the market.
    active: bool = True
    amount_precision: Optional[float] = None
    price_precision: Optional[float] = None
    tick_size: Optional[float] = None
    amount_step: Optional[float] = None
    min_amount: float = 0.0
    min_notional: float = 0.0
    taker_fee: Optional[float] = None
    maker_fee: Optional[float] = None
    order_types: Tuple[str, ...] = ()

    # Transient backoff bookkeeping.
    consecutive_failures: int = 0

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return (
            self.venue,
            self.canonical_symbol,
            self.market_type,
            self.environment,
        )

    @property
    def expired(self) -> bool:
        return self.next_refresh_at <= _now()

    @property
    def blocking(self) -> bool:
        """Should a caller skip the venue entirely for this market right now?"""
        if self.state == LISTED:
            return False
        if self.expired:
            # Due for revalidation: stop suppressing so the truth can be
            # rechecked rather than assumed indefinitely.
            return False
        return self.state in BLOCKING_STATES or self.state in TRANSIENT_STATES

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["order_types"] = list(self.order_types)
        data["expired"] = self.expired
        return data


@dataclass
class Resolution:
    """The answer to 'may I call this venue for this market, and why'."""

    venue: str
    canonical_symbol: str
    venue_symbol: str
    market_type: str
    environment: str
    callable: bool
    classification: str
    detail: str = ""
    record: Optional[CapabilityRecord] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "venue": self.venue,
            "canonical_symbol": self.canonical_symbol,
            "venue_symbol": self.venue_symbol,
            "market_type": self.market_type,
            "environment": self.environment,
            "callable": self.callable,
            "classification": self.classification,
            "detail": self.detail,
        }


def _order_types(market: Dict[str, Any], exchange_has: Dict[str, Any]) -> Tuple[str, ...]:
    """Order types a venue advertises for this market."""
    types: List[str] = []
    info = market.get("info") if isinstance(market.get("info"), dict) else {}
    declared = info.get("orderTypes") or info.get("order_types")
    if isinstance(declared, (list, tuple)):
        types = [str(t).lower() for t in declared]
    if not types:
        for name, flag in (
            ("market", "createMarketOrder"),
            ("limit", "createLimitOrder"),
            ("stop", "createStopOrder"),
        ):
            if exchange_has.get(flag) or (name in {"market", "limit"} and exchange_has.get("createOrder")):
                types.append(name)
    seen: List[str] = []
    for t in types:
        if t and t not in seen:
            seen.append(t)
    return tuple(seen)


def market_type_of(market: Dict[str, Any]) -> str:
    """Classify a ccxt market by instrument kind.

    The same textual pair means different instruments on different venues and
    within one venue: BTC/USDT spot, BTC/USDT:USDT linear perpetual and a
    dated future are three markets, and conflating them would route an order
    to the wrong instrument.
    """
    if market.get("spot"):
        return SPOT
    if market.get("option"):
        return "option"
    if market.get("swap"):
        return "inverse_swap" if market.get("inverse") else "linear_swap"
    if market.get("future"):
        return "inverse_future" if market.get("inverse") else "linear_future"
    declared = market.get("type")
    return str(declared).lower() if declared else "unknown"


class CapabilityRegistry:
    """Venue capabilities, plus what has been learned about their absences.

    One instance per process (see ``capabilities``). Thread-safe: discovery
    runs in worker threads while routing is read from the event loop.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: Dict[Tuple[str, str, str, str], CapabilityRecord] = {}
        # Transient backoff is held apart from capability facts. Overwriting
        # a LISTED record with a timeout would destroy what the venue told us
        # about the market, and clearing the timeout would then leave nothing
        # -- so a single network blip could harden into "not listed".
        self._transient: Dict[Tuple[str, str, str, str], CapabilityRecord] = {}
        self._venue_metadata_at: Dict[Tuple[str, str], float] = {}
        self._venue_market_counts: Dict[Tuple[str, str], int] = {}
        self._suppressed_calls: Dict[str, int] = {}
        # Observational evidence about routing, persisted so another process
        # can see it. Never a source of trading authority.
        self._routing_counters: Dict[str, int] = {
            "market_calls_considered": 0,
            "market_calls_allowed": 0,
            "market_calls_suppressed_not_listed": 0,
            "market_calls_suppressed_asset_class": 0,
            "market_calls_suppressed_timeframe": 0,
            "market_calls_suppressed_cached_negative": 0,
            "capability_refreshes": 0,
            "capability_revalidations": 0,
        }
        self._alias: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------- ingest

    def record_venue_markets(
        self,
        venue: str,
        markets: Optional[Dict[str, Any]],
        environment: str = "live",
        exchange_has: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Rebuild one venue's capability view from its own market metadata.

        This is the authoritative source. Everything it lists becomes LISTED
        with real precision and limits; anything previously recorded as listed
        for this venue and environment that is now absent becomes DELISTED
        rather than silently vanishing.

        ``markets`` of None means the metadata could not be read, which is not
        evidence that anything is absent -- existing records are left alone and
        the venue is marked unavailable for a short while.
        """
        venue = str(venue or "").strip().lower()
        environment = str(environment or "live").strip().lower()
        if not venue:
            return 0

        if markets is None:
            with self._lock:
                self._remember(
                    CapabilityRecord(
                        venue=venue,
                        canonical_symbol="*",
                        market_type=SPOT,
                        environment=environment,
                        state=VENUE_METADATA_UNAVAILABLE,
                        evidence="load_markets failed",
                    )
                )
            return 0

        from .registry import normalize_symbol

        exchange_has = exchange_has or {}
        seen: Dict[Tuple[str, str, str, str], None] = {}
        now = _now()

        with self._lock:
            for venue_symbol, market in markets.items():
                if not isinstance(market, dict):
                    continue
                canonical = normalize_symbol(
                    market.get("symbol") or venue_symbol
                )
                if not canonical:
                    continue

                market_type = market_type_of(market)
                key = (venue, canonical, market_type, environment)
                seen[key] = None
                self._alias[(venue, canonical)] = str(
                    market.get("id") or venue_symbol
                )

                limits = market.get("limits") or {}
                precision = market.get("precision") or {}
                active = market.get("active") is not False

                existing = self._records.get(key)
                record = CapabilityRecord(
                    venue=venue,
                    canonical_symbol=canonical,
                    market_type=market_type,
                    environment=environment,
                    state=LISTED if active else TEMPORARILY_SUSPENDED,
                    venue_symbol=str(market.get("symbol") or venue_symbol),
                    evidence="venue market metadata",
                    first_observed=existing.first_observed if existing else now,
                    last_verified=now,
                    verification_count=(
                        existing.verification_count + 1 if existing else 1
                    ),
                    active=active,
                    amount_precision=_optional_float(precision.get("amount")),
                    price_precision=_optional_float(precision.get("price")),
                    tick_size=_optional_float(precision.get("price")),
                    amount_step=_optional_float(precision.get("amount")),
                    min_amount=_float((limits.get("amount") or {}).get("min")),
                    min_notional=_float((limits.get("cost") or {}).get("min")),
                    taker_fee=_optional_float(market.get("taker")),
                    maker_fee=_optional_float(market.get("maker")),
                    order_types=_order_types(market, exchange_has),
                )
                self._remember(record)

            # Anything this venue and environment previously listed and no
            # longer does has been delisted. Recording it keeps the router
            # from calling a market that has gone away.
            for key, record in list(self._records.items()):
                if key in seen:
                    continue
                if key[0] != venue or key[3] != environment:
                    continue
                if record.canonical_symbol == "*":
                    continue
                if record.state == LISTED:
                    record.state = DELISTED
                    record.evidence = "absent from venue market metadata"
                    record.last_verified = now
                    record.next_refresh_at = now + DEFAULT_TTL_SECONDS[DELISTED]

            self._routing_counters["capability_refreshes"] = (
                self._routing_counters.get("capability_refreshes", 0) + 1
            )
            self._venue_metadata_at[(venue, environment)] = now
            self._venue_market_counts[(venue, environment)] = len(seen)
            self._records.pop((venue, "*", SPOT, environment), None)

        return len(seen)

    def _remember(self, record: CapabilityRecord) -> None:
        ttl = min(
            DEFAULT_TTL_SECONDS.get(record.state, 3600.0), MAX_TTL_SECONDS
        )
        if record.state in TRANSIENT_STATES:
            # Exponential, bounded, jittered: a venue outage must not produce
            # a synchronised retry herd when it recovers.
            ttl = min(
                ttl * (2 ** max(0, record.consecutive_failures - 1)),
                MAX_TTL_SECONDS,
            )
            ttl = _jitter(ttl)
        record.next_refresh_at = _now() + ttl
        self._records[record.key] = record

    # ------------------------------------------------------------ observe

    def count_routing(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._routing_counters[name] = (
                self._routing_counters.get(name, 0) + amount
            )

    def routing_counters(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._routing_counters)

    def record_absence(
        self,
        venue: str,
        canonical_symbol: str,
        state: str = NOT_LISTED,
        market_type: str = SPOT,
        environment: str = "live",
        evidence: str = "",
    ) -> CapabilityRecord:
        """Record authoritative evidence that a market is unusable here."""
        venue = str(venue or "").strip().lower()
        key = (venue, canonical_symbol, market_type, environment)

        with self._lock:
            existing = self._records.get(key)
            record = CapabilityRecord(
                venue=venue,
                canonical_symbol=canonical_symbol,
                market_type=market_type,
                environment=environment,
                state=state,
                venue_symbol=existing.venue_symbol if existing else "",
                evidence=evidence or "observed",
                first_observed=existing.first_observed if existing else _now(),
                last_verified=_now(),
                verification_count=(
                    existing.verification_count + 1
                    if existing and existing.state == state
                    else 1
                ),
            )
            self._remember(record)
            return record

    def record_transient(
        self,
        venue: str,
        canonical_symbol: str,
        state: str = NETWORK_ERROR,
        market_type: str = SPOT,
        environment: str = "live",
        evidence: str = "",
    ) -> CapabilityRecord:
        """Record a failure of the connection, never of the market.

        A timeout, a rate limit or an auth failure says nothing about whether
        a market exists. Recording these as absences would make LeanTrader
        permanently wrong about a market it simply could not reach once.
        """
        venue = str(venue or "").strip().lower()
        key = (venue, canonical_symbol, market_type, environment)

        with self._lock:
            existing = self._transient.get(key)
            failures = 1
            if existing and existing.state == state:
                failures = existing.consecutive_failures + 1

            known = self._records.get(key)
            record = CapabilityRecord(
                venue=venue,
                canonical_symbol=canonical_symbol,
                market_type=market_type,
                environment=environment,
                state=state,
                venue_symbol=(
                    known.venue_symbol
                    if known
                    else (existing.venue_symbol if existing else "")
                ),
                evidence=evidence or state.lower(),
                first_observed=existing.first_observed if existing else _now(),
                last_verified=_now(),
                verification_count=(
                    existing.verification_count + 1 if existing else 1
                ),
                consecutive_failures=failures,
            )
            ttl = min(
                DEFAULT_TTL_SECONDS.get(state, 60.0)
                * (2 ** max(0, failures - 1)),
                MAX_TTL_SECONDS,
            )
            record.next_refresh_at = _now() + _jitter(ttl)
            self._transient[key] = record
            return record

    def clear_transient(
        self,
        venue: str,
        canonical_symbol: str,
        market_type: str = SPOT,
        environment: str = "live",
    ) -> None:
        """A success cancels any backoff that was in effect."""
        key = (str(venue).lower(), canonical_symbol, market_type, environment)
        with self._lock:
            self._transient.pop(key, None)

    # ------------------------------------------------------------- resolve

    def resolve(
        self,
        venue: str,
        canonical_symbol: str,
        market_type: str = SPOT,
        environment: str = "live",
    ) -> Resolution:
        """May this venue be called for this market, and if not, why not?

        Answered entirely from memory. This is the check that belongs in front
        of every ticker, OHLCV, order-book, preflight and order call: a market
        the venue does not list resolves here, locally, instead of becoming an
        exchange error.
        """
        venue = str(venue or "").strip().lower()
        key = (venue, canonical_symbol, market_type, environment)

        with self._lock:
            self._routing_counters["market_calls_considered"] = (
                self._routing_counters.get("market_calls_considered", 0) + 1
            )
            record = self._records.get(key)
            venue_known = (venue, environment) in self._venue_metadata_at

            backoff = self._transient.get(key)
            if backoff is not None:
                if backoff.next_refresh_at > _now():
                    self._suppressed_calls[backoff.state] = (
                        self._suppressed_calls.get(backoff.state, 0) + 1
                    )
                    self._routing_counters[
                        "market_calls_suppressed_cached_negative"
                    ] = (
                        self._routing_counters.get(
                            "market_calls_suppressed_cached_negative", 0
                        )
                        + 1
                    )
                    return Resolution(
                        venue=venue,
                        canonical_symbol=canonical_symbol,
                        venue_symbol=backoff.venue_symbol,
                        market_type=market_type,
                        environment=environment,
                        callable=False,
                        classification=backoff.state,
                        detail=backoff.evidence,
                        record=backoff,
                    )
                # The backoff has elapsed; drop it and fall through to what
                # is actually known about the market.
                del self._transient[key]

            if record is not None and record.blocking:
                self._suppressed_calls[record.state] = (
                    self._suppressed_calls.get(record.state, 0) + 1
                )
                self._routing_counters[
                    "market_calls_suppressed_not_listed"
                ] = (
                    self._routing_counters.get(
                        "market_calls_suppressed_not_listed", 0
                    )
                    + 1
                )
                return Resolution(
                    venue=venue,
                    canonical_symbol=canonical_symbol,
                    venue_symbol=record.venue_symbol,
                    market_type=market_type,
                    environment=environment,
                    callable=False,
                    classification=record.state,
                    detail=record.evidence,
                    record=record,
                )

            if record is not None and record.state == LISTED:
                self._routing_counters["market_calls_allowed"] = (
                    self._routing_counters.get("market_calls_allowed", 0) + 1
                )
                return Resolution(
                    venue=venue,
                    canonical_symbol=canonical_symbol,
                    venue_symbol=record.venue_symbol,
                    market_type=market_type,
                    environment=environment,
                    callable=True,
                    classification=LISTED,
                    detail=record.evidence,
                    record=record,
                )

            if venue_known and record is None:
                # The venue's own metadata has been read and does not contain
                # this market. That is authoritative, so remember it rather
                # than letting the next caller discover it by failing.
                remembered = CapabilityRecord(
                    venue=venue,
                    canonical_symbol=canonical_symbol,
                    market_type=market_type,
                    environment=environment,
                    state=NOT_LISTED,
                    evidence="absent from venue market metadata",
                )
                self._remember(remembered)
                self._suppressed_calls[NOT_LISTED] = (
                    self._suppressed_calls.get(NOT_LISTED, 0) + 1
                )
                self._routing_counters[
                    "market_calls_suppressed_not_listed"
                ] = (
                    self._routing_counters.get(
                        "market_calls_suppressed_not_listed", 0
                    )
                    + 1
                )
                return Resolution(
                    venue=venue,
                    canonical_symbol=canonical_symbol,
                    venue_symbol="",
                    market_type=market_type,
                    environment=environment,
                    callable=False,
                    classification=NOT_LISTED,
                    detail="absent from venue market metadata",
                    record=remembered,
                )

        # Nothing known yet, or a record that has expired and is due to be
        # rechecked. Allow the call; the caller's result becomes the evidence.
        return Resolution(
            venue=venue,
            canonical_symbol=canonical_symbol,
            venue_symbol=(record.venue_symbol if record else ""),
            market_type=market_type,
            environment=environment,
            callable=True,
            classification="UNKNOWN" if record is None else "REVALIDATE",
            detail=(
                "venue metadata not yet read"
                if record is None
                else f"{record.state} expired; due for revalidation"
            ),
            record=record,
        )

    def venue_metadata_known(self, venue: str, environment: str) -> bool:
        """Has this venue's market metadata ever been read or loaded?

        The difference between "this venue does not list the market" and "we
        have never looked" is the whole of the unknown-vs-absent distinction.
        Callers that have no metadata for a venue must report UNKNOWN, because
        absence of evidence is not evidence of absence.
        """
        with self._lock:
            return (
                str(venue or "").strip().lower(),
                str(environment or "").strip().lower(),
            ) in self._venue_metadata_at

    def venue_symbol(self, venue: str, canonical_symbol: str) -> str:
        """The venue's own notation for a canonical pair, if known."""
        with self._lock:
            record = None
            for market_type in (SPOT, "linear_swap", "inverse_swap"):
                for environment in ("testnet", "live"):
                    record = self._records.get(
                        (
                            str(venue).lower(),
                            canonical_symbol,
                            market_type,
                            environment,
                        )
                    )
                    if record is not None and record.venue_symbol:
                        return record.venue_symbol
        return canonical_symbol

    def venues_listing(
        self,
        canonical_symbol: str,
        market_type: str = SPOT,
        environment: Optional[str] = None,
    ) -> List[str]:
        """Every venue known to list this market right now."""
        with self._lock:
            found = []
            for (venue, symbol, kind, env), record in self._records.items():
                if symbol != canonical_symbol or kind != market_type:
                    continue
                if environment is not None and env != environment:
                    continue
                if record.state == LISTED:
                    found.append(venue)
        return sorted(set(found))

    def record_for(
        self,
        venue: str,
        canonical_symbol: str,
        market_type: str = SPOT,
        environment: str = "live",
    ) -> Optional[CapabilityRecord]:
        """The capability fact. Transient backoff is separate; see backoff_for."""
        key = (str(venue).lower(), canonical_symbol, market_type, environment)
        with self._lock:
            return self._records.get(key) or self._transient.get(key)

    def backoff_for(
        self,
        venue: str,
        canonical_symbol: str,
        market_type: str = SPOT,
        environment: str = "live",
    ) -> Optional[CapabilityRecord]:
        key = (str(venue).lower(), canonical_symbol, market_type, environment)
        with self._lock:
            return self._transient.get(key)

    # ----------------------------------------------------------- telemetry

    def telemetry(self) -> Dict[str, Any]:
        with self._lock:
            by_venue: Dict[str, int] = {}
            by_state: Dict[str, int] = {}
            not_listed_by_venue: Dict[str, int] = {}

            for (venue, _symbol, _kind, environment), record in self._records.items():
                if record.state == LISTED:
                    label = f"{venue}:{environment}"
                    by_venue[label] = by_venue.get(label, 0) + 1
                by_state[record.state] = by_state.get(record.state, 0) + 1
                if record.state == NOT_LISTED:
                    not_listed_by_venue[venue] = (
                        not_listed_by_venue.get(venue, 0) + 1
                    )

            return {
                "records": len(self._records),
                "listed_by_venue_environment": dict(sorted(by_venue.items())),
                "by_state": dict(
                    sorted(by_state.items(), key=lambda kv: kv[1], reverse=True)
                ),
                "known_not_listed_by_venue": dict(
                    sorted(not_listed_by_venue.items())
                ),
                "suppressed_calls_by_reason": dict(
                    sorted(
                        self._suppressed_calls.items(),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                ),
                "suppressed_calls_total": sum(self._suppressed_calls.values()),
                "routing_counters": dict(self._routing_counters),
                "venue_metadata_read": {
                    f"{venue}:{environment}": count
                    for (venue, environment), count
                    in sorted(self._venue_market_counts.items())
                },
            }

    def explain(
        self,
        canonical_symbol: str,
        market_type: str = SPOT,
    ) -> Dict[str, Any]:
        """Every venue's position on one market, with reasons."""
        with self._lock:
            venues: Dict[str, Any] = {}
            for (venue, symbol, kind, environment), record in self._records.items():
                if symbol != canonical_symbol or kind != market_type:
                    continue
                venues.setdefault(venue, {})[environment] = {
                    "state": record.state,
                    "venue_symbol": record.venue_symbol,
                    "evidence": record.evidence,
                    "verification_count": record.verification_count,
                    "last_verified_age_seconds": round(
                        _now() - record.last_verified, 1
                    ),
                    "next_refresh_in_seconds": round(
                        max(0.0, record.next_refresh_at - _now()), 1
                    ),
                    "min_notional": record.min_notional,
                    "min_amount": record.min_amount,
                    "taker_fee": record.taker_fee,
                    "order_types": list(record.order_types),
                }
        return {
            "canonical_symbol": canonical_symbol,
            "market_type": market_type,
            "venues": dict(sorted(venues.items())),
        }

    # --------------------------------------------------------- persistence

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "written_at": _now(),
                "records": [r.as_dict() for r in self._records.values()],
                "venue_market_counts": {
                    f"{venue}:{environment}": count
                    for (venue, environment), count
                    in self._venue_market_counts.items()
                },
                "suppressed_calls_by_reason": dict(self._suppressed_calls),
                "routing_counters": dict(self._routing_counters),
            }

    def load_snapshot(self, payload: Dict[str, Any]) -> int:
        """Restore from a snapshot written by another process."""
        records = payload.get("records") or []
        restored = 0
        with self._lock:
            for raw in records:
                if not isinstance(raw, dict):
                    continue
                raw = dict(raw)
                raw.pop("expired", None)
                raw["order_types"] = tuple(raw.get("order_types") or ())
                try:
                    record = CapabilityRecord(**raw)
                except TypeError:
                    continue
                self._records[record.key] = record
                restored += 1
            for label, count in (payload.get("venue_market_counts") or {}).items():
                venue, _, environment = str(label).partition(":")
                if venue and environment:
                    self._venue_market_counts[(venue, environment)] = int(count)
                    self._venue_metadata_at[(venue, environment)] = _now()
            for reason, count in (
                payload.get("suppressed_calls_by_reason") or {}
            ).items():
                self._suppressed_calls[reason] = int(count)
            for name, count in (payload.get("routing_counters") or {}).items():
                self._routing_counters[name] = int(count)
        return restored

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._transient.clear()
            self._venue_metadata_at.clear()
            self._venue_market_counts.clear()
            self._suppressed_calls.clear()
            self._routing_counters = {k: 0 for k in self._routing_counters}
            self._alias.clear()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# One registry per process.
capabilities = CapabilityRegistry()


RUN_ID = f"{int(time.time())}-{os.getpid()}"


def snapshot_path() -> Path:
    """Where the canonical snapshot lives.

    Defaults to the shared runtime data directory mounted into the
    container, so a separate process can read it. Falls back to the local
    runtime directory when that mount is not present, which is what happens
    outside the container.
    """
    explicit = os.getenv("UNIVERSE_SNAPSHOT_PATH", "").strip()
    if explicit:
        return Path(explicit)

    # One authoritative contract: <data dir>/runtime/universe_snapshot.json.
    # The runtime subdirectory is created if the mount exists, so the
    # location does not depend on the image having pre-made it.
    shared = Path(os.getenv("LEANTRADER_DATA_DIR", "/app/data"))
    if shared.is_dir() and os.access(shared, os.W_OK):
        runtime_dir = shared / "runtime"
        try:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            return runtime_dir / "universe_snapshot.json"
        except OSError:
            return shared / "universe_snapshot.json"

    return Path("runtime/universe_snapshot.json")


def persist_discovery_state(extra: Optional[Dict[str, Any]] = None) -> bool:
    """Write the canonical snapshot. Call this after every discovery refresh.

    This is the integration point the discovery lifecycle uses, named so it
    is obvious at the call site what it is for. Persistence had existed only
    as a helper the maintenance loop called -- so when that loop could not
    resolve a broker, discovery succeeded and nothing was ever written, and a
    separate process saw an empty universe.
    """
    from .registry import universe as market_universe

    # Two layers, deliberately. write_snapshot puts the authoritative state at
    # the top level -- the full per-market records that load_persisted_state
    # restores from -- and these keys are the readable summaries a status
    # report prints without having to re-aggregate anything.
    payload: Dict[str, Any] = {
        "universe": market_universe.telemetry(),
        "capabilities": capabilities.telemetry(),
        "routing_counters": capabilities.routing_counters(),
    }
    if extra:
        payload.update(extra)
    return write_snapshot(payload)


def snapshot_age_seconds(payload: Optional[Dict[str, Any]]) -> Optional[float]:
    """How old a snapshot is, or None when it carries no timestamp.

    Stale state may be displayed as stale. It must never be presented as
    fresh truth.
    """
    if not payload:
        return None
    written = payload.get("written_at")
    try:
        return max(0.0, time.time() - float(written))
    except (TypeError, ValueError):
        return None


def canonical_market_rows() -> List[Dict[str, Any]]:
    """One row per (venue, symbol, market type, environment), for persistence.

    Deliberately flat and self-describing: another process reads this without
    importing anything from here.
    """
    from .instruments import classify_asset
    from .registry import universe as market_universe

    rows: List[Dict[str, Any]] = []
    with capabilities._lock:  # noqa: SLF001 - same module
        records = list(capabilities._records.values())  # noqa: SLF001
        transient = dict(capabilities._transient)  # noqa: SLF001

    for record in records:
        if record.canonical_symbol == "*":
            continue

        market = market_universe.get(record.canonical_symbol)
        backoff = transient.get(record.key)

        rows.append(
            {
                "canonical_symbol": record.canonical_symbol,
                "asset_class": classify_asset(record.canonical_symbol),
                "market_type": record.market_type,
                "venue": record.venue,
                "environment": record.environment,
                "venue_symbol": record.venue_symbol,
                "listed": record.state == LISTED,
                "active": record.active,
                "state": record.state,
                "public_data": record.state == LISTED,
                # Paper never needs a venue listing; testnet and live do, in
                # the environment they were recorded for.
                "paper_capable": True,
                "testnet_capable": (
                    record.state == LISTED and record.environment == "testnet"
                ),
                "live_capable": (
                    record.state == LISTED and record.environment == "live"
                ),
                "amount_precision": record.amount_precision,
                "price_precision": record.price_precision,
                "tick_size": record.tick_size,
                "amount_step": record.amount_step,
                "min_amount": record.min_amount,
                "min_notional": record.min_notional,
                "taker_fee": record.taker_fee,
                "maker_fee": record.maker_fee,
                "order_types": list(record.order_types),
                "negative_state": (backoff.state if backoff else ""),
                "negative_expires_in": (
                    round(max(0.0, backoff.next_refresh_at - _now()), 1)
                    if backoff
                    else None
                ),
                "next_refresh_in": round(
                    max(0.0, record.next_refresh_at - _now()), 1
                ),
                "evidence": record.evidence,
                "verification_count": record.verification_count,
                "first_observed": record.first_observed,
                "last_verified": record.last_verified,
                "execution_eligible": (
                    market.execution_eligible if market else None
                ),
                "research_eligible": True,
            }
        )

    return rows


def write_snapshot(extra: Optional[Dict[str, Any]] = None) -> bool:
    """Persist capability state so another process can read it.

    The registry is an in-process singleton, so tools/execution_status --
    which runs separately -- saw an empty universe no matter how much the
    runtime had discovered. This is the bridge, and it carries real state
    rather than a static fallback.
    """
    path = snapshot_path()
    payload = capabilities.snapshot()
    payload["schema_version"] = 3
    payload["generated_at"] = _now()
    payload["source_run_id"] = RUN_ID
    payload["source_pid"] = os.getpid()
    payload["snapshot_path"] = str(path)
    try:
        payload["markets"] = canonical_market_rows()
    except Exception:
        payload["markets"] = []
    if extra:
        payload.update(extra)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: a reader in another process never sees a partial file.
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def read_snapshot() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(snapshot_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_persisted_state(
    payload: Optional[Dict[str, Any]] = None,
) -> int:
    """Restore capability state written by another process. Returns rows read.

    The symmetric half of persist_discovery_state. Without it, restoring meant
    knowing that the authoritative records sit at the top level of the
    snapshot rather than under its readable "capabilities" summary -- which is
    exactly the kind of thing a caller gets wrong once and then carries.
    """
    payload = read_snapshot() if payload is None else payload
    if not isinstance(payload, dict):
        return 0
    return capabilities.load_snapshot(payload)
