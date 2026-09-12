"""Execution preflight: turn a strategy intent into a submittable order.

Between a signal and BrokerCCXT there is a run of checks that each have to
pass before an exchange will accept anything: the venue has to be the one the
runtime is authenticated against, the symbol has to name a market that venue
actually lists, that market has to be spot and active, there has to be free
quote balance, the size has to clear the venue's minimum notional and minimum
amount, and it has to survive rounding to the venue's amount precision.

Callers used to do none of this. They sized from a fixed table, passed the
result straight to the router, and read the refusal receipt as a fill. This
module does the checks in order and returns either a PreparedOrder or a
Blocked carrying the first class that stopped it, so a run that submits
nothing can say which stage it stopped at and how often.

Nothing here submits an order. Placement stays with route_order.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .broker_ccxt import BrokerCCXT


# Blocker classes, ordered by the pipeline stage that raises them. A run that
# produces no authenticated order should be explainable by exactly one of
# these per attempt -- the first stage that refused.
STRATEGY_REJECT = "STRATEGY_REJECT"
CONFIDENCE_BELOW_THRESHOLD = "CONFIDENCE_BELOW_THRESHOLD"
SIGNAL_STALE = "SIGNAL_STALE"
INVALID_INTENT = "INVALID_INTENT"
SYMBOL_NOT_NORMALIZED = "SYMBOL_NOT_NORMALIZED"
VENUE_NOT_ELIGIBLE = "VENUE_NOT_ELIGIBLE"
NO_EXECUTION_AUTHORITY = "NO_EXECUTION_AUTHORITY"
MARKET_METADATA_UNAVAILABLE = "MARKET_METADATA_UNAVAILABLE"
MARKET_NOT_LISTED = "MARKET_NOT_LISTED"
MARKET_NOT_SPOT = "MARKET_NOT_SPOT"
MARKET_INACTIVE = "MARKET_INACTIVE"
REFERENCE_PRICE_UNAVAILABLE = "REFERENCE_PRICE_UNAVAILABLE"
BALANCE_UNAVAILABLE = "BALANCE_UNAVAILABLE"
INSUFFICIENT_FREE_BALANCE = "INSUFFICIENT_FREE_BALANCE"
RISK_REJECT = "RISK_REJECT"
BELOW_MIN_NOTIONAL = "BELOW_MIN_NOTIONAL"
BELOW_MIN_AMOUNT = "BELOW_MIN_AMOUNT"
PRECISION_COLLAPSED_TO_ZERO = "PRECISION_COLLAPSED_TO_ZERO"
ROUTER_REFUSED = "ROUTER_REFUSED"
EXCHANGE_REJECT = "EXCHANGE_REJECT"
NO_ORDER_ID = "NO_ORDER_ID"
CAPITAL_BELOW_EXECUTABLE_MINIMUM = "CAPITAL_BELOW_EXECUTABLE_MINIMUM"

BLOCKER_CLASSES: Tuple[str, ...] = (
    STRATEGY_REJECT,
    CONFIDENCE_BELOW_THRESHOLD,
    SIGNAL_STALE,
    INVALID_INTENT,
    SYMBOL_NOT_NORMALIZED,
    VENUE_NOT_ELIGIBLE,
    NO_EXECUTION_AUTHORITY,
    MARKET_METADATA_UNAVAILABLE,
    MARKET_NOT_LISTED,
    MARKET_NOT_SPOT,
    MARKET_INACTIVE,
    REFERENCE_PRICE_UNAVAILABLE,
    BALANCE_UNAVAILABLE,
    INSUFFICIENT_FREE_BALANCE,
    RISK_REJECT,
    BELOW_MIN_NOTIONAL,
    BELOW_MIN_AMOUNT,
    PRECISION_COLLAPSED_TO_ZERO,
    ROUTER_REFUSED,
    EXCHANGE_REJECT,
    NO_ORDER_ID,
    CAPITAL_BELOW_EXECUTABLE_MINIMUM,
)


# ------------------------------------------------------------------ counters


_COUNTER_LOCK = threading.Lock()

# Identifies this process's counters, so a restart does not make last run's
# submissions look like this one's.
RUN_ID = f"{int(time.time())}-{os.getpid()}"


def _counter_path() -> Path:
    return Path(
        os.getenv("EXECUTION_TELEMETRY_PATH", "runtime/execution_telemetry.json")
    )


# One order lifecycle narrows monotonically: every acknowledgement had a
# submission, every fill had an acknowledgement. A violation means either a
# counter is being incremented at the wrong place or two different
# populations are being added together -- both worth knowing about.
LIFECYCLE_ORDER = (
    "attempts",
    "prepared",
    "submitted",
    "acknowledged",
    "fills",
    "closes",
)

# Orders that belong to a previous process. They are real evidence and must
# not be discarded, but adding them to this run's submissions would make the
# funnel describe something that never happened.
RECOVERED_KEYS = (
    "recovered_orders",
    "reconciled_external_orders",
)


def _blank_state() -> Dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "attempts": 0,
        "prepared": 0,
        "submitted": 0,
        "acknowledged": 0,
        "fills": 0,
        "closes": 0,
        "recovered_orders": 0,
        "reconciled_external_orders": 0,
        "blockers": {},
        "stages": {},
        "runs": {},
        "started_at": time.time(),
        "updated_at": time.time(),
    }


def _load_state() -> Dict[str, Any]:
    try:
        state = json.loads(_counter_path().read_text(encoding="utf-8"))
    except Exception:
        return _blank_state()
    if not isinstance(state, dict):
        return _blank_state()
    blank = _blank_state()
    blank.update(state)
    for key in ("blockers", "stages"):
        if not isinstance(blank.get(key), dict):
            blank[key] = {}
    return blank


def _store_state(state: Dict[str, Any]) -> None:
    path = _counter_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Telemetry must never take down execution.
        pass


def record_blocker(blocker: str, detail: str = "") -> None:
    """Count one blocked attempt.

    ``detail`` is a short, non-secret description (a symbol, a venue, an
    exchange error class). Credentials never reach this path because callers
    pass exchange error type names rather than raw responses.
    """
    blocker = str(blocker or "").strip().upper() or "UNCLASSIFIED"
    with _COUNTER_LOCK:
        state = _load_state()
        entry = state["blockers"].get(blocker)
        if not isinstance(entry, dict):
            entry = {"count": 0}
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["last_seen"] = time.time()
        if detail:
            entry["last_detail"] = str(detail)[:200]
        state["blockers"][blocker] = entry
        state["updated_at"] = time.time()
        _store_state(state)


def record_event(name: str, count: int = 1) -> None:
    """Increment one lifecycle counter, for all time and for this run.

    The persisted totals survive restarts, which is what makes them evidence.
    The per-run block is what makes the funnel honest: after a restart the
    cumulative totals still carry last run's acknowledgements, and reading
    those as this run's activity is how a funnel comes to show more
    acknowledgements than submissions.
    """
    with _COUNTER_LOCK:
        state = _load_state()
        state[name] = int(state.get(name, 0)) + int(count)

        runs = state.setdefault("runs", {})
        current = runs.setdefault(RUN_ID, {"started_at": time.time()})
        current[name] = int(current.get(name, 0)) + int(count)
        current["updated_at"] = time.time()

        # Keep the last few runs only; this is a counter file, not a log.
        if len(runs) > 5:
            for stale in sorted(
                runs, key=lambda r: runs[r].get("started_at", 0)
            )[:-5]:
                del runs[stale]

        state["updated_at"] = time.time()
        _store_state(state)


def current_run_counters() -> Dict[str, int]:
    """This process's counters, separate from anything inherited."""
    with _COUNTER_LOCK:
        state = _load_state()
        return dict((state.get("runs") or {}).get(RUN_ID, {}))


def lifecycle_violations(counters: Dict[str, Any]) -> List[str]:
    """Where the funnel widens when it should only narrow.

    Returns a description per violation. An empty list means the counters
    describe a lifecycle that could actually have happened.
    """
    violations: List[str] = []
    previous_name = None
    previous_value = None

    for name in LIFECYCLE_ORDER:
        value = int(counters.get(name, 0) or 0)
        if previous_value is not None and value > previous_value:
            violations.append(
                f"{name}={value} exceeds {previous_name}={previous_value}"
            )
        previous_name, previous_value = name, value

    return violations


def record_stage_latency(stage: str, seconds: float) -> None:
    """Accumulate wall time for one pipeline stage."""
    with _COUNTER_LOCK:
        state = _load_state()
        entry = state["stages"].get(stage)
        if not isinstance(entry, dict):
            entry = {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0}
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["total_seconds"] = float(entry.get("total_seconds", 0.0)) + float(seconds)
        entry["max_seconds"] = max(float(entry.get("max_seconds", 0.0)), float(seconds))
        state["stages"][stage] = entry
        state["updated_at"] = time.time()
        _store_state(state)


def telemetry_snapshot() -> Dict[str, Any]:
    """Read the persisted counters. Used by status reporting."""
    with _COUNTER_LOCK:
        return _load_state()


def record_recovered_order(external: bool = False) -> None:
    """An order found on the exchange that this run did not submit.

    Counted apart from the run's own submissions. Discarding it would throw
    away real exchange evidence; adding it to submitted would describe a
    submission that never happened here.
    """
    record_event(
        "reconciled_external_orders" if external else "recovered_orders"
    )


# ------------------------------------------------------------ shared clients


_BROKER_CACHE: Dict[Tuple[str, str, str], BrokerCCXT] = {}
_BROKER_LOCK = threading.Lock()


def shared_broker(
    execution_mode: Optional[str] = None,
    exchange_id: Optional[str] = None,
    market_mode: Optional[str] = None,
) -> BrokerCCXT:
    """Reuse one BrokerCCXT per (mode, venue, market type).

    route_order builds a fresh broker per call, so every preflight check would
    otherwise re-create a ccxt client and re-probe the environment. Markets and
    balances are read through this instance so the ccxt client, its loaded
    markets and its resolved mode survive between attempts.

    Credentials are resolved inside BrokerCCXT from the runtime environment;
    this cache holds no credential material of its own and is keyed only by
    non-secret routing facts.
    """
    key = (
        str(execution_mode or "").lower(),
        str(exchange_id or "").lower(),
        str(market_mode or "").lower(),
    )
    with _BROKER_LOCK:
        broker = _BROKER_CACHE.get(key)
        if broker is None:
            broker = BrokerCCXT(
                execution_mode=execution_mode,
                exchange_id=exchange_id,
                market_mode=market_mode,
            )
            _BROKER_CACHE[key] = broker
        return broker


def reset_shared_brokers() -> None:
    """Drop cached clients. Tests and credential rotation use this."""
    with _BROKER_LOCK:
        _BROKER_CACHE.clear()


# ------------------------------------------------------------------- caching


_MARKETS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_BALANCE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_LOCK = threading.Lock()


def _markets_ttl() -> float:
    return float(os.getenv("EXECUTION_MARKETS_TTL_SECONDS", "900"))


def _balance_ttl() -> float:
    return float(os.getenv("EXECUTION_BALANCE_TTL_SECONDS", "5"))


def load_markets_cached(broker: BrokerCCXT) -> Dict[str, Any]:
    key = f"{broker.exchange_id}:{broker.market_mode}:{broker.resolve_mode()}"
    now = time.time()
    with _CACHE_LOCK:
        cached = _MARKETS_CACHE.get(key)
        if cached and now - cached[0] < _markets_ttl():
            return cached[1]
    started = time.time()
    markets = broker.load_markets() or {}
    record_stage_latency("load_markets", time.time() - started)
    with _CACHE_LOCK:
        _MARKETS_CACHE[key] = (now, markets)
    return markets


def fetch_balance_cached(broker: BrokerCCXT) -> Dict[str, Any]:
    key = f"{broker.exchange_id}:{broker.market_mode}:{broker.resolve_mode()}"
    now = time.time()
    with _CACHE_LOCK:
        cached = _BALANCE_CACHE.get(key)
        if cached and now - cached[0] < _balance_ttl():
            return cached[1]
    started = time.time()
    balance = broker.fetch_balance() or {}
    record_stage_latency("fetch_balance", time.time() - started)
    with _CACHE_LOCK:
        _BALANCE_CACHE[key] = (now, balance)
    return balance


_TICKER_CACHE: Dict[str, Tuple[float, float]] = {}


def _ticker_ttl() -> float:
    return float(os.getenv("EXECUTION_TICKER_TTL_SECONDS", "2"))


def fetch_last_price_cached(broker: BrokerCCXT, symbol: str) -> Optional[float]:
    """Last traded price for ``symbol`` on the venue we execute against.

    Priced through the same broker that will place the order, so a position on
    one venue is never marked against another venue's book, and the ccxt client
    is reused instead of built per call.
    """
    key = f"{broker.exchange_id}:{broker.resolve_mode()}:{symbol}"
    now = time.time()
    with _CACHE_LOCK:
        cached = _TICKER_CACHE.get(key)
        if cached and now - cached[0] < _ticker_ttl():
            return cached[1]

    # A venue that does not list this market has no price for it. Asking
    # produces an exchange error rather than an answer, so the question is
    # settled locally.
    if not _venue_may_serve(broker, symbol):
        return None

    started = time.time()
    ticker = broker.fetch_ticker(symbol) or {}
    record_stage_latency("fetch_ticker", time.time() - started)

    price = None
    for field_name in ("last", "close", "bid"):
        value = ticker.get(field_name)
        if value:
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if candidate > 0.0:
                price = candidate
                break

    if price is None:
        return None

    with _CACHE_LOCK:
        _TICKER_CACHE[key] = (now, price)
    return price


_INFEASIBLE: Dict[str, Tuple[float, str]] = {}
_INFEASIBLE_LOCK = threading.Lock()


def _infeasible_ttl() -> float:
    return float(os.getenv("EXECUTION_INFEASIBLE_TTL_SECONDS", "300"))


def _bucket(value: float, step: float) -> int:
    """Coarse buckets, so drift within the noise is not a new question."""
    if value <= 0 or step <= 0:
        return 0
    return int(value / step)


def _infeasible_key(
    venue: str,
    symbol: str,
    capital: float,
    price: float = 0.0,
    min_notional: float = 0.0,
    min_amount: float = 0.0,
    risk_budget: float = 0.0,
) -> str:
    """Identify the exact question that was answered "not fundable".

    The cooldown must lift when anything that could change the answer
    changes: the balance, the price, the venue's minimums, or the risk
    budget. Keying on capital alone would keep refusing a market whose
    minimum had since dropped, or whose price had halved.
    """
    return ":".join(
        (
            venue,
            symbol,
            str(_bucket(capital, 0.5)),
            str(_bucket(price, max(price * 0.02, 1e-12))),
            str(min_notional),
            str(min_amount),
            str(_bucket(risk_budget, 0.5)),
        )
    )


@dataclass
class MinimumTicket:
    """The smallest spend this venue will actually accept for this market."""

    symbol: str
    venue: str
    price: float
    min_notional: float
    min_amount: float
    fee_rate: float
    amount: float = 0.0
    notional: float = 0.0
    total_cost: float = 0.0
    fee_allowance: float = 0.0
    slippage_allowance: float = 0.0
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "venue": self.venue,
            "amount": self.amount,
            "notional": self.notional,
            "total_cost": self.total_cost,
            "fee_allowance": self.fee_allowance,
            "slippage_allowance": self.slippage_allowance,
            "min_notional": self.min_notional,
            "min_amount": self.min_amount,
            "detail": self.detail,
        }


def minimum_executable_ticket(
    symbol: str,
    venue: str,
    price: float,
    min_notional: float,
    min_amount: float,
    fee_rate: float = 0.001,
    exchange: Any = None,
    slippage_bps: float = 0.0,
) -> MinimumTicket:
    """What it would actually cost to place the smallest legal order here.

    Computed before a candidate is produced, not discovered in preflight
    dozens of times. A proposal of 0.919 USDT against a 1.00 minimum is not a
    near miss to be retried -- it is a question with a known answer, and the
    answer is the number this returns.

    Includes both fee legs and a price-movement buffer, because an order
    sized at exactly the minimum with no headroom fails the moment the price
    ticks.
    """
    price = float(price or 0.0)
    if price <= 0:
        return MinimumTicket(
            symbol=symbol,
            venue=venue,
            price=price,
            min_notional=min_notional,
            min_amount=min_amount,
            fee_rate=fee_rate,
            detail="no reference price",
        )

    if slippage_bps <= 0:
        try:
            slippage_bps = float(os.getenv("EXECUTION_SLIPPAGE_BPS", "10"))
        except (TypeError, ValueError):
            slippage_bps = 10.0

    # The larger of the two floors the venue publishes.
    floor_notional = max(
        float(min_notional or 0.0), float(min_amount or 0.0) * price
    )
    amount = floor_notional / price if floor_notional > 0 else float(min_amount or 0.0)

    # Venues truncate to their amount step, so the rounded amount is almost
    # always slightly below what was asked for. That only matters when it
    # drops the order back under the venue's own floor -- so bump by exactly
    # one step in that case, and leave it alone otherwise. Bumping
    # unconditionally turned a 5 USDT minimum into a 642 USDT order.
    if exchange is not None and amount > 0:
        rounded = _amount_to_precision(exchange, symbol, amount)

        if rounded > 0 and rounded * price + 1e-12 < floor_notional:
            step = _infer_amount_step(exchange, symbol, rounded)
            if step > 0:
                bumped = _amount_to_precision(exchange, symbol, rounded + step)
                if bumped > rounded:
                    rounded = bumped

        if rounded > 0:
            amount = rounded
        elif floor_notional > 0:
            # Precision collapsed the minimum to zero: the smallest amount
            # this venue can express is one step.
            step = _infer_amount_step(exchange, symbol, 0.0)
            if step > 0:
                amount = step

    notional = amount * price
    fee_allowance = notional * max(0.0, fee_rate) * 2.0
    slippage_allowance = notional * (max(0.0, slippage_bps) / 10_000.0)

    return MinimumTicket(
        symbol=symbol,
        venue=venue,
        price=price,
        min_notional=min_notional,
        min_amount=min_amount,
        fee_rate=fee_rate,
        amount=amount,
        notional=notional,
        total_cost=notional + fee_allowance + slippage_allowance,
        fee_allowance=fee_allowance,
        slippage_allowance=slippage_allowance,
        detail=(
            f"{amount} @ {price} = {notional:.8f} "
            f"(+{fee_allowance:.8f} fees, +{slippage_allowance:.8f} buffer)"
            + (
                "; venue amount precision forces this size"
                if floor_notional > 0 and notional > floor_notional * 1.5
                else ""
            )
        ),
    )


def _infer_amount_step(exchange: Any, symbol: str, amount: float) -> float:
    """Smallest increment that actually changes the venue-rounded amount.

    Probed rather than assumed: precision is expressed as decimals on some
    venues and as a tick size on others, and the only portable question is
    "what is the smallest thing that moves this".
    """
    base = _amount_to_precision(exchange, symbol, amount)
    for magnitude in range(12, -3, -1):
        step = 10.0 ** -magnitude
        if _amount_to_precision(exchange, symbol, amount + step) > base:
            return step
    return 0.0


def note_economically_infeasible(
    venue: str,
    symbol: str,
    capital: float,
    reason: str,
    price: float = 0.0,
    min_notional: float = 0.0,
    min_amount: float = 0.0,
    risk_budget: float = 0.0,
) -> None:
    """Remember that this balance cannot fund this market's minimum.

    A candidate sized at 0.99 against a 1.00 minimum is not a transient
    failure: it will produce the same impossible ticket every cycle until
    either the balance or the venue minimum changes. Remembering it for a
    bounded window stops the strategy layer re-proposing it, without
    hardening into a permanent exclusion -- balances move and so do minimums.
    """
    key = _infeasible_key(
        venue, symbol, capital, price, min_notional, min_amount, risk_budget
    )
    with _INFEASIBLE_LOCK:
        _INFEASIBLE[key] = (time.time() + _infeasible_ttl(), reason)


def economically_infeasible(
    venue: str,
    symbol: str,
    capital: float,
    price: float = 0.0,
    min_notional: float = 0.0,
    min_amount: float = 0.0,
    risk_budget: float = 0.0,
) -> Optional[str]:
    key = _infeasible_key(
        venue, symbol, capital, price, min_notional, min_amount, risk_budget
    )
    with _INFEASIBLE_LOCK:
        entry = _INFEASIBLE.get(key)
        if entry is None:
            return None
        expires_at, reason = entry
        if expires_at <= time.time():
            del _INFEASIBLE[key]
            return None
        return reason


def clear_infeasible_memory() -> None:
    with _INFEASIBLE_LOCK:
        _INFEASIBLE.clear()


def _venue_may_serve(broker: BrokerCCXT, symbol: str) -> bool:
    """Whether this broker's venue lists this market, answered from memory.

    The capability registry is consulted before any per-symbol venue call.
    An absent market resolves here instead of becoming an exchange error.
    """
    try:
        from ..universe.routing import may_call_venue

        allowed, classification, _detail = may_call_venue(
            broker.exchange_id,
            symbol,
            environment=broker.resolve_mode(),
        )
    except Exception:
        # The registry is an optimisation, not a gate. If it cannot answer,
        # fall through to the venue rather than blocking execution.
        return True

    if not allowed:
        record_blocker(MARKET_NOT_LISTED, f"{symbol}@{broker.exchange_id}")
    return allowed


def invalidate_balance_cache() -> None:
    """Call after a fill: the cached free balance is now wrong."""
    with _CACHE_LOCK:
        _BALANCE_CACHE.clear()


def reset_caches() -> None:
    with _CACHE_LOCK:
        _MARKETS_CACHE.clear()
        _BALANCE_CACHE.clear()
        _TICKER_CACHE.clear()
    # The infeasibility memory is process-global state of the same kind, so
    # it resets with the caches rather than leaking between runs and tests.
    clear_infeasible_memory()


# ------------------------------------------------------------------- results


@dataclass
class Blocked:
    """Why this intent did not become a submittable order."""

    blocker: str
    detail: str = ""
    stage: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "prepared": False,
            "blocker": self.blocker,
            "detail": self.detail,
            "stage": self.stage,
        }


@dataclass
class PreparedOrder:
    """An order that has cleared every venue constraint we can check offline."""

    symbol: str
    side: str
    amount: float
    price: float
    order_type: str
    exchange_id: str
    execution_mode: str
    notional: float
    quote_currency: str
    free_quote: float
    min_notional: float
    min_amount: float
    fee_rate: float
    sizing_reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """The dict route_order consumes."""
        payload: Dict[str, Any] = {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.amount,
            "order_type": self.order_type,
            "exchange_id": self.exchange_id,
            "execution_mode": self.execution_mode,
            "backend": "ccxt",
            "reference_price": self.price,
        }
        if self.order_type != "market":
            payload["price"] = self.price
        return payload


# ------------------------------------------------------------- normalization


def normalize_symbol(value: Any) -> str:
    """Return a ccxt-style BASE/QUOTE symbol, or "" if it is not one.

    Accepts the spellings the engines actually emit: "BTCUSDT", "btc-usdt",
    "BTC_USDT", "BTC/USDT:USDT". Anything that does not resolve to two
    non-empty segments is rejected rather than guessed at.
    """
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = text.split(":", 1)[0]
    for separator in ("-", "_"):
        text = text.replace(separator, "/")
    if "/" not in text:
        for quote in ("USDT", "USDC", "USD", "BTC", "ETH", "EUR", "GBP"):
            if text.endswith(quote) and len(text) > len(quote):
                text = f"{text[: -len(quote)]}/{quote}"
                break
    if text.count("/") != 1:
        return ""
    base, quote = text.split("/")
    if not base or not quote:
        return ""
    return f"{base}/{quote}"


def _limit(market: Dict[str, Any], group: str, bound: str) -> Optional[float]:
    limits = market.get("limits")
    if not isinstance(limits, dict):
        return None
    section = limits.get(group)
    if not isinstance(section, dict):
        return None
    value = section.get(bound)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _amount_to_precision(exchange: Any, symbol: str, amount: float) -> float:
    """Round down to the venue's amount step, using ccxt when available."""
    try:
        rounded = exchange.amount_to_precision(symbol, amount)
        return float(rounded)
    except Exception:
        return float(amount)


def _price_to_precision(exchange: Any, symbol: str, price: float) -> float:
    try:
        rounded = exchange.price_to_precision(symbol, price)
        return float(rounded)
    except Exception:
        return float(price)


# --------------------------------------------------------------------- sizing


@dataclass
class SizingPolicy:
    """How much of a small balance one order may use.

    Defaults are deliberately conservative and come from the environment so an
    operator can change them without a code change. They size an order; they do
    not decide whether to trade, and they never raise size above what the free
    balance can actually fund.
    """

    allocation_fraction: float = 0.25
    max_fraction: float = 0.5
    fee_reserve_multiple: float = 3.0
    absolute_reserve_quote: float = 0.0

    @classmethod
    def from_env(cls) -> "SizingPolicy":
        def _float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except Exception:
                return default

        return cls(
            allocation_fraction=_float("EXECUTION_ALLOCATION_FRACTION", 0.25),
            max_fraction=_float("EXECUTION_MAX_FRACTION", 0.5),
            fee_reserve_multiple=_float("EXECUTION_FEE_RESERVE_MULTIPLE", 3.0),
            absolute_reserve_quote=_float("EXECUTION_RESERVE_QUOTE", 0.0),
        )


def paper_equity() -> float:
    """Quote-currency equity a paper run sizes against.

    Paper has no account to query, so this is a declared number rather than a
    discovered one. It exists so paper and authenticated runs go through the
    same sizing, precision and minimum-notional checks; it is never used for
    an authenticated environment, where the real free balance is read.
    """
    try:
        return float(os.getenv("PAPER_EQUITY_QUOTE", "1000"))
    except (TypeError, ValueError):
        return 1000.0


def size_order(
    free_quote: float,
    price: float,
    min_notional: float,
    min_amount: float,
    fee_rate: float,
    confidence: float = 0.0,
    policy: Optional[SizingPolicy] = None,
) -> Tuple[float, str]:
    """Return (notional, reason) for one order against a real free balance.

    The account this runs against holds single-digit USDT, so a percentage of
    equity usually lands under the venue minimum. Rather than give up, the
    notional is raised to the venue minimum when -- and only when -- the
    spendable balance can still cover it along with the round-trip fee
    reserve. If it cannot, the caller gets 0.0 and blocks on min notional
    instead of sending something the exchange will reject.
    """
    policy = policy or SizingPolicy.from_env()

    if free_quote <= 0.0 or price <= 0.0:
        return 0.0, "no_free_balance"

    reserve = max(0.0, policy.absolute_reserve_quote)
    fee_reserve = free_quote * max(0.0, fee_rate) * max(0.0, policy.fee_reserve_multiple)
    spendable = free_quote - reserve - fee_reserve
    if spendable <= 0.0:
        return 0.0, "balance_consumed_by_reserve"

    fraction = policy.allocation_fraction
    if confidence > 0.0:
        # Confidence moves size within the configured band. It never lifts the
        # cap and never lowers the floor below the configured allocation.
        fraction = policy.allocation_fraction + (
            max(0.0, min(1.0, confidence)) * (policy.max_fraction - policy.allocation_fraction)
        )
    fraction = max(0.0, min(fraction, policy.max_fraction))

    notional = spendable * fraction
    reason = f"fraction={fraction:.4f} of spendable={spendable:.6f}"

    floor = max(float(min_notional or 0.0), float(min_amount or 0.0) * price)
    if floor > 0.0 and notional < floor:
        if spendable + 1e-12 >= floor:
            notional = floor
            reason = f"raised to venue minimum {floor:.6f} (spendable={spendable:.6f})"
        else:
            return 0.0, (
                f"venue minimum {floor:.6f} exceeds spendable {spendable:.6f}"
            )

    if notional > spendable:
        notional = spendable
        reason = f"capped at spendable={spendable:.6f}"

    return notional, reason


# ------------------------------------------------------------------ preflight


def _free_quote_balance(balance: Dict[str, Any], quote: str) -> Optional[float]:
    free = balance.get("free")
    if isinstance(free, dict) and quote in free:
        try:
            return float(free[quote] or 0.0)
        except Exception:
            return None
    entry = balance.get(quote)
    if isinstance(entry, dict) and entry.get("free") is not None:
        try:
            return float(entry["free"] or 0.0)
        except Exception:
            return None
    return None


def prepare_order(
    intent: Dict[str, Any],
    *,
    broker: Optional[BrokerCCXT] = None,
    policy: Optional[SizingPolicy] = None,
) -> Tuple[Optional[PreparedOrder], Optional[Blocked]]:
    """Run the intent through every venue check, in pipeline order.

    Returns exactly one of (PreparedOrder, None) or (None, Blocked). Every
    Blocked is counted, so a quiet run is explainable from the persisted
    telemetry rather than from log archaeology.
    """
    record_event("attempts")
    started = time.time()

    def blocked(blocker: str, detail: str, stage: str) -> Tuple[None, Blocked]:
        record_blocker(blocker, detail)
        record_stage_latency(f"blocked.{stage}", time.time() - started)
        return None, Blocked(blocker=blocker, detail=detail, stage=stage)

    side = str(intent.get("side") or intent.get("action") or "").strip().lower()
    if side not in {"buy", "sell"}:
        return blocked(INVALID_INTENT, f"side={side!r}", "intent")

    symbol = normalize_symbol(
        intent.get("symbol") or intent.get("pair") or intent.get("market")
    )
    if not symbol:
        raw = str(intent.get("symbol") or intent.get("pair") or "")[:64]
        return blocked(SYMBOL_NOT_NORMALIZED, raw, "symbol")

    try:
        confidence = float(intent.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0

    if broker is None:
        broker = shared_broker(
            execution_mode=intent.get("execution_mode"),
            exchange_id=intent.get("exchange_id") or intent.get("exchange"),
            market_mode=intent.get("market_mode"),
        )

    authority = broker.authority
    if authority not in {"testnet", "live", "paper"}:
        return blocked(
            NO_EXECUTION_AUTHORITY,
            f"{broker.exchange_id}:{broker.resolve_mode()}",
            "authority",
        )

    try:
        markets = load_markets_cached(broker)
    except Exception as exc:
        return blocked(
            MARKET_METADATA_UNAVAILABLE, type(exc).__name__, "markets"
        )
    if not markets:
        return blocked(MARKET_METADATA_UNAVAILABLE, broker.exchange_id, "markets")

    market = markets.get(symbol)
    if not isinstance(market, dict):
        # Remember it, so the next attempt resolves from memory rather than
        # re-reading the venue's whole market list to learn the same thing.
        try:
            from ..universe.venues import NOT_LISTED, capabilities

            capabilities.record_absence(
                broker.exchange_id,
                symbol,
                state=NOT_LISTED,
                environment=broker.resolve_mode(),
                evidence="absent from venue market metadata (preflight)",
            )
        except Exception:
            pass

        return blocked(
            MARKET_NOT_LISTED, f"{symbol}@{broker.exchange_id}", "market"
        )

    if broker.market_mode in {"spot", ""} and market.get("spot") is not True:
        return blocked(MARKET_NOT_SPOT, symbol, "market")

    if market.get("active") is False:
        return blocked(MARKET_INACTIVE, symbol, "market")

    quote = str(market.get("quote") or symbol.split("/")[1]).upper()

    price = None
    for key in ("price", "reference_price", "last", "entry_price"):
        value = intent.get(key)
        if value is None:
            continue
        try:
            candidate = float(value)
        except Exception:
            continue
        if candidate > 0.0:
            price = candidate
            break

    if price is None:
        try:
            ticker = broker.fetch_ticker(symbol) or {}
            for key in ("last", "close", "bid"):
                value = ticker.get(key)
                if value:
                    price = float(value)
                    break
        except Exception as exc:
            return blocked(
                REFERENCE_PRICE_UNAVAILABLE, type(exc).__name__, "price"
            )

    if not price or price <= 0.0:
        return blocked(REFERENCE_PRICE_UNAVAILABLE, symbol, "price")

    # A sell spends base, a buy spends quote. Both are checked against the
    # currency actually leaving the account.
    spend_currency = quote if side == "buy" else str(
        market.get("base") or symbol.split("/")[0]
    ).upper()

    if authority == "paper":
        # Paper has no account to read. The equity a paper run sizes against
        # is a declared configuration value, not a balance discovered
        # anywhere, and it is expressed in the quote currency.
        free_quote = paper_equity()
        if free_quote <= 0.0:
            return blocked(
                INSUFFICIENT_FREE_BALANCE,
                f"paper_equity={free_quote}",
                "balance",
            )
    else:
        try:
            balance = fetch_balance_cached(broker)
        except Exception as exc:
            return blocked(BALANCE_UNAVAILABLE, type(exc).__name__, "balance")

        if not balance:
            return blocked(BALANCE_UNAVAILABLE, broker.exchange_id, "balance")

        free_spend = _free_quote_balance(balance, spend_currency)
        if free_spend is None:
            return blocked(BALANCE_UNAVAILABLE, spend_currency, "balance")
        free_quote = free_spend if side == "buy" else free_spend * price
        if free_quote <= 0.0:
            return blocked(
                INSUFFICIENT_FREE_BALANCE, f"{spend_currency}=0", "balance"
            )

    min_notional = _limit(market, "cost", "min") or 0.0
    min_amount = _limit(market, "amount", "min") or 0.0
    try:
        fee_rate = float(market.get("taker") or 0.001)
    except Exception:
        fee_rate = 0.001

    try:
        risk_budget = float(intent.get("risk_budget") or 0.0)
    except (TypeError, ValueError):
        risk_budget = 0.0

    remembered = economically_infeasible(
        broker.exchange_id,
        symbol,
        free_quote,
        price=price,
        min_notional=min_notional,
        min_amount=min_amount,
        risk_budget=risk_budget,
    )
    if remembered:
        return blocked(
            CAPITAL_BELOW_EXECUTABLE_MINIMUM, remembered, "minimum_ticket"
        )

    precision_client = None
    try:
        precision_client = broker._make_exchange(  # noqa: SLF001
            broker.resolve_mode(), authenticated=False
        )
    except Exception:
        precision_client = None

    # What the smallest legal order here would actually cost, decided once
    # rather than rediscovered by proposing 0.919 against a 1.00 minimum
    # every cycle.
    ticket = minimum_executable_ticket(
        symbol=symbol,
        venue=broker.exchange_id,
        price=price,
        min_notional=min_notional,
        min_amount=min_amount,
        fee_rate=fee_rate,
        exchange=precision_client,
    )

    spendable = free_quote
    if risk_budget > 0:
        spendable = min(spendable, risk_budget)

    if ticket.total_cost > 0 and ticket.total_cost > spendable:
        reason = (
            f"smallest executable ticket costs {ticket.total_cost:.8f} "
            f"{quote} (min notional {ticket.min_notional}, min amount "
            f"{ticket.min_amount}); spendable {spendable:.8f}"
            + (" within risk budget" if risk_budget > 0 else "")
            + (
                "; venue amount precision forces this size"
                if "precision forces" in ticket.detail
                else ""
            )
        )
        note_economically_infeasible(
            broker.exchange_id,
            symbol,
            free_quote,
            reason,
            price=price,
            min_notional=min_notional,
            min_amount=min_amount,
            risk_budget=risk_budget,
        )
        return blocked(
            CAPITAL_BELOW_EXECUTABLE_MINIMUM, reason, "minimum_ticket"
        )

    # Size against the risk budget when one is set, not merely check the
    # minimum against it. A budget that gates the floor but not the size
    # would pass a 0.50 budget and then place a 12.46 order.
    notional, sizing_reason = size_order(
        free_quote=spendable,
        price=price,
        min_notional=min_notional,
        min_amount=min_amount,
        fee_rate=fee_rate,
        confidence=confidence,
        policy=policy,
    )

    if notional <= 0.0:
        blocker = (
            BELOW_MIN_NOTIONAL
            if "minimum" in sizing_reason
            else INSUFFICIENT_FREE_BALANCE
        )
        if blocker == BELOW_MIN_NOTIONAL:
            note_economically_infeasible(
                broker.exchange_id,
                symbol,
                free_quote,
                sizing_reason,
                price=price,
                min_notional=min_notional,
                min_amount=min_amount,
                risk_budget=risk_budget,
            )
            blocker = CAPITAL_BELOW_EXECUTABLE_MINIMUM
        return blocked(blocker, sizing_reason, "sizing")

    amount = notional / price

    exchange = precision_client

    if exchange is not None:
        amount = _amount_to_precision(exchange, symbol, amount)
        price = _price_to_precision(exchange, symbol, price)

    if not amount or amount <= 0.0 or math.isnan(amount):
        return blocked(
            PRECISION_COLLAPSED_TO_ZERO,
            f"{symbol} notional={notional:.8f}",
            "precision",
        )

    if min_amount and amount < min_amount:
        return blocked(
            BELOW_MIN_AMOUNT,
            f"{symbol} amount={amount:.10f} min={min_amount:.10f}",
            "precision",
        )

    final_notional = amount * price
    if min_notional and final_notional + 1e-12 < min_notional:
        return blocked(
            BELOW_MIN_NOTIONAL,
            f"{symbol} notional={final_notional:.8f} min={min_notional:.8f}",
            "precision",
        )

    if final_notional > spendable + 1e-12:
        return blocked(
            INSUFFICIENT_FREE_BALANCE,
            f"{symbol} needs={final_notional:.8f} spendable={spendable:.8f}",
            "precision",
        )

    record_event("prepared")
    record_stage_latency("prepare_order", time.time() - started)

    order_type = str(intent.get("order_type") or intent.get("type") or "market").lower()

    return (
        PreparedOrder(
            symbol=symbol,
            side=side,
            amount=amount,
            price=price,
            order_type=order_type,
            exchange_id=broker.exchange_id,
            execution_mode=broker.resolve_mode(),
            notional=final_notional,
            quote_currency=quote,
            free_quote=free_quote,
            min_notional=min_notional,
            min_amount=min_amount,
            fee_rate=fee_rate,
            sizing_reason=sizing_reason,
            meta={"authority": authority, "confidence": confidence},
        ),
        None,
    )


def classify_receipt(receipt: Optional[Dict[str, Any]]) -> Optional[str]:
    """Map a route_order receipt onto a blocker class, or None if it filled.

    A receipt is only an acknowledgement when the router says it executed AND
    the exchange returned an order id. Anything else is a refusal, whatever the
    dict's truthiness suggests.
    """
    if not isinstance(receipt, dict):
        return ROUTER_REFUSED

    if not receipt.get("ok"):
        error = str(receipt.get("error") or receipt.get("reason") or "").lower()
        if not error:
            return ROUTER_REFUSED
        if "authority" in error or "credential" in error or "authenticated" in error:
            return NO_EXECUTION_AUTHORITY
        if "invalid_order_request" in error:
            return INVALID_INTENT
        if "route_unavailable" in error or "execution_route" in error:
            return ROUTER_REFUSED
        return EXCHANGE_REJECT

    if not receipt.get("executed"):
        return ROUTER_REFUSED

    order = receipt.get("order")
    order_id = order.get("id") if isinstance(order, dict) else None
    if not order_id:
        return NO_ORDER_ID

    return None
