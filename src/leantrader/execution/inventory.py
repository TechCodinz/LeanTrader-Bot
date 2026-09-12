"""What the account is actually holding, who owns it, and how it gets out.

Authenticated Testnet holds several non-zero assets and no open orders. Most
of the free USDT went into ETH and SOL buys. None of that is a reason to
liquidate: it is a reason to reconcile. An asset bought by an engine that has
since stopped running still represents capital, and capital with no owner and
no exit path is capital trapped.

Every non-zero balance is classified, valued, and given an exit verdict:

    ACTIVE_MANAGED_POSITION    a live engine holds it and is managing it
    LEGACY_MANAGED_POSITION    recorded by this system, no live manager
    ORPHANED_POSITION          meaningful size, no record of who opened it
    DUST                       below what the venue will let us sell
    NON_EXECUTABLE_INVENTORY   the market is no longer tradable here
    EXIT_ELIGIBLE              can be closed now, if a decision says to
    EXIT_PENDING               a close is already working

Nothing here sells anything. It reports, so that something with authority can
decide. Dust is never counted as a managed position, and a position is never
counted as reclaimable capital until the venue would actually accept the
closing order.
"""

from __future__ import annotations

import os
from decimal import Decimal, ROUND_DOWN
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

ACTIVE_MANAGED_POSITION = "ACTIVE_MANAGED_POSITION"
LEGACY_MANAGED_POSITION = "LEGACY_MANAGED_POSITION"
ORPHANED_POSITION = "ORPHANED_POSITION"
DUST = "DUST"
NON_EXECUTABLE_INVENTORY = "NON_EXECUTABLE_INVENTORY"
EXIT_PENDING = "EXIT_PENDING"
EXIT_ELIGIBLE = "EXIT_ELIGIBLE"
DUST_BELOW_SELL_MINIMUM = "DUST_BELOW_SELL_MINIMUM"

# Classifications that represent capital we could actually get back.
RECLAIMABLE = frozenset(
    {ACTIVE_MANAGED_POSITION, LEGACY_MANAGED_POSITION, ORPHANED_POSITION, EXIT_ELIGIBLE}
)


@dataclass
class InventoryItem:
    """One non-zero asset, and everything known about it."""

    asset: str
    symbol: str
    free: float
    used: float = 0.0
    total: float = 0.0

    # Valuation, kept as distinct concepts rather than one overloaded
    # "value". The aggregate and the rows could not be reconciled while dust
    # and executable holdings shared a field whose meaning differed by row.
    mark_price: float = 0.0
    raw_quote_value: float = 0.0          # quantity x mark, always
    rounded_sell_amount: float = 0.0      # what the venue would accept
    executable_quote_value: float = 0.0   # rounded amount x mark, or 0
    included_in_inventory_total: bool = False
    included_in_reclaimable_capital: bool = False
    dust_reason: str = ""
    sellable: bool = False

    # Retained for callers that predate the split; always equal to
    # raw_quote_value so nothing silently changes meaning.
    liquidation_value: float = 0.0

    acquisition_price: Optional[float] = None
    acquisition_time: Optional[float] = None
    acquisition_cost: Optional[float] = None
    acquisition_fees: Optional[float] = None
    order_ids: List[str] = field(default_factory=list)
    unrealized_pnl: Optional[float] = None

    owner: str = ""
    owner_alive: bool = False
    classification: str = ""
    exit_state: str = ""
    exit_blocked_reason: str = ""

    market_executable: bool = False
    min_sell_amount: float = 0.0
    min_sell_notional: float = 0.0
    amount_precision: Optional[float] = None
    can_close_now: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _round_down(
    amount: float,
    precision: Optional[float],
) -> float:
    """Round an amount down using canonical CCXT amount precision.

    LeanTrader's authenticated Bybit adapter runs with CCXT TICK_SIZE
    precision mode. Values such as 1e-05 and 0.001 are therefore amount
    steps, not counts of decimal places.

    Exit quantities are always rounded DOWN so the system can never claim
    it owns more base asset than the exchange balance actually contains.

    Integer precision values greater than one retain compatibility with
    older decimal-place metadata.
    """
    if precision is None or amount <= 0:
        return amount

    try:
        value = Decimal(str(amount))
        p = Decimal(str(precision))
    except Exception:
        return amount

    if p <= 0:
        return amount

    # CCXT TICK_SIZE form:
    # ETH amount precision 0.00001
    # SOL amount precision 0.001
    # A step of 1 is also valid TICK_SIZE metadata.
    if p <= Decimal("1"):
        units = (
            value / p
        ).to_integral_value(
            rounding=ROUND_DOWN
        )

        return float(
            units * p
        )

    # Compatibility for historical metadata that represented
    # precision as an integer decimal-place count.
    try:
        places = int(p)
    except Exception:
        return amount

    if p != Decimal(places) or places < 0:
        return amount

    quantum = Decimal("1").scaleb(
        -places
    )

    return float(
        value.quantize(
            quantum,
            rounding=ROUND_DOWN,
        )
    )


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def dust_threshold_quote() -> float:
    """Below this value an asset is not worth a decision, let alone an order."""
    try:
        return float(os.getenv("INVENTORY_DUST_QUOTE", "1.0"))
    except (TypeError, ValueError):
        return 1.0


def reconcile(
    balance: Dict[str, Any],
    markets: Optional[Dict[str, Any]] = None,
    tickers: Optional[Dict[str, Any]] = None,
    open_orders: Optional[List[Dict[str, Any]]] = None,
    known_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    quote: str = "USDT",
    venue: str = "",
) -> List[InventoryItem]:
    """Classify every non-zero asset the account holds.

    ``known_positions`` is what the running system believes it opened, keyed
    by symbol. An asset present on the exchange but absent from it was opened
    by something that is no longer running -- that is the orphan case, and it
    is the one that traps capital silently.

    Nothing is sold, and nothing is assumed: an asset whose market this venue
    no longer lists is reported as non-executable rather than valued as if it
    could be closed.
    """
    markets = markets or {}
    tickers = tickers or {}
    open_orders = open_orders or []
    known_positions = known_positions or {}
    quote = str(quote or "USDT").upper()

    free_map = balance.get("free") if isinstance(balance.get("free"), dict) else {}
    used_map = balance.get("used") if isinstance(balance.get("used"), dict) else {}
    total_map = balance.get("total") if isinstance(balance.get("total"), dict) else {}

    pending_symbols = {
        str(order.get("symbol") or "")
        for order in open_orders
        if str(order.get("side") or "").lower() == "sell"
    }

    items: List[InventoryItem] = []

    for asset, amount in sorted(free_map.items()):
        asset = str(asset).upper()
        if asset == quote:
            continue

        free = _f(amount)
        used = _f(used_map.get(asset))
        total = _f(total_map.get(asset), free + used)
        if total <= 0:
            continue

        symbol = f"{asset}/{quote}"
        market = markets.get(symbol)
        ticker = tickers.get(symbol) or {}

        mark = _f(ticker.get("last") or ticker.get("close"))
        raw_value = total * mark if mark > 0 else 0.0
        item = InventoryItem(
            asset=asset,
            symbol=symbol,
            free=free,
            used=used,
            total=total,
            mark_price=mark,
            raw_quote_value=raw_value,
            liquidation_value=raw_value,
        )

        record = known_positions.get(symbol)
        if record:
            item.owner = str(record.get("owner") or record.get("side") or "recorded")
            item.owner_alive = bool(record.get("owner_alive"))
            item.acquisition_price = _f(record.get("entry_price")) or None
            item.acquisition_time = record.get("timestamp")
            item.acquisition_cost = (
                item.acquisition_price * total if item.acquisition_price else None
            )
            item.acquisition_fees = record.get("fees")
            order_id = record.get("order_id")
            if order_id:
                item.order_ids = [str(order_id)]
            if item.acquisition_price and mark > 0:
                item.unrealized_pnl = (mark - item.acquisition_price) * total

        if not isinstance(market, dict) or market.get("active") is False:
            item.classification = NON_EXECUTABLE_INVENTORY
            item.exit_state = ""
            item.exit_blocked_reason = (
                f"{symbol} is not an active market on {venue or 'this venue'}"
            )
            items.append(item)
            continue

        item.market_executable = True
        limits = market.get("limits") or {}
        precision = market.get("precision") or {}
        item.min_sell_amount = _f((limits.get("amount") or {}).get("min"))
        item.min_sell_notional = _f((limits.get("cost") or {}).get("min"))
        item.amount_precision = (
            _f(precision.get("amount")) if precision.get("amount") is not None
            else None
        )

        # What the venue would actually accept, which is not the same as what
        # is held: an amount is truncated to the venue's step before it can be
        # sold, and the value of that truncated amount is the only figure that
        # represents recoverable capital.
        item.rounded_sell_amount = _round_down(total, item.amount_precision)
        item.executable_quote_value = (
            item.rounded_sell_amount * mark if mark > 0 else 0.0
        )

        below_amount = (
            item.min_sell_amount > 0
            and item.rounded_sell_amount < item.min_sell_amount
        )
        below_notional = (
            item.min_sell_notional > 0
            and item.executable_quote_value < item.min_sell_notional
        )
        below_dust = (
            item.raw_quote_value > 0
            and item.raw_quote_value < dust_threshold_quote()
        )

        if below_amount or below_notional or below_dust:
            item.classification = DUST
            item.exit_blocked_reason = DUST_BELOW_SELL_MINIMUM
            item.dust_reason = (
                "below venue minimum amount"
                if below_amount
                else "below venue minimum notional"
                if below_notional
                else f"below dust threshold {dust_threshold_quote()}"
            )
            item.can_close_now = False
            item.sellable = False
            item.executable_quote_value = 0.0
            items.append(item)
            continue

        item.can_close_now = True
        item.sellable = True
        item.included_in_inventory_total = True
        item.included_in_reclaimable_capital = True

        if symbol in pending_symbols:
            item.exit_state = EXIT_PENDING
        else:
            item.exit_state = EXIT_ELIGIBLE

        if record and item.owner_alive:
            item.classification = ACTIVE_MANAGED_POSITION
        elif record:
            item.classification = LEGACY_MANAGED_POSITION
        else:
            item.classification = ORPHANED_POSITION
            item.owner = item.owner or "unknown"

        items.append(item)

    return items


def summarize(items: List[InventoryItem]) -> Dict[str, Any]:
    """Totals that reconcile exactly against their component rows.

    The inclusion rules are explicit, because one field called "value" could
    not carry them: mark-to-market is every row; executable inventory is only
    rows the venue would accept a sell for; dust is the remainder. An
    invariant test checks these add up rather than trusting that they do.
    """
    by_class: Dict[str, int] = {}

    total_mark_to_market = 0.0
    total_executable = 0.0
    total_dust = 0.0
    reclaimable = 0.0

    for item in items:
        by_class[item.classification] = by_class.get(item.classification, 0) + 1
        total_mark_to_market += item.raw_quote_value

        if item.classification == DUST:
            total_dust += item.raw_quote_value
            continue

        if item.included_in_inventory_total:
            total_executable += item.executable_quote_value
        if (
            item.included_in_reclaimable_capital
            and item.classification in RECLAIMABLE
            and item.can_close_now
        ):
            reclaimable += item.executable_quote_value

    non_executable = sum(
        item.raw_quote_value
        for item in items
        if item.classification == NON_EXECUTABLE_INVENTORY
    )

    return {
        "nonzero_assets": len(items),
        "by_classification": dict(sorted(by_class.items())),
        "managed_positions": by_class.get(ACTIVE_MANAGED_POSITION, 0)
        + by_class.get(LEGACY_MANAGED_POSITION, 0),
        "orphaned_positions": by_class.get(ORPHANED_POSITION, 0),
        "dust_assets": by_class.get(DUST, 0),
        "non_executable": by_class.get(NON_EXECUTABLE_INVENTORY, 0),
        "exit_eligible": sum(
            1 for item in items if item.exit_state == EXIT_ELIGIBLE
        ),
        "exit_pending": sum(1 for item in items if item.exit_state == EXIT_PENDING),
        "total_mark_to_market_value": round(total_mark_to_market, 8),
        "total_executable_inventory_value": round(total_executable, 8),
        "total_dust_mark_to_market_value": round(total_dust, 8),
        "total_non_executable_value": round(non_executable, 8),
        "total_reclaimable_capital": round(reclaimable, 8),
    }


def valuation_invariants(
    items: List[InventoryItem], summary: Dict[str, Any], tolerance: float = 1e-6
) -> List[str]:
    """Where the printed rows and the printed totals disagree.

    Returns a description per discrepancy; empty means the report is
    internally consistent. This is what makes the aggregate safe to act on.
    """
    problems: List[str] = []

    row_mark = sum(item.raw_quote_value for item in items)
    if abs(row_mark - summary["total_mark_to_market_value"]) > tolerance:
        problems.append(
            f"mark-to-market rows sum to {row_mark:.8f} but the total says "
            f"{summary['total_mark_to_market_value']:.8f}"
        )

    row_dust = sum(
        item.raw_quote_value for item in items if item.classification == DUST
    )
    if abs(row_dust - summary["total_dust_mark_to_market_value"]) > tolerance:
        problems.append(
            f"dust rows sum to {row_dust:.8f} but the total says "
            f"{summary['total_dust_mark_to_market_value']:.8f}"
        )

    row_executable = sum(
        item.executable_quote_value
        for item in items
        if item.included_in_inventory_total
    )
    if abs(row_executable - summary["total_executable_inventory_value"]) > tolerance:
        problems.append(
            f"executable rows sum to {row_executable:.8f} but the total says "
            f"{summary['total_executable_inventory_value']:.8f}"
        )

    row_reclaimable = sum(
        item.executable_quote_value
        for item in items
        if item.included_in_reclaimable_capital
        and item.classification in RECLAIMABLE
        and item.can_close_now
    )
    if abs(row_reclaimable - summary["total_reclaimable_capital"]) > tolerance:
        problems.append(
            f"reclaimable rows sum to {row_reclaimable:.8f} but the total says "
            f"{summary['total_reclaimable_capital']:.8f}"
        )

    for item in items:
        if item.classification == DUST and item.included_in_reclaimable_capital:
            problems.append(f"{item.asset}: dust counted as reclaimable")
        if item.executable_quote_value > item.raw_quote_value + tolerance:
            problems.append(
                f"{item.asset}: executable value exceeds mark-to-market"
            )

    return problems


def spendable_capital(
    free_quote: float,
    items: Optional[List[InventoryItem]] = None,
    reserve_fraction: float = 0.0,
) -> Dict[str, float]:
    """Cash, and everything that is not cash, kept apart.

    Reclaimable capital is what an exit *would* return if one filled. It is
    not spendable, and a new buy must never be sized against it. The fields
    are named so that using the wrong one is a visible mistake rather than an
    easy one.
    """
    items = items or []
    summary = summarize(items)

    if reserve_fraction <= 0:
        try:
            reserve_fraction = float(os.getenv("EXECUTION_CASH_RESERVE", "0"))
        except (TypeError, ValueError):
            reserve_fraction = 0.0

    cash_spendable_now = max(
        0.0, free_quote * (1.0 - max(0.0, min(reserve_fraction, 1.0)))
    )

    committed = summary["total_mark_to_market_value"]
    exit_eligible = sum(
        item.executable_quote_value
        for item in items
        if item.exit_state == EXIT_ELIGIBLE and item.can_close_now
    )
    exit_pending = sum(
        item.executable_quote_value
        for item in items
        if item.exit_state == EXIT_PENDING
    )

    return {
        "free_quote_balance": round(free_quote, 8),
        "cash_spendable_now": round(cash_spendable_now, 8),
        "capital_committed": round(committed, 8),
        "capital_exit_eligible": round(exit_eligible, 8),
        "capital_exit_pending": round(exit_pending, 8),
        "capital_recovered": 0.0,
        "reclaimable_capital": summary["total_reclaimable_capital"],
        "portfolio_value": round(free_quote + committed, 8),
        "cash_fraction": round(free_quote / (free_quote + committed), 6)
        if (free_quote + committed) > 0
        else 0.0,
        # Retained for callers that predate the split. Deliberately equal to
        # cash_spendable_now, never to reclaimable capital.
        "spendable_quote": round(cash_spendable_now, 8),
        "free_quote": round(free_quote, 8),
        "inventory_value": round(committed, 8),
    }


def reconcile_from_broker(
    broker: Any,
    known_positions: Optional[Dict[str, Dict[str, Any]]] = None,
    quote: str = "USDT",
) -> Dict[str, Any]:
    """Read the account and reconcile it. Read-only; places no orders.

    Returns the items, the summary and the capital picture, or a structured
    reason when the account cannot be read -- never a guess.
    """
    try:
        if broker.authority not in {"testnet", "live"}:
            return {
                "available": False,
                "reason": "no authenticated account to reconcile",
            }

        balance = broker.fetch_balance() or {}
        markets = broker.load_markets() or {}
    except Exception as exc:
        return {
            "available": False,
            "reason": f"account read failed: {type(exc).__name__}",
        }

    tickers: Dict[str, Any] = {}
    free_map = balance.get("free") or {}
    for asset in free_map:
        if str(asset).upper() == quote.upper():
            continue
        symbol = f"{str(asset).upper()}/{quote.upper()}"
        if symbol not in markets:
            continue
        try:
            tickers[symbol] = broker.fetch_ticker(symbol) or {}
        except Exception:
            continue

    items = reconcile(
        balance=balance,
        markets=markets,
        tickers=tickers,
        known_positions=known_positions,
        quote=quote,
        venue=getattr(broker, "exchange_id", ""),
    )

    free_quote = _f((balance.get("free") or {}).get(quote.upper()))

    return {
        "available": True,
        "venue": getattr(broker, "exchange_id", ""),
        "environment": broker.resolve_mode(),
        "reconciled_at": time.time(),
        "items": [item.as_dict() for item in items],
        "summary": summarize(items),
        "capital": spendable_capital(free_quote, items),
        # Checked here rather than only in tests: if the totals ever stop
        # reconciling against their rows, the report says so instead of
        # printing a number nobody can derive.
        "valuation_problems": valuation_invariants(items, summarize(items)),
        "objects": items,
    }
