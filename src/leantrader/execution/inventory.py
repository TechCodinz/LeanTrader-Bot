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

    mark_price: float = 0.0
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
    can_close_now: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        item = InventoryItem(
            asset=asset,
            symbol=symbol,
            free=free,
            used=used,
            total=total,
            mark_price=mark,
            liquidation_value=total * mark if mark > 0 else 0.0,
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
        item.min_sell_amount = _f((limits.get("amount") or {}).get("min"))
        item.min_sell_notional = _f((limits.get("cost") or {}).get("min"))

        below_amount = item.min_sell_amount > 0 and total < item.min_sell_amount
        below_notional = (
            item.min_sell_notional > 0
            and item.liquidation_value > 0
            and item.liquidation_value < item.min_sell_notional
        )
        below_dust = (
            item.liquidation_value > 0
            and item.liquidation_value < dust_threshold_quote()
        )

        if below_amount or below_notional or below_dust:
            item.classification = DUST
            item.exit_blocked_reason = DUST_BELOW_SELL_MINIMUM
            item.can_close_now = False
            items.append(item)
            continue

        item.can_close_now = True

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
    """Counts and reclaimable capital, with dust kept out of both."""
    by_class: Dict[str, int] = {}
    reclaimable = 0.0
    dust_value = 0.0

    for item in items:
        by_class[item.classification] = by_class.get(item.classification, 0) + 1
        if item.classification == DUST:
            dust_value += item.liquidation_value
        elif item.classification in RECLAIMABLE and item.can_close_now:
            reclaimable += item.liquidation_value

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
        "reclaimable_capital": round(reclaimable, 8),
        "dust_value": round(dust_value, 8),
    }


def spendable_capital(
    free_quote: float,
    items: Optional[List[InventoryItem]] = None,
    reserve_fraction: float = 0.0,
) -> Dict[str, float]:
    """What a new buy may actually spend, given what is already held.

    Free cash is the only thing a new buy can spend. Inventory is reported
    alongside it so the caller can see that capital is committed rather than
    missing -- repeatedly proposing orders as though the original balance
    were still cash is what produced dozens of identical sub-minimum
    attempts.
    """
    items = items or []
    inventory_value = sum(
        item.liquidation_value
        for item in items
        if item.classification != DUST
    )
    reclaimable = sum(
        item.liquidation_value
        for item in items
        if item.classification in RECLAIMABLE and item.can_close_now
    )

    if reserve_fraction <= 0:
        try:
            reserve_fraction = float(os.getenv("EXECUTION_CASH_RESERVE", "0"))
        except (TypeError, ValueError):
            reserve_fraction = 0.0

    spendable = max(0.0, free_quote * (1.0 - max(0.0, min(reserve_fraction, 1.0))))

    return {
        "free_quote": round(free_quote, 8),
        "spendable_quote": round(spendable, 8),
        "inventory_value": round(inventory_value, 8),
        "reclaimable_capital": round(reclaimable, 8),
        "portfolio_value": round(free_quote + inventory_value, 8),
        "cash_fraction": round(
            free_quote / (free_quote + inventory_value), 6
        )
        if (free_quote + inventory_value) > 0
        else 0.0,
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
    }
