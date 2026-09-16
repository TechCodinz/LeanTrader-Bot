"""Owned-position lifecycle for REAL_PROFIT_BOT.

REAL_PROFIT_BOT has always executed for real, but across every historical
version it never recorded what it bought. A BUY fired and the loop moved on, so
nothing could monitor a position or decide when to leave it, and the only exit
that could ever happen was a coincidental momentum SELL on the same symbol.
That is the missing half of the compounding cycle, and it is the only thing
this module adds.

REAL_PROFIT_BOT remains the execution owner. Nothing here places an order.
Exits are decided here and handed back to the bot's own ``execute_trade`` so
that every order still goes through the same path, with the same owned-
inventory sizing, dust skipping, precision and minimum-notional guards that are
already working.

The exit taxonomy is transplanted from the mature fast lane
(fast_collective_testnet / velocity_sniper_testnet): take profit, trailing
profit, stop loss, momentum decay, velocity decay, dynamic timeout and
microstructure deterioration. The donor's gating is deliberately NOT
transplanted -- this module only decides when to leave a position it already
owns, so it can never prevent an entry.

Nothing here fabricates a fill, a price or a profit. A position is created only
from an authenticated non-zero filled quantity, and realized PnL is computed
from actual fills and actual fees.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

# Exit reasons, matching the historical fast-lane vocabulary.
TAKE_PROFIT = "take_profit"
TRAILING_PROFIT = "trailing_profit"
STOP_LOSS = "stop_loss"
MOMENTUM_DECAY = "momentum_decay"
VELOCITY_DECAY = "velocity_decay"
DYNAMIC_TIMEOUT = "dynamic_timeout"
MICROSTRUCTURE_DETERIORATION = "microstructure_deterioration"

TERMINAL_ORDER_STATUSES = frozenset({"closed", "canceled", "cancelled", "rejected", "expired"})


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return float(default)


# --------------------------------------------------------------- fill truth


def reconcile_fill(
    exchange: Any,
    symbol: str,
    order: Optional[Dict[str, Any]],
    *,
    attempts: int = 3,
    delay_seconds: float = 0.4,
) -> Dict[str, Any]:
    """Turn a submitted order into authenticated fill facts.

    An acknowledgement is not a fill. Bybit can return an order id with status
    ``open`` and ``filled`` 0; treating that as execution is how a position
    that does not exist gets recorded. This re-reads the order from the venue
    until it is terminal or the attempts run out, and reports exactly what the
    exchange says.

    ``fee_base`` matters on spot: a BUY fee is commonly charged in the base
    asset, so the quantity actually available to sell later is the filled
    quantity minus that fee. Ignoring it produces a SELL for more than is held.
    """
    result = {
        "order_id": "",
        "client_order_id": "",
        "status": "unknown",
        "filled": 0.0,
        "average": 0.0,
        "cost": 0.0,
        "fee_quote": 0.0,
        "fee_base": 0.0,
        "reconciled": False,
        "raw_status": "",
    }
    if not isinstance(order, dict):
        return result

    order_id = str(order.get("id") or "")
    result["order_id"] = order_id
    result["client_order_id"] = str(
        order.get("clientOrderId") or (order.get("info") or {}).get("orderLinkId") or ""
    )

    latest = order
    for attempt in range(max(1, int(attempts))):
        status = str(latest.get("status") or "").lower()
        filled = _f(latest.get("filled"))
        if status in TERMINAL_ORDER_STATUSES or filled > 0:
            break
        if not order_id:
            break
        time.sleep(max(0.0, delay_seconds))
        try:
            latest = exchange.fetch_order(order_id, symbol) or latest
        except Exception:
            # A venue that cannot re-read the order leaves us with what we
            # have. Reporting unreconciled is correct; inventing a fill is not.
            break

    result["raw_status"] = str(latest.get("status") or "")
    result["status"] = str(latest.get("status") or "unknown").lower()
    result["filled"] = _f(latest.get("filled"))
    result["average"] = _f(latest.get("average")) or _f(latest.get("price"))
    result["cost"] = _f(latest.get("cost"))
    result["reconciled"] = True

    base = symbol.split("/")[0].upper()
    fee = latest.get("fee")
    fees = latest.get("fees") if isinstance(latest.get("fees"), list) else []
    for entry in ([fee] if isinstance(fee, dict) else []) + [f for f in fees if isinstance(f, dict)]:
        cost = _f(entry.get("cost"))
        currency = str(entry.get("currency") or "").upper()
        if currency == base:
            result["fee_base"] += cost
        else:
            result["fee_quote"] += cost

    if result["cost"] <= 0 and result["filled"] > 0 and result["average"] > 0:
        result["cost"] = result["filled"] * result["average"]

    return result


# ------------------------------------------------------------- the ledger


class PositionLedger:
    """What the bot owns, with the evidence that it owns it.

    Persisted so a restart does not orphan inventory the bot bought. Only
    positions this bot opened appear here, which is what lets the exit loop
    tell its own inventory apart from unrelated balances and dust.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else self._default_path()
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed: List[Dict[str, Any]] = []
        self.load()

    @staticmethod
    def _default_path() -> Path:
        base = os.getenv("LEANTRADER_DATA_DIR", "").strip()
        for candidate in ([Path(base) / "runtime"] if base else []) + [Path("runtime")]:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate / "rpb_positions.json"
            except OSError:
                continue
        return Path("rpb_positions.json")

    def load(self) -> int:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if not isinstance(payload, dict):
            return 0
        self.positions = {
            str(k): v for k, v in (payload.get("positions") or {}).items() if isinstance(v, dict)
        }
        self.closed = [c for c in (payload.get("closed") or []) if isinstance(c, dict)][-200:]
        return len(self.positions)

    def save(self) -> bool:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": time.time(),
            "positions": self.positions,
            "closed": self.closed[-200:],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
            return True
        except Exception:
            return False

    def open_position(
        self,
        symbol: str,
        fill: Dict[str, Any],
        *,
        strategy: str = "",
        confidence: float = 0.0,
        signal_price: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Record ownership from an authenticated fill, or refuse.

        A zero-fill acknowledgement creates nothing. That refusal is the point:
        a position recorded from an ack is inventory the bot does not have.
        """
        filled = _f(fill.get("filled"))
        if filled <= 0:
            return None

        average = _f(fill.get("average")) or _f(signal_price)
        if average <= 0:
            return None

        # The BUY fee is often taken in the base asset, so what can actually be
        # sold later is less than what filled.
        sellable = max(0.0, filled - _f(fill.get("fee_base")))
        cost_basis = _f(fill.get("cost")) or (filled * average)

        record = {
            "symbol": symbol,
            "order_id": fill.get("order_id", ""),
            "client_order_id": fill.get("client_order_id", ""),
            "filled_quantity": filled,
            "sellable_quantity": sellable,
            "remaining_quantity": sellable,
            "average_entry": average,
            "entry_cost": cost_basis,
            "entry_fee_quote": _f(fill.get("fee_quote")),
            "entry_fee_base": _f(fill.get("fee_base")),
            "entry_timestamp": time.time(),
            "strategy": strategy,
            "confidence": confidence,
            "peak_price": average,
            "exit_state": "open",
        }
        self.positions[symbol] = record
        self.save()
        return record

    def close_position(
        self,
        symbol: str,
        exit_fill: Dict[str, Any],
        exit_reason: str,
    ) -> Optional[Dict[str, Any]]:
        """Settle a position from an authenticated SELL fill."""
        record = self.positions.get(symbol)
        if record is None:
            return None

        sold = _f(exit_fill.get("filled"))
        if sold <= 0:
            return None

        proceeds = _f(exit_fill.get("cost")) or (sold * _f(exit_fill.get("average")))
        exit_fee = _f(exit_fill.get("fee_quote"))

        # Cost basis for the portion actually sold.
        owned = _f(record.get("sellable_quantity")) or _f(record.get("filled_quantity"))
        portion = min(1.0, sold / owned) if owned > 0 else 1.0
        cost_basis = _f(record.get("entry_cost")) * portion
        entry_fee = _f(record.get("entry_fee_quote")) * portion

        gross = proceeds - cost_basis
        fees = entry_fee + exit_fee
        net = gross - fees

        settled = {
            **record,
            "exit_order_id": exit_fill.get("order_id", ""),
            "exit_quantity": sold,
            "exit_average": _f(exit_fill.get("average")),
            "exit_proceeds": proceeds,
            "exit_fee_quote": exit_fee,
            "exit_reason": exit_reason,
            "exit_timestamp": time.time(),
            "hold_seconds": time.time() - _f(record.get("entry_timestamp")),
            "gross_pnl": gross,
            "total_fees": fees,
            "realized_net_pnl": net,
            "exit_state": "closed",
        }

        remaining = max(0.0, owned - sold)
        if remaining > 0 and portion < 0.999:
            record["remaining_quantity"] = remaining
            record["sellable_quantity"] = remaining
            record["entry_cost"] = _f(record.get("entry_cost")) * (1.0 - portion)
            record["entry_fee_quote"] = _f(record.get("entry_fee_quote")) * (1.0 - portion)
            self.positions[symbol] = record
        else:
            self.positions.pop(symbol, None)

        self.closed.append(settled)
        self.save()
        return settled

    def owns(self, symbol: str) -> bool:
        return symbol in self.positions

    def open_symbols(self) -> List[str]:
        return list(self.positions)

    def realized_net_total(self) -> float:
        return sum(_f(c.get("realized_net_pnl")) for c in self.closed)

    def stats(self) -> Dict[str, Any]:
        wins = [c for c in self.closed if _f(c.get("realized_net_pnl")) > 0]
        return {
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed),
            "realized_net_pnl": self.realized_net_total(),
            "authentic_wins": len(wins),
            "authentic_win_rate": (100.0 * len(wins) / len(self.closed)) if self.closed else 0.0,
            "total_fees": sum(_f(c.get("total_fees")) for c in self.closed),
        }


# --------------------------------------------------------------- exits


class ExitEvaluator:
    """When to leave a position the bot already owns.

    Transplanted from the historical fast lane's exit taxonomy. Thresholds are
    in basis points and tunable by environment; the defaults are the historical
    micro-scalping values, not new ones.

    This can only ever say "leave" -- it is never consulted before an entry, so
    it cannot starve execution.
    """

    def __init__(self) -> None:
        self.take_profit_bps = _env_float("RPB_TAKE_PROFIT_BPS", 50.0)
        self.stop_loss_bps = _env_float("RPB_STOP_LOSS_BPS", 30.0)
        self.trailing_arm_bps = _env_float("RPB_TRAILING_ARM_BPS", 35.0)
        self.trailing_give_back_bps = _env_float("RPB_TRAILING_GIVEBACK_BPS", 15.0)
        self.max_hold_seconds = _env_float("RPB_MAX_HOLD_SECONDS", 300.0)
        self.momentum_decay_bps = _env_float("RPB_MOMENTUM_DECAY_BPS", 20.0)
        self.max_spread_bps = _env_float("RPB_MAX_EXIT_SPREAD_BPS", 60.0)

    def evaluate(
        self,
        record: Dict[str, Any],
        price: float,
        *,
        now: Optional[float] = None,
        change_pct: Optional[float] = None,
        spread_bps: Optional[float] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Return (should_exit, reason, detail).

        ``price`` must be a real current price. A missing or non-positive price
        yields no exit rather than a guessed one.
        """
        now = time.time() if now is None else now
        entry = _f(record.get("average_entry"))
        if entry <= 0 or price <= 0:
            return False, "", {"reason": "no_usable_mark"}

        move_bps = ((price - entry) / entry) * 10_000.0
        peak = max(_f(record.get("peak_price")) or entry, price)
        record["peak_price"] = peak
        give_back_bps = ((peak - price) / peak) * 10_000.0 if peak > 0 else 0.0
        held = now - _f(record.get("entry_timestamp"), now)

        detail = {
            "move_bps": round(move_bps, 2),
            "give_back_bps": round(give_back_bps, 2),
            "held_seconds": round(held, 1),
            "peak_price": peak,
        }

        # Compared with a small tolerance: a price computed to land exactly on
        # the threshold lands a fraction below it in binary floating point, and
        # a scalper that hits take-profit exactly must still take it.
        epsilon = 1e-9

        if move_bps + epsilon >= self.take_profit_bps:
            return True, TAKE_PROFIT, detail

        if move_bps - epsilon <= -self.stop_loss_bps:
            return True, STOP_LOSS, detail

        # Trailing only arms once the position has actually been in profit.
        peak_bps = ((peak - entry) / entry) * 10_000.0
        if (
            peak_bps + epsilon >= self.trailing_arm_bps
            and give_back_bps + epsilon >= self.trailing_give_back_bps
        ):
            return True, TRAILING_PROFIT, detail

        if spread_bps is not None and spread_bps > self.max_spread_bps:
            detail["spread_bps"] = spread_bps
            return True, MICROSTRUCTURE_DETERIORATION, detail

        # Momentum that produced the entry has reversed while we are flat-ish.
        if (
            change_pct is not None
            and change_pct < 0
            and move_bps - epsilon <= -self.momentum_decay_bps
        ):
            detail["change_pct"] = change_pct
            return True, MOMENTUM_DECAY, detail

        if held + epsilon >= self.max_hold_seconds:
            # A position that has gone nowhere is capital that is not working.
            return True, DYNAMIC_TIMEOUT, detail

        return False, "", detail
