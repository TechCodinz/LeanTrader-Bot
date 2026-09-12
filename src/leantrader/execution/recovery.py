"""Recovery coordination for inventory nobody is managing.

Four assets sat in the account -- bought by processes that no longer exist --
with no owner, no stop, no target and no record of what they cost. They were
not a bug in isolation; they were capital the system could neither use nor
account for, and the classification work that found them stopped at naming
them.

This coordinator adopts them. Its first responsibility is explicitly NOT
selling: an orphan is first understood (what is it, where did it come from,
what did it cost, can it legally be exited at all) and brought under
management. Only then, and only with explicit per-asset authorization, does an
exit become possible -- and that exit goes through the same central authority
every other order goes through, so it is subject to every check a normal order
is subject to.

What this module will never do:

* liquidate inventory because it happens to be an orphan;
* build its own exchange client, or call create_order directly;
* report a close that the exchange did not acknowledge;
* treat the proceeds an exit *might* return as money available to spend.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import intent as execution_intent
from . import inventory as inventory_reconciler
from . import preflight

SCHEMA_VERSION = 1

# Recovery states, in the order a recovered holding passes through them. The
# first four are about understanding; nothing is sold before the fifth, and
# the fifth requires an explicit authorization that this module never grants
# to itself.
ORPHAN_DISCOVERED = "ORPHAN_DISCOVERED"
PROVENANCE_RECONSTRUCTED = "PROVENANCE_RECONSTRUCTED"
EXECUTABILITY_ASSESSED = "EXECUTABILITY_ASSESSED"
ADOPTED_UNDER_MANAGEMENT = "ADOPTED_UNDER_MANAGEMENT"
EXIT_AUTHORIZED = "EXIT_AUTHORIZED"
EXIT_INTENT_PREPARED = "EXIT_INTENT_PREPARED"
EXIT_SUBMITTED = "EXIT_SUBMITTED"
EXIT_ACKNOWLEDGED = "EXIT_ACKNOWLEDGED"
EXIT_FILLED = "EXIT_FILLED"
RECOVERED_RECONCILED = "RECOVERED_RECONCILED"

# Non-progress outcomes. A holding in one of these is still adopted and still
# reported; it simply has no legal exit right now.
RECOVERY_BLOCKED_NOT_EXECUTABLE = "RECOVERY_BLOCKED_NOT_EXECUTABLE"
RECOVERY_BLOCKED_BELOW_MINIMUM = "RECOVERY_BLOCKED_BELOW_MINIMUM"
RECOVERY_AWAITING_AUTHORIZATION = "RECOVERY_AWAITING_AUTHORIZATION"
RECOVERY_EXIT_REFUSED = "RECOVERY_EXIT_REFUSED"

RECOVERY_STATES: Tuple[str, ...] = (
    ORPHAN_DISCOVERED,
    PROVENANCE_RECONSTRUCTED,
    EXECUTABILITY_ASSESSED,
    ADOPTED_UNDER_MANAGEMENT,
    RECOVERY_AWAITING_AUTHORIZATION,
    RECOVERY_BLOCKED_NOT_EXECUTABLE,
    RECOVERY_BLOCKED_BELOW_MINIMUM,
    EXIT_AUTHORIZED,
    EXIT_INTENT_PREPARED,
    EXIT_SUBMITTED,
    EXIT_ACKNOWLEDGED,
    EXIT_FILLED,
    RECOVERY_EXIT_REFUSED,
    RECOVERED_RECONCILED,
)

# States from which no further progress is attempted without new input.
TERMINAL_STATES = frozenset(
    {
        RECOVERED_RECONCILED,
        RECOVERY_BLOCKED_NOT_EXECUTABLE,
    }
)

# Classes this coordinator takes responsibility for. A position an engine is
# actively managing is left alone: adopting it would give it two owners.
ADOPTABLE_CLASSES = frozenset(
    {
        inventory_reconciler.ORPHANED_POSITION,
        inventory_reconciler.LEGACY_MANAGED_POSITION,
        inventory_reconciler.DUST,
        inventory_reconciler.NON_EXECUTABLE_INVENTORY,
    }
)


@dataclass
class RecoveryRecord:
    """Everything known about one piece of unmanaged inventory.

    The fields are deliberately separate rather than collapsed into a single
    "value": what the holding is worth on paper, what a venue would accept a
    sell for, and what an exit would actually return are three different
    numbers, and confusing them is how trapped capital came to be reported as
    spendable.
    """

    asset: str
    symbol: str
    venue: str = ""
    environment: str = ""

    # Responsibility 1: what is held.
    quantity: float = 0.0
    free: float = 0.0
    used: float = 0.0
    classification: str = ""

    # Responsibility 2-4: where it came from and what it cost.
    owner: str = ""
    owner_alive: bool = False
    provenance: str = "UNKNOWN"
    acquisition_price: Optional[float] = None
    acquisition_time: Optional[float] = None
    acquisition_cost: Optional[float] = None
    acquisition_fees: Optional[float] = None
    order_ids: List[str] = field(default_factory=list)
    trade_ids: List[str] = field(default_factory=list)

    # Responsibility 6-8: whether it can be exited, and for how much.
    market_executable: bool = False
    mark_price: float = 0.0
    mark_to_market_value: float = 0.0
    min_sell_amount: float = 0.0
    min_sell_notional: float = 0.0
    amount_precision: Optional[float] = None
    minimum_legal_exit_amount: float = 0.0
    minimum_legal_exit_notional: float = 0.0
    executable_exit_amount: float = 0.0
    executable_exit_value: float = 0.0
    exit_possible: bool = False
    exit_blocked_reason: str = ""
    unrealized_pnl: Optional[float] = None

    # Responsibility 9: where recovery has got to.
    state: str = ORPHAN_DISCOVERED
    state_reason: str = ""
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    authorized: bool = False
    authorized_by: str = ""
    authorized_at: Optional[float] = None

    # Responsibility 10: the audit trail of the exit itself.
    exit_intent_id: str = ""
    exit_correlation_id: str = ""
    exit_order_id: str = ""
    exit_filled_amount: float = 0.0
    exit_fill_price: Optional[float] = None
    exit_fees: Optional[float] = None
    recovered_quote: float = 0.0

    discovered_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def transition(self, state: str, reason: str = "") -> "RecoveryRecord":
        """Move to a new recovery state, keeping the path that got there."""
        self.state = state
        self.state_reason = reason
        self.updated_at = time.time()
        self.state_history.append(
            {"state": state, "reason": reason, "at": self.updated_at}
        )
        return self

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _snapshot_path() -> Path:
    explicit = os.getenv("INVENTORY_RECOVERY_PATH", "").strip()
    if explicit:
        return Path(explicit)

    shared = Path(os.getenv("LEANTRADER_DATA_DIR", "/app/data"))
    if shared.is_dir() and os.access(shared, os.W_OK):
        runtime_dir = shared / "runtime"
        try:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            return runtime_dir / "inventory_recovery.json"
        except OSError:
            return shared / "inventory_recovery.json"

    return Path("runtime/inventory_recovery.json")


def exit_allowlist() -> frozenset:
    """Assets an operator has authorized recovery exits for.

    Empty by default, and that default is the point: adoption happens
    automatically, selling never does.
    """
    raw = os.getenv("INVENTORY_RECOVERY_EXIT_ALLOWLIST", "")
    return frozenset(
        part.strip().upper() for part in str(raw).split(",") if part.strip()
    )


class InventoryRecoveryCoordinator:
    """Adopts unmanaged inventory and, when authorized, recovers its capital.

    Construct one per process. It keeps its records in memory and persists
    them atomically so another process -- and this one after a restart -- can
    see what has been adopted and how far each recovery has got.
    """

    def __init__(
        self,
        venue: str = "",
        environment: str = "",
        route: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.venue = str(venue or "")
        self.environment = str(environment or "")
        self._records: Dict[str, RecoveryRecord] = {}
        self._lock = threading.RLock()
        # Injected only so tests can observe the call without an exchange.
        # Production leaves it None, which resolves to route_order -- the one
        # final order authority. There is no other path out of this module.
        self._route = route

    # ------------------------------------------------------------ discovery

    def discover(
        self,
        items: List[inventory_reconciler.InventoryItem],
    ) -> List[RecoveryRecord]:
        """Adopt every unmanaged holding in a reconciled inventory.

        Positions an engine is actively managing are skipped: they already
        have an owner, and a second one would fight it.
        """
        adopted: List[RecoveryRecord] = []
        with self._lock:
            for item in items:
                if item.classification not in ADOPTABLE_CLASSES:
                    continue
                record = self._records.get(item.symbol)
                if record is None:
                    record = RecoveryRecord(
                        asset=item.asset,
                        symbol=item.symbol,
                        venue=self.venue,
                        environment=self.environment,
                    )
                    record.transition(
                        ORPHAN_DISCOVERED,
                        f"classified {item.classification} with no live owner",
                    )
                    self._records[item.symbol] = record
                self._absorb(record, item)
                adopted.append(record)
        return adopted

    def _absorb(
        self,
        record: RecoveryRecord,
        item: inventory_reconciler.InventoryItem,
    ) -> None:
        """Copy the reconciled facts onto the recovery record.

        Every number here came from the exchange or from the venue's own
        market metadata. Nothing is estimated except the acquisition cost,
        which is marked UNKNOWN when there is no record to reconstruct it
        from rather than being invented.
        """
        record.quantity = item.total
        record.free = item.free
        record.used = item.used
        record.classification = item.classification
        record.owner = item.owner or "unknown"
        record.owner_alive = bool(item.owner_alive)
        record.mark_price = item.mark_price
        record.mark_to_market_value = item.raw_quote_value
        record.market_executable = bool(item.market_executable)
        record.min_sell_amount = item.min_sell_amount
        record.min_sell_notional = item.min_sell_notional
        record.amount_precision = item.amount_precision
        record.executable_exit_amount = item.rounded_sell_amount
        record.executable_exit_value = item.executable_quote_value
        record.unrealized_pnl = item.unrealized_pnl

        if item.acquisition_price:
            record.acquisition_price = item.acquisition_price
            record.acquisition_cost = item.acquisition_cost
            record.acquisition_time = item.acquisition_time
            record.acquisition_fees = item.acquisition_fees
            record.provenance = "RECONSTRUCTED_FROM_POSITION_RECORD"
        if item.order_ids:
            record.order_ids = list(
                dict.fromkeys(list(record.order_ids) + list(item.order_ids))
            )

        record.updated_at = time.time()

    # -------------------------------------------------- provenance recovery

    def reconstruct_provenance(
        self,
        record: RecoveryRecord,
        trades: Optional[List[Dict[str, Any]]] = None,
    ) -> RecoveryRecord:
        """Rebuild what this holding cost from the venue's own trade history.

        ``trades`` is whatever the venue returned for this symbol. When there
        is nothing to reconstruct from, the provenance stays UNKNOWN and the
        cost stays None: an unknown entry price is reported as unknown, never
        defaulted to the mark, because that would silently show zero PnL on a
        position that may be well underwater.
        """
        with self._lock:
            buys = [
                trade
                for trade in (trades or [])
                if str(trade.get("side") or "").lower() == "buy"
            ]

            if buys:
                total_amount = 0.0
                total_cost = 0.0
                total_fees = 0.0
                order_ids: List[str] = list(record.order_ids)
                trade_ids: List[str] = list(record.trade_ids)

                for trade in buys:
                    amount = inventory_reconciler._f(trade.get("amount"))
                    price = inventory_reconciler._f(
                        trade.get("price") or trade.get("average")
                    )
                    cost = inventory_reconciler._f(trade.get("cost"))
                    if cost <= 0 and amount > 0 and price > 0:
                        cost = amount * price
                    fee = trade.get("fee") or {}
                    if isinstance(fee, dict):
                        total_fees += inventory_reconciler._f(fee.get("cost"))

                    total_amount += amount
                    total_cost += cost

                    order_id = trade.get("order")
                    if order_id:
                        order_ids.append(str(order_id))
                    trade_id = trade.get("id")
                    if trade_id:
                        trade_ids.append(str(trade_id))

                record.order_ids = list(dict.fromkeys(order_ids))
                record.trade_ids = list(dict.fromkeys(trade_ids))
                record.acquisition_fees = total_fees or record.acquisition_fees

                if total_amount > 0 and total_cost > 0:
                    record.acquisition_price = total_cost / total_amount
                    record.acquisition_cost = record.acquisition_price * record.quantity
                    record.provenance = "RECONSTRUCTED_FROM_VENUE_TRADES"
                    times = [
                        inventory_reconciler._f(trade.get("timestamp"))
                        for trade in buys
                        if trade.get("timestamp")
                    ]
                    if times:
                        record.acquisition_time = min(times) / 1000.0

                if record.acquisition_price and record.mark_price > 0:
                    record.unrealized_pnl = (
                        record.mark_price - record.acquisition_price
                    ) * record.quantity

            if record.provenance == "UNKNOWN":
                record.transition(
                    PROVENANCE_RECONSTRUCTED,
                    "no acquisition record and no venue trade history; cost "
                    "basis is unknown and is reported as unknown",
                )
            else:
                record.transition(
                    PROVENANCE_RECONSTRUCTED,
                    f"cost basis {record.provenance.lower()}",
                )
        return record

    # ------------------------------------------------------- executability

    def assess_executability(self, record: RecoveryRecord) -> RecoveryRecord:
        """Decide whether an exit is legally possible, and at what minimum.

        This answers "could this be sold at all", not "should it be". The
        minimum legal exit is the venue's own floor: the larger of its minimum
        amount and the amount needed to clear its minimum notional, rounded up
        to something the venue's precision will accept.
        """
        with self._lock:
            if not record.market_executable:
                record.exit_possible = False
                record.exit_blocked_reason = (
                    f"{record.symbol} is not an active market on "
                    f"{record.venue or 'this venue'}; capital cannot be "
                    "recovered here at all"
                )
                record.transition(
                    RECOVERY_BLOCKED_NOT_EXECUTABLE, record.exit_blocked_reason
                )
                return record

            by_amount = record.min_sell_amount
            by_notional = 0.0
            if record.min_sell_notional > 0 and record.mark_price > 0:
                by_notional = record.min_sell_notional / record.mark_price

            floor_amount = max(by_amount, by_notional)
            record.minimum_legal_exit_amount = floor_amount
            record.minimum_legal_exit_notional = floor_amount * record.mark_price

            # The holding must cover the floor *after* the venue truncates it.
            record.exit_possible = (
                record.executable_exit_amount > 0
                and record.executable_exit_amount + 1e-12 >= floor_amount
                and (
                    record.min_sell_notional <= 0
                    or record.executable_exit_value + 1e-12
                    >= record.min_sell_notional
                )
            )

            if record.exit_possible:
                record.exit_blocked_reason = ""
                record.transition(
                    EXECUTABILITY_ASSESSED,
                    f"a legal exit exists: {record.executable_exit_amount} "
                    f"{record.asset} for ~{record.executable_exit_value:.8f}",
                )
            else:
                record.exit_blocked_reason = (
                    f"holding {record.quantity} {record.asset} is below the "
                    f"venue's minimum sellable size "
                    f"({floor_amount:.10f} {record.asset} / "
                    f"{record.min_sell_notional} notional); the capital is "
                    "stranded until the balance or the minimum changes"
                )
                record.transition(
                    RECOVERY_BLOCKED_BELOW_MINIMUM, record.exit_blocked_reason
                )
            return record

    def adopt(self, record: RecoveryRecord) -> RecoveryRecord:
        """Take responsibility for a holding without selling anything.

        This is the coordinator's normal resting state for an orphan: it is
        understood, it is accounted for, it is visible, and it is not being
        liquidated.
        """
        with self._lock:
            if record.state in (
                RECOVERY_BLOCKED_NOT_EXECUTABLE,
                RECOVERY_BLOCKED_BELOW_MINIMUM,
            ):
                # Still adopted -- just with no legal exit to authorize.
                return record
            record.transition(
                ADOPTED_UNDER_MANAGEMENT,
                "under coordinator management; no exit proposed",
            )
            if record.asset in exit_allowlist():
                record.authorized = True
                record.authorized_by = "INVENTORY_RECOVERY_EXIT_ALLOWLIST"
                record.authorized_at = time.time()
                record.transition(
                    EXIT_AUTHORIZED, "asset is on the operator exit allowlist"
                )
            else:
                record.transition(
                    RECOVERY_AWAITING_AUTHORIZATION,
                    "a legal exit exists but no operator authorization does",
                )
            return record

    def authorize_exit(self, symbol: str, authorized_by: str) -> bool:
        """Explicitly permit one recovery exit.

        Authorization is per holding and is never self-granted: something
        outside this module -- an operator, or a policy that names the asset --
        has to say so.
        """
        with self._lock:
            record = self._records.get(symbol)
            if record is None:
                return False
            if not record.exit_possible:
                record.transition(
                    record.state,
                    "authorization ignored: no legal exit exists for this "
                    "holding",
                )
                return False
            record.authorized = True
            record.authorized_by = str(authorized_by or "unattributed")
            record.authorized_at = time.time()
            record.transition(
                EXIT_AUTHORIZED, f"authorized by {record.authorized_by}"
            )
            return True

    # --------------------------------------------------------------- exits

    def _router(self) -> Callable[..., Dict[str, Any]]:
        if self._route is not None:
            return self._route
        from .router import route_order

        return route_order

    def recover(
        self,
        symbol: str,
        broker: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Submit the authorized exit for one adopted holding.

        The path is fixed and is the same one every other order takes:
        coordinator -> ExecutionIntent -> preflight.prepare_order ->
        route_order -> BrokerCCXT. There is no direct create_order here, no
        second exchange client, and no synthetic close: if the exchange does
        not acknowledge, this reports that it did not.
        """
        with self._lock:
            record = self._records.get(symbol)

        if record is None:
            return {
                "ok": False,
                "reason": "NOT_ADOPTED",
                "detail": f"{symbol} is not under recovery management",
            }
        if not record.exit_possible:
            return {
                "ok": False,
                "reason": record.state,
                "detail": record.exit_blocked_reason,
            }
        if not record.authorized:
            return {
                "ok": False,
                "reason": RECOVERY_AWAITING_AUTHORIZATION,
                "detail": (
                    "recovery exits require explicit authorization; this "
                    "coordinator does not liquidate inventory on its own "
                    "judgement"
                ),
            }

        exit_intent = execution_intent.ExecutionIntent(
            symbol=record.symbol,
            side="sell",
            source_engine="inventory_recovery_coordinator",
            source_strategy="orphan_capital_recovery",
            candidate_id=execution_intent.new_candidate_id(),
            decision_id=execution_intent.new_decision_id(),
            venue=record.venue,
            environment=record.environment,
            confidence=1.0,
        )
        record.exit_intent_id = exit_intent.intent_id
        record.exit_correlation_id = exit_intent.correlation_id

        exit_intent.advance(
            execution_intent.CANDIDATE,
            f"recovery exit for adopted orphan {record.symbol}",
        )
        exit_intent.advance(
            execution_intent.ECONOMIC_ELIGIBILITY,
            f"legal exit {record.executable_exit_amount} {record.asset}",
        )
        exit_intent.advance(execution_intent.PREFLIGHT)

        # An exit is sized by what is held, not by a risk budget: the capital
        # is already committed. amount is passed explicitly so preflight sizes
        # the position being closed rather than a new one.
        order_intent: Dict[str, Any] = {
            "symbol": record.symbol,
            "side": "sell",
            "confidence": 1.0,
            "order_type": "market",
            "amount": record.executable_exit_amount,
            "reduce_only_inventory": True,
            "source": "inventory_recovery_coordinator",
        }
        if record.mark_price > 0:
            order_intent["price"] = record.mark_price
        if record.venue:
            order_intent["exchange_id"] = record.venue

        prepared, blocked = preflight.prepare_order(order_intent, broker=broker)

        if prepared is None:
            exit_intent.stop(
                execution_intent.PREFLIGHT, blocked.blocker, blocked.detail
            )
            with self._lock:
                record.transition(
                    RECOVERY_EXIT_REFUSED,
                    f"preflight refused: {blocked.blocker} ({blocked.detail})",
                )
            self.persist()
            return {
                "ok": False,
                "reason": blocked.blocker,
                "detail": blocked.detail,
                "stage": blocked.stage,
                "intent_id": exit_intent.intent_id,
                "correlation_id": exit_intent.correlation_id,
            }

        with self._lock:
            record.transition(
                EXIT_INTENT_PREPARED,
                f"preflight prepared {prepared.amount} {record.asset} "
                f"notional {prepared.notional:.8f} {prepared.quote_currency}",
            )

        preflight.record_event("submitted")
        exit_intent.advance(execution_intent.ROUTE_ORDER)
        with self._lock:
            record.transition(EXIT_SUBMITTED, "handed to route_order")

        payload = prepared.to_payload()
        payload.setdefault("params", {}).update(exit_intent.to_payload())

        receipt = self._router()(payload)

        blocker = preflight.classify_receipt(receipt)
        if blocker is not None:
            detail = str((receipt or {}).get("error", ""))[:200]
            preflight.record_blocker(blocker, f"{record.symbol}:{detail}")
            exit_intent.stop(execution_intent.EXCHANGE_ORDER, blocker, detail)
            with self._lock:
                record.transition(
                    RECOVERY_EXIT_REFUSED, f"{blocker}: {detail}"
                )
            self.persist()
            return {
                "ok": False,
                "prepared": True,
                "reason": blocker,
                "detail": detail,
                "receipt": receipt,
                "intent_id": exit_intent.intent_id,
                "correlation_id": exit_intent.correlation_id,
            }

        order = (receipt or {}).get("order") or {}
        preflight.record_event("acknowledged")
        exit_intent.advance(
            execution_intent.EXCHANGE_ORDER, str(order.get("id") or "")
        )

        filled = inventory_reconciler._f(order.get("filled"))
        average = order.get("average") or order.get("price")

        with self._lock:
            record.exit_order_id = str(order.get("id") or "")
            record.transition(
                EXIT_ACKNOWLEDGED,
                f"exchange acknowledged order {record.exit_order_id}",
            )

            if filled > 0:
                preflight.record_event("fills")
                preflight.record_event("closes")
                exit_intent.succeed(execution_intent.FILL, str(filled))
                exit_intent.succeed(execution_intent.CLOSE, record.symbol)
                record.exit_filled_amount = filled
                record.exit_fill_price = (
                    float(average) if average is not None else None
                )
                fee = order.get("fee") or {}
                if isinstance(fee, dict):
                    record.exit_fees = inventory_reconciler._f(fee.get("cost")) or None
                record.recovered_quote = inventory_reconciler._f(
                    order.get("cost")
                ) or (filled * (record.exit_fill_price or record.mark_price))
                record.transition(
                    EXIT_FILLED,
                    f"filled {filled} {record.asset} for "
                    f"{record.recovered_quote:.8f}",
                )
            else:
                # Acknowledged is not filled. Saying otherwise would be the
                # synthetic close this module exists to avoid.
                record.transition(
                    EXIT_ACKNOWLEDGED,
                    "order acknowledged with no fill reported yet; recovery "
                    "is not complete and no proceeds are claimed",
                )

        preflight.invalidate_balance_cache()
        self.persist()

        return {
            "ok": True,
            "order_id": record.exit_order_id,
            "filled": filled,
            "recovered_quote": record.recovered_quote,
            "state": record.state,
            "receipt": receipt,
            "intent_id": exit_intent.intent_id,
            "correlation_id": exit_intent.correlation_id,
        }

    def reconcile_recovery(
        self,
        symbol: str,
        items: List[inventory_reconciler.InventoryItem],
    ) -> Optional[RecoveryRecord]:
        """Confirm from a fresh inventory read that the holding is gone.

        A recovery is only complete when the exchange's own balance agrees.
        Until then the record stays at EXIT_FILLED, which is a claim about one
        order, not about the account.
        """
        with self._lock:
            record = self._records.get(symbol)
            if record is None:
                return None

            remaining = next(
                (item for item in items if item.symbol == symbol), None
            )
            still_held = remaining.total if remaining is not None else 0.0

            if still_held <= 0:
                record.quantity = 0.0
                record.transition(
                    RECOVERED_RECONCILED,
                    f"balance confirms the position is closed; "
                    f"{record.recovered_quote:.8f} returned to free quote",
                )
            else:
                record.quantity = still_held
                record.transition(
                    record.state,
                    f"{still_held} {record.asset} still held; recovery is "
                    "partial and remains open",
                )
            return record

    # -------------------------------------------------------- observability

    def records(self) -> List[RecoveryRecord]:
        with self._lock:
            return list(self._records.values())

    def record_for(self, symbol: str) -> Optional[RecoveryRecord]:
        with self._lock:
            return self._records.get(symbol)

    def summary(self) -> Dict[str, Any]:
        """What is adopted, what is exitable, and what is stranded.

        ``stranded_mark_to_market_value`` is the answer to "how much capital
        can this account not currently recover", which is a number worth
        knowing and is not the same as either total inventory or spendable
        cash.
        """
        with self._lock:
            records = list(self._records.values())

        by_state: Dict[str, int] = {}
        exitable_value = 0.0
        stranded_value = 0.0
        recovered = 0.0
        unknown_cost = 0

        for record in records:
            by_state[record.state] = by_state.get(record.state, 0) + 1
            if record.exit_possible:
                exitable_value += record.executable_exit_value
            else:
                stranded_value += record.mark_to_market_value
            recovered += record.recovered_quote
            if record.provenance == "UNKNOWN":
                unknown_cost += 1

        return {
            "schema_version": SCHEMA_VERSION,
            "adopted": len(records),
            "by_state": dict(
                sorted(by_state.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "exitable_executable_value": exitable_value,
            "stranded_mark_to_market_value": stranded_value,
            "recovered_quote": recovered,
            "holdings_with_unknown_cost_basis": unknown_cost,
            "exit_allowlist": sorted(exit_allowlist()),
            "awaiting_authorization": sum(
                1
                for record in records
                if record.exit_possible and not record.authorized
            ),
        }

    def persist(self) -> bool:
        """Write recovery state atomically, so another process can read it."""
        path = _snapshot_path()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": time.time(),
            "source_run_id": preflight.RUN_ID,
            "source_pid": os.getpid(),
            "venue": self.venue,
            "environment": self.environment,
            "summary": self.summary(),
            "records": [record.as_dict() for record in self.records()],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.replace(tmp, path)
            return True
        except Exception:
            return False

    def load(self) -> int:
        """Restore previously adopted records. Returns how many were read."""
        try:
            payload = json.loads(_snapshot_path().read_text(encoding="utf-8"))
        except Exception:
            return 0
        if not isinstance(payload, dict):
            return 0

        restored = 0
        with self._lock:
            for raw in payload.get("records") or []:
                if not isinstance(raw, dict):
                    continue
                try:
                    record = RecoveryRecord(**raw)
                except TypeError:
                    continue
                self._records[record.symbol] = record
                restored += 1
        return restored


def snapshot_path() -> Path:
    """Where recovery state is written. Exposed for status reporting."""
    return _snapshot_path()


def read_snapshot() -> Dict[str, Any]:
    """Read recovery state from another process, or {} if none exists."""
    try:
        payload = json.loads(_snapshot_path().read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def coordinate(
    items: List[inventory_reconciler.InventoryItem],
    venue: str = "",
    environment: str = "",
    trades_by_symbol: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    coordinator: Optional[InventoryRecoveryCoordinator] = None,
) -> InventoryRecoveryCoordinator:
    """Run the full understanding pass over a reconciled inventory.

    Discovery, provenance, executability and adoption -- in that order, and
    stopping there. Nothing in this function can place an order.
    """
    coordinator = coordinator or InventoryRecoveryCoordinator(
        venue=venue, environment=environment
    )
    trades_by_symbol = trades_by_symbol or {}

    for record in coordinator.discover(items):
        coordinator.reconstruct_provenance(
            record, trades_by_symbol.get(record.symbol)
        )
        coordinator.assess_executability(record)
        coordinator.adopt(record)

    coordinator.persist()
    return coordinator
