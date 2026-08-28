"""v1.60.29 terminal pending-event reconciliation and restart-safe cycle recovery.

Bybit Testnet only. This module never submits an exchange order, never
fabricates a fill or a close, never mutates global exchange-realized PnL, and
never relaxes an executability, freshness, price-limit, reconciliation, or
idempotency gate. Every payload asserts ``live_authority`` is ``False``.

P0  Terminal pending-event latch. A persisted fast-lane ``pending_event`` whose
    deterministic ``orderLinkId`` is already terminal-closed with a real fill is
    reconciled instead of resubmitted. Open/submitting/ambiguous orders remain
    fail-closed and keep the latch.

P1  Restart-safe completed-cycle recovery from durable closed filled buy/sell
    orders, rather than the ephemeral ``position_cycle_pnl_usd`` key that
    ``_record_non_tradeable_dust`` destroys.

P2  Canonical Bybit executable minimums, resolved from CCXT limits and
    cross-checked against the raw ``lotSizeFilter``, failing closed when a
    minimum cannot be proven, and evaluated on the raw quantity so that
    ``amount_to_precision`` is never called with a sub-precision amount.
"""

from __future__ import annotations

import copy
import datetime as dt
from typing import Any

from .testnet_exit_recycle import _record_non_tradeable_dust
from .testnet_residual_dust_cycle_v1627 import (
    _current_cycle_evidence,
    _n,
    _timestamp,
)

VERSION = "1.60.29"

# Orders in these states are unresolved. They must never be reconciled away and
# must never release the pending latch.
NON_TERMINAL_ORDER_STATES = frozenset({"open", "submitting"})

# Bounded startup recovery.
MAX_RECOVERY_SYMBOLS = 50
MAX_RECORDED_CYCLES = 250

# Residual cost-basis match tolerance (strict).
COST_BASIS_ABS_TOLERANCE = 1e-9
COST_BASIS_REL_TOLERANCE = 1e-6

# Residual quantity tolerance is derived from the canonical minimum amount so it
# can never be large enough to mistake a tradeable remainder for dust.
QUANTITY_TOLERANCE_FRACTION = 1e-2
QUANTITY_TOLERANCE_FLOOR = 1e-9

# Canonical v1.60.27 bounded retention. Rows roll out sooner than keys, which is
# why the key ledger is authoritative for idempotency.
MAX_RECORDED_CYCLE_KEYS = 500


def canonical_executable_minimums(
    exchange: Any,
    symbol: str,
) -> tuple[float, float, bool]:
    """Return ``(min_amount, min_cost, resolved)`` for a Bybit spot market.

    CCXT's normalized ``limits`` are preferred and are cross-checked against
    Bybit's raw ``info.lotSizeFilter``. The strictest proven value from either
    source wins, so the helper can only ever tighten a constraint.

    ``resolved`` is ``False`` when neither source proves a minimum. Callers must
    treat that as "cannot prove executable" rather than "unconstrained".
    """

    try:
        market = exchange.market(symbol)
    except Exception:
        return 0.0, 0.0, False

    if not isinstance(market, dict):
        return 0.0, 0.0, False

    limits = market.get("limits") or {}
    min_amount = max(0.0, _n((limits.get("amount") or {}).get("min")))
    min_cost = max(0.0, _n((limits.get("cost") or {}).get("min")))

    lot_size_filter = (market.get("info") or {}).get("lotSizeFilter") or {}
    raw_min_amount = max(0.0, _n(lot_size_filter.get("minOrderQty")))
    raw_min_cost = max(0.0, _n(lot_size_filter.get("minOrderAmt")))

    # Strictest proven minimum from either source.
    min_amount = max(min_amount, raw_min_amount)
    min_cost = max(min_cost, raw_min_cost)

    resolved = bool(min_amount > 0.0 or min_cost > 0.0)
    return min_amount, min_cost, resolved


def below_canonical_minimum(
    exchange: Any,
    symbol: str,
    quantity: float,
    reference_price: float,
) -> dict[str, Any]:
    """Decide whether ``quantity`` is provably non-tradeable.

    Evaluated on the raw quantity, before any precision conversion, because
    Bybit raises ``InvalidOrder`` when ``amount_to_precision`` is handed an
    amount below the market's amount precision.
    """

    min_amount, min_cost, resolved = canonical_executable_minimums(
        exchange,
        symbol,
    )

    quantity = max(0.0, _n(quantity))
    reference_price = max(0.0, _n(reference_price))

    if not resolved:
        return {
            "below_minimum": False,
            "resolved": False,
            "minimum_amount": min_amount,
            "minimum_cost_usd": min_cost,
            "reason": "canonical_minimums_unresolved",
            "live_authority": False,
        }

    below_amount = bool(min_amount > 0.0 and quantity < min_amount)
    below_cost = bool(
        min_cost > 0.0
        and reference_price > 0.0
        and quantity * reference_price < min_cost
    )

    return {
        "below_minimum": bool(quantity <= 0.0 or below_amount or below_cost),
        "resolved": True,
        "below_minimum_amount": below_amount,
        "below_minimum_cost": below_cost,
        "minimum_amount": min_amount,
        "minimum_cost_usd": min_cost,
        "quantity": quantity,
        "reference_price": reference_price,
        "estimated_value_usd": quantity * reference_price,
        "precision_conversion_skipped": bool(below_amount),
        "live_authority": False,
    }


def _deterministic_client_order_id(event: dict[str, Any]) -> str:
    """Derive the persisted event's idempotent ``orderLinkId``."""

    from .testnet_execution import BybitTestnetExecutionEngine

    return BybitTestnetExecutionEngine._client_order_id(event)


def _authoritative_order(
    testnet: Any,
    client_order_id: str,
) -> dict[str, Any] | None:
    state = getattr(testnet, "state", None)
    lock = getattr(testnet, "_io_lock", None)

    if not isinstance(state, dict) or lock is None:
        return None

    with lock:
        row = (state.get("orders") or {}).get(client_order_id)
        return copy.deepcopy(row) if isinstance(row, dict) else None


def _has_later_filled_buy(
    state: dict[str, Any],
    symbol: str,
    after_timestamp: float,
) -> bool:
    """True when a real filled buy exists after ``after_timestamp``.

    A later buy means the reconstructed cycle is stale and must not be counted.
    """

    normalized = str(symbol or "").upper()

    for row in (state.get("orders") or {}).values():
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol") or "").upper() != normalized:
            continue
        if str(row.get("side") or "").lower() != "buy":
            continue
        if str(row.get("status") or "").lower() != "closed":
            continue
        if _n(row.get("filled")) <= 0.0:
            continue
        if _timestamp(row) > after_timestamp:
            return True

    return False


def _recorded_cycle_keys(state: dict[str, Any]) -> set[str]:
    """Idempotency ledger shared with v1.60.27 so a cycle counts at most once.

    The canonical ``v1627_completed_cycle_keys`` ledger is authoritative and is
    retained longer (500) than the completed-cycle rows (250). Consulting only
    the rows would let a cycle be recounted once its row rolled out of the
    bounded row list, so both sources are unioned.
    """

    keys: set[str] = set()

    for key in state.get("v1627_completed_cycle_keys") or []:
        text = str(key or "")
        if text:
            keys.add(text)

    for row in state.get("v1627_completed_executable_cycles") or []:
        if isinstance(row, dict):
            text = str(row.get("cycle_key") or "")
            if text:
                keys.add(text)

    return keys


def _dust_recorded_at_timestamp(dust: dict[str, Any]) -> float:
    raw = dust.get("recorded_at")

    if raw in {None, ""}:
        return 0.0

    try:
        parsed = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0.0

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)

    return parsed.timestamp()


def evaluate_recoverable_cycle(
    *,
    exchange: Any,
    state: dict[str, Any],
    symbol: str,
    dust: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Fail-closed predicate for counting a previously uncounted dust cycle.

    Every condition must hold. Any failure returns ``eligible=False`` with the
    precise reason and never counts anything.
    """

    def reject(reason: str) -> dict[str, Any]:
        return {
            "eligible": False,
            "reason": reason,
            "symbol": str(symbol or "").upper(),
            "live_authority": False,
        }

    if not isinstance(dust, dict) or not isinstance(evidence, dict) or not evidence:
        return reject("missing_dust_or_evidence")

    if dust.get("counted_as_executed_close") is True:
        return reject("already_counted_as_executed_close")

    if int(_n(evidence.get("buy_order_count"))) <= 0:
        return reject("no_real_filled_buy")

    if int(_n(evidence.get("sell_order_count"))) <= 0:
        return reject("no_real_filled_sell")

    executed_sell_quantity = _n(evidence.get("executed_sell_quantity"))
    if executed_sell_quantity <= 0.0:
        return reject("no_executed_sell_quantity")

    buy_at = _n(evidence.get("buy_submitted_at"))
    sell_at = _n(evidence.get("sell_submitted_at"))

    if buy_at <= 0.0 or sell_at <= 0.0:
        return reject("incomplete_cycle_timestamps")

    if sell_at < buy_at:
        return reject("sell_not_after_buy")

    if _has_later_filled_buy(state, symbol, sell_at):
        return reject("later_buy_invalidates_cycle")

    cycle_key = str(evidence.get("cycle_key") or "")
    if not cycle_key:
        return reject("missing_cycle_key")

    if cycle_key in _recorded_cycle_keys(state):
        return reject("cycle_key_already_recorded")

    recorded_at = _dust_recorded_at_timestamp(dust)
    if recorded_at <= 0.0:
        return reject("dust_recorded_at_unavailable")

    # Forensic invariant: dust is a consequence of the sell, so it can never be
    # recorded before it. Both values are persisted UTC timestamps, so any
    # ordering that is not strictly non-decreasing fails closed. No slack.
    if recorded_at < sell_at:
        return reject("dust_recorded_before_sell")

    # Residual quantity match, tolerance derived from the canonical minimum.
    dust_quantity = max(0.0, _n(dust.get("quantity")))
    residual_estimate = max(0.0, _n(evidence.get("residual_base_estimate")))

    min_amount, _min_cost, resolved = canonical_executable_minimums(
        exchange,
        symbol,
    )

    if not resolved:
        return reject("canonical_minimums_unresolved")

    quantity_tolerance = max(
        QUANTITY_TOLERANCE_FLOOR,
        min_amount * QUANTITY_TOLERANCE_FRACTION,
    )

    if abs(residual_estimate - dust_quantity) > quantity_tolerance:
        return reject("residual_quantity_mismatch")

    # Residual cost basis match (strict).
    effective_buy_quantity = _n(evidence.get("effective_buy_quantity"))
    entry_cost_usd = max(0.0, _n(evidence.get("entry_cost_usd")))

    if effective_buy_quantity <= 0.0:
        return reject("effective_buy_quantity_unavailable")

    reconstructed_cost_basis = entry_cost_usd * (
        residual_estimate / effective_buy_quantity
    )
    dust_cost_basis = max(0.0, _n(dust.get("cost_basis_usd")))

    cost_tolerance = max(
        COST_BASIS_ABS_TOLERANCE,
        dust_cost_basis * COST_BASIS_REL_TOLERANCE,
    )

    if abs(reconstructed_cost_basis - dust_cost_basis) > cost_tolerance:
        return reject("residual_cost_basis_mismatch")

    realized_sell_pnl = _n(evidence.get("reconstructed_realized_sell_pnl_usd"))
    net_after_dust = realized_sell_pnl - dust_cost_basis

    return {
        "eligible": True,
        "reason": "recoverable_completed_cycle",
        "symbol": str(symbol or "").upper(),
        "cycle_key": cycle_key,
        "realized_sell_pnl_usd": realized_sell_pnl,
        "residual_dust_cost_basis_usd": dust_cost_basis,
        "net_realized_after_dust_usd": net_after_dust,
        "is_win": bool(net_after_dust > 0.0),
        "winning_after_dust": bool(net_after_dust > 0.0),
        "residual_quantity": residual_estimate,
        "quantity_tolerance": quantity_tolerance,
        "cost_basis_tolerance": cost_tolerance,
        "counted_as_executed_close": True,
        "residual_dust_counted_as_sale": False,
        "global_realized_pnl_mutated": False,
        "testnet_only": True,
        "live_authority": False,
    }


def install_testnet_terminal_pending_recovery_v1629() -> None:
    from .fast_collective_hyper import HyperSpeedCollectiveTestnetLane
    from .testnet_execution import BybitTestnetExecutionEngine
    from .velocity_sniper_testnet import VelocitySniperTestnetLane

    if getattr(
        BybitTestnetExecutionEngine,
        "_v1629_terminal_pending_recovery_installed",
        False,
    ):
        return

    original_submit_pending = HyperSpeedCollectiveTestnetLane._submit_pending
    original_prepare_sell = BybitTestnetExecutionEngine.prepare_sell
    original_engine_health = BybitTestnetExecutionEngine.health
    original_engine_start = BybitTestnetExecutionEngine.start
    original_lane_health = HyperSpeedCollectiveTestnetLane.health

    def _has_canonical_cycle(self: Any, symbol: str) -> bool:
        normalized = str(symbol or "").upper()
        for row in self.state.get("v1627_completed_executable_cycles") or []:
            if (
                isinstance(row, dict)
                and str(row.get("symbol") or "").upper() == normalized
            ):
                return True
        return False

    def prepare_sell(
        self: Any,
        symbol: str,
        requested_quantity: float,
        reference_price: float,
    ) -> dict[str, Any]:
        """Finalize a durable cycle as soon as dust is persisted.

        v1.60.27 marks ``completed_executable_cycle`` from the ephemeral
        ``position_cycle_pnl_usd`` key, which ``_record_non_tradeable_dust``
        destroys. Without this, a genuinely completed cycle proven by durable
        closed filled orders would never reach v1.60.27's retirement path and it
        would emit a zeroed closed record. Finalization is idempotent.
        """

        result = original_prepare_sell(
            self,
            symbol,
            requested_quantity,
            reference_price,
        )

        if str((result or {}).get("status") or "") != "dust":
            return result

        normalized = str(symbol or "").upper()

        # Shared single-symbol finalizer: same evidence gates as startup
        # recovery, counts at most once, submits nothing.
        _finalize_symbol_cycle(self, normalized)

        result = dict(result)

        with self._io_lock:
            durable_cycle = _has_canonical_cycle(self, normalized)

        if durable_cycle:
            result["completed_executable_cycle"] = True
            result["completed_cycle_source"] = "durable_closed_filled_orders"

        result["residual_dust_counted_as_sale"] = False
        result["live_authority"] = False
        return result

    def _finalize_symbol_cycle(self: Any, symbol: str) -> dict[str, Any]:
        """Durably finalize one symbol's completed cycle, at most once.

        Shared by bounded startup recovery and P0 immediate finalization so the
        two paths cannot drift. Writes the canonical v1.60.27 completed-cycle
        schema, appends to the canonical key ledger, marks the dust row counted,
        and never mutates global exchange-realized PnL or submits an order.
        """

        normalized = str(symbol or "").upper()

        with self._io_lock:
            dust_row = (self.state.get("non_tradeable_dust") or {}).get(normalized)
            dust = copy.deepcopy(dust_row) if isinstance(dust_row, dict) else {}
            evidence = _current_cycle_evidence(self.state, normalized)

            assessment = evaluate_recoverable_cycle(
                exchange=getattr(self, "exchange", None),
                state=self.state,
                symbol=normalized,
                dust=dust,
                evidence=evidence,
            )

            if assessment.get("eligible") is not True:
                return assessment

            realized_sell_pnl = assessment["realized_sell_pnl_usd"]
            dust_cost_basis = assessment["residual_dust_cost_basis_usd"]
            actual_net = assessment["net_realized_after_dust_usd"]
            winning_after_dust = bool(actual_net > 0.0)
            cycle_key = assessment["cycle_key"]

            cycle = {
                **copy.deepcopy(evidence),
                # Canonical v1.60.27 completed-cycle schema. Existing consumers
                # (engine health aggregation and retire_fast_state) read exactly
                # these field names.
                "actual_realized_sell_pnl_usd": realized_sell_pnl,
                "residual_dust_cost_basis_usd": dust_cost_basis,
                "actual_cycle_net_after_dust_usd": actual_net,
                "winning_after_dust": winning_after_dust,
                "residual_dust_quantity": assessment["residual_quantity"],
                "completed_executable_cycle": True,
                "counted_as_executed_close": True,
                "residual_dust_counted_as_sale": False,
                "recovered_by": VERSION,
                "recovery_source": "durable_closed_filled_orders",
                "global_realized_pnl_mutated": False,
                "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
                "testnet_only": True,
                "live_authority": False,
            }

            rows = list(self.state.get("v1627_completed_executable_cycles") or [])
            rows.append(cycle)
            self.state["v1627_completed_executable_cycles"] = rows[
                -MAX_RECORDED_CYCLES:
            ]

            # Canonical key ledger, bounded exactly as v1.60.27 bounds it.
            keys = list(self.state.get("v1627_completed_cycle_keys") or [])
            if cycle_key not in keys:
                keys.append(cycle_key)
            self.state["v1627_completed_cycle_keys"] = keys[-MAX_RECORDED_CYCLE_KEYS:]

            self.state["closed_positions"] = (
                int(self.state.get("closed_positions") or 0) + 1
            )
            if winning_after_dust:
                self.state["winning_positions"] = (
                    int(self.state.get("winning_positions") or 0) + 1
                )

            self.state["v1629_recovered_cycles"] = (
                int(self.state.get("v1629_recovered_cycles") or 0) + 1
            )

            if isinstance(dust_row, dict):
                dust_row["counted_as_executed_close"] = True
                dust_row["counted_by"] = VERSION
                dust_row["actual_cycle_net_after_dust_usd"] = actual_net

            self._save_state()

        return {**assessment, "winning_after_dust": winning_after_dust}

    def recover_uncounted_dust_cycles(
        self: Any,
        *,
        limit: int = MAX_RECOVERY_SYMBOLS,
    ) -> dict[str, Any]:
        """Bounded startup recovery of previously recorded uncounted dust cycles.

        Delegates every symbol to the shared single-symbol finalizer so startup
        recovery and P0 immediate finalization enforce identical evidence gates.
        """

        recovered: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        with self._io_lock:
            dust_symbols = [
                symbol
                for symbol, row in (self.state.get("non_tradeable_dust") or {}).items()
                if isinstance(row, dict)
                and row.get("counted_as_executed_close") is not True
            ][: max(0, int(limit))]

        for symbol in dust_symbols:
            assessment = _finalize_symbol_cycle(self, symbol)
            if assessment.get("eligible") is True:
                recovered.append(assessment)
            else:
                rejected.append(assessment)

        return {
            "version": VERSION,
            "inspected": len(dust_symbols),
            "recovered": len(recovered),
            "rejected": len(rejected),
            "cycles": recovered,
            "rejections": rejected[-20:],
            "global_realized_pnl_mutated": False,
            "testnet_only": True,
            "live_authority": False,
        }

    def start(self: Any) -> None:
        """Start the executor, then perform bounded uncounted-cycle recovery.

        Recovery reads only durable persisted state. A failure here must never
        prevent the executor from starting, so it is recorded and swallowed.
        """

        original_engine_start(self)

        try:
            outcome = recover_uncounted_dust_cycles(self)
        except Exception as exc:  # pragma: no cover - defensive
            with self._io_lock:
                self.state["v1629_startup_recovery"] = {
                    "version": VERSION,
                    "ok": False,
                    "error": type(exc).__name__,
                    "live_authority": False,
                }
                self._save_state()
            return

        with self._io_lock:
            self.state["v1629_startup_recovery"] = {
                "version": VERSION,
                "ok": True,
                "inspected": outcome.get("inspected"),
                "recovered": outcome.get("recovered"),
                "rejected": outcome.get("rejected"),
                "bounded_limit": MAX_RECOVERY_SYMBOLS,
                "global_realized_pnl_mutated": False,
                "live_authority": False,
            }
            self._save_state()

    def _submit_pending(
        self: Any,
        pending: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        """P0: reconcile a terminal pending event instead of resubmitting it."""

        event = (pending or {}).get("event") or {}
        symbol = str(event.get("symbol") or "").upper()

        testnet_state = getattr(self.testnet, "state", None)
        testnet_lock = getattr(self.testnet, "_io_lock", None)

        # Legacy/test adapters keep their original behavior.
        if not isinstance(testnet_state, dict) or testnet_lock is None or not symbol:
            return original_submit_pending(self, pending, now=now)

        try:
            client_order_id = _deterministic_client_order_id(event)
        except Exception:
            return original_submit_pending(self, pending, now=now)

        order = _authoritative_order(self.testnet, client_order_id)

        if not isinstance(order, dict):
            # No authoritative record for this orderLinkId. Preserve the existing
            # submission path, which carries its own idempotency.
            return original_submit_pending(self, pending, now=now)

        status = str(order.get("status") or "").lower()
        filled = max(0.0, _n(order.get("filled")))

        # Unresolved orders are fail-closed: never resubmit, never reconcile,
        # never clear the latch.
        if status in NON_TERMINAL_ORDER_STATES:
            with self._lock:
                self.state["last_action"] = {
                    "action": "terminal_pending_unresolved_fail_closed",
                    "symbol": symbol,
                    "client_order_id": client_order_id,
                    "status": status,
                    "timestamp": now,
                    "order_submitted": False,
                    "pending_latch_cleared": False,
                    "live_authority": False,
                }
                self._save_locked()

            return self._decision(
                "terminal_pending_unresolved_order_fail_closed",
                details={
                    "kind": pending.get("kind"),
                    "symbol": symbol,
                    "client_order_id": client_order_id,
                    "status": status,
                    "order_submitted": False,
                    "pending_latch_cleared": False,
                    "position_remains_active": True,
                    "live_authority": False,
                },
            )

        # Only a terminal close with a real fill is reconcilable here.
        if status != "closed" or filled <= 0.0:
            return original_submit_pending(self, pending, now=now)

        # Authoritative executor reconciliation. Ambiguity is fail-closed.
        try:
            self.testnet.reconcile_required()
            snapshot = self.testnet.safe_snapshot()
        except Exception:
            return self._decision(
                "terminal_pending_reconciliation_ambiguous",
                details={
                    "kind": pending.get("kind"),
                    "symbol": symbol,
                    "client_order_id": client_order_id,
                    "order_submitted": False,
                    "pending_latch_cleared": False,
                    "position_remains_active": True,
                    "live_authority": False,
                },
            )

        remaining = max(0.0, _n((snapshot.get("positions") or {}).get(symbol)))
        reference_price = max(
            0.0,
            _n(order.get("average"), _n(event.get("price"))),
        )

        exchange = getattr(self.testnet, "exchange", None)

        if exchange is None:
            dust_assessment = {
                "below_minimum": False,
                "resolved": False,
                "reason": "exchange_unavailable",
                "live_authority": False,
            }
        else:
            dust_assessment = below_canonical_minimum(
                exchange,
                symbol,
                remaining,
                reference_price,
            )

        residual_recorded_as_dust = False

        # Route a provably non-tradeable residual through the canonical dust
        # path. No precision conversion, no exchange order, no fabricated close.
        if remaining > 0.0 and dust_assessment.get("below_minimum") is True:
            with testnet_lock:
                free_quantity = max(
                    0.0,
                    _n(
                        (
                            (testnet_state.get("account_balance") or {}).get("free")
                            or {}
                        ).get(symbol.split("/", 1)[0])
                    ),
                )
                dust_result = _record_non_tradeable_dust(
                    self.testnet,
                    symbol=symbol,
                    quantity=remaining,
                    reference_price=reference_price,
                    minimum_amount=dust_assessment.get("minimum_amount") or 0.0,
                    minimum_cost=dust_assessment.get("minimum_cost_usd") or 0.0,
                    free_quantity=free_quantity,
                    reason="terminal_pending_residual_below_canonical_minimum",
                )
            residual_recorded_as_dust = (
                str((dust_result or {}).get("status") or "") == "dust"
            )
            if residual_recorded_as_dust:
                remaining = 0.0

        # The symbol is fully retired only when nothing tradeable remains.
        position_retired = bool(remaining <= 0.0)

        with self._lock:
            active = self.state.setdefault("active", {})
            record = active.get(symbol)

            if record is not None and not position_retired:
                # Retain the authoritative remaining executor quantity.
                record["quantity"] = remaining

            if position_retired:
                # Clear every symbol-scoped stale fast state, matching the
                # cleanup semantics already proven by v1.60.27
                # retire_fast_state(). Strictly scoped to this symbol.
                active.pop(symbol, None)

                (self.state.get("deferred_exit_recoveries") or {}).pop(symbol, None)
                (self.state.get("v1615_price_limit_watch") or {}).pop(symbol, None)

                self.state.setdefault("last_exit_by_symbol", {})[symbol] = now

            # Clear the stale global pending latch only when it is still this
            # symbol's event; a concurrently installed latch for another symbol
            # must survive untouched.
            pending_latch = self.state.get("pending_event")
            pending_latch_cleared = False

            if isinstance(pending_latch, dict):
                latch_event = (
                    pending_latch.get("event")
                    or pending_latch.get("source_event")
                    or {}
                )
                if str(latch_event.get("symbol") or "").upper() == symbol:
                    self.state["pending_event"] = None
                    pending_latch_cleared = True
            elif pending_latch is None:
                pending_latch_cleared = True

            self.state["last_error"] = None

            reconciliations = list(
                self.state.get("v1629_terminal_pending_reconciliations") or []
            )
            reconciliations.append(
                {
                    "symbol": symbol,
                    "client_order_id": client_order_id,
                    "event_id": event.get("event_id"),
                    "order_id": order.get("order_id") or order.get("id"),
                    "status": status,
                    "filled": filled,
                    "authoritative_remaining_quantity": remaining,
                    "residual_recorded_as_dust": residual_recorded_as_dust,
                    "dust_assessment": copy.deepcopy(dust_assessment),
                    "position_retired": position_retired,
                    "order_submitted": False,
                    "resubmission_suppressed": True,
                    "pending_latch_cleared": pending_latch_cleared,
                    "stale_symbol_state_cleared": position_retired,
                    "fabricated_close": False,
                    "global_realized_pnl_mutated": False,
                    "observed_at": now,
                    "live_authority": False,
                }
            )
            self.state["v1629_terminal_pending_reconciliations"] = reconciliations[
                -100:
            ]

            self.state["last_action"] = {
                "action": "terminal_pending_reconciled",
                "symbol": symbol,
                "client_order_id": client_order_id,
                "status": status,
                "filled": filled,
                "timestamp": now,
                "order_submitted": False,
                "pending_latch_cleared": pending_latch_cleared,
                "live_authority": False,
            }

            self._save_locked()

        # v1.60.29 P0->P1: finalize the residual-dust cycle now rather than
        # waiting for the next restart. Same evidence gates as startup recovery,
        # via the shared single-symbol finalizer. Counts at most once.
        finalization: dict[str, Any] = {}
        if residual_recorded_as_dust:
            finalization = _finalize_symbol_cycle(self.testnet, symbol)

        return self._decision(
            "terminal_pending_reconciled",
            details={
                "kind": pending.get("kind"),
                "symbol": symbol,
                "client_order_id": client_order_id,
                "status": status,
                "filled": filled,
                "authoritative_remaining_quantity": remaining,
                "residual_recorded_as_dust": residual_recorded_as_dust,
                "dust_assessment": dust_assessment,
                "position_retired": position_retired,
                "stale_symbol_state_cleared": position_retired,
                "immediate_cycle_finalization": finalization,
                "cycle_counted_immediately": bool(
                    finalization.get("eligible") is True
                ),
                "order_submitted": False,
                "resubmission_suppressed": True,
                "pending_latch_cleared": pending_latch_cleared,
                "fabricated_close": False,
                "residual_dust_counted_as_sale": False,
                "global_realized_pnl_mutated": False,
                "live_authority": False,
            },
        )

    def engine_health(self: Any) -> dict[str, Any]:
        payload = original_engine_health(self)

        payload["terminal_pending_cycle_recovery"] = {
            "version": VERSION,
            "enabled": True,
            "recovered_cycles": int(self.state.get("v1629_recovered_cycles") or 0),
            "startup_recovery": copy.deepcopy(
                self.state.get("v1629_startup_recovery") or {}
            ),
            "startup_recovery_is_bounded": True,
            "requires_real_filled_buy_and_sell": True,
            "requires_residual_quantity_match": True,
            "requires_residual_cost_basis_match": True,
            "later_buy_invalidates_cycle": True,
            "win_loss_is_net_of_residual_dust": True,
            "writes_canonical_v1627_cycle_schema": True,
            "uses_canonical_v1627_key_ledger": True,
            "dust_must_not_precede_sell": True,
            "immediate_finalization_without_restart": True,
            "global_realized_pnl_mutated": False,
            "fake_close_allowed": False,
            "testnet_only": True,
            "live_authority": False,
        }
        payload["live_authority"] = False
        return payload

    def lane_health(self: Any) -> dict[str, Any]:
        payload = original_lane_health(self)

        with self._lock:
            reconciliations = copy.deepcopy(
                self.state.get("v1629_terminal_pending_reconciliations") or []
            )

        payload["terminal_pending_latch"] = {
            "version": VERSION,
            "enabled": True,
            "terminal_closed_pending_event_resubmitted": False,
            "unresolved_orders_fail_closed": True,
            "reconciliation_ambiguity_fail_closed": True,
            "authoritative_remaining_quantity_retained": True,
            "deterministic_order_link_id_idempotency": True,
            "reconciliations": len(reconciliations),
            "recent": reconciliations[-20:],
            "live_authority": False,
        }
        payload["live_authority"] = False
        return payload

    BybitTestnetExecutionEngine.prepare_sell = prepare_sell
    BybitTestnetExecutionEngine.finalize_symbol_cycle = _finalize_symbol_cycle
    BybitTestnetExecutionEngine.recover_uncounted_dust_cycles = (
        recover_uncounted_dust_cycles
    )
    BybitTestnetExecutionEngine.start = start
    BybitTestnetExecutionEngine.health = engine_health
    HyperSpeedCollectiveTestnetLane._submit_pending = _submit_pending
    HyperSpeedCollectiveTestnetLane.health = lane_health

    BybitTestnetExecutionEngine.VERSION = "3.5"
    HyperSpeedCollectiveTestnetLane.VERSION = VERSION
    VelocitySniperTestnetLane.VERSION = VERSION

    BybitTestnetExecutionEngine._v1629_terminal_pending_recovery_installed = True
