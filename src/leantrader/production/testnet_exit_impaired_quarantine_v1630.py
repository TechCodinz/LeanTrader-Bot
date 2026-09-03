"""v1.60.30 exit-impaired symbol quarantine and liquid-capital integrity.

Bybit Testnet only. This module never submits an exchange order, never sells or
deletes a position, never fabricates a fill or close, never bypasses a price
limit, and never relaxes reconciliation, idempotency or the >=30 bps modeled
round-trip cost floor. Every payload asserts ``live_authority`` is ``False``.

P0  Quarantine a symbol only from authoritative *current* runtime evidence: a
    real positive executor position plus a persistent symbol-scoped exit
    obstruction. A losing position or a stale historical record is never
    sufficient. Quarantine lifts automatically once the position is gone or the
    obstruction genuinely clears.

P1  An exit-impaired symbol can never receive another buy. Because impairment
    requires a live executor position, the fast entry loop's existing
    ``symbol in positions`` guard already excludes these symbols from candidate
    assessment; this module adds an explicit, telemetered block at the order
    submission boundary as defence in depth. Exits are never blocked. The
    existing prospective exit-price-limit entry gate (v1.60.13) is left intact.

P2  Executor inventory, healthy reusable routing slots and impaired inventory
    are separated. Quarantine may free a routing slot, but never converts
    trapped inventory into liquid capital: any freed capacity is capped by
    reconciled free quote balance and existing executor risk limits.

P3  ``free_balance_not_executable`` is classified as exit-impaired only when the
    FULL executor position is executable in principle - clearing both the
    canonical amount minimum and, against a fresh executable bid, the canonical
    cost/notional minimum - while the reconciled free base balance is not, and
    no open/submitting order explains the difference. An unavailable fresh bid,
    unresolved minimums or an unprovable notional all fail closed. A genuinely
    non-executable residual is reported as such and never mislabeled as trapped
    healthy inventory.

P4  Compounding integrity is already enforced by v1.60.8, which gates on actual
    Testnet realized PnL net of residual dust cost basis and the modeled
    round-trip cost floor. This module deliberately does not re-decide it -
    duplicating that gate risks wrongly clamping legitimate compounding - and
    instead republishes the verdict under stable telemetry keys and guarantees
    that quarantine never manufactures compoundable profit.
"""

from __future__ import annotations

import copy
from typing import Any

from .testnet_residual_dust_cycle_v1627 import _n
from .testnet_terminal_pending_recovery_v1629 import (
    canonical_executable_minimums,
)

VERSION = "1.60.30"

# Orders in these states make exit state ambiguous. Never quarantine on them.
UNRESOLVED_ORDER_STATES = frozenset({"open", "submitting"})

# A single transient deferral is not an impairment. Require persistence.
MIN_DEFERRED_ATTEMPTS_FOR_IMPAIRMENT = 2

# Impairment classifications surfaced in telemetry.
REASON_PRICE_LIMIT = "price_limit_impaired"
REASON_FREE_BALANCE = "free_balance_impaired"
REASON_DEFERRED_EXIT = "deferred_exit_obstruction"

# Non-impairment classifications, surfaced so they stay distinguishable.
STATE_UNRESOLVED_ORDER = "unresolved_order_blocked"
STATE_TRUE_DUST = "true_dust"
STATE_HEALTHY = "healthy"
STATE_NOT_EXECUTABLE_RESIDUAL = "not_executable_residual"
STATE_EXECUTABILITY_UNPROVABLE = "executability_unprovable"


def _has_unresolved_order(state: dict[str, Any], symbol: str) -> bool:
    normalized = str(symbol or "").upper()

    for row in (state.get("orders") or {}).values():
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol") or "").upper() != normalized:
            continue
        if str(row.get("status") or "").lower() in UNRESOLVED_ORDER_STATES:
            return True

    return False


def _deferred_attempts(lane_state: dict[str, Any], symbol: str) -> int:
    row = (lane_state.get("deferred_exit_recoveries") or {}).get(symbol)

    if not isinstance(row, dict):
        return 0

    for key in ("attempts", "attempt", "deferrals", "count"):
        value = int(_n(row.get(key)))
        if value > 0:
            return value

    # A present row with no counter still evidences one deferral.
    return 1


def _price_limit_obstructed(lane_state: dict[str, Any], symbol: str) -> bool:
    """True when v1.60.15 currently reports a non-executable sell boundary."""

    row = (lane_state.get("v1615_price_limit_watch") or {}).get(symbol)

    if not isinstance(row, dict):
        return False

    return row.get("executable_boundary") is False


def _position_executable_in_principle(
    *,
    exchange: Any,
    symbol: str,
    quantity: float,
    min_amount: float,
    min_cost: float,
    resolved: bool,
    fresh_bid_provider: Any = None,
) -> dict[str, Any]:
    """Prove the full position clears every applicable exchange minimum.

    Both the amount minimum and the cost/notional minimum must be satisfied. A
    fresh executable bid is required whenever a cost minimum applies. Anything
    that cannot be proven returns ``provable=False`` so the caller fails closed.
    No order is submitted here.
    """

    if not resolved or (min_amount <= 0.0 and min_cost <= 0.0):
        return {
            "provable": False,
            "reason": "canonical_minimums_unresolved",
        }

    if min_amount > 0.0 and quantity < min_amount:
        return {
            "provable": True,
            "executable": False,
            "reason": "position_below_minimum_amount",
        }

    if min_cost <= 0.0:
        return {
            "provable": True,
            "executable": True,
            "reason": "amount_minimum_satisfied",
            "fresh_bid": 0.0,
        }

    if not callable(fresh_bid_provider):
        return {
            "provable": False,
            "reason": "fresh_executable_bid_unavailable",
        }

    try:
        fresh_bid = max(0.0, _n(fresh_bid_provider(symbol)))
    except Exception:
        return {
            "provable": False,
            "reason": "fresh_executable_bid_unavailable",
        }

    if fresh_bid <= 0.0:
        return {
            "provable": False,
            "reason": "fresh_executable_bid_unavailable",
        }

    # Safe: only reached once quantity already clears the amount minimum, so
    # precision conversion cannot be handed a sub-precision amount.
    precise = quantity
    if exchange is not None and hasattr(exchange, "amount_to_precision"):
        try:
            precise = max(0.0, _n(exchange.amount_to_precision(symbol, quantity)))
        except Exception:
            return {
                "provable": False,
                "reason": "precision_conversion_unavailable",
            }

    value = precise * fresh_bid

    if value + 1e-12 < min_cost:
        return {
            "provable": True,
            "executable": False,
            "reason": "position_below_minimum_cost",
            "fresh_bid": fresh_bid,
            "estimated_value_usd": value,
        }

    return {
        "provable": True,
        "executable": True,
        "reason": "position_executable_in_principle",
        "fresh_bid": fresh_bid,
        "estimated_value_usd": value,
    }


def classify_symbol_exit_state(
    *,
    engine_state: dict[str, Any],
    lane_state: dict[str, Any],
    symbol: str,
    exchange: Any = None,
    fresh_bid_provider: Any = None,
) -> dict[str, Any]:
    """Classify one symbol's current exit state from authoritative evidence.

    Returns a payload whose ``impaired`` flag is only ever ``True`` for a real
    positive executor position with a live, symbol-scoped exit obstruction.
    """

    normalized = str(symbol or "").upper()

    quantity = max(
        0.0,
        _n((engine_state.get("positions") or {}).get(normalized)),
    )

    base = {
        "symbol": normalized,
        "impaired": False,
        "quantity": quantity,
        "testnet_only": True,
        "live_authority": False,
    }

    # Recorded dust is not impairment; it is handled by v1.60.27/v1.60.29.
    dust = (engine_state.get("non_tradeable_dust") or {}).get(normalized)
    if quantity <= 0.0 and isinstance(dust, dict):
        return {**base, "state": STATE_TRUE_DUST}

    # No live position means nothing to quarantine. Quarantine lifts here.
    if quantity <= 0.0:
        return {**base, "state": STATE_HEALTHY, "reason": "no_open_position"}

    # Ambiguity fails closed: an unresolved order explains the state instead.
    if _has_unresolved_order(engine_state, normalized):
        return {
            **base,
            "state": STATE_UNRESOLVED_ORDER,
            "reason": "symbol_has_unresolved_order",
            "fail_closed": True,
        }

    min_amount, min_cost, resolved = canonical_executable_minimums(
        exchange,
        normalized,
    ) if exchange is not None else (0.0, 0.0, False)

    # Price-limit impairment: the sell boundary is currently not executable.
    if _price_limit_obstructed(lane_state, normalized):
        return {
            **base,
            "impaired": True,
            "state": REASON_PRICE_LIMIT,
            "reason": REASON_PRICE_LIMIT,
            "evidence": "v1615_price_limit_watch.executable_boundary=false",
            "minimum_amount": min_amount,
            "minimum_cost_usd": min_cost,
            "minimums_resolved": resolved,
        }

    # Free-balance impairment (P3): the whole position is executable in
    # principle, but the reconciled free base balance is not.
    base_asset = normalized.split("/", 1)[0]
    free_base = max(
        0.0,
        _n(
            ((engine_state.get("account_balance") or {}).get("free") or {}).get(
                base_asset
            )
        ),
    )

    # Only a free balance that provably cannot be sold is worth investigating.
    # This is a cheap, price-free signal, so healthy symbols never trigger a
    # market-data fetch here.
    if resolved and min_amount > 0.0 and free_base < min_amount:
        # Low free base is only an impairment when the FULL executor position is
        # itself executable in principle. That requires proving both applicable
        # minimums, so a genuinely non-executable residual is never mislabeled
        # as trapped healthy inventory. Anything unprovable fails closed.
        executable = _position_executable_in_principle(
            exchange=exchange,
            symbol=normalized,
            quantity=quantity,
            min_amount=min_amount,
            min_cost=min_cost,
            resolved=resolved,
            fresh_bid_provider=fresh_bid_provider,
        )

        if executable.get("provable") is not True:
            return {
                **base,
                "state": STATE_EXECUTABILITY_UNPROVABLE,
                "reason": executable.get("reason"),
                "fail_closed": True,
                "minimum_amount": min_amount,
                "minimum_cost_usd": min_cost,
                "minimums_resolved": resolved,
            }

        if executable.get("executable") is not True:
            # Not executable at full size: this is a min-cost/min-amount
            # residual, handled by the dust path. Never an impairment.
            return {
                **base,
                "state": STATE_NOT_EXECUTABLE_RESIDUAL,
                "reason": executable.get("reason"),
                "minimum_amount": min_amount,
                "minimum_cost_usd": min_cost,
                "estimated_value_usd": executable.get("estimated_value_usd"),
            }

        if free_base < min_amount or (
            min_cost > 0.0
            and free_base * _n(executable.get("fresh_bid")) < min_cost
        ):
            return {
                **base,
                "impaired": True,
                "state": REASON_FREE_BALANCE,
                "reason": REASON_FREE_BALANCE,
                "evidence": "free_balance_not_executable",
                "free_base_quantity": free_base,
                "minimum_amount": min_amount,
                "minimum_cost_usd": min_cost,
                "minimums_resolved": True,
                "position_executable_in_principle": True,
                "fresh_bid": executable.get("fresh_bid"),
                "position_value_usd": executable.get("estimated_value_usd"),
            }

    # Persistent deferred exit recovery is impairment once it stops being a
    # single transient attempt.
    attempts = _deferred_attempts(lane_state, normalized)
    if attempts >= MIN_DEFERRED_ATTEMPTS_FOR_IMPAIRMENT:
        return {
            **base,
            "impaired": True,
            "state": REASON_DEFERRED_EXIT,
            "reason": REASON_DEFERRED_EXIT,
            "evidence": f"deferred_exit_recoveries.attempts={attempts}",
            "deferred_attempts": attempts,
        }

    return {**base, "state": STATE_HEALTHY, "deferred_attempts": attempts}


def exit_impaired_snapshot(
    *,
    engine_state: dict[str, Any],
    lane_state: dict[str, Any],
    exchange: Any = None,
    fresh_bid_provider: Any = None,
) -> dict[str, Any]:
    """Classify every currently held symbol and summarize impairment."""

    impaired: dict[str, dict[str, Any]] = {}
    other: dict[str, dict[str, Any]] = {}

    symbols = [
        str(symbol).upper()
        for symbol, quantity in (engine_state.get("positions") or {}).items()
        if _n(quantity) > 0.0
    ]

    for symbol in symbols:
        row = classify_symbol_exit_state(
            engine_state=engine_state,
            lane_state=lane_state,
            symbol=symbol,
            exchange=exchange,
            fresh_bid_provider=fresh_bid_provider,
        )
        if row.get("impaired") is True:
            impaired[symbol] = row
        else:
            other[symbol] = row

    impaired_notional = 0.0
    for symbol, row in impaired.items():
        cost = _n((engine_state.get("position_cost_usd") or {}).get(symbol))
        impaired_notional += max(0.0, cost)

    return {
        "version": VERSION,
        "exit_impaired_symbols": sorted(impaired),
        "reasons": {symbol: row.get("reason") for symbol, row in impaired.items()},
        "detail": impaired,
        "non_impaired": other,
        "impaired_inventory_count": len(impaired),
        "impaired_inventory_notional_usd": impaired_notional,
        "executor_inventory_count": len(symbols),
        "healthy_inventory_count": max(0, len(symbols) - len(impaired)),
        "impaired_capital_is_not_liquid": True,
        "testnet_only": True,
        "live_authority": False,
    }


def actual_realized_net_usd(engine_state: dict[str, Any]) -> dict[str, Any]:
    """Actual realized Testnet net eligible for compounding.

    Exchange-realized fills only, less residual dust cost basis. Unrealized
    gains, trapped inventory value and modeled PnL are excluded by construction.
    """

    realized = _n(engine_state.get("realized_pnl_usd"))
    dust_cost = max(0.0, _n(engine_state.get("dust_cost_basis_usd_total")))
    net = realized - dust_cost

    return {
        "exchange_realized_pnl_usd": realized,
        "residual_dust_cost_basis_usd_total": dust_cost,
        "actual_realized_net_usd": net,
        "eligible_for_compounding": bool(net > 0.0),
        "excludes_unrealized_gains": True,
        "excludes_trapped_inventory_value": True,
        "excludes_modeled_pnl": True,
        "live_authority": False,
    }


def install_testnet_exit_impaired_quarantine_v1630() -> None:
    from .fast_collective_hyper import HyperSpeedCollectiveTestnetLane
    from .testnet_execution import BybitTestnetExecutionEngine
    from .velocity_sniper_testnet import VelocitySniperTestnetLane

    if getattr(
        BybitTestnetExecutionEngine,
        "_v1630_exit_impaired_quarantine_installed",
        False,
    ):
        return

    original_capacity = (
        HyperSpeedCollectiveTestnetLane._adaptive_position_capacity
    )
    original_compound = (
        HyperSpeedCollectiveTestnetLane._compound_order_notional
    )
    original_submit_pending = (
        HyperSpeedCollectiveTestnetLane._submit_pending
    )
    original_lane_health = HyperSpeedCollectiveTestnetLane.health
    original_engine_health = BybitTestnetExecutionEngine.health

    def _lane_impairment(self: Any) -> dict[str, Any]:
        # Legacy/lightweight lane instances may not carry an executor at all.
        # They have no inventory, so they have no impairment, and the original
        # behavior must be preserved exactly.
        testnet = getattr(self, "testnet", None)
        engine_state = getattr(testnet, "state", None)
        if not isinstance(engine_state, dict):
            return exit_impaired_snapshot(engine_state={}, lane_state={})

        lock = getattr(testnet, "_io_lock", None)
        exchange = getattr(testnet, "exchange", None)

        if lock is not None:
            with lock:
                engine_copy = {
                    "positions": dict(engine_state.get("positions") or {}),
                    "position_cost_usd": dict(
                        engine_state.get("position_cost_usd") or {}
                    ),
                    "orders": dict(engine_state.get("orders") or {}),
                    "account_balance": copy.deepcopy(
                        engine_state.get("account_balance") or {}
                    ),
                    "non_tradeable_dust": dict(
                        engine_state.get("non_tradeable_dust") or {}
                    ),
                }
        else:  # pragma: no cover - defensive
            engine_copy = dict(engine_state)

        lane_lock = getattr(self, "_lock", None)
        lane_state = getattr(self, "state", None)

        if lane_lock is not None and isinstance(lane_state, dict):
            with lane_lock:
                lane_copy = {
                    "deferred_exit_recoveries": dict(
                        lane_state.get("deferred_exit_recoveries") or {}
                    ),
                    "v1615_price_limit_watch": dict(
                        lane_state.get("v1615_price_limit_watch") or {}
                    ),
                }
        else:  # pragma: no cover - defensive
            lane_copy = {}

        def _fresh_bid_provider(symbol: str) -> float:
            """Lazily fetch a fresh executable bid. Read-only, no order."""

            from .testnet_exit_price_guard_v1611 import _fresh_bid

            bid, _ask = _fresh_bid(testnet, symbol)
            return bid

        return exit_impaired_snapshot(
            engine_state=engine_copy,
            lane_state=lane_copy,
            exchange=exchange,
            fresh_bid_provider=_fresh_bid_provider,
        )

    def exit_impaired_state(self: Any) -> dict[str, Any]:
        """Public quarantine snapshot for telemetry and routing decisions."""

        return _lane_impairment(self)

    def _reconciled_free_quote(self: Any) -> float:
        testnet = getattr(self, "testnet", None)
        engine_state = getattr(testnet, "state", None)
        if not isinstance(engine_state, dict):
            return 0.0

        lock = getattr(testnet, "_io_lock", None)
        quote = str(getattr(testnet, "quote_asset", "USDT") or "USDT").upper()

        if lock is not None:
            with lock:
                free = (
                    (engine_state.get("account_balance") or {}).get("free") or {}
                )
                return max(0.0, _n(free.get(quote)))

        free = (engine_state.get("account_balance") or {}).get("free") or {}
        return max(0.0, _n(free.get(quote)))

    def _submit_pending(
        self: Any,
        pending: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        """P1: an exit-impaired symbol can never receive another buy.

        The fast entry loop already skips symbols holding a live executor
        position, so this is defence in depth at the submission boundary: it
        also holds if a quarantined symbol ever reaches submission by another
        path. Exits are never blocked - a quarantined symbol must keep being
        managed until it can genuinely leave.
        """

        event = (pending or {}).get("event") or {}
        symbol = str(event.get("symbol") or "").upper()
        side = str(event.get("side") or "").lower()
        kind = str((pending or {}).get("kind") or "").lower()

        is_entry = bool(kind == "entry" or side == "buy")

        if symbol and is_entry:
            impairment = _lane_impairment(self)
            if symbol in set(impairment.get("exit_impaired_symbols") or []):
                reason = (impairment.get("reasons") or {}).get(symbol)

                with self._lock:
                    self.state["last_action"] = {
                        "action": "entry_blocked_exit_impaired",
                        "symbol": symbol,
                        "quarantine_reason": reason,
                        "timestamp": now,
                        "order_submitted": False,
                        "live_authority": False,
                    }
                    self._save_locked()

                return self._decision(
                    f"exit_impaired_quarantine:{reason}",
                    details={
                        "kind": kind,
                        "symbol": symbol,
                        "exit_impaired": True,
                        "quarantine_reason": reason,
                        "order_submitted": False,
                        "position_retained": True,
                        "counted_as_dust": False,
                        "live_authority": False,
                    },
                )

        return original_submit_pending(self, pending, now=now)

    def _adaptive_position_capacity(
        self: Any,
        supervisor: dict[str, Any],
        snapshot: dict[str, Any],
        *,
        candidate_count: int,
        entries_today: int,
    ) -> dict[str, Any]:
        """P2: separate impaired inventory from healthy reusable routing slots.

        Trapped inventory may stop occupying a routing slot, but the capacity it
        frees is capped by reconciled free quote balance. Quarantine never
        creates deployable capital.
        """

        result = original_capacity(
            self,
            supervisor,
            snapshot,
            candidate_count=candidate_count,
            entries_today=entries_today,
        )

        if not isinstance(result, dict):  # pragma: no cover - defensive
            return result

        impairment = _lane_impairment(self)
        impaired_count = int(impairment.get("impaired_inventory_count") or 0)

        result = dict(result)
        result["executor_inventory_count"] = int(
            impairment.get("executor_inventory_count") or 0
        )
        result["impaired_inventory_count"] = impaired_count
        result["impaired_inventory_notional_usd"] = _n(
            impairment.get("impaired_inventory_notional_usd")
        )
        result["exit_impaired_symbols"] = list(
            impairment.get("exit_impaired_symbols") or []
        )
        result["impaired_capital_is_not_liquid"] = True

        base_slots = max(0, int(_n(result.get("available_slots"))))
        result["healthy_reusable_routing_slots"] = base_slots

        if impaired_count <= 0:
            result["quarantine_released_slots"] = 0
            result["reconciled_free_quote_usd"] = _reconciled_free_quote(self)
            result["live_authority"] = False
            return result

        # Recompute capacity with routing occupancy set to healthy inventory
        # only, by re-running the ORIGINAL capacity function against a snapshot
        # whose positions exclude impaired symbols. Every pre-existing ceiling
        # (candidate room, capital slots, daily entry room, executor order room,
        # daily submitted-notional room, risk multiplier, maximum adaptive
        # positions) is therefore reapplied by the original code and remains
        # authoritative. The supervisor - and so capital accounting, including
        # impaired notional - is passed through unchanged.
        impaired_symbols = {
            str(symbol).upper()
            for symbol in (impairment.get("exit_impaired_symbols") or [])
        }

        healthy_snapshot = dict(snapshot or {})
        healthy_snapshot["positions"] = {
            symbol: quantity
            for symbol, quantity in (
                (snapshot or {}).get("positions") or {}
            ).items()
            if str(symbol).upper() not in impaired_symbols
        }

        healthy_result = original_capacity(
            self,
            supervisor,
            healthy_snapshot,
            candidate_count=candidate_count,
            entries_today=entries_today,
        )

        healthy_slots = max(
            0,
            int(_n((healthy_result or {}).get("available_slots"))),
        )

        # Quarantine may only ever remove impaired inventory from routing
        # occupancy. It can never grant more than the original constraints
        # already permit for that reduced occupancy.
        release_ceiling = max(0, healthy_slots - base_slots)

        # And it is additionally bounded by real, reconciled free quote money.
        free_quote = _reconciled_free_quote(self)
        minimum_ticket = max(
            0.0,
            min(
                _n(getattr(self, "maximum_order_usd", 0.0)) or float("inf"),
                _n(getattr(self, "order_usd", 0.0)),
            ),
        )

        affordable = 0 if minimum_ticket <= 0.0 else int(free_quote // minimum_ticket)

        released = max(0, min(impaired_count, release_ceiling, affordable))

        result["reconciled_free_quote_usd"] = free_quote
        result["quarantine_release_candidates"] = impaired_count
        result["quarantine_release_ceiling"] = release_ceiling
        result["quarantine_released_slots"] = released
        result["quarantine_release_capped_by_free_quote"] = bool(
            affordable < min(impaired_count, release_ceiling)
        )
        result["quarantine_release_capped_by_risk_envelope"] = bool(
            release_ceiling < impaired_count
        )
        result["healthy_occupancy_available_slots"] = healthy_slots
        result["original_available_slots"] = base_slots
        result["risk_authority_unchanged"] = True

        # By construction this never exceeds what the original constraints allow
        # for healthy occupancy.
        result["available_slots"] = min(base_slots + released, healthy_slots)
        result["healthy_reusable_routing_slots"] = result["available_slots"]
        result["live_authority"] = False
        return result

    def _compound_order_notional(
        self: Any,
        supervisor: dict[str, Any],
        *,
        slots: int | None = None,
        snapshot: dict[str, Any] | None = None,
        entries: int | None = None,
    ) -> dict[str, Any]:
        """P4 telemetry only. The compounding decision belongs to v1.60.8.

        v1.60.8 already enforces the required invariant: compounding is gated on
        actual Testnet realized PnL net of residual dust cost basis and the
        modeled round-trip cost floor
        (``actual_testnet_profit_compounding_eligible``,
        ``actual_testnet_net_after_modeled_cost_usd``). Re-deciding it here would
        duplicate that gate and could wrongly clamp legitimate compounding, so
        this wrapper never mutates ``compounding`` or ``order_notional_usd``. It
        only republishes the decision under stable v1.60.30 telemetry keys.
        """

        sizing = original_compound(
            self,
            supervisor,
            slots=slots,
            snapshot=snapshot,
            entries=entries,
        )

        if not isinstance(sizing, dict):  # pragma: no cover - defensive
            return sizing

        sizing = dict(sizing)

        eligible = sizing.get("actual_testnet_profit_compounding_eligible")

        if eligible is None:
            # No v1.60.8 verdict on this path (e.g. the pre-snapshot fixed
            # fallback). Report, never override.
            sizing["compounding_gate"] = "no_actual_testnet_verdict_available"
        else:
            sizing["compounding_gate"] = (
                "actual_realized_net_positive"
                if eligible
                else "actual_realized_net_not_positive"
            )

        sizing["actual_realized_net_usd"] = _n(
            sizing.get("actual_testnet_net_after_modeled_cost_usd")
        )
        sizing["compounding_decided_by"] = "v1.60.8_actual_testnet_gate"
        sizing["quarantine_never_creates_compoundable_profit"] = True
        sizing["live_authority"] = False
        return sizing

    def lane_health(self: Any) -> dict[str, Any]:
        payload = original_lane_health(self)

        impairment = _lane_impairment(self)
        free_quote = _reconciled_free_quote(self)

        testnet = getattr(self, "testnet", None)
        engine_state = getattr(testnet, "state", None)
        lock = getattr(testnet, "_io_lock", None)
        if isinstance(engine_state, dict) and lock is not None:
            with lock:
                realized = actual_realized_net_usd(engine_state)
        else:  # pragma: no cover - defensive
            realized = actual_realized_net_usd(engine_state or {})

        payload["exit_impaired_quarantine"] = {
            "version": VERSION,
            "enabled": True,
            "exit_impaired_symbols": impairment.get("exit_impaired_symbols"),
            "reason_by_symbol": impairment.get("reasons"),
            "impaired_inventory_count": impairment.get(
                "impaired_inventory_count"
            ),
            "impaired_inventory_notional_usd": impairment.get(
                "impaired_inventory_notional_usd"
            ),
            "executor_inventory_count": impairment.get(
                "executor_inventory_count"
            ),
            "healthy_inventory_count": impairment.get("healthy_inventory_count"),
            "reconciled_free_quote_usd": free_quote,
            "actual_realized_net_usd": realized.get("actual_realized_net_usd"),
            "compounding_active": bool(
                realized.get("eligible_for_compounding")
            ),
            "compounding_reason": (
                "actual_realized_net_positive"
                if realized.get("eligible_for_compounding")
                else "actual_realized_net_not_positive"
            ),
            "impaired_capital_is_not_liquid": True,
            "quarantined_positions_retained": True,
            "quarantined_positions_never_dust": True,
            "quarantine_lifts_on_real_exit": True,
            "testnet_only": True,
            "live_authority": False,
        }
        payload["live_authority"] = False
        return payload

    def engine_health(self: Any) -> dict[str, Any]:
        payload = original_engine_health(self)

        with self._io_lock:
            realized = actual_realized_net_usd(self.state)

        payload["actual_realized_net"] = {
            "version": VERSION,
            **realized,
        }
        payload["live_authority"] = False
        return payload

    HyperSpeedCollectiveTestnetLane.exit_impaired_state = exit_impaired_state
    HyperSpeedCollectiveTestnetLane._submit_pending = _submit_pending
    HyperSpeedCollectiveTestnetLane._adaptive_position_capacity = (
        _adaptive_position_capacity
    )
    HyperSpeedCollectiveTestnetLane._compound_order_notional = (
        _compound_order_notional
    )
    HyperSpeedCollectiveTestnetLane.health = lane_health
    BybitTestnetExecutionEngine.health = engine_health

    HyperSpeedCollectiveTestnetLane.VERSION = VERSION
    VelocitySniperTestnetLane.VERSION = VERSION
    BybitTestnetExecutionEngine._v1630_exit_impaired_quarantine_installed = True
