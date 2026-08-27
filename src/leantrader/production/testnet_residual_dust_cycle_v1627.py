from __future__ import annotations

import copy
import datetime as dt
import hashlib
from typing import Any

from .testnet_exit_price_guard_v1611 import (
    _fresh_bid,
)


def _n(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _timestamp(
    row: dict[str, Any],
) -> float:
    raw = (
        row.get("submitted_at")
        or row.get("decision_at")
        or row.get("paper_event_timestamp")
    )

    if raw in {None, ""}:
        return 0.0

    try:
        value = float(raw)

        if value > 10_000_000_000:
            value /= 1000.0

        return value

    except (TypeError, ValueError):
        pass

    try:
        parsed = dt.datetime.fromisoformat(
            str(raw).replace(
                "Z",
                "+00:00",
            )
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=dt.UTC
            )

        return parsed.timestamp()

    except (TypeError, ValueError):
        return 0.0


def _current_cycle_evidence(
    state: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    """
    Reconstruct the currently completing fast-lane cycle.

    The supported fast lane does not re-enter a symbol while that
    symbol remains in executor positions. Consecutive buys before
    the first sell are nevertheless aggregated defensively so a
    future bounded scale-in cannot silently under-count cost basis.
    """

    normalized = str(
        symbol
        or ""
    ).upper()

    rows: list[
        tuple[
            float,
            str,
            dict[str, Any],
        ]
    ] = []

    for client_id, row in (
        state.get("orders")
        or {}
    ).items():
        if not isinstance(
            row,
            dict,
        ):
            continue

        if (
            str(
                row.get("symbol")
                or ""
            ).upper()
            != normalized
        ):
            continue

        if (
            str(
                row.get("status")
                or ""
            ).lower()
            != "closed"
        ):
            continue

        if _n(
            row.get("filled")
        ) <= 0.0:
            continue

        side = str(
            row.get("side")
            or ""
        ).lower()

        if side not in {
            "buy",
            "sell",
        }:
            continue

        observed = _timestamp(
            row
        )

        if observed <= 0.0:
            continue

        rows.append(
            (
                observed,
                str(client_id),
                row,
            )
        )

    rows.sort(
        key=lambda item: item[0]
    )

    sell_indexes = [
        index
        for index, item in enumerate(
            rows
        )
        if str(
            item[2].get("side")
            or ""
        ).lower()
        == "sell"
    ]

    if not sell_indexes:
        return {}

    last_sell_index = (
        sell_indexes[-1]
    )

    buy_indexes = [
        index
        for index in range(
            0,
            last_sell_index + 1,
        )
        if str(
            rows[index][2].get(
                "side"
            )
            or ""
        ).lower()
        == "buy"
    ]

    if not buy_indexes:
        return {}

    last_buy_index = (
        buy_indexes[-1]
    )

    previous_sell_indexes = [
        index
        for index in sell_indexes
        if index < last_buy_index
    ]

    previous_sell_index = (
        previous_sell_indexes[-1]
        if previous_sell_indexes
        else -1
    )

    cycle_rows = rows[
        previous_sell_index + 1:
        last_sell_index + 1
    ]

    buys = [
        item
        for item in cycle_rows
        if str(
            item[2].get("side")
            or ""
        ).lower()
        == "buy"
    ]

    sells = [
        item
        for item in cycle_rows
        if str(
            item[2].get("side")
            or ""
        ).lower()
        == "sell"
    ]

    if (
        not buys
        or not sells
    ):
        return {}

    base_asset, quote_asset = (
        normalized.split(
            "/",
            1,
        )
    )

    effective_buy_quantity = 0.0
    total_buy_cost = 0.0

    buy_ids: list[str] = []

    for (
        _observed,
        client_id,
        row,
    ) in buys:
        filled = max(
            0.0,
            _n(
                row.get("filled")
            ),
        )

        average = max(
            0.0,
            _n(
                row.get("average"),
                _n(
                    row.get(
                        "reference_price"
                    )
                ),
            ),
        )

        cost = max(
            0.0,
            _n(
                row.get(
                    "filled_cost"
                ),
                filled * average,
            ),
        )

        fee = max(
            0.0,
            _n(
                row.get("fee")
            ),
        )

        fee_currency = str(
            row.get(
                "fee_currency"
            )
            or quote_asset
        ).upper()

        effective = filled

        if (
            fee_currency
            == base_asset
        ):
            effective = max(
                0.0,
                filled - fee,
            )

        if (
            fee_currency
            == quote_asset
        ):
            cost += fee

        effective_buy_quantity += (
            effective
        )

        total_buy_cost += cost

        buy_ids.append(
            client_id
        )

    total_sell_quantity = 0.0
    gross_sell_value = 0.0
    net_sell_proceeds = 0.0

    sell_ids: list[str] = []

    for (
        _observed,
        client_id,
        row,
    ) in sells:
        filled = max(
            0.0,
            _n(
                row.get("filled")
            ),
        )

        average = max(
            0.0,
            _n(
                row.get("average"),
                _n(
                    row.get(
                        "reference_price"
                    )
                ),
            ),
        )

        value = max(
            0.0,
            _n(
                row.get(
                    "filled_cost"
                ),
                filled * average,
            ),
        )

        fee = max(
            0.0,
            _n(
                row.get("fee")
            ),
        )

        fee_currency = str(
            row.get(
                "fee_currency"
            )
            or quote_asset
        ).upper()

        proceeds = value

        if (
            fee_currency
            == quote_asset
        ):
            proceeds -= fee

        total_sell_quantity += (
            filled
        )

        gross_sell_value += value

        net_sell_proceeds += (
            proceeds
        )

        sell_ids.append(
            client_id
        )

    if (
        effective_buy_quantity
        <= 0.0
        or total_sell_quantity
        <= 0.0
    ):
        return {}

    executed_quantity = min(
        effective_buy_quantity,
        total_sell_quantity,
    )

    entry_price = (
        total_buy_cost
        / effective_buy_quantity
    )

    exit_price = (
        gross_sell_value
        / total_sell_quantity
    )

    allocated_buy_cost = (
        total_buy_cost
        * (
            executed_quantity
            / effective_buy_quantity
        )
    )

    allocated_sell_proceeds = (
        net_sell_proceeds
        * (
            executed_quantity
            / total_sell_quantity
        )
    )

    reconstructed_realized = (
        allocated_sell_proceeds
        - allocated_buy_cost
    )

    residual_estimate = max(
        0.0,
        effective_buy_quantity
        - executed_quantity,
    )

    identifiers = (
        buy_ids
        + sell_ids
    )

    cycle_key = hashlib.sha256(
        "|".join(
            identifiers
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:24]

    return {
        "cycle_key": cycle_key,
        "symbol": normalized,
        "buy_client_order_ids": (
            buy_ids
        ),
        "sell_client_order_ids": (
            sell_ids
        ),
        "buy_order_count": len(
            buys
        ),
        "sell_order_count": len(
            sells
        ),
        "scale_in_buys_aggregated": (
            len(buys) > 1
        ),
        "effective_buy_quantity": (
            effective_buy_quantity
        ),
        "executed_sell_quantity": (
            executed_quantity
        ),
        "residual_base_estimate": (
            residual_estimate
        ),
        "entry_cost_usd": (
            total_buy_cost
        ),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "reconstructed_realized_sell_pnl_usd": (
            reconstructed_realized
        ),
        "buy_submitted_at": (
            buys[0][0]
        ),
        "sell_submitted_at": (
            sells[-1][0]
        ),
        "testnet_only": True,
        "live_authority": False,
    }


def install_testnet_residual_dust_cycle_v1627() -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )

    from .testnet_execution import (
        BybitTestnetExecutionEngine,
    )

    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        BybitTestnetExecutionEngine,
        "_v1627_residual_dust_cycle_installed",
        False,
    ):
        return

    original_prepare_sell = (
        BybitTestnetExecutionEngine.prepare_sell
    )

    original_engine_health = (
        BybitTestnetExecutionEngine.health
    )

    original_manage = (
        HyperSpeedCollectiveTestnetLane._manage_active
    )

    original_lane_health = (
        HyperSpeedCollectiveTestnetLane.health
    )

    def prepare_sell(
        self: Any,
        symbol: str,
        requested_quantity: float,
        reference_price: float,
    ) -> dict[str, Any]:
        normalized = str(
            symbol
            or ""
        ).upper()

        with self._io_lock:
            cycle_map = (
                self.state.get(
                    "position_cycle_pnl_usd"
                )
                or {}
            )

            had_realized_sell = (
                normalized
                in cycle_map
            )

            realized_sell_pnl = (
                _n(
                    cycle_map.get(
                        normalized
                    )
                )
            )

            evidence = (
                _current_cycle_evidence(
                    self.state,
                    normalized,
                )
            )

        result = original_prepare_sell(
            self,
            normalized,
            requested_quantity,
            reference_price,
        )

        if (
            str(
                result.get("status")
                or ""
            )
            != "dust"
        ):
            return result

        dust_cost = max(
            0.0,
            _n(
                result.get(
                    "cost_basis_usd"
                )
            ),
        )

        cycle_net_after_dust = (
            realized_sell_pnl
            - dust_cost
        )

        cycle_key = str(
            evidence.get(
                "cycle_key"
            )
            or ""
        )

        completed = bool(
            had_realized_sell
            and cycle_key
            and _n(
                evidence.get(
                    "effective_buy_quantity"
                )
            )
            > 0.0
            and _n(
                evidence.get(
                    "executed_sell_quantity"
                )
            )
            > 0.0
        )

        winning_after_dust = bool(
            completed
            and cycle_net_after_dust
            > 1e-12
        )

        with self._io_lock:
            dust = (
                self.state.get(
                    "non_tradeable_dust"
                )
                or {}
            ).get(
                normalized
            )

            if isinstance(
                dust,
                dict,
            ):
                dust[
                    "completed_executable_cycle"
                ] = completed

                dust[
                    "counted_as_executed_close"
                ] = False

                dust[
                    "residual_dust_counted_as_sale"
                ] = False

                dust[
                    "actual_realized_sell_pnl_usd"
                ] = realized_sell_pnl

                dust[
                    "residual_dust_cost_basis_usd"
                ] = dust_cost

                dust[
                    "actual_cycle_net_after_dust_usd"
                ] = cycle_net_after_dust

                dust[
                    "winning_after_dust"
                ] = (
                    winning_after_dust
                )

                if evidence:
                    dust[
                        "executed_cycle_evidence"
                    ] = copy.deepcopy(
                        evidence
                    )

            keys = list(
                self.state.get(
                    "v1627_completed_cycle_keys"
                )
                or []
            )

            if (
                completed
                and cycle_key
                not in keys
            ):
                rows = (
                    self.state.setdefault(
                        "v1627_completed_executable_cycles",
                        [],
                    )
                )

                cycle = {
                    **copy.deepcopy(
                        evidence
                    ),
                    "actual_realized_sell_pnl_usd": (
                        realized_sell_pnl
                    ),
                    "residual_dust_cost_basis_usd": (
                        dust_cost
                    ),
                    "actual_cycle_net_after_dust_usd": (
                        cycle_net_after_dust
                    ),
                    "winning_after_dust": (
                        winning_after_dust
                    ),
                    "completed_executable_cycle": True,
                    "residual_dust_counted_as_sale": False,
                    "counted_as_executed_close": True,
                    "realized_pnl_global_mutated_for_dust": False,
                    "recorded_at": (
                        dt.datetime.now(
                            dt.UTC
                        ).isoformat()
                    ),
                    "testnet_only": True,
                    "live_authority": False,
                }

                rows.append(
                    cycle
                )

                self.state[
                    "v1627_completed_executable_cycles"
                ] = rows[-250:]

                keys.append(
                    cycle_key
                )

                self.state[
                    "v1627_completed_cycle_keys"
                ] = keys[-500:]

                self.state[
                    "closed_positions"
                ] = (
                    int(
                        self.state.get(
                            "closed_positions"
                        )
                        or 0
                    )
                    + 1
                )

                if winning_after_dust:
                    self.state[
                        "winning_positions"
                    ] = (
                        int(
                            self.state.get(
                                "winning_positions"
                            )
                            or 0
                        )
                        + 1
                    )

                self.state[
                    "v1627_closed_with_residual_dust"
                ] = (
                    int(
                        self.state.get(
                            "v1627_closed_with_residual_dust"
                        )
                        or 0
                    )
                    + 1
                )

            self._save_state()

        result = dict(
            result
        )

        result[
            "completed_executable_cycle"
        ] = completed

        result[
            "actual_realized_sell_pnl_usd"
        ] = realized_sell_pnl

        result[
            "residual_dust_cost_basis_usd"
        ] = dust_cost

        result[
            "actual_cycle_net_after_dust_usd"
        ] = cycle_net_after_dust

        result[
            "winning_after_dust"
        ] = winning_after_dust

        result[
            "residual_dust_counted_as_sale"
        ] = False

        result[
            "live_authority"
        ] = False

        return result

    def engine_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_engine_health(
                self
            )
        )

        rows = copy.deepcopy(
            self.state.get(
                "v1627_completed_executable_cycles"
            )
            or []
        )

        performance = dict(
            payload.get(
                "performance"
            )
            or {}
        )

        performance[
            "completed_executable_cycles_with_residual_dust"
        ] = len(
            rows
        )

        performance[
            "completed_cycle_net_after_dust_usd"
        ] = sum(
            _n(
                row.get(
                    "actual_cycle_net_after_dust_usd"
                )
            )
            for row in rows
            if isinstance(
                row,
                dict,
            )
        )

        performance[
            "residual_dust_is_not_counted_as_sale"
        ] = True

        performance[
            "realized_pnl_remains_exchange_fill_only"
        ] = True

        payload[
            "performance"
        ] = performance

        payload[
            "residual_dust_cycle_finalization"
        ] = {
            "version": "1.60.27",
            "enabled": True,
            "completed_cycles": len(
                rows
            ),
            "cycles": rows[-20:],
            "winning_classification_basis": (
                "actual_realized_sell_pnl_minus_"
                "residual_dust_cost_basis"
            ),
            "realized_pnl_global_mutated_for_dust": False,
            "requires_real_filled_buy": True,
            "requires_real_filled_sell": True,
            "residual_dust_counted_as_sale": False,
            "idempotent_cycle_keys": True,
            "fake_close_allowed": False,
            "testnet_only": True,
            "live_authority": False,
        }

        payload[
            "live_authority"
        ] = False

        return payload

    def _latest_completed_cycle(
        self: Any,
        symbol: str,
    ) -> dict[str, Any]:
        normalized = str(
            symbol
        ).upper()

        rows = (
            self.testnet.state.get(
                "v1627_completed_executable_cycles"
            )
            or []
        )

        matches = [
            row
            for row in rows
            if (
                isinstance(
                    row,
                    dict,
                )
                and str(
                    row.get(
                        "symbol"
                    )
                    or ""
                ).upper()
                == normalized
            )
        ]

        return (
            copy.deepcopy(
                matches[-1]
            )
            if matches
            else {}
        )

    def retire_fast_state(
        self: Any,
        *,
        symbol: str,
        now: float,
        reason: str,
        preparation: (
            dict[str, Any]
            | None
        ) = None,
    ) -> dict[str, Any]:
        normalized = str(
            symbol
        ).upper()

        preparation = dict(
            preparation
            or {}
        )

        completed = bool(
            preparation.get(
                "completed_executable_cycle"
            )
        )

        cycle = (
            _latest_completed_cycle(
                self,
                normalized,
            )
            if completed
            else {}
        )

        dust = copy.deepcopy(
            (
                self.testnet.state.get(
                    "non_tradeable_dust"
                )
                or {}
            ).get(
                normalized
            )
            or {}
        )

        if dust and cycle:
            final_reason = (
                "residual_dust_cycle_finalized"
            )
        elif (
            dust
            and reason
            == "active_exit_reclassified_dust_preboundary"
        ):
            final_reason = (
                "active_exit_reclassified_"
                "dust_preboundary"
            )
        elif dust:
            final_reason = (
                "non_tradeable_dust_state_retired"
            )
        else:
            final_reason = (
                "authoritative_executor_"
                "position_absent_retired"
            )

        with self._lock:
            (
                self.state.get(
                    "active"
                )
                or {}
            ).pop(
                normalized,
                None,
            )

            (
                self.state.get(
                    "deferred_exit_recoveries"
                )
                or {}
            ).pop(
                normalized,
                None,
            )

            (
                self.state.get(
                    "v1615_price_limit_watch"
                )
                or {}
            ).pop(
                normalized,
                None,
            )

            pending = (
                self.state.get(
                    "pending_event"
                )
            )

            if isinstance(
                pending,
                dict,
            ):
                pending_event = (
                    pending.get(
                        "event"
                    )
                    or pending.get(
                        "source_event"
                    )
                    or {}
                )

                if (
                    str(
                        pending_event.get(
                            "symbol"
                        )
                        or ""
                    ).upper()
                    == normalized
                ):
                    self.state[
                        "pending_event"
                    ] = None

            self.state.setdefault(
                "last_exit_by_symbol",
                {},
            )[normalized] = now

            closed_added = False

            if cycle:
                cycle_key = str(
                    cycle.get(
                        "cycle_key"
                    )
                    or ""
                )

                history = list(
                    self.state.get(
                        "closed"
                    )
                    or []
                )

                existing_keys = {
                    str(
                        row.get(
                            "executor_cycle_key"
                        )
                        or ""
                    )
                    for row in history
                    if isinstance(
                        row,
                        dict,
                    )
                }

                if (
                    cycle_key
                    and cycle_key
                    not in existing_keys
                ):
                    entry_price = max(
                        0.0,
                        _n(
                            cycle.get(
                                "entry_price"
                            )
                        ),
                    )

                    exit_price = max(
                        0.0,
                        _n(
                            cycle.get(
                                "exit_price"
                            )
                        ),
                    )

                    entry_cost = max(
                        0.0,
                        _n(
                            cycle.get(
                                "entry_cost_usd"
                            )
                        ),
                    )

                    actual_net = _n(
                        cycle.get(
                            "actual_cycle_net_after_dust_usd"
                        )
                    )

                    actual_return_bps = (
                        (
                            actual_net
                            / entry_cost
                            * 10_000.0
                        )
                        if entry_cost > 0.0
                        else 0.0
                    )

                    gross_bps = (
                        (
                            exit_price
                            / entry_price
                            - 1.0
                        )
                        * 10_000.0
                        if (
                            entry_price
                            > 0.0
                            and exit_price
                            > 0.0
                        )
                        else 0.0
                    )

                    modeled_floor = max(
                        30.0,
                        _n(
                            getattr(
                                self,
                                "round_trip_cost_bps",
                                30.0,
                            ),
                            30.0,
                        ),
                    )

                    modeled_net_bps = (
                        gross_bps
                        - modeled_floor
                    )

                    # Never let the evidence stream look better
                    # than either the actual after-dust outcome
                    # or the >=30 bps modeled-cost outcome.
                    conservative_net_bps = min(
                        modeled_net_bps,
                        actual_return_bps,
                    )

                    close_row = {
                        "symbol": normalized,
                        "quantity": (
                            _n(
                                cycle.get(
                                    "effective_buy_quantity"
                                )
                            )
                        ),
                        "executed_quantity": (
                            _n(
                                cycle.get(
                                    "executed_sell_quantity"
                                )
                            )
                        ),
                        "residual_dust_quantity": (
                            _n(
                                dust.get(
                                    "quantity"
                                ),
                                _n(
                                    cycle.get(
                                        "residual_base_estimate"
                                    )
                                ),
                            )
                        ),
                        "entry_price": (
                            entry_price
                        ),
                        "exit_price": (
                            exit_price
                        ),
                        "gross_bps": (
                            gross_bps
                        ),
                        "modeled_net_bps_before_dust": (
                            modeled_net_bps
                        ),
                        "actual_return_bps_after_dust": (
                            actual_return_bps
                        ),
                        "net_bps_after_model": (
                            conservative_net_bps
                        ),
                        "actual_realized_sell_pnl_usd": (
                            _n(
                                cycle.get(
                                    "actual_realized_sell_pnl_usd"
                                )
                            )
                        ),
                        "residual_dust_cost_basis_usd": (
                            _n(
                                cycle.get(
                                    "residual_dust_cost_basis_usd"
                                )
                            )
                        ),
                        "actual_cycle_net_after_dust_usd": (
                            actual_net
                        ),
                        "winning_after_dust": bool(
                            cycle.get(
                                "winning_after_dust"
                            )
                        ),
                        "entry_notional_usd": (
                            entry_cost
                        ),
                        "entered_at": (
                            cycle.get(
                                "buy_submitted_at"
                            )
                        ),
                        "exited_at": (
                            cycle.get(
                                "sell_submitted_at"
                            )
                        ),
                        "hold_seconds": max(
                            0.0,
                            _n(
                                cycle.get(
                                    "sell_submitted_at"
                                )
                            )
                            - _n(
                                cycle.get(
                                    "buy_submitted_at"
                                )
                            ),
                        ),
                        "exit_reason": (
                            reason
                        ),
                        "executor_cycle_key": (
                            cycle_key
                        ),
                        "completed_executable_cycle": True,
                        "residual_dust_counted_as_sale": False,
                        "counted_as_executed_close": True,
                        "modeled_round_trip_cost_bps": (
                            modeled_floor
                        ),
                        "outcome_basis": (
                            "conservative_min_of_"
                            "modeled_cost_and_actual_"
                            "net_after_dust"
                        ),
                        "testnet_only": True,
                        "live_authority": False,
                    }

                    history.append(
                        close_row
                    )

                    self.state[
                        "closed"
                    ] = history[-250:]

                    self.state[
                        "exits_today"
                    ] = (
                        int(
                            self.state.get(
                                "exits_today"
                            )
                            or 0
                        )
                        + 1
                    )

                    closed_added = True

            retirements = (
                self.state.setdefault(
                    "v1627_authoritative_cycle_retirements",
                    [],
                )
            )

            retirements.append(
                {
                    "symbol": normalized,
                    "reason": reason,
                    "closed_record_added": (
                        closed_added
                    ),
                    "cycle": cycle,
                    "dust": dust,
                    "preparation": (
                        copy.deepcopy(
                            preparation
                        )
                    ),
                    "order_submitted": False,
                    "position_remains_active": False,
                    "residual_dust_counted_as_sale": False,
                    "live_authority": False,
                    "observed_at": now,
                }
            )

            self.state[
                "v1627_authoritative_cycle_retirements"
            ] = retirements[-100:]

            self.state[
                "last_action"
            ] = {
                "action": final_reason,
                "symbol": normalized,
                "timestamp": now,
                "order_submitted": False,
                "position_remains_active": False,
                "live_authority": False,
            }

            self._save_locked()

        return self._decision(
            final_reason,
            details={
                "kind": "exit",
                "symbol": normalized,
                "cycle": cycle,
                "dust": dust,
                "preparation": preparation,
                "position_remains_active": False,
                "order_submitted": False,
                "residual_dust_counted_as_sale": False,
                "reentry_cooldown_restored": True,
                "live_authority": False,
            },
        )

    def manage_active(
        self: Any,
        service: Any,
        snapshot: dict[str, Any],
        symbol: str,
        record: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        normalized = str(
            symbol
        ).upper()

        current = max(
            0.0,
            _n(
                (
                    snapshot.get(
                        "positions"
                    )
                    or {}
                ).get(
                    normalized
                )
            ),
        )

        # Run this reconciliation/cleanup before the old
        # deferred price-limit watcher can intercept the symbol.
        if current <= 0.0:
            try:
                self.testnet.reconcile_required()

                refreshed = (
                    self.testnet.safe_snapshot()
                )

                current = max(
                    0.0,
                    _n(
                        (
                            refreshed.get(
                                "positions"
                            )
                            or {}
                        ).get(
                            normalized
                        )
                    ),
                )

                snapshot = refreshed

            except Exception:
                # Reconciliation ambiguity remains fail closed.
                return original_manage(
                    self,
                    service,
                    snapshot,
                    symbol,
                    record,
                    now=now,
                )

            if current <= 0.0:
                return retire_fast_state(
                    self,
                    symbol=normalized,
                    now=now,
                    reason=(
                        "authoritative_executor_position_absent"
                    ),
                )

        # v1.60.27 must not intercept ordinary pre-entry/pre-exit
        # dust. Only the real Bybit Testnet executor exposes the
        # persistent cycle-PnL state used as authoritative proof that
        # an actual sell fill has already been applied to this open
        # cycle. Legacy/test adapters keep their original behavior.
        testnet_state = getattr(
            self.testnet,
            "state",
            None,
        )

        testnet_lock = getattr(
            self.testnet,
            "_io_lock",
            None,
        )

        if (
            not isinstance(
                testnet_state,
                dict,
            )
            or testnet_lock is None
        ):
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        # v1.60.28: authoritative non-tradeable dust must be
        # released before the old deferred/price-limit watcher can
        # occupy an execution slot indefinitely. This does not count
        # as a completed trade unless v1.60.27 can prove a real filled
        # buy and real filled sell for the current cycle.
        exchange = getattr(
            self.testnet,
            "exchange",
            None,
        )

        if exchange is not None:
            try:
                market = exchange.market(
                    normalized
                )

                limits = (
                    market.get("limits")
                    or {}
                )

                minimum_amount = max(
                    0.0,
                    _n(
                        (
                            limits.get("amount")
                            or {}
                        ).get("min")
                    ),
                )

                minimum_cost = max(
                    0.0,
                    _n(
                        (
                            limits.get("cost")
                            or {}
                        ).get("min")
                    ),
                )

                precise = max(
                    0.0,
                    _n(
                        exchange.amount_to_precision(
                            normalized,
                            current,
                        )
                    ),
                )

                bid, _ask = _fresh_bid(
                    self.testnet,
                    normalized,
                )

                potential_dust = bool(
                    bid > 0.0
                    and (
                        precise <= 0.0
                        or (
                            minimum_amount > 0.0
                            and precise < minimum_amount
                        )
                        or (
                            minimum_cost > 0.0
                            and precise * bid + 1e-12
                            < minimum_cost
                        )
                    )
                )

            except Exception:
                potential_dust = False
                bid = 0.0

            if potential_dust:
                try:
                    preparation = (
                        self.testnet.prepare_sell(
                            normalized,
                            current,
                            bid,
                        )
                    )
                except Exception:
                    preparation = {}

                if (
                    str(
                        preparation.get("status")
                        or ""
                    )
                    == "dust"
                ):
                    return retire_fast_state(
                        self,
                        symbol=normalized,
                        now=now,
                        reason=(
                            "active_exit_reclassified_"
                            "dust_preboundary"
                        ),
                        preparation=preparation,
                    )

        with testnet_lock:
            executed_sell_in_current_cycle = (
                normalized
                in (
                    testnet_state.get(
                        "position_cycle_pnl_usd"
                    )
                    or {}
                )
            )

        if not executed_sell_in_current_cycle:
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        exchange = getattr(
            self.testnet,
            "exchange",
            None,
        )

        if exchange is not None:
            try:
                market = (
                    exchange.market(
                        normalized
                    )
                )

                limits = (
                    market.get(
                        "limits"
                    )
                    or {}
                )

                minimum_amount = max(
                    0.0,
                    _n(
                        (
                            limits.get(
                                "amount"
                            )
                            or {}
                        ).get(
                            "min"
                        )
                    ),
                )

                minimum_cost = max(
                    0.0,
                    _n(
                        (
                            limits.get(
                                "cost"
                            )
                            or {}
                        ).get(
                            "min"
                        )
                    ),
                )

                precise = max(
                    0.0,
                    _n(
                        exchange.amount_to_precision(
                            normalized,
                            current,
                        )
                    ),
                )

                bid, _ask = (
                    _fresh_bid(
                        self.testnet,
                        normalized,
                    )
                )

                potential_dust = bool(
                    bid > 0.0
                    and (
                        precise <= 0.0
                        or (
                            minimum_amount
                            > 0.0
                            and precise
                            < minimum_amount
                        )
                        or (
                            minimum_cost
                            > 0.0
                            and precise
                            * bid
                            + 1e-12
                            < minimum_cost
                        )
                    )
                )

            except Exception:
                potential_dust = False
                bid = 0.0

            if potential_dust:
                try:
                    preparation = (
                        self.testnet.prepare_sell(
                            normalized,
                            current,
                            bid,
                        )
                    )
                except Exception:
                    preparation = {}

                if (
                    str(
                        preparation.get(
                            "status"
                        )
                        or ""
                    )
                    == "dust"
                ):
                    return retire_fast_state(
                        self,
                        symbol=normalized,
                        now=now,
                        reason=(
                            "executed_exit_residual_dust"
                        ),
                        preparation=preparation,
                    )

        return original_manage(
            self,
            service,
            snapshot,
            symbol,
            record,
            now=now,
        )

    def lane_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_lane_health(
                self
            )
        )

        with self._lock:
            retirements = (
                copy.deepcopy(
                    self.state.get(
                        "v1627_authoritative_cycle_retirements"
                    )
                    or []
                )
            )

        payload[
            "residual_dust_cycle_finalization"
        ] = {
            "version": "1.60.27",
            "enabled": True,
            "cleanup_runs_before_price_limit_watch": True,
            "authoritative_executor_absence_cleanup": True,
            "stale_active_state_can_block_reentry": False,
            "reentry_cooldown_restored": True,
            "requires_real_filled_sell_for_completed_cycle": True,
            "winning_basis": (
                "actual_cycle_net_after_dust"
            ),
            "evidence_basis": (
                "conservative_min_of_modeled_"
                "cost_and_actual_net_after_dust"
            ),
            "residual_dust_counted_as_sale": False,
            "idempotent_cycle_keys": True,
            "retirements": len(
                retirements
            ),
            "recent": (
                retirements[-20:]
            ),
            "fake_close_allowed": False,
            "live_authority": False,
        }

        payload[
            "live_authority"
        ] = False

        return payload

    BybitTestnetExecutionEngine.prepare_sell = (
        prepare_sell
    )

    BybitTestnetExecutionEngine.health = (
        engine_health
    )

    HyperSpeedCollectiveTestnetLane._manage_active = (
        manage_active
    )

    HyperSpeedCollectiveTestnetLane.health = (
        lane_health
    )

    BybitTestnetExecutionEngine.VERSION = (
        "3.4"
    )

    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.27"
    )

    VelocitySniperTestnetLane.VERSION = (
        "1.60.27"
    )

    BybitTestnetExecutionEngine._v1627_residual_dust_cycle_installed = (
        True
    )
