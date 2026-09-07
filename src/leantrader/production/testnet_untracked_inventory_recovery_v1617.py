from __future__ import annotations

import copy
import time
from typing import Any

from .testnet_exit_price_guard_v1611 import (
    _fresh_bid,
    _price_limit,
)

VERSION = "1.61.7"
SCAN_SECONDS = 5.0
MAX_PREFLIGHTS_PER_SCAN = 6


def _n(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _full_balance_maps(
    balance: dict[str, Any],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    totals_raw = (
        balance.get("total")
        if isinstance(balance.get("total"), dict)
        else {}
    )
    free_raw = (
        balance.get("free")
        if isinstance(balance.get("free"), dict)
        else {}
    )
    used_raw = (
        balance.get("used")
        if isinstance(balance.get("used"), dict)
        else {}
    )

    names = set(totals_raw) | set(free_raw) | set(used_raw)

    totals: dict[str, float] = {}
    free: dict[str, float] = {}
    used: dict[str, float] = {}

    for raw_name in names:
        asset = str(raw_name).upper()

        total = _n(totals_raw.get(raw_name))
        free_qty = _n(free_raw.get(raw_name))
        used_qty = _n(used_raw.get(raw_name))

        nested = balance.get(raw_name)
        if isinstance(nested, dict):
            if total == 0.0:
                total = _n(nested.get("total"))
            if free_qty == 0.0:
                free_qty = _n(nested.get("free"))
            if used_qty == 0.0:
                used_qty = _n(nested.get("used"))

        if max(
            abs(total),
            abs(free_qty),
            abs(used_qty),
        ) <= 1e-12:
            continue

        totals[asset] = total
        free[asset] = free_qty
        used[asset] = used_qty

    return totals, free, used


def _tracked_assets(
    state: dict[str, Any],
) -> set[str]:
    output = {"USDT"}

    for key in (
        "positions",
        "non_tradeable_dust",
    ):
        for symbol in (
            state.get(key)
            or {}
        ):
            output.add(
                str(symbol)
                .upper()
                .split("/", 1)[0]
            )

    return output


def _supported(testnet: Any) -> bool:
    return bool(
        getattr(testnet, "exchange", None)
        is not None
        and isinstance(
            getattr(testnet, "state", None),
            dict,
        )
        and getattr(
            testnet,
            "_io_lock",
            None,
        ) is not None
        and callable(
            getattr(
                testnet,
                "mirror_events",
                None,
            )
        )
        and callable(
            getattr(
                testnet,
                "prepare_sell",
                None,
            )
        )
    )


def _augment_balance(
    self: Any,
    balance: dict[str, Any],
) -> None:
    totals, free, used = (
        _full_balance_maps(balance)
    )

    existing = dict(
        self.state.get("account_balance")
        or {}
    )

    existing.update(
        {
            "assets": totals,
            "free": free,
            "used": used,
            "all_nonzero_assets_visible": True,
            "inventory_visibility_version": VERSION,
        }
    )

    self.state["account_balance"] = existing

    tracked = _tracked_assets(
        self.state
    )

    untracked = []

    for asset, quantity in free.items():
        if (
            asset == "USDT"
            or quantity <= 1e-12
            or asset in tracked
        ):
            continue

        symbol = f"{asset}/USDT"

        untracked.append(
            {
                "asset": asset,
                "symbol": symbol,
                "free_quantity": quantity,
                "spot_usdt_market": (
                    symbol
                    in getattr(
                        self,
                        "_eligible_symbols",
                        set(),
                    )
                ),
                "cost_basis_known": False,
                "trading_pnl_eligible": False,
                "principal_recovery_only": True,
                "live_authority": False,
            }
        )

    self.state[
        "v1617_untracked_exchange_inventory"
    ] = untracked


def _candidate_symbols(
    testnet: Any,
) -> list[str]:
    state = testnet.state
    result: list[str] = []

    recovered = (
        state.get(
            "v1617_recovered_inventory"
        )
        or {}
    )

    for symbol, row in recovered.items():
        if not isinstance(row, dict):
            continue

        if (
            _n(
                row.get(
                    "remaining_quantity"
                )
            )
            > 1e-12
            and str(
                row.get("status")
                or ""
            )
            not in {
                "liquidated",
                "dust",
            }
        ):
            result.append(
                str(symbol).upper()
            )

    for row in (
        state.get(
            "v1617_untracked_exchange_inventory"
        )
        or []
    ):
        if not isinstance(row, dict):
            continue

        symbol = str(
            row.get("symbol")
            or ""
        ).upper()

        if (
            symbol
            and row.get(
                "spot_usdt_market"
            )
            is True
            and symbol not in result
        ):
            result.append(symbol)

    return result


def _assess(
    testnet: Any,
    symbol: str,
) -> dict[str, Any]:
    symbol = str(symbol).upper()

    account = (
        testnet.state.get(
            "account_balance"
        )
        or {}
    )

    free = (
        account.get("free")
        or {}
    )

    base = symbol.split("/", 1)[0]

    free_quantity = max(
        0.0,
        _n(free.get(base)),
    )

    if free_quantity <= 0.0:
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": "free_balance_absent",
        }

    if (
        symbol
        not in getattr(
            testnet,
            "_eligible_symbols",
            set(),
        )
    ):
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": "spot_market_unavailable",
        }

    market = testnet.exchange.market(
        symbol
    )

    limits = market.get("limits") or {}

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

    bid, ask = _fresh_bid(
        testnet,
        symbol,
    )

    if bid <= 0.0:
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": "fresh_bid_unavailable",
        }

    quantity = max(
        0.0,
        _n(
            testnet.exchange
            .amount_to_precision(
                symbol,
                free_quantity,
            )
        ),
    )

    value = quantity * bid

    if (
        quantity <= 0.0
        or (
            minimum_amount > 0.0
            and quantity
            < minimum_amount
        )
        or (
            minimum_cost > 0.0
            and value + 1e-12
            < minimum_cost
        )
    ):
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": (
                "untracked_inventory_below_exchange_minimum"
            ),
            "quantity": quantity,
            "fresh_bid": bid,
            "fresh_ask": ask,
            "estimated_value_usd": value,
            "minimum_amount": minimum_amount,
            "minimum_cost_usd": minimum_cost,
        }

    price_limit = _price_limit(
        testnet,
        symbol,
    )

    if (
        price_limit.get("supported")
        is True
    ):
        if (
            price_limit.get("ok")
            is not True
        ):
            return {
                "eligible": False,
                "symbol": symbol,
                "reason": (
                    "bybit_price_limit_unavailable"
                ),
                "quantity": quantity,
                "estimated_value_usd": value,
            }

        sell_limit = max(
            0.0,
            _n(
                price_limit.get(
                    "sell_limit"
                )
            ),
        )

        if (
            sell_limit > 0.0
            and bid + 1e-12
            < sell_limit
        ):
            return {
                "eligible": False,
                "symbol": symbol,
                "reason": (
                    "bybit_sell_boundary_unexecutable"
                ),
                "quantity": quantity,
                "fresh_bid": bid,
                "sell_limit": sell_limit,
                "estimated_value_usd": value,
            }

    return {
        "eligible": True,
        "symbol": symbol,
        "asset": base,
        "quantity": quantity,
        "free_quantity": free_quantity,
        "fresh_bid": bid,
        "fresh_ask": ask,
        "estimated_value_usd": value,
        "minimum_amount": minimum_amount,
        "minimum_cost_usd": minimum_cost,
        "price_limit": price_limit,
        "cost_basis_known": False,
        "trading_pnl_eligible": False,
        "principal_recovery_only": True,
        "live_authority": False,
    }


def _adopt(
    testnet: Any,
    row: dict[str, Any],
    *,
    now: float,
) -> None:
    symbol = str(
        row["symbol"]
    ).upper()

    quantity = max(
        0.0,
        _n(row.get("quantity")),
    )

    reference_value = max(
        0.0,
        _n(
            row.get(
                "estimated_value_usd"
            )
        ),
    )

    with testnet._io_lock:
        recovered = (
            testnet.state.setdefault(
                "v1617_recovered_inventory",
                {},
            )
        )

        existing = recovered.get(
            symbol
        )

        if isinstance(existing, dict):
            existing[
                "remaining_quantity"
            ] = max(
                _n(
                    existing.get(
                        "remaining_quantity"
                    )
                ),
                quantity,
            )
            return

        # This reference basis exists only so the normal sell
        # preparation/risk machinery can operate. v1.61.7
        # explicitly reverses all resulting P&L attribution.
        testnet.state.setdefault(
            "positions",
            {},
        )[symbol] = quantity

        testnet.state.setdefault(
            "position_cost_usd",
            {},
        )[symbol] = reference_value

        testnet.state.setdefault(
            "position_cycle_pnl_usd",
            {},
        )[symbol] = 0.0

        recovered[symbol] = {
            "symbol": symbol,
            "asset": row.get("asset"),
            "quantity_adopted": quantity,
            "remaining_quantity": quantity,
            "reference_bid": row.get(
                "fresh_bid"
            ),
            "reference_value_usd": (
                reference_value
            ),
            "adopted_at": now,
            "status": "adopted_for_principal_recovery",
            "historical_cost_basis_known": False,
            "reference_basis_is_trading_cost_basis": False,
            "trading_pnl_eligible": False,
            "principal_recovery_only": True,
            "live_authority": False,
        }

        testnet.state[
            "v1617_inventory_adoptions"
        ] = (
            int(
                testnet.state.get(
                    "v1617_inventory_adoptions"
                )
                or 0
            )
            + 1
        )

        testnet._save_state()


def _recover_once(
    lane: Any,
    *,
    now: float,
) -> dict[str, Any]:
    testnet = lane.testnet

    if not _supported(testnet):
        return {
            "supported": False,
            "submitted": False,
        }

    next_scan = max(
        0.0,
        _n(
            testnet.state.get(
                "v1617_next_scan_at"
            )
        ),
    )

    if next_scan > now:
        return {
            "supported": True,
            "submitted": False,
            "reason": "scan_cooldown",
        }

    with testnet._io_lock:
        testnet.state[
            "v1617_next_scan_at"
        ] = now + SCAN_SECONDS
        testnet._save_state()

    try:
        testnet.reconcile_required()
    except Exception as exc:
        return {
            "supported": True,
            "submitted": False,
            "reason": "reconciliation_blocked",
            "error": type(exc).__name__,
        }

    candidates = _candidate_symbols(
        testnet
    )

    assessments = []
    selected = None

    for symbol in candidates[
        :MAX_PREFLIGHTS_PER_SCAN
    ]:
        try:
            row = _assess(
                testnet,
                symbol,
            )
        except Exception as exc:
            row = {
                "eligible": False,
                "symbol": symbol,
                "reason": (
                    "inventory_preflight_error"
                ),
                "error": type(exc).__name__,
            }

        assessments.append(
            copy.deepcopy(row)
        )

        if (
            row.get("eligible")
            is True
            and (
                selected is None
                or _n(
                    row.get(
                        "estimated_value_usd"
                    )
                )
                > _n(
                    selected.get(
                        "estimated_value_usd"
                    )
                )
            )
        ):
            selected = row

    with testnet._io_lock:
        testnet.state[
            "v1617_last_inventory_scan"
        ] = {
            "observed_at": now,
            "candidate_count": len(
                candidates
            ),
            "assessments": (
                assessments[-20:]
            ),
            "live_authority": False,
        }
        testnet._save_state()

    if selected is None:
        return {
            "supported": True,
            "submitted": False,
            "reason": (
                "no_executable_untracked_inventory"
            ),
            "assessments": assessments,
        }

    _adopt(
        testnet,
        selected,
        now=now,
    )

    prepared = testnet.prepare_sell(
        selected["symbol"],
        selected["quantity"],
        selected["fresh_bid"],
    )

    if (
        str(
            prepared.get("status")
            or ""
        )
        != "executable"
    ):
        with testnet._io_lock:
            row = (
                testnet.state
                .setdefault(
                    "v1617_recovered_inventory",
                    {},
                )
                .get(
                    selected["symbol"]
                )
            )

            if isinstance(row, dict):
                row["status"] = (
                    "dust"
                    if prepared.get(
                        "status"
                    )
                    == "dust"
                    else "waiting_for_executable_exit"
                )
                row[
                    "last_prepare_result"
                ] = copy.deepcopy(
                    prepared
                )
                testnet._save_state()

        return {
            "supported": True,
            "submitted": False,
            "reason": (
                "recovered_inventory_exit_not_executable"
            ),
            "selected": selected,
            "preparation": prepared,
        }

    event = lane._new_event(
        symbol=selected["symbol"],
        side="sell",
        quantity=_n(
            prepared.get(
                "executable_quantity"
            )
        ),
        price=_n(
            prepared.get(
                "fresh_bid"
            ),
            selected["fresh_bid"],
        ),
        reason=(
            "untracked_testnet_inventory_principal_recovery"
        ),
        now=now,
        remaining_quantity=0.0,
    )

    rows = testnet.mirror_events(
        [event]
    )

    result = (
        dict(rows[0])
        if rows
        else {}
    )

    with testnet._io_lock:
        testnet.state[
            "v1617_recovery_attempts"
        ] = (
            int(
                testnet.state.get(
                    "v1617_recovery_attempts"
                )
                or 0
            )
            + 1
        )

        testnet.state[
            "v1617_last_recovery_result"
        ] = {
            "symbol": selected[
                "symbol"
            ],
            "result": copy.deepcopy(
                result
            ),
            "preparation": copy.deepcopy(
                prepared
            ),
            "observed_at": now,
            "recovered_proceeds_are_principal": True,
            "trading_pnl_eligible": False,
            "live_authority": False,
        }

        testnet._save_state()

    return {
        "supported": True,
        "submitted": str(
            result.get("status")
            or ""
        ).lower()
        not in {
            "",
            "skipped",
            "rejected",
        },
        "reason": (
            "untracked_inventory_recovery_processed"
        ),
        "selected": selected,
        "result": result,
    }


def install_testnet_untracked_inventory_recovery_v1617(
) -> None:
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
        "_v1617_untracked_inventory_recovery_installed",
        False,
    ):
        return

    original_balance = (
        BybitTestnetExecutionEngine
        ._update_balance_snapshot
    )

    original_apply = (
        BybitTestnetExecutionEngine
        ._apply_new_fill
    )

    original_health = (
        BybitTestnetExecutionEngine
        .health
    )

    original_step = (
        HyperSpeedCollectiveTestnetLane
        .step
    )

    original_lane_health = (
        HyperSpeedCollectiveTestnetLane
        .health
    )

    def update_balance_snapshot(
        self: Any,
        balance: dict[str, Any],
    ) -> None:
        original_balance(
            self,
            balance,
        )
        _augment_balance(
            self,
            balance,
        )

    def apply_new_fill(
        self: Any,
        record: dict[str, Any],
    ) -> None:
        symbol = str(
            record.get("symbol")
            or ""
        ).upper()

        side = str(
            record.get("side")
            or ""
        ).lower()

        with self._io_lock:
            recovered = copy.deepcopy(
                (
                    self.state.get(
                        "v1617_recovered_inventory"
                    )
                    or {}
                ).get(symbol)
            )

            before_realized = _n(
                self.state.get(
                    "realized_pnl_usd"
                )
            )

            before_closed = int(
                self.state.get(
                    "closed_positions"
                )
                or 0
            )

            before_wins = int(
                self.state.get(
                    "winning_positions"
                )
                or 0
            )

            before_last_closed = (
                copy.deepcopy(
                    self.state.get(
                        "last_closed_cycle"
                    )
                )
            )

            before_cycle = _n(
                (
                    self.state.get(
                        "position_cycle_pnl_usd"
                    )
                    or {}
                ).get(symbol)
            )

            before_applied = _n(
                record.get(
                    "applied_filled"
                )
            )

            before_fill_cost = _n(
                record.get(
                    "applied_fill_cost"
                )
            )

            before_fee = _n(
                record.get(
                    "applied_fee"
                )
            )

        original_apply(
            self,
            record,
        )

        if (
            side != "sell"
            or not isinstance(
                recovered,
                dict,
            )
        ):
            return

        after_applied = _n(
            record.get(
                "applied_filled"
            )
        )

        delta = max(
            0.0,
            after_applied
            - before_applied,
        )

        remaining_before = max(
            0.0,
            _n(
                recovered.get(
                    "remaining_quantity"
                ),
                _n(
                    recovered.get(
                        "quantity_adopted"
                    )
                ),
            ),
        )

        recovered_sold = min(
            delta,
            remaining_before,
        )

        if recovered_sold <= 0.0:
            return

        fill_cost_delta = max(
            0.0,
            _n(
                record.get(
                    "applied_fill_cost"
                )
            )
            - before_fill_cost,
        )

        fee_delta = max(
            0.0,
            _n(
                record.get(
                    "applied_fee"
                )
            )
            - before_fee,
        )

        proportion = (
            recovered_sold / delta
            if delta > 0.0
            else 0.0
        )

        proceeds = (
            fill_cost_delta
            * proportion
        )

        quote = (
            symbol.split("/", 1)[1]
            if "/" in symbol
            else "USDT"
        )

        if (
            str(
                record.get(
                    "fee_currency"
                )
                or ""
            ).upper()
            == quote
        ):
            proceeds = max(
                0.0,
                proceeds
                - fee_delta
                * proportion,
            )

        with self._io_lock:
            # The historical acquisition basis is unknown.
            # Therefore this liquidation is principal recovery,
            # not a LeanTrader trading win/loss.
            self.state[
                "realized_pnl_usd"
            ] = before_realized

            self.state[
                "closed_positions"
            ] = before_closed

            self.state[
                "winning_positions"
            ] = before_wins

            self.state[
                "last_closed_cycle"
            ] = before_last_closed

            if (
                symbol
                in (
                    self.state.get(
                        "positions"
                    )
                    or {}
                )
            ):
                self.state.setdefault(
                    "position_cycle_pnl_usd",
                    {},
                )[symbol] = before_cycle
            else:
                (
                    self.state.setdefault(
                        "position_cycle_pnl_usd",
                        {},
                    )
                    .pop(
                        symbol,
                        None,
                    )
                )

            live = (
                self.state
                .setdefault(
                    "v1617_recovered_inventory",
                    {},
                )
                .get(symbol)
            )

            if isinstance(live, dict):
                live[
                    "remaining_quantity"
                ] = max(
                    0.0,
                    remaining_before
                    - recovered_sold,
                )

                live[
                    "recovered_quote_usd"
                ] = (
                    _n(
                        live.get(
                            "recovered_quote_usd"
                        )
                    )
                    + proceeds
                )

                live[
                    "last_fill_quantity"
                ] = recovered_sold

                live[
                    "last_recovered_quote_usd"
                ] = proceeds

                live[
                    "last_recovery_at"
                ] = time.time()

                if (
                    live[
                        "remaining_quantity"
                    ]
                    <= 1e-12
                ):
                    live[
                        "status"
                    ] = "liquidated"

            self.state[
                "v1617_recovered_principal_usd"
            ] = (
                _n(
                    self.state.get(
                        "v1617_recovered_principal_usd"
                    )
                )
                + proceeds
            )

            self.state[
                "v1617_recovery_fills"
            ] = (
                int(
                    self.state.get(
                        "v1617_recovery_fills"
                    )
                    or 0
                )
                + 1
            )

            record[
                "v1617_recovered_principal_sale"
            ] = True

            record[
                "v1617_trading_pnl_counted"
            ] = False

            self._save_state()

    def engine_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = original_health(
            self
        )

        with self._io_lock:
            untracked = copy.deepcopy(
                self.state.get(
                    "v1617_untracked_exchange_inventory"
                )
                or []
            )

            recovered = copy.deepcopy(
                self.state.get(
                    "v1617_recovered_inventory"
                )
                or {}
            )

            principal = _n(
                self.state.get(
                    "v1617_recovered_principal_usd"
                )
            )

        payload[
            "untracked_inventory_recovery"
        ] = {
            "version": VERSION,
            "all_nonzero_exchange_assets_visible": True,
            "untracked_assets": untracked,
            "recovered_inventory": recovered,
            "recovered_principal_usd": principal,
            "recovery_attempts": int(
                self.state.get(
                    "v1617_recovery_attempts"
                )
                or 0
            ),
            "recovery_fills": int(
                self.state.get(
                    "v1617_recovery_fills"
                )
                or 0
            ),
            "last_scan": copy.deepcopy(
                self.state.get(
                    "v1617_last_inventory_scan"
                )
                or {}
            ),
            "last_result": copy.deepcopy(
                self.state.get(
                    "v1617_last_recovery_result"
                )
                or {}
            ),
            "recovered_principal_is_realized_trading_profit": False,
            "compounding_profit_fabricated": False,
            "authenticated_testnet_executor_only": True,
            "live_authority": False,
        }

        payload[
            "live_authority"
        ] = False

        return payload

    def step(
        self: Any,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        current = (
            time.time()
            if now is None
            else float(now)
        )

        if (
            _supported(
                self.testnet
            )
            and self._pending()
            is None
        ):
            recovery = _recover_once(
                self,
                now=current,
            )

            if (
                recovery.get(
                    "submitted"
                )
                is True
            ):
                return self._decision(
                    "untracked_testnet_inventory_recovery",
                    details=recovery,
                )

        return original_step(
            self,
            now=current,
        )

    def lane_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_lane_health(
                self
            )
        )

        try:
            snapshot = (
                self.testnet
                .safe_snapshot()
            )
        except Exception:
            snapshot = {}

        payload[
            "untracked_inventory_recovery"
        ] = copy.deepcopy(
            snapshot.get(
                "untracked_inventory_recovery"
            )
            or {}
        )

        payload[
            "live_authority"
        ] = False

        return payload

    BybitTestnetExecutionEngine._update_balance_snapshot = (
        update_balance_snapshot
    )

    BybitTestnetExecutionEngine._apply_new_fill = (
        apply_new_fill
    )

    BybitTestnetExecutionEngine.health = (
        engine_health
    )

    HyperSpeedCollectiveTestnetLane.step = (
        step
    )

    HyperSpeedCollectiveTestnetLane.health = (
        lane_health
    )

    BybitTestnetExecutionEngine._v1617_untracked_inventory_recovery_installed = (
        True
    )

    HyperSpeedCollectiveTestnetLane.VERSION = VERSION
    VelocitySniperTestnetLane.VERSION = VERSION
