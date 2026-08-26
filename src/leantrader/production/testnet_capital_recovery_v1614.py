from __future__ import annotations

import copy
import datetime as dt
import time
from typing import Any

from .testnet_exit_price_guard_v1611 import _fresh_bid
from .testnet_exit_recycle import MODELED_ROUND_TRIP_COST_FLOOR_BPS

FAST_QUOTE_RESERVE_USD = 6.0
PROFIT_RECYCLE_MIN_NET_BPS = 5.0
STALE_RECYCLE_AFTER_SECONDS = 900.0
MAX_CONTROLLED_RECYCLE_LOSS_BPS = 75.0
SCAN_COOLDOWN_SECONDS = 15.0
ZERO_FILL_COOLDOWN_SECONDS = 120.0


def _n(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _epoch(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        raw = float(value)
        return raw / 1000.0 if raw > 10_000_000_000 else raw
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        parsed = dt.datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return None


def _supported(testnet: Any) -> bool:
    exchange = getattr(testnet, "exchange", None)
    return bool(
        exchange is not None
        and callable(getattr(exchange, "fetch_ticker", None))
        and callable(getattr(testnet, "prepare_sell", None))
        and callable(getattr(testnet, "mirror_events", None))
        and callable(getattr(testnet, "reconcile_required", None))
    )


def _free_usdt(snapshot: dict[str, Any]) -> float:
    return max(
        0.0,
        _n(
            (
                (
                    snapshot.get("account_balance")
                    or {}
                ).get("free")
                or {}
            ).get("USDT")
        ),
    )


def _reserve(testnet: Any) -> float:
    return max(
        FAST_QUOTE_RESERVE_USD,
        min(
            10.0,
            max(
                0.0,
                _n(
                    getattr(
                        testnet,
                        "max_order_usd",
                        0.0,
                    )
                ),
            ),
        ),
    )


def _age_seconds(
    testnet: Any,
    symbol: str,
    now: float,
) -> float:
    latest = None

    for record in (
        testnet.state.get("orders") or {}
    ).values():
        if not isinstance(record, dict):
            continue

        if (
            str(
                record.get("symbol") or ""
            ).upper()
            != symbol
            or str(
                record.get("side") or ""
            ).lower()
            != "buy"
        ):
            continue

        if _n(record.get("filled")) <= 0.0:
            continue

        observed = _epoch(
            record.get("submitted_at")
            or record.get("decision_at")
            or record.get(
                "paper_event_timestamp"
            )
        )

        if observed is not None:
            latest = (
                observed
                if latest is None
                else max(latest, observed)
            )

    if latest is not None:
        return max(0.0, now - latest)

    with testnet._io_lock:
        seen = testnet.state.setdefault(
            "v1614_position_first_seen",
            {},
        )

        if symbol not in seen:
            seen[symbol] = now
            testnet._save_state()

        return max(
            0.0,
            now - _n(seen.get(symbol), now),
        )


def _assess_position(
    testnet: Any,
    snapshot: dict[str, Any],
    symbol: str,
    quantity: float,
    *,
    now: float,
    quote_starved: bool,
) -> dict[str, Any]:
    symbol = str(symbol).upper()
    quantity = max(0.0, _n(quantity))

    if quantity <= 0.0:
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": "position_absent",
        }

    bid, ask = _fresh_bid(
        testnet,
        symbol,
    )

    if bid <= 0.0:
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": (
                "fresh_bid_unavailable"
            ),
        }

    preparation = testnet.prepare_sell(
        symbol,
        quantity,
        bid,
    )

    status = str(
        preparation.get("status") or ""
    )

    if status == "dust":
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": (
                "position_reclassified_as_dust"
            ),
            "preparation": preparation,
        }

    if status != "executable":
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": str(
                preparation.get("reason")
                or "exit_not_executable"
            ),
            "preparation": preparation,
        }

    executable = max(
        0.0,
        _n(
            preparation.get(
                "executable_quantity"
            )
        ),
    )

    if executable <= 0.0:
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": (
                "zero_executable_quantity"
            ),
        }

    total_cost = max(
        0.0,
        _n(
            (
                snapshot.get(
                    "position_cost_usd"
                )
                or {}
            ).get(symbol)
        ),
    )

    proportional_cost = (
        total_cost
        * executable
        / quantity
        if total_cost > 0.0
        else 0.0
    )

    if proportional_cost <= 0.0:
        return {
            "eligible": False,
            "symbol": symbol,
            "reason": (
                "cost_basis_unavailable"
            ),
            "preparation": preparation,
        }

    exit_value = executable * bid

    gross_bps = (
        (
            exit_value
            / proportional_cost
            - 1.0
        )
        * 10_000.0
    )

    net_bps = (
        gross_bps
        - MODELED_ROUND_TRIP_COST_FLOOR_BPS
    )

    age = _age_seconds(
        testnet,
        symbol,
        now,
    )

    profit_ready = (
        net_bps
        >= PROFIT_RECYCLE_MIN_NET_BPS
    )

    controlled = bool(
        quote_starved
        and age
        >= STALE_RECYCLE_AFTER_SECONDS
        and net_bps
        >= -MAX_CONTROLLED_RECYCLE_LOSS_BPS
    )

    return {
        "eligible": bool(
            profit_ready or controlled
        ),
        "symbol": symbol,
        "reason": (
            "profit_recycle"
            if profit_ready
            else (
                "quote_starved_controlled_recycle"
                if controlled
                else "waiting_for_profitable_recycle"
            )
        ),
        "quantity": executable,
        "fresh_bid": bid,
        "fresh_ask": ask,
        "exit_value_usd": exit_value,
        "cost_basis_usd": proportional_cost,
        "gross_bps": gross_bps,
        "net_bps_after_model": net_bps,
        "age_seconds": age,
        "profit_ready": profit_ready,
        "controlled_recycle": controlled,
        "preparation": preparation,
        "live_authority": False,
    }


def _reactivate_tradeable_dust(
    testnet: Any,
    snapshot: dict[str, Any],
) -> list[str]:
    reactivated = []

    dust_rows = (
        snapshot.get(
            "non_tradeable_dust"
        )
        or {}
    )

    free_assets = (
        (
            snapshot.get(
                "account_balance"
            )
            or {}
        ).get("free")
        or {}
    )

    for symbol, dust in list(
        dust_rows.items()
    ):
        if not isinstance(dust, dict):
            continue

        normalized = str(
            symbol
        ).upper()

        quantity = max(
            0.0,
            _n(dust.get("quantity")),
        )

        base = normalized.split(
            "/",
            1,
        )[0]

        free_quantity = max(
            0.0,
            _n(free_assets.get(base)),
        )

        raw_available = min(
            quantity,
            free_quantity,
        )

        if raw_available <= 0.0:
            continue

        market = testnet.exchange.market(
            normalized
        )

        limits = (
            market.get("limits")
            or {}
        )

        min_amount = max(
            0.0,
            _n(
                (
                    limits.get("amount")
                    or {}
                ).get("min")
            ),
        )

        min_cost = max(
            0.0,
            _n(
                (
                    limits.get("cost")
                    or {}
                ).get("min")
            ),
        )

        if (
            min_amount > 0.0
            and raw_available
            < min_amount
        ):
            continue

        precise = max(
            0.0,
            _n(
                testnet.exchange.amount_to_precision(
                    normalized,
                    raw_available,
                )
            ),
        )

        bid, _ = _fresh_bid(
            testnet,
            normalized,
        )

        if (
            precise <= 0.0
            or bid <= 0.0
            or (
                min_amount > 0.0
                and precise < min_amount
            )
            or (
                min_cost > 0.0
                and precise * bid
                < min_cost
            )
        ):
            continue

        cost_basis = max(
            0.0,
            _n(
                dust.get(
                    "cost_basis_usd"
                )
            ),
        )

        with testnet._io_lock:
            current = (
                testnet.state.get(
                    "non_tradeable_dust"
                )
                or {}
            ).get(normalized)

            if not isinstance(
                current,
                dict,
            ):
                continue

            testnet.state.setdefault(
                "positions",
                {},
            )[normalized] = max(
                quantity,
                _n(
                    (
                        testnet.state.get(
                            "positions"
                        )
                        or {}
                    ).get(normalized)
                ),
            )

            testnet.state.setdefault(
                "position_cost_usd",
                {},
            )[normalized] = max(
                cost_basis,
                _n(
                    (
                        testnet.state.get(
                            "position_cost_usd"
                        )
                        or {}
                    ).get(normalized)
                ),
            )

            testnet.state.setdefault(
                "position_cycle_pnl_usd",
                {},
            ).setdefault(
                normalized,
                0.0,
            )

            testnet.state.get(
                "non_tradeable_dust",
                {},
            ).pop(
                normalized,
                None,
            )

            testnet.state[
                "dust_cost_basis_usd_total"
            ] = max(
                0.0,
                _n(
                    testnet.state.get(
                        "dust_cost_basis_usd_total"
                    )
                )
                - cost_basis,
            )

            testnet.state[
                "v1614_dust_reactivated"
            ] = (
                int(
                    testnet.state.get(
                        "v1614_dust_reactivated"
                    )
                    or 0
                )
                + 1
            )

            testnet.state[
                "v1614_last_dust_reactivation"
            ] = {
                "symbol": normalized,
                "quantity": quantity,
                "fresh_bid": bid,
                "fresh_value_usd": (
                    precise * bid
                ),
                "cost_basis_usd": (
                    cost_basis
                ),
                "reactivated_at": (
                    time.time()
                ),
                "live_authority": False,
            }

            testnet._save_state()

        reactivated.append(
            normalized
        )

    return reactivated


def _record_scan(
    testnet: Any,
    payload: dict[str, Any],
    now: float,
) -> None:
    with testnet._io_lock:
        testnet.state[
            "v1614_last_recovery_scan"
        ] = copy.deepcopy(payload)

        testnet.state[
            "v1614_next_scan_at"
        ] = (
            now
            + SCAN_COOLDOWN_SECONDS
        )

        testnet._save_state()


def _record_result(
    testnet: Any,
    selected: dict[str, Any],
    result: dict[str, Any],
    now: float,
) -> None:
    filled = max(
        0.0,
        _n(result.get("filled")),
    )

    average = max(
        0.0,
        _n(
            result.get("average"),
            _n(
                selected.get(
                    "fresh_bid"
                )
            ),
        ),
    )

    recovered = (
        filled * average
    )

    status = str(
        result.get("status") or ""
    ).lower()

    with testnet._io_lock:
        testnet.state[
            "v1614_recovery_attempts"
        ] = (
            int(
                testnet.state.get(
                    "v1614_recovery_attempts"
                )
                or 0
            )
            + 1
        )

        if filled > 0.0:
            testnet.state[
                "v1614_recovery_fills"
            ] = (
                int(
                    testnet.state.get(
                        "v1614_recovery_fills"
                    )
                    or 0
                )
                + 1
            )

            testnet.state[
                "v1614_recovered_quote_usd"
            ] = (
                _n(
                    testnet.state.get(
                        "v1614_recovered_quote_usd"
                    )
                )
                + recovered
            )

            if (
                selected.get(
                    "controlled_recycle"
                )
                is True
            ):
                testnet.state[
                    "v1614_controlled_loss_recycles"
                ] = (
                    int(
                        testnet.state.get(
                            "v1614_controlled_loss_recycles"
                        )
                        or 0
                    )
                    + 1
                )

        testnet.state[
            "v1614_last_recovery_result"
        ] = {
            "symbol": selected.get(
                "symbol"
            ),
            "status": status,
            "filled": filled,
            "average": (
                average or None
            ),
            "recovered_quote_usd": (
                recovered
            ),
            "assessment": copy.deepcopy(
                selected
            ),
            "observed_at": now,
            "live_authority": False,
        }

        if (
            status
            in {"canceled", "rejected"}
            and filled <= 0.0
        ):
            testnet.state[
                "v1614_next_scan_at"
            ] = (
                now
                + ZERO_FILL_COOLDOWN_SECONDS
            )

        testnet._save_state()


def _capital_recovery_once(
    lane: Any,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    now = (
        time.time()
        if now is None
        else float(now)
    )

    testnet = lane.testnet

    if not _supported(testnet):
        return {
            "supported": False,
            "clear": True,
            "submitted": False,
            "quote_starved": False,
            "live_authority": False,
        }

    try:
        testnet.reconcile_required()
    except Exception as exc:
        return {
            "supported": True,
            "clear": False,
            "submitted": False,
            "quote_starved": True,
            "reason": (
                "reconciliation_blocked"
            ),
            "error": type(exc).__name__,
            "live_authority": False,
        }

    snapshot = (
        testnet.safe_snapshot()
    )

    reactivated_dust = (
        _reactivate_tradeable_dust(
            testnet,
            snapshot,
        )
    )

    if reactivated_dust:
        snapshot = (
            testnet.safe_snapshot()
        )

    free_usdt = _free_usdt(
        snapshot
    )

    reserve = _reserve(
        testnet
    )

    quote_starved = (
        free_usdt + 1e-12
        < reserve
    )

    base = {
        "supported": True,
        "clear": True,
        "submitted": False,
        "quote_starved": quote_starved,
        "free_usdt": free_usdt,
        "fast_quote_reserve_usd": (
            reserve
        ),
        "slower_lane_surplus_usd": max(
            0.0,
            free_usdt - reserve,
        ),
        "reactivated_dust": (
            reactivated_dust
        ),
        "live_authority": False,
    }

    next_scan = max(
        0.0,
        _n(
            testnet.state.get(
                "v1614_next_scan_at"
            )
        ),
    )

    if next_scan > now:
        return {
            **base,
            "reason": (
                "capital_recovery_scan_cooldown"
            ),
            "next_scan_at": next_scan,
        }

    active = {
        str(symbol).upper()
        for symbol in (
            lane._active_snapshot()
            or {}
        )
    }

    assessments = []
    candidates = []

    for symbol, quantity in (
        snapshot.get("positions")
        or {}
    ).items():
        normalized = str(
            symbol
        ).upper()

        if normalized in active:
            continue

        row = _assess_position(
            testnet,
            snapshot,
            normalized,
            _n(quantity),
            now=now,
            quote_starved=quote_starved,
        )

        assessments.append(
            copy.deepcopy(row)
        )

        if (
            row.get("eligible")
            is True
        ):
            candidates.append(row)

    scan = {
        **base,
        "observed_at": now,
        "assessments": (
            assessments[-20:]
        ),
        "candidate_count": (
            len(candidates)
        ),
    }

    _record_scan(
        testnet,
        scan,
        now,
    )

    if not candidates:
        return {
            **scan,
            "reason": (
                "quote_reserve_below_fast_minimum"
                if quote_starved
                else "no_recovery_exit_ready"
            ),
        }

    selected = sorted(
        candidates,
        key=lambda row: (
            bool(
                row.get(
                    "profit_ready"
                )
            ),
            _n(
                row.get(
                    "net_bps_after_model"
                ),
                -1_000_000.0,
            ),
            _n(
                row.get(
                    "exit_value_usd"
                )
            ),
        ),
        reverse=True,
    )[0]

    event = lane._new_event(
        symbol=str(
            selected["symbol"]
        ),
        side="sell",
        quantity=_n(
            selected["quantity"]
        ),
        price=_n(
            selected["fresh_bid"]
        ),
        reason=(
            "capital_recovery_testnet_exit:"
            + str(
                selected["reason"]
            )
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

    _record_result(
        testnet,
        selected,
        result,
        now,
    )

    return {
        **scan,
        "submitted": (
            str(
                result.get(
                    "status"
                )
                or ""
            ).lower()
            not in {"", "skipped"}
        ),
        "reason": (
            "capital_recovery_exit_processed"
        ),
        "selected": selected,
        "result": result,
    }


def install_testnet_capital_recovery_v1614() -> None:
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
        "_v1614_capital_recovery_installed",
        False,
    ):
        return

    original_engine_health = (
        BybitTestnetExecutionEngine.health
    )

    original_step = (
        HyperSpeedCollectiveTestnetLane.step
    )

    original_hyper_health = (
        HyperSpeedCollectiveTestnetLane.health
    )

    def engine_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_engine_health(self)
        )

        free_usdt = _free_usdt(
            payload
        )

        reserve = _reserve(self)

        payload[
            "capital_recovery"
        ] = {
            "version": "1.60.14",
            "enabled": True,
            "free_usdt": free_usdt,
            "fast_quote_reserve_usd": (
                reserve
            ),
            "quote_starved": (
                free_usdt + 1e-12
                < reserve
            ),
            "slower_lane_surplus_usd": max(
                0.0,
                free_usdt - reserve,
            ),
            "profit_recycle_min_net_bps": (
                PROFIT_RECYCLE_MIN_NET_BPS
            ),
            "stale_recycle_after_seconds": (
                STALE_RECYCLE_AFTER_SECONDS
            ),
            "maximum_controlled_recycle_loss_bps": (
                MAX_CONTROLLED_RECYCLE_LOSS_BPS
            ),
            "recovery_attempts": int(
                self.state.get(
                    "v1614_recovery_attempts"
                )
                or 0
            ),
            "recovery_fills": int(
                self.state.get(
                    "v1614_recovery_fills"
                )
                or 0
            ),
            "recovered_quote_usd": _n(
                self.state.get(
                    "v1614_recovered_quote_usd"
                )
            ),
            "controlled_loss_recycles": int(
                self.state.get(
                    "v1614_controlled_loss_recycles"
                )
                or 0
            ),
            "dust_reactivated": int(
                self.state.get(
                    "v1614_dust_reactivated"
                )
                or 0
            ),
            "last_scan": copy.deepcopy(
                self.state.get(
                    "v1614_last_recovery_scan"
                )
                or {}
            ),
            "last_result": copy.deepcopy(
                self.state.get(
                    "v1614_last_recovery_result"
                )
                or {}
            ),
            "micro_lane_capital_priority": True,
            "slower_lanes_use_surplus_only": True,
            "ambiguous_resubmission_allowed": False,
            "live_authority": False,
        }

        payload["live_authority"] = False
        return payload

    def step(
        self: Any,
        now: float | None = None,
    ) -> dict[str, Any]:
        current = (
            time.time()
            if now is None
            else float(now)
        )

        if (
            not _supported(
                self.testnet
            )
            or self._pending()
            is not None
            or bool(
                self._active_snapshot()
            )
        ):
            return original_step(
                self,
                now=current,
            )

        recovery = (
            _capital_recovery_once(
                self,
                now=current,
            )
        )

        if (
            recovery.get("clear")
            is False
        ):
            return self._decision(
                "capital_recovery_reconciliation_blocked",
                details=recovery,
            )

        if (
            recovery.get("submitted")
            is True
        ):
            return self._decision(
                "capital_recovery_exit_processed",
                details=recovery,
            )

        if (
            recovery.get(
                "quote_starved"
            )
            is True
        ):
            return self._decision(
                "capital_recovery_quote_reserve",
                details=recovery,
            )

        return original_step(
            self,
            now=current,
        )

    def hyper_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_hyper_health(self)
        )

        try:
            snapshot = (
                self.testnet.safe_snapshot()
            )
        except Exception:
            snapshot = {}

        payload["version"] = "1.60.14"

        payload[
            "capital_recovery"
        ] = copy.deepcopy(
            snapshot.get(
                "capital_recovery"
            )
            or {}
        )

        payload[
            "live_authority"
        ] = False

        return payload

    BybitTestnetExecutionEngine.health = (
        engine_health
    )

    BybitTestnetExecutionEngine.VERSION = (
        "3.0"
    )

    BybitTestnetExecutionEngine._v1614_capital_recovery_installed = (
        True
    )

    HyperSpeedCollectiveTestnetLane.step = (
        step
    )

    HyperSpeedCollectiveTestnetLane.health = (
        hyper_health
    )

    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.14"
    )

    VelocitySniperTestnetLane.VERSION = (
        "1.60.14"
    )
