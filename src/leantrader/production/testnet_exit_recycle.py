from __future__ import annotations

import copy
import datetime as dt
import time
from typing import Any


MODELED_ROUND_TRIP_COST_FLOOR_BPS = 30.0
EXIT_RECYCLE_MIN_COOLDOWN_SECONDS = 15.0


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _execution_day(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    try:
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        return dt.datetime.fromtimestamp(timestamp, tz=dt.UTC).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.astimezone(dt.UTC).date().isoformat()
    except (TypeError, ValueError):
        return None


def _today_order_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    today = str(state.get("day") or dt.datetime.now(dt.UTC).date().isoformat())
    rows: list[dict[str, Any]] = []
    for record in (state.get("orders") or {}).values():
        if not isinstance(record, dict):
            continue
        observed_at = record.get("submitted_at") or record.get("decision_at")
        if _execution_day(observed_at) == today:
            rows.append(record)
    return rows


def _reconstruct_entry_budget(state: dict[str, Any]) -> tuple[int, float]:
    count = 0
    submitted = 0.0
    for record in _today_order_rows(state):
        if str(record.get("side") or "").lower() != "buy":
            continue
        if record.get("submitted_at") in {None, ""}:
            continue
        notional = max(0.0, _number(record.get("submitted_usd")))
        if notional <= 0.0:
            continue
        count += 1
        submitted += notional
    return count, submitted


def _current_day_execution_quality(state: dict[str, Any]) -> dict[str, Any]:
    rows = _today_order_rows(state)
    submitted = [row for row in rows if row.get("submitted_at") not in {None, ""}]
    status_counts = {
        status: sum(
            1
            for row in rows
            if str(row.get("status") or "").lower() == status
        )
        for status in ("closed", "open", "submitting", "canceled", "rejected", "skipped")
    }
    return {
        "date": str(state.get("day") or dt.datetime.now(dt.UTC).date().isoformat()),
        "decisions": len(rows),
        "submitted_orders": len(submitted),
        "buy_entries_submitted": sum(
            1 for row in submitted if str(row.get("side") or "").lower() == "buy"
        ),
        "protective_exits_submitted": sum(
            1 for row in submitted if str(row.get("side") or "").lower() == "sell"
        ),
        "submitted_notional_usd": sum(
            max(0.0, _number(row.get("submitted_usd"))) for row in submitted
        ),
        "filled_orders": sum(
            1 for row in rows if _number(row.get("filled")) > 0.0
        ),
        "status_counts": status_counts,
        "historical_rows_excluded": max(
            0,
            len(state.get("orders") or {}) - len(rows),
        ),
    }


def _load_state_v1608(original: Any, self: Any) -> dict[str, Any]:
    state = original(self)
    state.setdefault("daily_entry_order_count", None)
    state.setdefault("daily_entry_submitted_usd", None)
    state.setdefault("non_tradeable_dust", {})
    state.setdefault("dust_cost_basis_usd_total", 0.0)
    state.setdefault("dust_positions_closed", 0)
    if state.get("daily_entry_order_count") is None or state.get("daily_entry_submitted_usd") is None:
        count, submitted = _reconstruct_entry_budget(state)
        state["daily_entry_order_count"] = count
        state["daily_entry_submitted_usd"] = submitted
    return state


def _refresh_day_v1608(original: Any, self: Any) -> None:
    before = str(self.state.get("day") or "")
    original(self)
    after = str(self.state.get("day") or "")
    if after != before:
        self.state["daily_entry_order_count"] = 0
        self.state["daily_entry_submitted_usd"] = 0.0


def _update_balance_snapshot_v1608(self: Any, balance: dict[str, Any]) -> None:
    totals = balance.get("total") or {}
    free = balance.get("free") or {}
    watched_assets = {"USDT"}
    for symbol in self.state.get("positions", {}):
        watched_assets.update(str(symbol).split("/", 1))
    for symbol in self.state.get("non_tradeable_dust", {}):
        watched_assets.update(str(symbol).split("/", 1))

    assets: dict[str, float] = {}
    free_assets: dict[str, float] = {}
    for asset in sorted(watched_assets):
        total_value = totals.get(asset)
        free_value = free.get(asset)
        nested = balance.get(asset)
        if isinstance(nested, dict):
            if total_value is None:
                total_value = nested.get("total")
            if free_value is None:
                free_value = nested.get("free")
        if total_value is not None:
            assets[asset] = float(total_value)
        if free_value is not None:
            free_assets[asset] = float(free_value)

    self.state["account_balance"] = {
        "timestamp": dt.datetime.now(dt.UTC).isoformat(),
        "assets": assets,
        "free": free_assets,
    }


def _skip_v1608(original: Any, self: Any, client_id: str, symbol: str, side: str, reason: str) -> dict[str, Any]:
    result = original(self, client_id, symbol, side, reason)
    record = (self.state.get("orders") or {}).get(client_id)
    if isinstance(record, dict):
        record.setdefault("decision_at", dt.datetime.now(dt.UTC).isoformat())
        self._save_state()
    return result


def _record_non_tradeable_dust(
    self: Any,
    *,
    symbol: str,
    quantity: float,
    reference_price: float,
    minimum_amount: float,
    minimum_cost: float,
    free_quantity: float,
    reason: str,
) -> dict[str, Any]:
    current = max(0.0, _number((self.state.get("positions") or {}).get(symbol)))
    current_cost = max(0.0, _number((self.state.get("position_cost_usd") or {}).get(symbol)))
    if current <= 0.0:
        return {
            "status": "absent",
            "symbol": symbol,
            "reason": "no_testnet_position",
            "live_authority": False,
        }

    dust = {
        "symbol": symbol,
        "quantity": current,
        "free_quantity": max(0.0, free_quantity),
        "reference_price": max(0.0, reference_price),
        "estimated_value_usd": max(0.0, current * reference_price),
        "cost_basis_usd": current_cost,
        "minimum_amount": max(0.0, minimum_amount),
        "minimum_cost_usd": max(0.0, minimum_cost),
        "reason": reason,
        "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
        "tradeable": False,
        "removed_from_active_risk_capacity": True,
        "testnet_only": True,
        "live_authority": False,
    }
    self.state.setdefault("non_tradeable_dust", {})[symbol] = dust
    self.state["dust_cost_basis_usd_total"] = (
        _number(self.state.get("dust_cost_basis_usd_total")) + current_cost
    )
    self.state["dust_positions_closed"] = int(self.state.get("dust_positions_closed") or 0) + 1
    self.state.get("positions", {}).pop(symbol, None)
    self.state.get("position_cost_usd", {}).pop(symbol, None)
    cycle_pnl = _number(self.state.get("position_cycle_pnl_usd", {}).pop(symbol, 0.0))
    self.state["closed_positions"] = int(self.state.get("closed_positions") or 0) + 1
    if cycle_pnl - current_cost > 0.0:
        self.state["winning_positions"] = int(self.state.get("winning_positions") or 0) + 1
    self._save_state()
    return {
        "status": "dust",
        "symbol": symbol,
        "quantity": current,
        "free_quantity": max(0.0, free_quantity),
        "cost_basis_usd": current_cost,
        "minimum_amount": max(0.0, minimum_amount),
        "minimum_cost_usd": max(0.0, minimum_cost),
        "reason": reason,
        "live_authority": False,
    }


def _prepare_sell_v1608(
    self: Any,
    symbol: str,
    requested_quantity: float,
    reference_price: float,
) -> dict[str, Any]:
    with self._io_lock:
        self._require_started()
        symbol = str(symbol or "").upper()
        requested_quantity = max(0.0, _number(requested_quantity))
        reference_price = max(0.0, _number(reference_price))
        if not symbol or requested_quantity <= 0.0 or reference_price <= 0.0:
            return {
                "status": "blocked",
                "reason": "invalid_sell_preparation_input",
                "symbol": symbol,
                "live_authority": False,
            }

        self.reconcile_required()
        current = max(0.0, _number((self.state.get("positions") or {}).get(symbol)))
        if current <= 0.0:
            return {
                "status": "absent",
                "reason": "no_testnet_position",
                "symbol": symbol,
                "live_authority": False,
            }

        unresolved = [
            record
            for record in (self.state.get("orders") or {}).values()
            if isinstance(record, dict)
            and str(record.get("symbol") or "").upper() == symbol
            and str(record.get("status") or "").lower() in {"open", "submitting"}
        ]
        if unresolved:
            return {
                "status": "blocked",
                "reason": "symbol_has_unresolved_order",
                "symbol": symbol,
                "live_authority": False,
            }

        base_asset = symbol.split("/", 1)[0]
        free_assets = (self.state.get("account_balance") or {}).get("free") or {}
        if base_asset not in free_assets:
            return {
                "status": "blocked",
                "reason": "free_base_balance_unavailable",
                "symbol": symbol,
                "position_quantity": current,
                "live_authority": False,
            }
        free_quantity = max(0.0, _number(free_assets.get(base_asset)))

        market = self.exchange.market(symbol)
        minimum_cost = max(
            0.0,
            _number(((market.get("limits") or {}).get("cost") or {}).get("min")),
        )
        minimum_amount = max(
            0.0,
            _number(((market.get("limits") or {}).get("amount") or {}).get("min")),
        )

        position_precise = max(
            0.0,
            _number(self.exchange.amount_to_precision(symbol, current)),
        )
        position_value = position_precise * reference_price
        position_is_dust = bool(
            position_precise <= 0.0
            or (minimum_amount > 0.0 and position_precise < minimum_amount)
            or (minimum_cost > 0.0 and position_value < minimum_cost)
        )
        if position_is_dust:
            return _record_non_tradeable_dust(
                self,
                symbol=symbol,
                quantity=current,
                reference_price=reference_price,
                minimum_amount=minimum_amount,
                minimum_cost=minimum_cost,
                free_quantity=free_quantity,
                reason="residual_below_exchange_executable_threshold",
            )

        available = min(current, free_quantity)
        available_precise = max(
            0.0,
            _number(self.exchange.amount_to_precision(symbol, available)),
        )
        if (
            available_precise <= 0.0
            or (minimum_amount > 0.0 and available_precise < minimum_amount)
            or (minimum_cost > 0.0 and available_precise * reference_price < minimum_cost)
        ):
            return {
                "status": "blocked",
                "reason": "free_balance_not_executable",
                "symbol": symbol,
                "position_quantity": current,
                "free_quantity": free_quantity,
                "minimum_amount": minimum_amount,
                "minimum_cost_usd": minimum_cost,
                "live_authority": False,
            }

        quantity = min(requested_quantity, available_precise)
        quantity = max(
            0.0,
            _number(self.exchange.amount_to_precision(symbol, quantity)),
        )
        if (
            quantity <= 0.0
            or (minimum_amount > 0.0 and quantity < minimum_amount)
            or (minimum_cost > 0.0 and quantity * reference_price < minimum_cost)
        ):
            return {
                "status": "blocked",
                "reason": "requested_exit_below_exchange_minimum",
                "symbol": symbol,
                "position_quantity": current,
                "free_quantity": free_quantity,
                "executable_available_quantity": available_precise,
                "minimum_amount": minimum_amount,
                "minimum_cost_usd": minimum_cost,
                "live_authority": False,
            }

        return {
            "status": "executable",
            "symbol": symbol,
            "requested_quantity": requested_quantity,
            "position_quantity": current,
            "free_quantity": free_quantity,
            "executable_available_quantity": available_precise,
            "executable_quantity": quantity,
            "submitted_usd": quantity * reference_price,
            "minimum_amount": minimum_amount,
            "minimum_cost_usd": minimum_cost,
            "reconciled_at": self.state.get("last_reconciliation"),
            "balance_timestamp": (self.state.get("account_balance") or {}).get("timestamp"),
            "testnet_only": True,
            "live_authority": False,
        }


def _mirror_event_v1608(self: Any, event: dict[str, Any]) -> dict[str, Any]:
    symbol = str(event["symbol"]).upper()
    side = str(event["side"]).lower()
    price = float(event["price"])
    requested_quantity = float(event["quantity"])
    if price <= 0 or requested_quantity <= 0:
        raise ValueError("positive testnet price and quantity are required")

    client_id = self._client_order_id(event)
    existing = self.state["orders"].get(client_id)
    if existing is not None:
        return {"client_order_id": client_id, "idempotent": True, **self._public_record(existing)}

    if symbol not in self._eligible_symbols:
        return self._skip(client_id, symbol, side, "market_unavailable_on_bybit_testnet")

    self._refresh_day()
    if side == "buy" and (self.state_path.parent / "TESTNET_HALT").exists():
        return self._skip(client_id, symbol, side, "testnet_kill_switch")
    if side == "buy" and int(self.state.get("daily_entry_order_count") or 0) >= self.max_orders_per_day:
        return self._skip(client_id, symbol, side, "daily_order_count_cap")
    if side == "buy" and _number(self.state.get("daily_entry_submitted_usd")) >= self.max_daily_submitted_usd:
        return self._skip(client_id, symbol, side, "daily_submitted_notional_cap")

    market = self.exchange.market(symbol)
    minimum_cost = max(0.0, _number(((market.get("limits") or {}).get("cost") or {}).get("min")))
    minimum_amount = max(0.0, _number(((market.get("limits") or {}).get("amount") or {}).get("min")))
    current_quantity = max(0.0, _number(self.state["positions"].get(symbol, 0.0)))

    sell_preparation: dict[str, Any] | None = None
    if side == "buy":
        submitted_usd = max(requested_quantity * price, minimum_cost, minimum_amount * price)
        if submitted_usd > self.max_order_usd:
            return self._skip(client_id, symbol, side, "exchange_minimum_exceeds_order_cap")
        reserved_notional = self._pending_buy_notional(symbol, price)
        if current_quantity * price + reserved_notional + submitted_usd > self.max_position_usd:
            return self._skip(client_id, symbol, side, "position_notional_cap")
        quantity = submitted_usd / price
    else:
        sell_preparation = self.prepare_sell(symbol, requested_quantity, price)
        prep_status = str(sell_preparation.get("status") or "")
        if prep_status == "dust":
            return self._skip(client_id, symbol, side, "non_tradeable_dust")
        if prep_status == "absent":
            return self._skip(client_id, symbol, side, "no_testnet_position")
        if prep_status != "executable":
            return self._skip(
                client_id,
                symbol,
                side,
                "sell_preparation:" + str(sell_preparation.get("reason") or "blocked"),
            )
        quantity = _number(sell_preparation.get("executable_quantity"))
        submitted_usd = quantity * price

    quantity = max(0.0, _number(self.exchange.amount_to_precision(symbol, quantity)))
    if side == "sell":
        quantity = min(quantity, _number((sell_preparation or {}).get("executable_available_quantity")))
    if quantity <= 0:
        return self._skip(client_id, symbol, side, "quantity_below_exchange_precision")
    submitted_usd = quantity * price
    if side == "buy" and submitted_usd > self.max_order_usd:
        return self._skip(client_id, symbol, side, "exchange_precision_exceeds_order_cap")
    if side == "buy" and _number(self.state.get("daily_entry_submitted_usd")) + submitted_usd > self.max_daily_submitted_usd:
        return self._skip(client_id, symbol, side, "daily_submitted_notional_cap")

    record = {
        "client_order_id": client_id,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "submitted_usd": submitted_usd,
        "reference_price": price,
        "reason": str(event.get("reason", "paper_event")),
        "paper_event_timestamp": str(event.get("timestamp", "")),
        "submitted_at": dt.datetime.now(dt.UTC).isoformat(),
        "status": "submitting",
        "order_id": None,
        "filled": 0.0,
        "applied_filled": 0.0,
        "filled_cost": 0.0,
        "applied_fill_cost": 0.0,
        "average": None,
        "fee": 0.0,
        "fee_currency": None,
        "applied_fee": 0.0,
        "fill_counted": False,
    }
    if sell_preparation is not None:
        record["sell_preparation"] = copy.deepcopy(sell_preparation)
    self.state["orders"][client_id] = record

    self.state["daily_submitted_usd"] = _number(self.state.get("daily_submitted_usd")) + submitted_usd
    self.state["daily_order_count"] = int(self.state.get("daily_order_count") or 0) + 1
    if side == "buy":
        self.state["daily_entry_submitted_usd"] = (
            _number(self.state.get("daily_entry_submitted_usd")) + submitted_usd
        )
        self.state["daily_entry_order_count"] = int(self.state.get("daily_entry_order_count") or 0) + 1
    self._save_state()

    self._verify_testnet_urls()
    observed = self.exchange.create_order(
        symbol,
        "market",
        side,
        quantity,
        None,
        {"orderLinkId": client_id},
    )
    self._merge_observed(record, observed)
    self._save_state()
    return {"client_order_id": client_id, "idempotent": False, **self._public_record(record)}


def _engine_health_v1608(original: Any, self: Any) -> dict[str, Any]:
    snapshot = original(self)
    performance = dict(snapshot.get("performance") or {})
    dust_cost = max(0.0, _number(self.state.get("dust_cost_basis_usd_total")))
    realized = _number(performance.get("realized_pnl_usd"))
    performance.update(
        {
            "non_tradeable_dust_cost_basis_usd": dust_cost,
            "realized_net_after_dust_usd": realized - dust_cost,
            "dust_positions_closed": int(self.state.get("dust_positions_closed") or 0),
        }
    )
    snapshot["performance"] = performance
    snapshot["daily_total_submitted_usd"] = _number(self.state.get("daily_submitted_usd"))
    snapshot["daily_total_order_count"] = int(self.state.get("daily_order_count") or 0)
    snapshot["daily_entry_submitted_usd"] = _number(self.state.get("daily_entry_submitted_usd"))
    snapshot["daily_entry_order_count"] = int(self.state.get("daily_entry_order_count") or 0)
    snapshot["current_day_execution_quality"] = _current_day_execution_quality(self.state)
    snapshot["non_tradeable_dust"] = copy.deepcopy(self.state.get("non_tradeable_dust") or {})
    snapshot["entry_budget_excludes_protective_exits"] = True
    snapshot["fresh_free_balance_exit_sizing"] = True
    snapshot["dust_recycles_risk_capacity"] = True
    snapshot["live_authority"] = False
    return snapshot


class _EligibleCandidateService:
    def __init__(self, service: Any, lane: Any) -> None:
        self._service = service
        self._lane = lane

    def __getattr__(self, name: str) -> Any:
        return getattr(self._service, name)

    def collective_candidates(self, limit: int = 8) -> list[str]:
        public = list(self._service.collective_candidates(limit=limit) or [])
        eligible_method = getattr(self._lane.testnet, "eligible_symbols", None)
        eligible: set[str] = set()
        error_type: str | None = None
        if callable(eligible_method):
            try:
                eligible = {str(symbol).upper() for symbol in eligible_method("USDT")}
            except Exception as exc:
                error_type = type(exc).__name__
        filtered = [
            str(symbol).upper()
            for symbol in public
            if str(symbol).upper() in eligible
        ]
        with self._lane._lock:
            self._lane.state["last_testnet_market_filter"] = {
                "public_candidate_count": len(public),
                "eligible_candidate_count": len(filtered),
                "filtered_out_count": max(0, len(public) - len(filtered)),
                "eligible_market_count": len(eligible),
                "error_type": error_type,
                "fail_closed": True,
                "timestamp": time.time(),
            }
            self._lane._save_locked()
        return filtered[:limit]


def _hyper_step_v1608(original: Any, self: Any, *, now: float | None = None) -> dict[str, Any]:
    original_provider = self.service_provider

    def provider() -> Any:
        service = original_provider()
        if service is None:
            return None
        return _EligibleCandidateService(service, self)

    self.service_provider = provider
    try:
        return original(self, now=now)
    finally:
        self.service_provider = original_provider


def _adaptive_position_capacity_v1608(
    original: Any,
    self: Any,
    supervisor: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    candidate_count: int,
    entries_today: int,
) -> dict[str, Any]:
    entry_snapshot = copy.deepcopy(snapshot)
    entry_snapshot["daily_order_count"] = int(snapshot.get("daily_entry_order_count") or 0)
    entry_snapshot["daily_submitted_usd"] = _number(snapshot.get("daily_entry_submitted_usd"))
    result = original(
        self,
        supervisor,
        entry_snapshot,
        candidate_count=candidate_count,
        entries_today=entries_today,
    )
    result["entry_budget_order_count"] = entry_snapshot["daily_order_count"]
    result["entry_budget_submitted_usd"] = entry_snapshot["daily_submitted_usd"]
    result["protective_exits_excluded_from_entry_budget"] = True
    return result


def _compound_order_notional_v1608(
    original: Any,
    self: Any,
    supervisor: dict[str, Any],
    *,
    slots: int,
) -> dict[str, Any]:
    canonical = original(self, supervisor, slots=slots)
    if canonical.get("allowed") is not True:
        return canonical

    snapshot = self.testnet.safe_snapshot()
    performance = snapshot.get("performance") or {}
    actual_realized = _number(performance.get("realized_pnl_usd"))
    dust_cost = max(0.0, _number(performance.get("non_tradeable_dust_cost_basis_usd")))
    with self._lock:
        closed = [
            copy.deepcopy(row)
            for row in (self.state.get("closed") or [])
            if isinstance(row, dict)
        ]
    completed_entry_notional = sum(
        max(
            0.0,
            _number(row.get("entry_notional_usd"))
            or (_number(row.get("quantity")) * _number(row.get("entry_price"))),
        )
        for row in closed
    )
    modeled_floor_bps = max(MODELED_ROUND_TRIP_COST_FLOOR_BPS, _number(self.round_trip_cost_bps))
    modeled_cost_reserve = completed_entry_notional * modeled_floor_bps / 10_000.0
    actual_after_model = actual_realized - dust_cost - modeled_cost_reserve

    growth = supervisor.get("capital_growth") or {}
    risk_multiplier = max(0.0, min(1.0, _number(growth.get("risk_multiplier"), 1.0)))
    base_notional = min(self.maximum_order_usd, self.order_usd) * risk_multiplier
    if risk_multiplier <= 0.0 or base_notional < 0.50:
        return {
            **canonical,
            "allowed": False,
            "reason": "testnet_base_exploration_budget_blocked",
            "actual_testnet_realized_pnl_usd": actual_realized,
            "actual_testnet_net_after_modeled_cost_usd": actual_after_model,
            "actual_testnet_profit_compounding_eligible": False,
            "compounding": False,
            "live_authority": False,
        }

    canonical_notional = max(0.0, _number(canonical.get("order_notional_usd")))
    if actual_after_model <= 0.0:
        return {
            **canonical,
            "reason": "bounded_base_until_positive_testnet_realized_net_pnl",
            "order_notional_usd": base_notional,
            "compounding": False,
            "canonical_paper_compounding_available": bool(canonical.get("compounding")),
            "canonical_paper_order_notional_usd": canonical_notional,
            "actual_testnet_realized_pnl_usd": actual_realized,
            "testnet_non_tradeable_dust_cost_basis_usd": dust_cost,
            "testnet_modeled_cost_reserve_usd": modeled_cost_reserve,
            "actual_testnet_net_after_modeled_cost_usd": actual_after_model,
            "actual_testnet_profit_compounding_eligible": False,
            "modeled_round_trip_cost_floor_bps": modeled_floor_bps,
            "live_authority": False,
        }

    incremental_per_slot = actual_after_model / max(1, int(slots))
    actual_profit_cap = min(self.maximum_order_usd, base_notional + incremental_per_slot)
    order_notional = min(canonical_notional, actual_profit_cap)
    return {
        **canonical,
        "reason": "actual_testnet_profit_gated_compound_budget",
        "order_notional_usd": order_notional,
        "compounding": order_notional > base_notional + 1e-12,
        "canonical_paper_compounding_available": bool(canonical.get("compounding")),
        "canonical_paper_order_notional_usd": canonical_notional,
        "actual_testnet_realized_pnl_usd": actual_realized,
        "testnet_non_tradeable_dust_cost_basis_usd": dust_cost,
        "testnet_modeled_cost_reserve_usd": modeled_cost_reserve,
        "actual_testnet_net_after_modeled_cost_usd": actual_after_model,
        "actual_testnet_profit_compounding_eligible": True,
        "actual_testnet_profit_increment_cap_usd": actual_profit_cap,
        "modeled_round_trip_cost_floor_bps": modeled_floor_bps,
        "live_authority": False,
    }


def _exit_recycle_cooldown(self: Any) -> float:
    return max(
        EXIT_RECYCLE_MIN_COOLDOWN_SECONDS,
        max(1.0, _number(getattr(self, "cadence_seconds", 5.0))) * 3.0,
    )


def _close_dust_slot(self: Any, pending: dict[str, Any], preparation: dict[str, Any], now: float) -> dict[str, Any]:
    source_event = pending.get("event") or pending.get("source_event") or {}
    symbol = str(source_event.get("symbol") or preparation.get("symbol") or "").upper()
    with self._lock:
        record = (self.state.get("active") or {}).pop(symbol, None)
        dust_history = list(self.state.get("dust_recycles") or [])
        dust_history.append(
            {
                "symbol": symbol,
                "quantity": preparation.get("quantity"),
                "cost_basis_usd": preparation.get("cost_basis_usd"),
                "reason": preparation.get("reason"),
                "entry_event_id": (record or {}).get("entry_event_id"),
                "exit_reason": (pending.get("assessment") or {}).get("exit_reason"),
                "recycled_at": now,
                "testnet_only": True,
                "live_authority": False,
            }
        )
        self.state["dust_recycles"] = dust_history[-250:]
        self.state["pending_event"] = None
        self.state.setdefault("last_exit_by_symbol", {})[symbol] = now
        self.state["last_action"] = {
            "action": "dust_slot_recycle",
            "symbol": symbol,
            "timestamp": now,
            "quantity": preparation.get("quantity"),
            "live_authority": False,
        }
        self._save_locked()
    return self._decision(
        "non_tradeable_dust_slot_recycled",
        details={
            "kind": "exit",
            "symbol": symbol,
            "status": "dust",
            "preparation": copy.deepcopy(preparation),
        },
    )


def _submit_pending_v1608(
    original: Any,
    self: Any,
    pending: dict[str, Any],
    *,
    now: float,
) -> dict[str, Any]:
    kind = str(pending.get("kind") or "")
    if kind not in {"exit", "exit_recovery"}:
        return original(self, pending, now=now)

    retry_not_before = _number(pending.get("retry_not_before"))
    if retry_not_before > now:
        return self._decision(
            "exit_recycle_cooldown",
            details={
                "kind": "exit",
                "symbol": str(
                    ((pending.get("event") or pending.get("source_event") or {}).get("symbol") or "")
                ).upper(),
                "retry_in_seconds": max(0.0, retry_not_before - now),
                "recovery_attempt": int(pending.get("recovery_attempt") or 0),
            },
        )

    if kind == "exit_recovery":
        source = copy.deepcopy(pending.get("source_event") or {})
        preparation = self.testnet.prepare_sell(
            str(source.get("symbol") or ""),
            _number(source.get("quantity")),
            _number(source.get("price")),
        )
        if preparation.get("status") == "dust":
            return _close_dust_slot(self, pending, preparation, now)
        if preparation.get("status") != "executable":
            pending = copy.deepcopy(pending)
            pending["retry_not_before"] = now + _exit_recycle_cooldown(self)
            pending["last_sell_preparation"] = copy.deepcopy(preparation)
            self._set_pending(pending)
            return self._decision(
                "exit_reconciliation_waiting_for_executable_balance",
                details={
                    "kind": "exit",
                    "symbol": str(source.get("symbol") or "").upper(),
                    "preparation": preparation,
                },
            )
        new_event = self._new_event(
            symbol=str(source.get("symbol") or "").upper(),
            side="sell",
            quantity=_number(preparation.get("executable_quantity")),
            price=_number(source.get("price")),
            reason=str(source.get("reason") or "fast_collective_testnet_exit") + ":corrected_recycle",
            now=now,
            remaining_quantity=max(
                0.0,
                _number(preparation.get("position_quantity"))
                - _number(preparation.get("executable_quantity")),
            ),
        )
        pending = {
            "kind": "exit",
            "event": new_event,
            "assessment": copy.deepcopy(pending.get("assessment") or {}),
            "created_at": now,
            "recovery_attempt": int(pending.get("recovery_attempt") or 0),
            "recovery_of_event_id": source.get("event_id"),
            "sell_preparation": copy.deepcopy(preparation),
            "submitted_once": False,
        }
        self._set_pending(pending)

    if not pending.get("submitted_once"):
        event = copy.deepcopy(pending.get("event") or {})
        preparation = self.testnet.prepare_sell(
            str(event.get("symbol") or ""),
            _number(event.get("quantity")),
            _number(event.get("price")),
        )
        if preparation.get("status") == "dust":
            return _close_dust_slot(self, pending, preparation, now)
        if preparation.get("status") != "executable":
            hold = {
                "kind": "exit_recovery",
                "source_event": event,
                "assessment": copy.deepcopy(pending.get("assessment") or {}),
                "created_at": pending.get("created_at", now),
                "retry_not_before": now + _exit_recycle_cooldown(self),
                "recovery_attempt": int(pending.get("recovery_attempt") or 0),
                "last_sell_preparation": copy.deepcopy(preparation),
            }
            self._set_pending(hold)
            return self._decision(
                "exit_reconciliation_waiting_for_executable_balance",
                details={
                    "kind": "exit",
                    "symbol": str(event.get("symbol") or "").upper(),
                    "preparation": preparation,
                },
            )
        event["quantity"] = _number(preparation.get("executable_quantity"))
        event["remaining_quantity"] = max(
            0.0,
            _number(preparation.get("position_quantity")) - event["quantity"],
        )
        pending = copy.deepcopy(pending)
        pending["event"] = event
        pending["sell_preparation"] = copy.deepcopy(preparation)
        pending["submitted_once"] = True
        self._set_pending(pending)

    result = original(self, pending, now=now)
    details = result.get("details") or {}
    status = str(details.get("status") or "").lower()
    filled = max(0.0, _number(details.get("filled")))
    current_total = max(0.0, _number(details.get("current_total_quantity")))
    if status not in {"canceled", "rejected"} or filled > 0.0 or current_total <= 0.0:
        return result

    old_event = copy.deepcopy(pending.get("event") or {})
    preparation = self.testnet.prepare_sell(
        str(old_event.get("symbol") or ""),
        max(current_total, _number(old_event.get("quantity"))),
        _number(old_event.get("price")),
    )
    if preparation.get("status") == "dust":
        return _close_dust_slot(self, pending, preparation, now)

    recovery = {
        "kind": "exit_recovery",
        "source_event": old_event,
        "assessment": copy.deepcopy(pending.get("assessment") or {}),
        "created_at": now,
        "retry_not_before": now + _exit_recycle_cooldown(self),
        "recovery_attempt": int(pending.get("recovery_attempt") or 0) + 1,
        "terminal_zero_fill_status": status,
        "last_sell_preparation": copy.deepcopy(preparation),
    }
    self._set_pending(recovery)
    return self._decision(
        "zero_fill_exit_reconciled_for_corrected_retry",
        details={
            "kind": "exit",
            "symbol": str(old_event.get("symbol") or "").upper(),
            "status": status,
            "filled": filled,
            "current_total_quantity": current_total,
            "retry_not_before": recovery["retry_not_before"],
            "recovery_attempt": recovery["recovery_attempt"],
            "preparation": preparation,
            "ambiguous_order_resubmission_allowed": False,
        },
    )


def _hyper_health_v1608(original: Any, self: Any) -> dict[str, Any]:
    payload = original(self)
    snapshot = self.testnet.safe_snapshot()
    performance = snapshot.get("performance") or {}
    with self._lock:
        last_filter = copy.deepcopy(self.state.get("last_testnet_market_filter") or {})
        dust_recycles = copy.deepcopy(self.state.get("dust_recycles") or [])
        last_sizing = copy.deepcopy(self.state.get("last_sizing") or {})
    payload.update(
        {
            "version": "1.60.8",
            "testnet_market_candidate_filter": last_filter,
            "testnet_eligible_market_intersection": True,
            "exit_quantity_source": "fresh_reconciled_free_base_balance",
            "zero_fill_terminal_exit_recycle": {
                "enabled": True,
                "minimum_cooldown_seconds": _exit_recycle_cooldown(self),
                "new_order_only_after_terminal_zero_fill_and_fresh_reconciliation": True,
                "ambiguous_order_resubmission_allowed": False,
                "deterministic_order_link_id": True,
            },
            "dust_recycles": dust_recycles[-10:],
            "dust_recycle_count": len(dust_recycles),
            "canonical_paper_growth": copy.deepcopy(
                (self.supervisory_provider() or {}).get("capital_growth") or {}
            ),
            "actual_testnet_realized_pnl_usd": _number(performance.get("realized_pnl_usd")),
            "actual_testnet_realized_net_after_dust_usd": _number(
                performance.get("realized_net_after_dust_usd"),
                _number(performance.get("realized_pnl_usd")),
            ),
            "actual_testnet_profit_compounding_eligible": bool(
                last_sizing.get("actual_testnet_profit_compounding_eligible")
            ),
            "principal_protected_compounding": bool(last_sizing.get("compounding")),
            "modeled_round_trip_cost_floor_bps": max(
                MODELED_ROUND_TRIP_COST_FLOOR_BPS,
                _number(self.round_trip_cost_bps),
            ),
            "automatic_promotion": False,
            "live_authority": False,
        }
    )
    return payload


def install_testnet_exit_recycle_v1608() -> None:
    """Install the v1.60.8 Testnet exit/recycle correction idempotently."""

    from .fast_collective_hyper import HyperSpeedCollectiveTestnetLane
    from .testnet_execution import BybitTestnetExecutionEngine
    from .velocity_sniper_testnet import VelocitySniperTestnetLane

    if getattr(BybitTestnetExecutionEngine, "_testnet_exit_recycle_v1608_installed", False):
        return

    original_load_state = BybitTestnetExecutionEngine._load_state
    original_refresh_day = BybitTestnetExecutionEngine._refresh_day
    original_skip = BybitTestnetExecutionEngine._skip
    original_engine_health = BybitTestnetExecutionEngine.health

    BybitTestnetExecutionEngine.VERSION = "2.6"
    BybitTestnetExecutionEngine._load_state = lambda self: _load_state_v1608(original_load_state, self)
    BybitTestnetExecutionEngine._refresh_day = lambda self: _refresh_day_v1608(original_refresh_day, self)
    BybitTestnetExecutionEngine._update_balance_snapshot = _update_balance_snapshot_v1608
    BybitTestnetExecutionEngine._skip = lambda self, client_id, symbol, side, reason: _skip_v1608(
        original_skip, self, client_id, symbol, side, reason
    )
    BybitTestnetExecutionEngine.prepare_sell = _prepare_sell_v1608
    BybitTestnetExecutionEngine._mirror_event = _mirror_event_v1608
    BybitTestnetExecutionEngine.health = lambda self: _engine_health_v1608(original_engine_health, self)
    BybitTestnetExecutionEngine._testnet_exit_recycle_v1608_installed = True

    original_hyper_step = HyperSpeedCollectiveTestnetLane.step
    original_capacity = HyperSpeedCollectiveTestnetLane._adaptive_position_capacity
    original_compound = HyperSpeedCollectiveTestnetLane._compound_order_notional
    original_submit = HyperSpeedCollectiveTestnetLane._submit_pending
    original_hyper_health = HyperSpeedCollectiveTestnetLane.health

    HyperSpeedCollectiveTestnetLane.VERSION = "1.60.8"
    HyperSpeedCollectiveTestnetLane.step = lambda self, *, now=None: _hyper_step_v1608(
        original_hyper_step, self, now=now
    )
    HyperSpeedCollectiveTestnetLane._adaptive_position_capacity = (
        lambda self, supervisor, snapshot, *, candidate_count, entries_today: _adaptive_position_capacity_v1608(
            original_capacity,
            self,
            supervisor,
            snapshot,
            candidate_count=candidate_count,
            entries_today=entries_today,
        )
    )
    HyperSpeedCollectiveTestnetLane._compound_order_notional = (
        lambda self, supervisor, *, slots: _compound_order_notional_v1608(
            original_compound, self, supervisor, slots=slots
        )
    )
    HyperSpeedCollectiveTestnetLane._submit_pending = (
        lambda self, pending, *, now: _submit_pending_v1608(
            original_submit, self, pending, now=now
        )
    )
    HyperSpeedCollectiveTestnetLane.health = lambda self: _hyper_health_v1608(
        original_hyper_health, self
    )
    HyperSpeedCollectiveTestnetLane._testnet_exit_recycle_v1608_installed = True

    VelocitySniperTestnetLane.VERSION = "1.60.8"
