from __future__ import annotations

import copy
import time
from typing import Any

from .testnet_exit_price_guard_v1611 import _price_limit
from .testnet_exit_recycle import MODELED_ROUND_TRIP_COST_FLOOR_BPS


MIN_COST_HEADROOM_BPS = 500.0
EXIT_STRESS_BPS = 500.0
ZERO_FILL_COOLDOWN_SECONDS = 300.0
PREFLIGHT_COOLDOWN_SECONDS = 60.0
BOOK_LIMIT = 25
LIQUIDITY_BUFFER = 0.05


def _n(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _supported(engine: Any) -> bool:
    exchange = getattr(engine, "exchange", None)
    if exchange is None:
        return False
    return bool(
        callable(getattr(exchange, "fetch_ticker", None))
        and callable(getattr(exchange, "fetch_order_book", None))
        and any(
            callable(getattr(exchange, name, None))
            for name in (
                "public_get_v5_market_price_limit",
                "publicGetV5MarketPriceLimit",
            )
        )
    )


def _normalize_buy(
    engine: Any,
    event: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = copy.deepcopy(event)

    symbol = str(row.get("symbol") or "").upper()
    price = max(0.0, _n(row.get("price")))
    quantity = max(0.0, _n(row.get("quantity")))

    if not symbol or price <= 0.0 or quantity <= 0.0:
        return row, {"normalized": False}

    market = engine.exchange.market(symbol)
    limits = market.get("limits") or {}

    min_cost = max(
        0.0,
        _n((limits.get("cost") or {}).get("min")),
    )
    min_amount = max(
        0.0,
        _n((limits.get("amount") or {}).get("min")),
    )

    requested = quantity * price

    stress_fraction = max(
        0.01,
        1.0 - EXIT_STRESS_BPS / 10_000.0,
    )
    cost_fraction = max(
        0.01,
        1.0
        - MODELED_ROUND_TRIP_COST_FLOOR_BPS
        / 10_000.0,
    )

    safe_min_cost = (
        min_cost
        / (stress_fraction * cost_fraction)
        if min_cost > 0.0
        else 0.0
    )

    required = max(
        requested,
        safe_min_cost,
        min_amount * price,
    )

    max_order = max(
        0.0,
        _n(getattr(engine, "max_order_usd", 0.0)),
    )

    if (
        max_order > 0.0
        and required > max_order + 1e-12
    ):
        return row, {
            "normalized": False,
            "blocked_reason": (
                "safe_minimum_exceeds_order_cap"
            ),
            "requested_notional_usd": requested,
            "safe_required_notional_usd": required,
            "max_order_usd": max_order,
            "minimum_cost_usd": min_cost,
            "minimum_amount": min_amount,
            "exit_stress_bps": EXIT_STRESS_BPS,
            "modeled_round_trip_cost_floor_bps": (
                MODELED_ROUND_TRIP_COST_FLOOR_BPS
            ),
        }

    row["quantity"] = required / price

    return row, {
        "normalized": abs(required - requested) > 1e-12,
        "requested_notional_usd": requested,
        "normalized_notional_usd": required,
        "minimum_cost_usd": min_cost,
        "minimum_amount": min_amount,
        "minimum_cost_headroom_bps": MIN_COST_HEADROOM_BPS,
    }


def _top_book(engine: Any, symbol: str) -> dict[str, Any]:
    ticker = engine.exchange.fetch_ticker(symbol) or {}
    book = (
        engine.exchange.fetch_order_book(
            symbol,
            limit=BOOK_LIMIT,
        )
        or {}
    )

    bids = [
        row
        for row in (book.get("bids") or [])
        if (
            isinstance(row, (list, tuple))
            and len(row) >= 2
            and _n(row[0]) > 0.0
            and _n(row[1]) > 0.0
        )
    ]

    asks = [
        row
        for row in (book.get("asks") or [])
        if (
            isinstance(row, (list, tuple))
            and len(row) >= 2
            and _n(row[0]) > 0.0
            and _n(row[1]) > 0.0
        )
    ]

    bid = (
        _n(bids[0][0])
        if bids
        else _n(ticker.get("bid"))
    )

    ask = (
        _n(asks[0][0])
        if asks
        else _n(ticker.get("ask"))
    )

    return {
        "bid": max(0.0, bid),
        "ask": max(0.0, ask),
        "bids": bids,
        "asks": asks,
    }


def _buy_depth(
    asks: list[Any],
    quantity: float,
    buy_limit: float,
) -> tuple[float, float]:
    remaining = max(0.0, quantity)
    filled = 0.0
    cost = 0.0

    for row in asks:
        price = max(0.0, _n(row[0]))
        available = max(0.0, _n(row[1]))

        if price <= 0.0 or available <= 0.0:
            continue

        if (
            buy_limit > 0.0
            and price > buy_limit + 1e-12
        ):
            break

        take = min(remaining, available)

        filled += take
        cost += take * price
        remaining -= take

        if remaining <= 1e-12:
            break

    return filled, cost


def _blocked_until(engine: Any, symbol: str) -> float:
    return max(
        0.0,
        _n(
            (
                engine.state.get(
                    "v1613_entry_blocked_until"
                )
                or {}
            ).get(symbol)
        ),
    )


def _arm(
    engine: Any,
    symbol: str,
    seconds: float,
    reason: str,
    detail: dict[str, Any],
    *,
    zero_fill: bool = False,
) -> float:
    now = time.time()
    until = now + seconds

    with engine._io_lock:
        previous = _blocked_until(engine, symbol)

        engine.state.setdefault(
            "v1613_entry_blocked_until",
            {},
        )[symbol] = max(previous, until)

        if zero_fill:
            counter = "v1613_terminal_zero_fill_buys"
            last = "v1613_last_terminal_zero_fill_buy"
        else:
            counter = "v1613_entry_preflight_blocks"
            last = "v1613_last_entry_preflight_block"

        engine.state[counter] = (
            int(engine.state.get(counter) or 0) + 1
        )

        engine.state[last] = {
            "symbol": symbol,
            "reason": reason,
            "blocked_at": now,
            "blocked_until": max(previous, until),
            "detail": copy.deepcopy(detail),
            "live_authority": False,
        }

        engine._save_state()

    return max(previous, until)


def _preflight(
    engine: Any,
    event: dict[str, Any],
) -> dict[str, Any]:
    symbol = str(event.get("symbol") or "").upper()

    until = _blocked_until(engine, symbol)

    if until > time.time():
        return {
            "allowed": False,
            "reason": "entry_cooldown",
            "blocked_until": until,
        }

    market = engine.exchange.market(symbol)
    limits = market.get("limits") or {}

    min_cost = max(
        0.0,
        _n((limits.get("cost") or {}).get("min")),
    )
    min_amount = max(
        0.0,
        _n((limits.get("amount") or {}).get("min")),
    )

    raw_quantity = max(
        0.0,
        _n(event.get("quantity")),
    )

    quantity = max(
        0.0,
        _n(
            engine.exchange.amount_to_precision(
                symbol,
                raw_quantity,
            )
        ),
    )

    if (
        quantity <= 0.0
        or (
            min_amount > 0.0
            and quantity < min_amount
        )
    ):
        return {
            "allowed": False,
            "reason": "entry_quantity_below_minimum",
        }

    book = _top_book(engine, symbol)

    bid = _n(book.get("bid"))
    ask = _n(book.get("ask"))

    if bid <= 0.0 or ask <= 0.0:
        return {
            "allowed": False,
            "reason": "fresh_two_sided_book_unavailable",
        }

    limits_now = _price_limit(engine, symbol)

    if (
        limits_now.get("supported") is True
        and limits_now.get("ok") is not True
    ):
        return {
            "allowed": False,
            "reason": "bybit_price_limit_unavailable",
        }

    buy_limit = max(
        0.0,
        _n(limits_now.get("buy_limit")),
    )
    sell_limit = max(
        0.0,
        _n(limits_now.get("sell_limit")),
    )

    if (
        limits_now.get("supported") is True
        and (
            buy_limit <= 0.0
            or sell_limit <= 0.0
        )
    ):
        return {
            "allowed": False,
            "reason": "bybit_price_limit_incomplete",
        }

    if (
        buy_limit > 0.0
        and ask > buy_limit + 1e-12
    ):
        return {
            "allowed": False,
            "reason": "buy_price_limit_unexecutable",
            "fresh_ask": ask,
            "buy_limit": buy_limit,
        }

    if (
        sell_limit > 0.0
        and bid + 1e-12 < sell_limit
    ):
        return {
            "allowed": False,
            "reason": "prospective_exit_price_limit_unexecutable",
            "fresh_bid": bid,
            "sell_limit": sell_limit,
        }

    buffered_quantity = quantity * (
        1.0 + LIQUIDITY_BUFFER
    )

    fillable_buffered, _ = _buy_depth(
        book.get("asks") or [],
        buffered_quantity,
        buy_limit,
    )

    if (
        fillable_buffered + 1e-12
        < buffered_quantity
    ):
        return {
            "allowed": False,
            "reason": "insufficient_immediate_ask_liquidity",
            "quantity": quantity,
            "buffered_quantity": buffered_quantity,
            "fillable_quantity": fillable_buffered,
        }

    fillable, projected_cost = _buy_depth(
        book.get("asks") or [],
        quantity,
        buy_limit,
    )

    if fillable + 1e-12 < quantity:
        return {
            "allowed": False,
            "reason": "buy_not_fully_fillable",
        }

    free = (
        (
            engine.state.get("account_balance")
            or {}
        ).get("free")
        or {}
    )

    if "USDT" not in free:
        return {
            "allowed": False,
            "reason": "fresh_free_quote_unavailable",
        }

    free_usdt = max(
        0.0,
        _n(free.get("USDT")),
    )

    reserve = max(
        0.01,
        projected_cost * 0.005,
    )

    if (
        projected_cost
        > max(0.0, free_usdt - reserve)
        + 1e-12
    ):
        return {
            "allowed": False,
            "reason": "projected_buy_exceeds_free_quote",
            "projected_cost_usd": projected_cost,
            "free_usdt": free_usdt,
            "reserve_usd": reserve,
        }

    exit_raw = quantity * (
        1.0
        - MODELED_ROUND_TRIP_COST_FLOOR_BPS
        / 10_000.0
    )

    exit_quantity = max(
        0.0,
        _n(
            engine.exchange.amount_to_precision(
                symbol,
                exit_raw,
            )
        ),
    )

    stressed_bid = bid * max(
        0.0,
        1.0 - EXIT_STRESS_BPS / 10_000.0,
    )

    fresh_exit_value = exit_quantity * bid
    exit_value = exit_quantity * stressed_bid

    if (
        exit_quantity <= 0.0
        or (
            min_amount > 0.0
            and exit_quantity < min_amount
        )
        or (
            min_cost > 0.0
            and exit_value + 1e-12 < min_cost
        )
    ):
        return {
            "allowed": False,
            "reason": (
                "prospective_position_not_sellable_under_stress"
            ),
            "exit_quantity": exit_quantity,
            "fresh_exit_value_usd": fresh_exit_value,
            "stressed_exit_value_usd": exit_value,
            "stressed_bid": stressed_bid,
            "minimum_cost_usd": min_cost,
            "minimum_amount": min_amount,
        }

    return {
        "allowed": True,
        "reason": "round_trip_executable",
        "fresh_bid": bid,
        "fresh_ask": ask,
        "buy_limit": buy_limit,
        "sell_limit": sell_limit,
        "projected_cost_usd": projected_cost,
        "prospective_exit_value_usd": fresh_exit_value,
        "stressed_exit_value_usd": exit_value,
    }


def install_testnet_entry_roundtrip_v1613() -> None:
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
        "_v1613_entry_roundtrip_installed",
        False,
    ):
        return

    original_mirror = (
        BybitTestnetExecutionEngine._mirror_event
    )
    original_health = (
        BybitTestnetExecutionEngine.health
    )
    original_merge = (
        BybitTestnetExecutionEngine._merge_observed
    )
    original_hyper_submit = (
        HyperSpeedCollectiveTestnetLane._submit_pending
    )
    original_hyper_health = (
        HyperSpeedCollectiveTestnetLane.health
    )

    def mirror_event(
        self: Any,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            str(event.get("side") or "").lower()
            != "buy"
            or not _supported(self)
        ):
            return original_mirror(self, event)

        normalized, normalization = (
            _normalize_buy(self, event)
        )

        symbol = str(
            normalized.get("symbol") or ""
        ).upper()

        client_id = self._client_order_id(
            normalized
        )

        if client_id in (
            self.state.get("orders") or {}
        ):
            return original_mirror(
                self,
                normalized,
            )

        # Preserve canonical skip ordering.
        if (
            symbol not in self._eligible_symbols
            or (
                self.state_path.parent
                / "TESTNET_HALT"
            ).exists()
            or int(
                self.state.get(
                    "daily_entry_order_count"
                )
                or 0
            )
            >= self.max_orders_per_day
            or _n(
                self.state.get(
                    "daily_entry_submitted_usd"
                )
            )
            >= self.max_daily_submitted_usd
        ):
            return original_mirror(
                self,
                normalized,
            )

        normalization_block = str(
            normalization.get("blocked_reason") or ""
        )

        if normalization_block:
            preflight = {
                "allowed": False,
                "reason": normalization_block,
                "normalization": copy.deepcopy(
                    normalization
                ),
            }
        else:
            preflight = _preflight(
                self,
                normalized,
            )

        if preflight.get("allowed") is not True:
            reason = str(
                preflight.get("reason")
                or "entry_round_trip_blocked"
            )

            if reason == "entry_cooldown":
                with self._io_lock:
                    self.state[
                        "v1613_entry_cooldown_skips"
                    ] = (
                        int(
                            self.state.get(
                                "v1613_entry_cooldown_skips"
                            )
                            or 0
                        )
                        + 1
                    )
                    self._save_state()

                return {
                    "client_order_id": client_id,
                    "idempotent": False,
                    "symbol": symbol,
                    "side": "buy",
                    "status": "skipped",
                    "skip_reason": (
                        "entry_round_trip:"
                        + reason
                    ),
                    "order_id": None,
                    "filled": 0.0,
                    "average": None,
                    "fee": 0.0,
                }

            until = _arm(
                self,
                symbol,
                PREFLIGHT_COOLDOWN_SECONDS,
                reason,
                {
                    "preflight": preflight,
                    "normalization": normalization,
                },
            )

            result = self._skip(
                client_id,
                symbol,
                "buy",
                "entry_round_trip:" + reason,
            )

            record = (
                self.state.get("orders") or {}
            ).get(client_id)

            if isinstance(record, dict):
                record[
                    "v1613_entry_preflight"
                ] = copy.deepcopy(preflight)

                record[
                    "v1613_entry_normalization"
                ] = copy.deepcopy(normalization)

                record[
                    "v1613_entry_blocked_until"
                ] = until

                self._save_state()

            return result

        result = original_mirror(
            self,
            normalized,
        )

        status = str(
            result.get("status") or ""
        ).lower()

        filled = max(
            0.0,
            _n(result.get("filled")),
        )

        if (
            status in {"canceled", "rejected"}
            and filled <= 0.0
        ):
            record = (
                self.state.get("orders") or {}
            ).get(client_id)

            if (
                isinstance(record, dict)
                and record.get(
                    "v1613_zero_fill_counted"
                )
                is not True
            ):
                record[
                    "v1613_zero_fill_counted"
                ] = True

                _arm(
                    self,
                    symbol,
                    ZERO_FILL_COOLDOWN_SECONDS,
                    "terminal_zero_fill_buy",
                    {
                        "status": status,
                        "exchange_reject_reason": (
                            record.get(
                                "exchange_reject_reason"
                            )
                        ),
                        "reconciliation_resolution": (
                            record.get(
                                "reconciliation_resolution"
                            )
                        ),
                    },
                    zero_fill=True,
                )

                self._save_state()

        return result

    def merge_observed(
        self: Any,
        record: dict[str, Any],
        observed: dict[str, Any],
    ) -> None:
        original_merge(self, record, observed)

        if (
            str(record.get("side") or "").lower()
            != "buy"
            or str(record.get("status") or "").lower()
            not in {"canceled", "rejected"}
            or _n(record.get("filled")) > 0.0
            or record.get("v1613_zero_fill_counted")
            is True
        ):
            return

        record["v1613_zero_fill_counted"] = True

        _arm(
            self,
            str(record.get("symbol") or "").upper(),
            ZERO_FILL_COOLDOWN_SECONDS,
            "terminal_zero_fill_buy",
            {
                "status": record.get("status"),
                "exchange_reject_reason": (
                    record.get("exchange_reject_reason")
                ),
                "reconciliation_resolution": (
                    record.get(
                        "reconciliation_resolution"
                    )
                ),
            },
            zero_fill=True,
        )

    def health(self: Any) -> dict[str, Any]:
        payload = original_health(self)

        payload["entry_round_trip_guard"] = {
            "version": "1.60.13",
            "enabled": True,
            "adapter_supported": _supported(self),
            "minimum_cost_headroom_bps": (
                MIN_COST_HEADROOM_BPS
            ),
            "exit_stress_bps": EXIT_STRESS_BPS,
            "modeled_round_trip_cost_floor_bps": (
                MODELED_ROUND_TRIP_COST_FLOOR_BPS
            ),
            "liquidity_buffer_fraction": (
                LIQUIDITY_BUFFER
            ),
            "zero_fill_cooldown_seconds": (
                ZERO_FILL_COOLDOWN_SECONDS
            ),
            "preflight_blocks": int(
                self.state.get(
                    "v1613_entry_preflight_blocks"
                )
                or 0
            ),
            "terminal_zero_fill_buys": int(
                self.state.get(
                    "v1613_terminal_zero_fill_buys"
                )
                or 0
            ),
            "cooldown_skips": int(
                self.state.get(
                    "v1613_entry_cooldown_skips"
                )
                or 0
            ),
            "last_preflight_block": copy.deepcopy(
                self.state.get(
                    "v1613_last_entry_preflight_block"
                )
                or {}
            ),
            "last_terminal_zero_fill_buy": (
                copy.deepcopy(
                    self.state.get(
                        "v1613_last_terminal_zero_fill_buy"
                    )
                    or {}
                )
            ),
            "blocked_until": copy.deepcopy(
                self.state.get(
                    "v1613_entry_blocked_until"
                )
                or {}
            ),
            "buy_price_limit_preflight": True,
            "prospective_sell_limit_preflight": True,
            "ask_liquidity_preflight": True,
            "prospective_exit_minimum_preflight": True,
            "ambiguous_resubmission_allowed": False,
            "live_authority": False,
        }

        payload["live_authority"] = False
        return payload

    def hyper_submit(
        self: Any,
        pending: dict[str, Any],
        now: float,
    ) -> dict[str, Any]:
        if (
            str(pending.get("kind") or "")
            == "entry"
            and _supported(self.testnet)
        ):
            row = copy.deepcopy(pending)

            event, normalization = (
                _normalize_buy(
                    self.testnet,
                    row.get("event") or {},
                )
            )

            row["event"] = event

            assessment = dict(
                row.get("assessment") or {}
            )

            normalized_notional = _n(
                normalization.get(
                    "normalized_notional_usd"
                )
            )

            if normalized_notional > 0.0:
                assessment[
                    "order_notional_usd"
                ] = normalized_notional

            assessment[
                "v1613_entry_normalization"
            ] = copy.deepcopy(
                normalization
            )

            row["assessment"] = assessment

            return original_hyper_submit(
                self,
                row,
                now,
            )

        return original_hyper_submit(
            self,
            pending,
            now,
        )

    def hyper_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = original_hyper_health(self)

        try:
            snapshot = (
                self.testnet.safe_snapshot()
            )
        except Exception:
            snapshot = {}

        payload["version"] = "1.60.13"

        payload[
            "entry_round_trip_guard"
        ] = copy.deepcopy(
            snapshot.get(
                "entry_round_trip_guard"
            )
            or {}
        )

        payload["live_authority"] = False
        return payload

    BybitTestnetExecutionEngine._mirror_event = (
        mirror_event
    )
    BybitTestnetExecutionEngine.health = health
    BybitTestnetExecutionEngine._merge_observed = (
        merge_observed
    )
    BybitTestnetExecutionEngine.VERSION = "2.9"
    BybitTestnetExecutionEngine._v1613_entry_roundtrip_installed = True

    HyperSpeedCollectiveTestnetLane._submit_pending = (
        hyper_submit
    )
    HyperSpeedCollectiveTestnetLane.health = (
        hyper_health
    )
    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.13"
    )
    VelocitySniperTestnetLane.VERSION = (
        "1.60.13"
    )
