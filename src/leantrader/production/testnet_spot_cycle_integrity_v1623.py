from __future__ import annotations

import copy
import datetime as dt
from typing import Any

from .testnet_exit_price_guard_v1611 import _fresh_bid
from .testnet_price_limit_edge_exit_v1615 import _base_exit_reason


def _n(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def install_testnet_spot_cycle_integrity_v1623() -> None:
    from .fast_collective_hyper import HyperSpeedCollectiveTestnetLane
    from .testnet_execution import BybitTestnetExecutionEngine
    from .velocity_sniper_testnet import VelocitySniperTestnetLane

    if getattr(
        BybitTestnetExecutionEngine,
        "_v1623_spot_cycle_integrity_installed",
        False,
    ):
        return

    original_mirror = BybitTestnetExecutionEngine._mirror_event
    original_engine_health = BybitTestnetExecutionEngine.health
    original_manage = HyperSpeedCollectiveTestnetLane._manage_active
    original_lane_health = HyperSpeedCollectiveTestnetLane.health

    def mirror_event(
        self: Any,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        side = str(event.get("side") or "").lower()

        if side != "buy" or self.exchange is None:
            return original_mirror(self, event)

        exchange = self.exchange
        original_create = getattr(exchange, "create_order", None)
        quote_buy = getattr(
            exchange,
            "create_market_buy_order_with_cost",
            None,
        )

        if not callable(original_create):
            return original_mirror(self, event)

        def routed_create(
            symbol: str,
            order_type: str,
            order_side: str,
            amount: float,
            price: Any = None,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            params = dict(params or {})

            if (
                str(order_type).lower() != "market"
                or str(order_side).lower() != "buy"
            ):
                return original_create(
                    symbol,
                    order_type,
                    order_side,
                    amount,
                    price,
                    params,
                )

            client_id = str(
                params.get("orderLinkId") or ""
            )

            record = (
                self.state.get("orders") or {}
            ).get(client_id)

            if not isinstance(record, dict):
                raise RuntimeError(
                    "quote-cost market buy has no "
                    "persisted submission record"
                )

            cost = max(
                0.0,
                _n(record.get("submitted_usd")),
            )

            if cost <= 0.0:
                raise RuntimeError(
                    "quote-cost market buy has "
                    "invalid submitted notional"
                )

            record["submission_mode"] = (
                "quote_cost_market_buy"
            )
            record["requested_base_quantity"] = (
                _n(amount)
            )
            record["quote_cost_usd"] = cost

            self.state[
                "v1623_quote_cost_market_buy_attempts"
            ] = (
                int(
                    self.state.get(
                        "v1623_quote_cost_market_buy_attempts"
                    )
                    or 0
                )
                + 1
            )

            self._save_state()

            try:
                if not callable(quote_buy):
                    return original_create(
                        symbol,
                        order_type,
                        order_side,
                        amount,
                        price,
                        params,
                    )

                # Avoid recursion because CCXT's convenience
                # method may internally call create_order.
                exchange.create_order = original_create

                try:
                    return quote_buy(
                        symbol,
                        cost,
                        params,
                    )
                finally:
                    exchange.create_order = routed_create

            except Exception as exc:
                reason = self._redact(str(exc))

                record["submission_exception_type"] = (
                    type(exc).__name__
                )
                record["submission_exception_reason"] = (
                    reason[:500]
                )
                record["submission_exception_at"] = (
                    dt.datetime.now(dt.UTC).isoformat()
                )

                self.state[
                    "v1623_submission_exception_count"
                ] = (
                    int(
                        self.state.get(
                            "v1623_submission_exception_count"
                        )
                        or 0
                    )
                    + 1
                )

                self.state[
                    "v1623_last_submission_exception"
                ] = {
                    "symbol": str(symbol).upper(),
                    "side": "buy",
                    "client_order_id": client_id,
                    "submission_mode": (
                        "quote_cost_market_buy"
                    ),
                    "exception_type": type(exc).__name__,
                    "reason": reason[:500],
                    "observed_at": record[
                        "submission_exception_at"
                    ],
                    "live_authority": False,
                }

                self._save_state()
                raise

        exchange.create_order = routed_create

        try:
            return original_mirror(self, event)
        finally:
            exchange.create_order = original_create

    def manage_active(
        self: Any,
        service: Any,
        snapshot: dict[str, Any],
        symbol: str,
        record: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        normalized = str(symbol).upper()

        with self._lock:
            queue = copy.deepcopy(
                (
                    self.state.get(
                        "deferred_exit_recoveries"
                    )
                    or {}
                ).get(normalized)
            )

            # Stop the repeated :corrected_recycle suffix
            # from growing indefinitely.
            live_queue = (
                self.state.get(
                    "deferred_exit_recoveries"
                )
                or {}
            ).get(normalized)

            if isinstance(live_queue, dict):
                source = live_queue.get("source_event")

                if isinstance(source, dict):
                    before = str(
                        source.get("reason") or ""
                    )
                    after = _base_exit_reason(before)

                    if after != before:
                        source["reason"] = after
                        self._save_locked()

        if not isinstance(queue, dict):
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

        if exchange is None:
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        try:
            bid, _ask = _fresh_bid(
                self.testnet,
                normalized,
            )

            current = max(
                0.0,
                _n(
                    (
                        snapshot.get("positions")
                        or {}
                    ).get(normalized)
                ),
                _n(record.get("quantity")),
            )

            market = exchange.market(normalized)
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

            precise = max(
                0.0,
                _n(
                    exchange.amount_to_precision(
                        normalized,
                        current,
                    )
                ),
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
            current = 0.0

        if potential_dust:
            try:
                preparation = self.testnet.prepare_sell(
                    normalized,
                    current,
                    bid,
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
                with self._lock:
                    (
                        self.state.get("active")
                        or {}
                    ).pop(normalized, None)

                    (
                        self.state.get(
                            "deferred_exit_recoveries"
                        )
                        or {}
                    ).pop(normalized, None)

                    (
                        self.state.get(
                            "v1615_price_limit_watch"
                        )
                        or {}
                    ).pop(normalized, None)

                    rows = self.state.setdefault(
                        "v1623_preboundary_dust_"
                        "reclassifications",
                        [],
                    )

                    rows.append(
                        {
                            "symbol": normalized,
                            "recorded_at": now,
                            "preparation": copy.deepcopy(
                                preparation
                            ),
                            "counted_as_executed_close": False,
                            "live_authority": False,
                        }
                    )

                    self.state[
                        "v1623_preboundary_dust_"
                        "reclassifications"
                    ] = rows[-100:]

                    self._save_locked()

                return self._decision(
                    "active_exit_reclassified_"
                    "dust_preboundary",
                    details={
                        "kind": "exit",
                        "symbol": normalized,
                        "preparation": preparation,
                        "counted_as_executed_close": False,
                        "live_authority": False,
                    },
                )

        return original_manage(
            self,
            service,
            snapshot,
            symbol,
            record,
            now=now,
        )

    def engine_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = original_engine_health(self)

        payload["spot_market_buy_routing"] = {
            "version": "1.60.23",
            "quote_cost_capability": bool(
                callable(
                    getattr(
                        self.exchange,
                        "create_market_buy_order_with_cost",
                        None,
                    )
                )
            ),
            "quote_cost_market_buy_attempts": int(
                self.state.get(
                    "v1623_quote_cost_market_buy_attempts"
                )
                or 0
            ),
            "submission_exception_count": int(
                self.state.get(
                    "v1623_submission_exception_count"
                )
                or 0
            ),
            "last_submission_exception": (
                copy.deepcopy(
                    self.state.get(
                        "v1623_last_submission_exception"
                    )
                    or {}
                )
            ),
            "ambiguous_resubmission_allowed": False,
            "testnet_only": True,
            "live_authority": False,
        }

        payload["live_authority"] = False
        return payload

    def lane_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = original_lane_health(self)

        with self._lock:
            rows = copy.deepcopy(
                self.state.get(
                    "v1623_preboundary_dust_"
                    "reclassifications"
                )
                or []
            )

        payload["spot_cycle_integrity"] = {
            "version": "1.60.23",
            "dust_before_price_limit_watch": True,
            "preboundary_dust_count": len(rows),
            "preboundary_dust_reclassifications": rows[-20:],
            "corrected_recycle_reason_growth_bounded": True,
            "exchange_price_limit_bypass": False,
            "fake_close_allowed": False,
            "modeled_round_trip_cost_floor_bps": 30.0,
            "live_authority": False,
        }

        payload["live_authority"] = False
        return payload

    BybitTestnetExecutionEngine._mirror_event = mirror_event
    BybitTestnetExecutionEngine.health = engine_health

    HyperSpeedCollectiveTestnetLane._manage_active = (
        manage_active
    )
    HyperSpeedCollectiveTestnetLane.health = lane_health

    BybitTestnetExecutionEngine.VERSION = "3.3"
    HyperSpeedCollectiveTestnetLane.VERSION = "1.60.23"
    VelocitySniperTestnetLane.VERSION = "1.60.23"

    BybitTestnetExecutionEngine._v1623_spot_cycle_integrity_installed = True
