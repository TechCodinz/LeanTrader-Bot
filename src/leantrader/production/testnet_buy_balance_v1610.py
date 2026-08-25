from __future__ import annotations

import copy
import datetime as dt
from typing import Any


BUY_QUOTE_RESERVE_FRACTION = 0.005
BUY_QUOTE_MINIMUM_RESERVE_USD = 0.01


def _number(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _required_buy_notional(
    self: Any,
    event: dict[str, Any],
) -> float:
    symbol = str(
        event.get("symbol") or ""
    ).upper()

    price = max(
        0.0,
        _number(event.get("price")),
    )

    requested_quantity = max(
        0.0,
        _number(event.get("quantity")),
    )

    if (
        not symbol
        or price <= 0.0
        or requested_quantity <= 0.0
    ):
        return 0.0

    market = self.exchange.market(symbol)

    limits = market.get("limits") or {}

    minimum_cost = max(
        0.0,
        _number(
            (
                limits.get("cost")
                or {}
            ).get("min")
        ),
    )

    minimum_amount = max(
        0.0,
        _number(
            (
                limits.get("amount")
                or {}
            ).get("min")
        ),
    )

    return max(
        requested_quantity * price,
        minimum_cost,
        minimum_amount * price,
    )


def _free_quote_balance(
    self: Any,
    quote: str,
) -> tuple[float | None, str | None]:
    balance = (
        self.state.get("account_balance")
        or {}
    )

    free = balance.get("free") or {}

    if quote not in free:
        return None, balance.get(
            "free_balance_source"
        )

    return (
        max(
            0.0,
            _number(free.get(quote)),
        ),
        balance.get(
            "free_balance_source"
        ),
    )


def _definitive_insufficient_balance(
    exc: Exception,
) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()

    return bool(
        "insufficientfunds" in name
        or "170131" in message
        or "insufficient balance" in message
    )


def _rollback_definitive_rejection(
    self: Any,
    *,
    client_id: str,
) -> dict[str, Any]:
    record = (
        self.state.get("orders")
        or {}
    ).get(client_id)

    if not isinstance(record, dict):
        raise RuntimeError(
            "definitive rejection has no "
            "persisted submission record"
        )

    if (
        record.get(
            "v1610_budget_rollback"
        )
        is not True
    ):
        attempted = max(
            0.0,
            _number(
                record.get(
                    "submitted_usd"
                )
            ),
        )

        self.state[
            "daily_order_count"
        ] = max(
            0,
            int(
                self.state.get(
                    "daily_order_count"
                )
                or 0
            )
            - 1,
        )

        self.state[
            "daily_submitted_usd"
        ] = max(
            0.0,
            _number(
                self.state.get(
                    "daily_submitted_usd"
                )
            )
            - attempted,
        )

        self.state[
            "daily_entry_order_count"
        ] = max(
            0,
            int(
                self.state.get(
                    "daily_entry_order_count"
                )
                or 0
            )
            - 1,
        )

        self.state[
            "daily_entry_submitted_usd"
        ] = max(
            0.0,
            _number(
                self.state.get(
                    "daily_entry_submitted_usd"
                )
            )
            - attempted,
        )

        record[
            "attempted_submission_usd"
        ] = attempted

        # Do not let restart-time budget reconstruction
        # count an exchange-definitive rejection as a
        # submitted/accepted Testnet order.
        record["submitted_usd"] = 0.0
        record[
            "submission_attempt_at"
        ] = record.get("submitted_at")
        record["submitted_at"] = None

        record[
            "v1610_budget_rollback"
        ] = True

    record["status"] = "rejected"
    record[
        "skip_reason"
    ] = (
        "insufficient_free_quote_balance"
    )
    record["filled"] = 0.0
    record[
        "decision_at"
    ] = dt.datetime.now(
        dt.UTC
    ).isoformat()
    record[
        "exchange_reject_code"
    ] = 170131
    record[
        "reconciliation_resolution"
    ] = (
        "definitive_bybit_"
        "insufficient_balance_rejection"
    )

    self.state[
        "buy_insufficient_balance_rejections"
    ] = (
        int(
            self.state.get(
                "buy_insufficient_balance_rejections"
            )
            or 0
        )
        + 1
    )

    self._save_state()

    return {
        "client_order_id": client_id,
        "idempotent": False,
        **self._public_record(record),
    }


def install_testnet_buy_balance_v1610() -> None:
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
        "_v1610_buy_balance_installed",
        False,
    ):
        return

    original_mirror = (
        BybitTestnetExecutionEngine
        ._mirror_event
    )

    original_health = (
        BybitTestnetExecutionEngine
        .health
    )

    original_hyper_health = (
        HyperSpeedCollectiveTestnetLane
        .health
    )

    def mirror_event(
        self: Any,
        event: dict[str, Any],
    ) -> dict[str, Any]:
        side = str(
            event.get("side") or ""
        ).lower()

        if side != "buy":
            return original_mirror(
                self,
                event,
            )

        symbol = str(
            event.get("symbol") or ""
        ).upper()

        client_id = self._client_order_id(
            event
        )

        # Preserve idempotency/reconciliation semantics for
        # an already-known exchange identity.
        if (
            client_id
            in (
                self.state.get(
                    "orders"
                )
                or {}
            )
        ):
            return original_mirror(
                self,
                event,
            )

        # Preserve canonical skip ordering for safety/cap
        # conditions already handled by v1.60.8.
        if (
            symbol
            not in self._eligible_symbols
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
            or _number(
                self.state.get(
                    "daily_entry_submitted_usd"
                )
            )
            >= self.max_daily_submitted_usd
        ):
            return original_mirror(
                self,
                event,
            )

        required = (
            _required_buy_notional(
                self,
                event,
            )
        )

        quote = (
            symbol.split("/", 1)[1]
            if "/" in symbol
            else "USDT"
        )

        free_quote, source = (
            _free_quote_balance(
                self,
                quote,
            )
        )

        if free_quote is None:
            self.state[
                "buy_balance_preflight_skips"
            ] = (
                int(
                    self.state.get(
                        "buy_balance_preflight_skips"
                    )
                    or 0
                )
                + 1
            )

            result = self._skip(
                client_id,
                symbol,
                side,
                "free_quote_balance_unavailable",
            )

            self._save_state()
            return result

        reserve = max(
            BUY_QUOTE_MINIMUM_RESERVE_USD,
            required
            * BUY_QUOTE_RESERVE_FRACTION,
        )

        usable = max(
            0.0,
            free_quote - reserve,
        )

        if required > usable + 1e-12:
            self.state[
                "buy_balance_preflight_skips"
            ] = (
                int(
                    self.state.get(
                        "buy_balance_preflight_skips"
                    )
                    or 0
                )
                + 1
            )

            result = self._skip(
                client_id,
                symbol,
                side,
                "insufficient_free_quote_balance_preflight",
            )

            record = (
                self.state.get("orders")
                or {}
            ).get(client_id)

            if isinstance(record, dict):
                record[
                    "required_quote_usd"
                ] = required
                record[
                    "free_quote_usd"
                ] = free_quote
                record[
                    "quote_reserve_usd"
                ] = reserve
                record[
                    "free_balance_source"
                ] = source
                self._save_state()

            return result

        try:
            return original_mirror(
                self,
                event,
            )

        except Exception as exc:
            if not (
                _definitive_insufficient_balance(
                    exc
                )
            ):
                raise

            # Bybit 170131 is a definitive exchange
            # rejection, not an ambiguous network state.
            # It is therefore safe to classify terminally,
            # undo the false entry-budget consumption and
            # allow the 0.5-second lane to move on.
            return (
                _rollback_definitive_rejection(
                    self,
                    client_id=client_id,
                )
            )

    def engine_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = original_health(self)

        quote_balance = (
            self.state.get(
                "account_balance"
            )
            or {}
        )

        free = (
            quote_balance.get("free")
            or {}
        )

        payload[
            "buy_balance_guard"
        ] = {
            "enabled": True,
            "fresh_reconciliation_balance": True,
            "quote_asset": "USDT",
            "free_quote_balance_usd": (
                _number(
                    free.get("USDT")
                )
                if "USDT" in free
                else None
            ),
            "free_balance_source": (
                quote_balance.get(
                    "free_balance_source"
                )
            ),
            "reserve_fraction": (
                BUY_QUOTE_RESERVE_FRACTION
            ),
            "minimum_reserve_usd": (
                BUY_QUOTE_MINIMUM_RESERVE_USD
            ),
            "preflight_skips": int(
                self.state.get(
                    "buy_balance_preflight_skips"
                )
                or 0
            ),
            "definitive_insufficient_balance_rejections": int(
                self.state.get(
                    "buy_insufficient_balance_rejections"
                )
                or 0
            ),
            "definitive_rejection_releases_entry_budget": True,
            "ambiguous_network_failure_resubmission_allowed": False,
            "live_authority": False,
        }

        payload[
            "live_authority"
        ] = False

        return payload

    def hyper_health(
        self: Any,
    ) -> dict[str, Any]:
        payload = original_hyper_health(
            self
        )

        try:
            testnet = (
                self.testnet.safe_snapshot()
            )
        except Exception:
            testnet = {}

        payload["version"] = "1.60.10"
        payload[
            "buy_balance_guard"
        ] = copy.deepcopy(
            testnet.get(
                "buy_balance_guard"
            )
            or {}
        )
        payload[
            "insufficient_balance_entry_nonblocking"
        ] = True
        payload[
            "live_authority"
        ] = False

        return payload

    BybitTestnetExecutionEngine._mirror_event = (
        mirror_event
    )
    BybitTestnetExecutionEngine.health = (
        engine_health
    )
    BybitTestnetExecutionEngine.VERSION = "2.7"

    HyperSpeedCollectiveTestnetLane.health = (
        hyper_health
    )
    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.10"
    )
    VelocitySniperTestnetLane.VERSION = (
        "1.60.10"
    )

    BybitTestnetExecutionEngine._v1610_buy_balance_installed = True
