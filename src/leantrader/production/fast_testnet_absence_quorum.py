from __future__ import annotations

import datetime as dt
import time
from typing import Any


FAST_ABSENCE_QUORUM_ROUNDS = 3
FAST_ABSENCE_QUORUM_DELAY_SECONDS = 0.5
FAST_ABSENCE_QUORUM_SOURCES = (
    "realtime_order",
    "order_history",
    "execution_history",
)


def _reset_absence_quorum(
    record: dict[str, Any],
    reason: str,
) -> None:
    record["fast_absence_quorum_consecutive_rounds"] = 0
    record["fast_absence_quorum_last_reset_reason"] = reason
    record["fast_absence_quorum_last_reset_at"] = (
        dt.datetime.now(dt.UTC).isoformat()
    )


def _recover_order_with_fast_absence_quorum(
    self: Any,
    record: dict[str, Any],
    client_id: str,
) -> dict[str, Any] | None:
    """Resolve recent Testnet ambiguity through a short fail-closed quorum.

    The same orderLinkId is queried repeatedly. This method never creates or
    resubmits an order. A recent ambiguous submission therefore does not need
    to wait five minutes before authoritative absence can be established.
    """

    for round_number in range(1, FAST_ABSENCE_QUORUM_ROUNDS + 1):
        if round_number > 1:
            self.reconciliation_retry_attempts += 1
            time.sleep(
                FAST_ABSENCE_QUORUM_DELAY_SECONDS
                * (round_number - 1)
            )
            self._verify_testnet_urls()

        observed = self._recover_order(
            record,
            client_id,
        )

        if observed is not None:
            if round_number > 1:
                self.reconciliation_retry_successes += 1
                record["automatic_reconciliation_retries"] = (
                    round_number - 1
                )
            return observed

    return None


def _recover_native_bybit_client_order_quorum(
    self: Any,
    record: dict[str, Any],
    symbol: str,
    client_id: str,
) -> dict[str, Any] | None:
    """Reconcile one orderLinkId using three independent Bybit V5 sources.

    Absence is authoritative only after realtime orders, order history and
    execution history all return successful negative results for multiple
    consecutive rounds. Any endpoint failure resets the negative streak and
    therefore keeps the executor fail-closed. Positive evidence is never
    converted into absence and the ambiguous order is never resubmitted.
    """

    try:
        market = self.exchange.market(symbol)
    except Exception:
        market = {}

    market_id = str(
        (market or {}).get("id")
        or symbol.replace("/", "")
    ).upper()

    params = {
        "category": "spot",
        "symbol": market_id,
        "orderLinkId": client_id,
    }

    source_success = {
        source: False
        for source in FAST_ABSENCE_QUORUM_SOURCES
    }

    for source, names in (
        (
            "realtime_order",
            (
                "private_get_v5_order_realtime",
                "privateGetV5OrderRealtime",
            ),
        ),
        (
            "order_history",
            (
                "private_get_v5_order_history",
                "privateGetV5OrderHistory",
            ),
        ),
    ):
        response = self._call_native_bybit(
            names,
            params,
        )

        if (
            response is None
            or not self._native_bybit_response_ok(response)
        ):
            continue

        source_success[source] = True

        for raw in self._native_bybit_rows(response):
            observed_link_id = str(
                raw.get("orderLinkId")
                or raw.get("clientOrderId")
                or ""
            )

            if observed_link_id != client_id:
                continue

            parsed = self._parse_native_bybit_order(
                raw,
                symbol,
                client_id,
            )

            if parsed is not None:
                _reset_absence_quorum(
                    record,
                    f"positive_{source}",
                )
                record["reconciliation_resolution"] = (
                    "native_bybit_order_link_id"
                )
                return parsed

    execution = self._call_native_bybit(
        (
            "private_get_v5_execution_list",
            "privateGetV5ExecutionList",
        ),
        params,
    )

    if (
        execution is not None
        and self._native_bybit_response_ok(execution)
    ):
        source_success["execution_history"] = True

        executions = [
            row
            for row in self._native_bybit_rows(execution)
            if str(
                row.get("orderLinkId")
                or ""
            ) == client_id
        ]

        if executions:
            quantity = sum(
                float(row.get("execQty") or 0.0)
                for row in executions
            )
            cost = sum(
                float(row.get("execValue") or 0.0)
                for row in executions
            )

            order_id = next(
                (
                    str(row.get("orderId"))
                    for row in executions
                    if row.get("orderId")
                ),
                "",
            )

            average = (
                cost / quantity
                if quantity > 0 and cost > 0
                else float(
                    executions[-1].get("execPrice")
                    or record.get("reference_price")
                    or 0.0
                )
            )

            _reset_absence_quorum(
                record,
                "positive_execution_history",
            )
            record["reconciliation_resolution"] = (
                "native_bybit_execution_link_id"
            )

            return {
                "id": order_id or None,
                "clientOrderId": client_id,
                "symbol": symbol,
                "side": record.get("side"),
                "status": "open",
                "filled": quantity,
                "average": average,
                "cost": cost,
                "info": {
                    "orderLinkId": client_id,
                },
            }

    if not all(source_success.values()):
        missing = ",".join(
            source
            for source, succeeded in source_success.items()
            if not succeeded
        )
        _reset_absence_quorum(
            record,
            f"endpoint_failure:{missing}",
        )
        return None

    streak = int(
        record.get(
            "fast_absence_quorum_consecutive_rounds",
            0,
        )
        or 0
    ) + 1

    record["fast_absence_quorum_consecutive_rounds"] = streak
    record["fast_absence_quorum_required_rounds"] = (
        FAST_ABSENCE_QUORUM_ROUNDS
    )
    record["fast_absence_quorum_sources"] = list(
        FAST_ABSENCE_QUORUM_SOURCES
    )
    record["fast_absence_quorum_last_negative_at"] = (
        dt.datetime.now(dt.UTC).isoformat()
    )

    if streak < FAST_ABSENCE_QUORUM_ROUNDS:
        return None

    record["reconciliation_resolution"] = (
        "native_bybit_fast_authoritative_absence_quorum"
    )

    return {
        "id": None,
        "clientOrderId": client_id,
        "symbol": symbol,
        "side": record.get("side"),
        "status": "rejected",
        "filled": 0.0,
        "cost": 0.0,
        "info": {
            "orderLinkId": client_id,
            "reconciliation": (
                "fast_authoritative_exchange_absence_quorum"
            ),
            "negative_quorum_rounds": streak,
        },
    }


def install_fast_testnet_absence_quorum() -> None:
    """Install the v1.60.7 reconciliation safety upgrade idempotently."""

    from .testnet_execution import BybitTestnetExecutionEngine

    if getattr(
        BybitTestnetExecutionEngine,
        "_fast_absence_quorum_v1607_installed",
        False,
    ):
        return

    original_health = BybitTestnetExecutionEngine.health

    def health_with_fast_absence_quorum(
        self: Any,
    ) -> dict[str, Any]:
        snapshot = original_health(self)
        recovery = dict(
            snapshot.get(
                "automatic_reconciliation_recovery",
                {},
            )
        )
        recovery.update(
            {
                "old_ambiguity_only": False,
                "maximum_retries": (
                    FAST_ABSENCE_QUORUM_ROUNDS - 1
                ),
                "resubmission_allowed": False,
                "fail_closed": True,
                "fast_absence_quorum": {
                    "enabled": True,
                    "required_consecutive_rounds": (
                        FAST_ABSENCE_QUORUM_ROUNDS
                    ),
                    "retry_delays_seconds": [
                        FAST_ABSENCE_QUORUM_DELAY_SECONDS,
                        FAST_ABSENCE_QUORUM_DELAY_SECONDS * 2,
                    ],
                    "authoritative_sources": list(
                        FAST_ABSENCE_QUORUM_SOURCES
                    ),
                    "fail_closed_on_endpoint_error": True,
                    "resubmission_allowed": False,
                },
            }
        )
        snapshot[
            "automatic_reconciliation_recovery"
        ] = recovery
        return snapshot

    BybitTestnetExecutionEngine._recover_order_with_bounded_retry = (
        _recover_order_with_fast_absence_quorum
    )
    BybitTestnetExecutionEngine._recover_native_bybit_client_order = (
        _recover_native_bybit_client_order_quorum
    )
    BybitTestnetExecutionEngine.health = (
        health_with_fast_absence_quorum
    )
    BybitTestnetExecutionEngine._fast_absence_quorum_v1607_installed = True
