from __future__ import annotations

import copy
from typing import Any

from .testnet_exit_price_guard_v1611 import _fresh_bid


MIN_FRESH_PROFIT_NET_BPS = 5.0

PROFIT_EXIT_REASONS = frozenset(
    {
        "take_profit",
        "velocity_take_profit",
        "velocity_trailing_profit",
        "velocity_trailing_take_profit",
        "velocity_profit_decay",
        "profit_decay",
    }
)


def _n(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _queue_exit_reason(
    queue: dict[str, Any],
) -> str:
    assessment = (
        queue.get("assessment")
        or {}
    )

    reason = str(
        assessment.get("exit_reason")
        or assessment.get("reason")
        or ""
    )

    if reason:
        return reason

    source_reason = str(
        (
            queue.get("source_event")
            or {}
        ).get("reason")
        or ""
    )

    for candidate in (
        PROFIT_EXIT_REASONS
    ):
        if candidate in source_reason:
            return candidate

    return ""


def install_testnet_stale_profit_exit_revalidation_v1617() -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )
    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_v1617_stale_profit_exit_revalidation_installed",
        False,
    ):
        return

    original_manage = (
        HyperSpeedCollectiveTestnetLane._manage_active
    )

    original_health = (
        HyperSpeedCollectiveTestnetLane.health
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

        with self._lock:
            queue = copy.deepcopy(
                (
                    self.state.get(
                        "deferred_exit_recoveries"
                    )
                    or {}
                ).get(normalized)
            )

        if not isinstance(
            queue,
            dict,
        ):
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        queued_reason = (
            _queue_exit_reason(
                queue
            )
        )

        if (
            queued_reason
            not in PROFIT_EXIT_REASONS
        ):
            # Protective stop-loss, reversal,
            # liquidity/risk and time exits retain
            # their existing behavior.
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        entry_price = max(
            0.0,
            _n(
                record.get(
                    "entry_price"
                )
            ),
        )

        try:
            fresh_bid, fresh_ask = (
                _fresh_bid(
                    self.testnet,
                    normalized,
                )
            )
        except Exception:
            # Cannot prove the old profit intent
            # stale, so leave the existing
            # fail-closed watcher untouched.
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        if (
            entry_price <= 0.0
            or fresh_bid <= 0.0
        ):
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        fresh_gross_bps = (
            (
                fresh_bid
                / entry_price
                - 1.0
            )
            * 10_000.0
        )

        modeled_cost_floor = max(
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

        minimum_valid_gross_bps = (
            modeled_cost_floor
            + MIN_FRESH_PROFIT_NET_BPS
        )

        if (
            fresh_gross_bps
            + 1e-12
            >= minimum_valid_gross_bps
        ):
            # Still demonstrates positive modeled
            # profit. Existing v1.60.15 boundary
            # and execution safety remain canonical.
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        retired = False

        with self._lock:
            live_queue = (
                self.state.get(
                    "deferred_exit_recoveries"
                )
                or {}
            ).get(normalized)

            if isinstance(
                live_queue,
                dict,
            ):
                live_reason = (
                    _queue_exit_reason(
                        live_queue
                    )
                )

                if (
                    live_reason
                    in PROFIT_EXIT_REASONS
                ):
                    (
                        self.state.get(
                            "deferred_exit_recoveries"
                        )
                        or {}
                    ).pop(
                        normalized,
                        None,
                    )

                    retired = True

            if retired:
                self.state[
                    "v1617_stale_profit_exit_retirements"
                ] = (
                    int(
                        self.state.get(
                            "v1617_stale_profit_exit_retirements"
                        )
                        or 0
                    )
                    + 1
                )

                self.state[
                    "v1617_last_stale_profit_exit_retirement"
                ] = {
                    "symbol": normalized,
                    "queued_exit_reason": (
                        queued_reason
                    ),
                    "entry_price": (
                        entry_price
                    ),
                    "fresh_bid": fresh_bid,
                    "fresh_ask": fresh_ask,
                    "fresh_gross_bps": (
                        fresh_gross_bps
                    ),
                    "modeled_round_trip_cost_floor_bps": (
                        modeled_cost_floor
                    ),
                    "minimum_fresh_profit_net_bps": (
                        MIN_FRESH_PROFIT_NET_BPS
                    ),
                    "minimum_valid_gross_bps": (
                        minimum_valid_gross_bps
                    ),
                    "position_remains_active": True,
                    "order_submitted": False,
                    "fresh_sentinel_reassessment_required": True,
                    "live_authority": False,
                    "observed_at": now,
                }

                self._save_locked()

        if not retired:
            return original_manage(
                self,
                service,
                snapshot,
                symbol,
                record,
                now=now,
            )

        return self._decision(
            "stale_profit_exit_retired_for_reassessment",
            details={
                "kind": "exit",
                "symbol": normalized,
                "queued_exit_reason": (
                    queued_reason
                ),
                "entry_price": entry_price,
                "fresh_bid": fresh_bid,
                "fresh_ask": fresh_ask,
                "fresh_gross_bps": (
                    fresh_gross_bps
                ),
                "modeled_round_trip_cost_floor_bps": (
                    modeled_cost_floor
                ),
                "minimum_fresh_profit_net_bps": (
                    MIN_FRESH_PROFIT_NET_BPS
                ),
                "minimum_valid_gross_bps": (
                    minimum_valid_gross_bps
                ),
                "queue_retired": True,
                "order_submitted": False,
                "position_remains_active": True,
                "fresh_sentinel_reassessment_required": True,
                "live_authority": False,
            },
        )

    def health(
        self: Any,
    ) -> dict[str, Any]:
        payload = (
            original_health(
                self
            )
        )

        with self._lock:
            last = copy.deepcopy(
                self.state.get(
                    "v1617_last_stale_profit_exit_retirement"
                )
                or {}
            )

            count = int(
                self.state.get(
                    "v1617_stale_profit_exit_retirements"
                )
                or 0
            )

        payload["version"] = (
            "1.60.17"
        )

        payload[
            "stale_profit_exit_revalidation"
        ] = {
            "version": "1.60.17",
            "enabled": True,
            "profit_exit_reasons": sorted(
                PROFIT_EXIT_REASONS
            ),
            "minimum_fresh_profit_net_bps": (
                MIN_FRESH_PROFIT_NET_BPS
            ),
            "modeled_round_trip_cost_floor_bps": max(
                30.0,
                _n(
                    getattr(
                        self,
                        "round_trip_cost_bps",
                        30.0,
                    ),
                    30.0,
                ),
            ),
            "retire_stale_profit_intent_without_closing_position": True,
            "protective_exit_reasons_preserved": True,
            "stale_profit_order_submission_allowed": False,
            "fresh_sentinel_reassessment_required": True,
            "retirements": count,
            "last_retirement": last,
            "live_authority": False,
        }

        payload[
            "live_authority"
        ] = False

        return payload

    HyperSpeedCollectiveTestnetLane._manage_active = (
        manage_active
    )

    HyperSpeedCollectiveTestnetLane.health = (
        health
    )

    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.17"
    )

    VelocitySniperTestnetLane.VERSION = (
        "1.60.17"
    )

    HyperSpeedCollectiveTestnetLane._v1617_stale_profit_exit_revalidation_installed = (
        True
    )
