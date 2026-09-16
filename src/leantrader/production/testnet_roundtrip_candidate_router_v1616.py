from __future__ import annotations

import copy
import time
from typing import Any

from .testnet_entry_roundtrip_v1613 import (
    PREFLIGHT_COOLDOWN_SECONDS,
    _arm,
    _blocked_until,
    _normalize_buy,
    _preflight,
    _supported,
)

ROUTE_ERROR_COOLDOWN_SECONDS = 5.0
FILTER_TELEMETRY_SAVE_SECONDS = 5.0


def _n(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _local_blocked_until(
    lane: Any,
    symbol: str,
) -> float:
    with lane._lock:
        return max(
            0.0,
            _n(
                (
                    lane.state.get(
                        "v1616_local_blocked_until"
                    )
                    or {}
                ).get(symbol)
            ),
        )


def _route_blocked_until(
    lane: Any,
    symbol: str,
) -> float:
    return max(
        _blocked_until(
            lane.testnet,
            symbol,
        ),
        _local_blocked_until(
            lane,
            symbol,
        ),
    )


def _record_filter(
    lane: Any,
    *,
    now: float,
    filtered: list[dict[str, Any]],
) -> None:
    if not filtered:
        return

    with lane._lock:
        lane.state[
            "v1616_route_cooldown_filtered"
        ] = (
            int(
                lane.state.get(
                    "v1616_route_cooldown_filtered"
                )
                or 0
            )
            + len(filtered)
        )

        lane.state[
            "v1616_last_filtered_candidates"
        ] = copy.deepcopy(
            filtered[-24:]
        )

        last_saved = _n(
            lane.state.get(
                "v1616_last_filter_saved_at"
            )
        )

        if (
            last_saved <= 0.0
            or now - last_saved
            >= FILTER_TELEMETRY_SAVE_SECONDS
        ):
            lane.state[
                "v1616_last_filter_saved_at"
            ] = now
            lane._save_locked()


def _clear_matching_pending(
    lane: Any,
    pending: dict[str, Any],
) -> bool:
    event_id = str(
        (
            pending.get("event")
            or {}
        ).get("event_id")
        or ""
    )

    with lane._lock:
        current = lane.state.get(
            "pending_event"
        )

        if not isinstance(
            current,
            dict,
        ):
            return True

        current_id = str(
            (
                current.get("event")
                or {}
            ).get("event_id")
            or ""
        )

        if (
            event_id
            and current_id
            and event_id != current_id
        ):
            return False

        lane.state[
            "pending_event"
        ] = None

        lane._save_locked()

        return True


def _record_preflight(
    lane: Any,
    *,
    symbol: str,
    allowed: bool,
    reason: str,
    blocked_until: float,
    detail: dict[str, Any],
    now: float,
) -> None:
    with lane._lock:
        lane.state[
            "v1616_route_preflight_checks"
        ] = (
            int(
                lane.state.get(
                    "v1616_route_preflight_checks"
                )
                or 0
            )
            + 1
        )

        if allowed:
            lane.state[
                "v1616_route_preflight_passes"
            ] = (
                int(
                    lane.state.get(
                        "v1616_route_preflight_passes"
                    )
                    or 0
                )
                + 1
            )
        else:
            lane.state[
                "v1616_route_preflight_blocks"
            ] = (
                int(
                    lane.state.get(
                        "v1616_route_preflight_blocks"
                    )
                    or 0
                )
                + 1
            )

            reasons = lane.state.setdefault(
                "v1616_route_block_reasons",
                {},
            )

            reasons[reason] = (
                int(
                    reasons.get(reason)
                    or 0
                )
                + 1
            )

        lane.state[
            "v1616_last_route_preflight"
        ] = {
            "symbol": symbol,
            "allowed": bool(allowed),
            "reason": reason,
            "blocked_until": (
                blocked_until
                if blocked_until > 0.0
                else None
            ),
            "detail": copy.deepcopy(
                detail
            ),
            "observed_at": now,
            "executor_order_created": False,
            "live_authority": False,
        }

        lane._save_locked()


class _CandidateProxy:
    def __init__(
        self,
        service: Any,
        lane: Any,
        now: float,
    ) -> None:
        self._service = service
        self._lane = lane
        self._now = now

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        return getattr(
            self._service,
            name,
        )

    def collective_candidates(
        self,
        limit: int = 8,
    ) -> list[str]:
        requested = max(
            1,
            min(
                64,
                int(limit) + 8,
            ),
        )

        raw = list(
            self._service.collective_candidates(
                limit=requested
            )
            or []
        )

        allowed: list[str] = []
        filtered: list[
            dict[str, Any]
        ] = []

        seen: set[str] = set()

        for value in raw:
            symbol = str(
                value or ""
            ).upper()

            if (
                not symbol
                or symbol in seen
            ):
                continue

            seen.add(symbol)

            until = _route_blocked_until(
                self._lane,
                symbol,
            )

            if until > self._now:
                filtered.append(
                    {
                        "symbol": symbol,
                        "blocked_until": until,
                        "retry_in_seconds": max(
                            0.0,
                            until - self._now,
                        ),
                    }
                )
                continue

            allowed.append(symbol)

        _record_filter(
            self._lane,
            now=self._now,
            filtered=filtered,
        )

        return allowed[: max(
            1,
            int(limit),
        )]


def install_testnet_roundtrip_candidate_router_v1616() -> None:
    from .fast_collective_hyper import (
        HyperSpeedCollectiveTestnetLane,
    )
    from .velocity_sniper_testnet import (
        VelocitySniperTestnetLane,
    )

    if getattr(
        HyperSpeedCollectiveTestnetLane,
        "_v1616_roundtrip_candidate_router_installed",
        False,
    ):
        return

    original_step = (
        HyperSpeedCollectiveTestnetLane.step
    )
    original_submit = (
        HyperSpeedCollectiveTestnetLane._submit_pending
    )
    original_health = (
        HyperSpeedCollectiveTestnetLane.health
    )

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

        # Preserve the legacy/non-exchange Testnet lane exactly.
        # v1.60.16 only applies when the canonical v1.60.13
        # round-trip exchange preflight is actually available.
        if not _supported(self.testnet):
            return original_step(
                self,
                now=current,
            )

        provider = self.service_provider

        try:
            service = provider()
        except Exception:
            return original_step(
                self,
                now=current,
            )

        if service is None:
            return original_step(
                self,
                now=current,
            )

        proxy = _CandidateProxy(
            service,
            self,
            current,
        )

        self.service_provider = (
            lambda: proxy
        )

        try:
            return original_step(
                self,
                now=current,
            )
        finally:
            self.service_provider = (
                provider
            )

    def submit_pending(
        self: Any,
        pending: dict[str, Any],
        *,
        now: float,
    ) -> dict[str, Any]:
        if (
            str(
                pending.get("kind")
                or ""
            )
            != "entry"
            or not _supported(
                self.testnet
            )
        ):
            return original_submit(
                self,
                pending,
                now=now,
            )

        event = copy.deepcopy(
            pending.get("event")
            or {}
        )

        symbol = str(
            event.get("symbol")
            or ""
        ).upper()

        if not symbol:
            return original_submit(
                self,
                pending,
                now=now,
            )

        normalization: dict[
            str,
            Any,
        ] = {}

        try:
            normalized, normalization = (
                _normalize_buy(
                    self.testnet,
                    event,
                )
            )

            normalization_block = str(
                normalization.get(
                    "blocked_reason"
                )
                or ""
            )

            if normalization_block:
                preflight = {
                    "allowed": False,
                    "reason": (
                        normalization_block
                    ),
                    "normalization": (
                        copy.deepcopy(
                            normalization
                        )
                    ),
                }
            else:
                preflight = _preflight(
                    self.testnet,
                    normalized,
                )

        except Exception as exc:
            until = (
                time.time()
                + ROUTE_ERROR_COOLDOWN_SECONDS
            )

            with self._lock:
                self.state.setdefault(
                    "v1616_local_blocked_until",
                    {},
                )[symbol] = until
                self._save_locked()

            cleared = (
                _clear_matching_pending(
                    self,
                    pending,
                )
            )

            detail = {
                "error_type": (
                    type(exc).__name__
                ),
                "pending_cleared": cleared,
                "fail_closed": True,
            }

            _record_preflight(
                self,
                symbol=symbol,
                allowed=False,
                reason=(
                    "route_preflight_error"
                ),
                blocked_until=until,
                detail=detail,
                now=now,
            )

            return self._decision(
                "entry_route_preflight_blocked",
                details={
                    "kind": "entry",
                    "symbol": symbol,
                    "reason": (
                        "route_preflight_error"
                    ),
                    "retry_in_seconds": (
                        ROUTE_ERROR_COOLDOWN_SECONDS
                    ),
                    "executor_order_created": False,
                    "pending_cleared": cleared,
                    "live_authority": False,
                },
            )

        if (
            preflight.get("allowed")
            is True
        ):
            _record_preflight(
                self,
                symbol=symbol,
                allowed=True,
                reason=(
                    "round_trip_executable"
                ),
                blocked_until=0.0,
                detail={
                    "preflight": (
                        copy.deepcopy(
                            preflight
                        )
                    ),
                    "normalization": (
                        copy.deepcopy(
                            normalization
                        )
                    ),
                },
                now=now,
            )

            # The v1.60.13 executor path
            # deliberately runs the same
            # checks again immediately
            # before the real Testnet order.
            return original_submit(
                self,
                pending,
                now=now,
            )

        reason = str(
            preflight.get("reason")
            or "entry_round_trip_blocked"
        )

        if reason == "entry_cooldown":
            until = _blocked_until(
                self.testnet,
                symbol,
            )
        else:
            until = _arm(
                self.testnet,
                symbol,
                PREFLIGHT_COOLDOWN_SECONDS,
                reason,
                {
                    "preflight": (
                        copy.deepcopy(
                            preflight
                        )
                    ),
                    "normalization": (
                        copy.deepcopy(
                            normalization
                        )
                    ),
                    "route_preflight": True,
                    "executor_order_created": False,
                },
            )

        cleared = (
            _clear_matching_pending(
                self,
                pending,
            )
        )

        detail = {
            "preflight": (
                copy.deepcopy(
                    preflight
                )
            ),
            "normalization": (
                copy.deepcopy(
                    normalization
                )
            ),
            "pending_cleared": cleared,
        }

        _record_preflight(
            self,
            symbol=symbol,
            allowed=False,
            reason=reason,
            blocked_until=until,
            detail=detail,
            now=now,
        )

        return self._decision(
            "entry_route_preflight_blocked",
            details={
                "kind": "entry",
                "symbol": symbol,
                "reason": reason,
                "blocked_until": until,
                "retry_in_seconds": max(
                    0.0,
                    until - time.time(),
                ),
                "executor_order_created": False,
                "pending_cleared": cleared,
                "same_cycle_can_continue": (
                    cleared
                ),
                "live_authority": False,
            },
        )

    def health(
        self: Any,
    ) -> dict[str, Any]:
        payload = original_health(
            self
        )

        with self._lock:
            route = {
                "version": "1.60.16",
                "enabled": True,
                "preflight_checks": int(
                    self.state.get(
                        "v1616_route_preflight_checks"
                    )
                    or 0
                ),
                "preflight_passes": int(
                    self.state.get(
                        "v1616_route_preflight_passes"
                    )
                    or 0
                ),
                "preflight_blocks": int(
                    self.state.get(
                        "v1616_route_preflight_blocks"
                    )
                    or 0
                ),
                "cooldown_candidates_filtered": int(
                    self.state.get(
                        "v1616_route_cooldown_filtered"
                    )
                    or 0
                ),
                "block_reasons": (
                    copy.deepcopy(
                        self.state.get(
                            "v1616_route_block_reasons"
                        )
                        or {}
                    )
                ),
                "last_preflight": (
                    copy.deepcopy(
                        self.state.get(
                            "v1616_last_route_preflight"
                        )
                        or {}
                    )
                ),
                "last_filtered_candidates": (
                    copy.deepcopy(
                        self.state.get(
                            "v1616_last_filtered_candidates"
                        )
                        or []
                    )
                ),
                "blocked_candidate_executor_order_created": False,
                "executor_rechecks_before_real_order": True,
                "v1613_roundtrip_guard_preserved": True,
                "modeled_cost_floor_lowered": False,
                "price_limit_protection_lowered": False,
                "liquidity_protection_lowered": False,
                "ambiguous_resubmission_allowed": False,
                "live_authority": False,
            }

        payload[
            "roundtrip_candidate_router"
        ] = route
        payload["version"] = "1.60.16"
        payload["live_authority"] = False

        return payload

    HyperSpeedCollectiveTestnetLane.step = (
        step
    )
    HyperSpeedCollectiveTestnetLane._submit_pending = (
        submit_pending
    )
    HyperSpeedCollectiveTestnetLane.health = (
        health
    )

    HyperSpeedCollectiveTestnetLane.VERSION = (
        "1.60.16"
    )
    VelocitySniperTestnetLane.VERSION = (
        "1.60.16"
    )

    HyperSpeedCollectiveTestnetLane._v1616_roundtrip_candidate_router_installed = (
        True
    )
