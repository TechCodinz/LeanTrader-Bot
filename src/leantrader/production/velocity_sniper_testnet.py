from __future__ import annotations

from typing import Any

from .fast_collective_hyper import (
    HyperSpeedCollectiveTestnetLane,
)


class VelocitySniperTestnetLane(
    HyperSpeedCollectiveTestnetLane
):
    """Fresh sub-second velocity exploration for Bybit Testnet only."""

    VERSION = "1.58.1"

    @classmethod
    def _velocity_state(
        cls,
        signal: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot = (
            signal.get("micro_velocity")
            or {}
        )

        def number(
            key: str,
            default: float = 0.0,
        ) -> float:
            return cls._number(
                snapshot.get(key),
                default,
            )

        age = number(
            "age_seconds",
            1_000_000.0,
        )
        midpoint = number("midpoint")
        spread = number(
            "spread_bps",
            1_000_000.0,
        )
        bid_depth = number(
            "bid_depth_usd"
        )
        ask_depth = number(
            "ask_depth_usd"
        )
        depth = bid_depth + ask_depth
        samples = int(
            number("temporal_samples")
        )
        velocity = number(
            "midpoint_velocity_bps_per_second"
        )
        acceleration = number(
            "midpoint_acceleration_bps_per_second2"
        )
        trend = number(
            "recent_midpoint_trend_bps_5s"
        )
        movement_range = max(
            0.0,
            number(
                "recent_midpoint_range_bps_5s"
            ),
        )
        imbalance = number(
            "depth_imbalance"
        )
        microprice = number(
            "microprice_shift_bps"
        )
        persistence = max(
            0.0,
            min(
                1.0,
                number(
                    "pressure_persistence"
                ),
            ),
        )
        score = max(
            0.0,
            number("velocity_score"),
        )

        positive_burst = bool(
            (
                trend >= 8.0
                and velocity >= 0.50
            )
            or (
                velocity >= 2.0
                and acceleration >= 0.15
            )
            or (
                trend >= 15.0
                and microprice >= 0.0
            )
        )

        qualified_long = bool(
            snapshot.get("fresh") is True
            and age <= 2.0
            and midpoint > 0.0
            and spread <= 20.0
            and depth >= 10_000.0
            and samples >= 3
            and movement_range >= 4.0
            and positive_burst
            and imbalance > -0.25
            and microprice > -2.0
        )

        projected_capture = max(
            0.0,
            trend,
            velocity * 5.0
            + max(
                0.0,
                acceleration
            )
            * 4.0,
        )

        return {
            "fresh": (
                snapshot.get("fresh")
                is True
            ),
            "qualified_long": (
                qualified_long
            ),
            "age_seconds": age,
            "price": midpoint,
            "spread_bps": spread,
            "depth_usd": depth,
            "temporal_samples": samples,
            "velocity_bps_per_second": (
                velocity
            ),
            "acceleration_bps_per_second2": (
                acceleration
            ),
            "trend_5s_bps": trend,
            "range_5s_bps": movement_range,
            "depth_imbalance": imbalance,
            "microprice_shift_bps": (
                microprice
            ),
            "pressure_persistence": (
                persistence
            ),
            "velocity_score": score,
            "projected_capture_bps_5s": (
                projected_capture
            ),
        }

    @classmethod
    def assess_candidate(
        cls,
        signal: dict[str, Any],
        supervisor_symbol: (
            dict[str, Any] | None
        ),
        *,
        relaxed: bool,
    ) -> dict[str, Any]:
        supervisor_symbol = (
            supervisor_symbol
            if isinstance(
                supervisor_symbol,
                dict,
            )
            else {}
        )

        base = super().assess_candidate(
            signal,
            supervisor_symbol,
            relaxed=relaxed,
        )

        velocity = cls._velocity_state(
            signal
        )

        # Existing cost-qualified or aligned collective signals stay
        # valid, but a genuine fresh velocity burst receives routing
        # priority.
        if base.get("allowed") is True:
            row = dict(base)
            row["velocity"] = velocity

            if velocity[
                "qualified_long"
            ]:
                row["velocity_sniper"] = True

                row["entry_mode"] = (
                    "velocity_sniper_"
                    + (
                        "cost_qualified"
                        if row.get(
                            "cost_qualified"
                        )
                        is True
                        else "probe"
                    )
                )

                row["decision_score"] = min(
                    1.0,
                    cls._number(
                        row.get(
                            "decision_score"
                        )
                    )
                    + min(
                        0.25,
                        velocity[
                            "velocity_score"
                        ]
                        / 120.0,
                    ),
                )

                groups = list(
                    row.get(
                        "support_groups"
                    )
                    or []
                )

                groups.append(
                    "subsecond_velocity_sniper"
                )

                row["support_groups"] = (
                    sorted(set(groups))
                )

            else:
                row["velocity_sniper"] = False

            return row

        # Never let the rapid exploration lane bypass its warmup.
        if (
            not relaxed
            or velocity[
                "qualified_long"
            ]
            is not True
        ):
            return base

        # Current short micro/MTF reversal remains a veto.
        if cls.strong_short_reversal(
            signal
        ):
            return {
                **base,
                "allowed": False,
                "reason": (
                    "velocity_short_conflict"
                ),
                "velocity": velocity,
            }

        route = (
            supervisor_symbol.get("route")
            or {}
        )

        temporal = (
            route.get("temporal_session")
            or {}
        )

        if (
            isinstance(temporal, dict)
            and temporal
            and temporal.get("allowed")
            is False
        ):
            return {
                **base,
                "allowed": False,
                "reason": (
                    "cached_temporal_safety_veto"
                ),
                "velocity": velocity,
            }

        protection = (
            route.get(
                "exchange_protection"
            )
            or {}
        )

        if (
            isinstance(
                protection,
                dict,
            )
            and protection
            and protection.get("allowed")
            is False
        ):
            return {
                **base,
                "allowed": False,
                "reason": (
                    "cached_exchange_protection_veto"
                ),
                "velocity": velocity,
            }

        route_reason = str(
            route.get("reason")
            or ""
        )

        if any(
            route_reason.startswith(prefix)
            for prefix in (
                cls.SAFETY_REASON_PREFIXES
            )
        ):
            return {
                **base,
                "allowed": False,
                "reason": (
                    "cached_governance_safety_veto"
                ),
                "velocity": velocity,
            }

        mtf_rows = [
            row
            for row in (
                signal.get(
                    "timeframe_assessments"
                )
                or {}
            ).values()
            if isinstance(row, dict)
        ]

        long_mtf_confidence = max(
            [
                cls._number(
                    row.get("confidence")
                )
                for row in mtf_rows
                if (
                    cls._direction(
                        row.get("direction")
                    )
                    == "long"
                    and cls._number(
                        row.get(
                            "expected_edge_bps"
                        )
                    )
                    > 0.0
                )
            ]
            or [0.0]
        )

        short_mtf_confidence = max(
            [
                cls._number(
                    row.get("confidence")
                )
                for row in mtf_rows
                if (
                    cls._direction(
                        row.get("direction")
                    )
                    == "short"
                    and cls._number(
                        row.get(
                            "expected_edge_bps"
                        )
                    )
                    > 0.0
                )
            ]
            or [0.0]
        )

        if (
            short_mtf_confidence >= 0.60
            and short_mtf_confidence
            > long_mtf_confidence + 0.05
        ):
            return {
                **base,
                "allowed": False,
                "reason": (
                    "velocity_mtf_short_conflict"
                ),
                "velocity": velocity,
            }

        cached_positive: list[str] = []

        if (
            cls._number(
                route.get("base_score")
            )
            >= 0.05
            and cls._number(
                route.get(
                    "base_confidence"
                )
            )
            >= 0.15
        ):
            cached_positive.append(
                "adaptive"
            )

        if (
            cls._number(
                route.get(
                    "advanced_score"
                )
            )
            >= 0.05
            and cls._number(
                route.get(
                    "advanced_confidence"
                )
            )
            >= 0.15
        ):
            cached_positive.append(
                "ultra_ensemble"
            )

        collective = (
            supervisor_symbol.get(
                "collective"
            )
            or {}
        )

        for group in (
            collective.get("groups")
            or []
        ):
            if not isinstance(
                group,
                dict,
            ):
                continue

            name = str(
                group.get("group")
                or ""
            )

            if name in {
                "fast_mtf",
                "microstructure",
            }:
                continue

            if (
                cls._number(
                    group.get("score")
                )
                >= 0.05
                and cls._number(
                    group.get(
                        "confidence"
                    )
                )
                >= 0.15
            ):
                cached_positive.append(
                    name
                )

        support_groups = [
            "subsecond_velocity_sniper",
        ]

        if long_mtf_confidence >= 0.50:
            support_groups.append(
                "multi_timeframe_minds"
            )

        support_groups.extend(
            cached_positive
        )

        # Velocity may originate only in Testnet exploration when at least
        # one independent slower intelligence family agrees. It never
        # becomes real-money authority.
        if (
            long_mtf_confidence < 0.50
            and not cached_positive
        ):
            return {
                **base,
                "allowed": False,
                "reason": (
                    "velocity_without_collective_support"
                ),
                "velocity": velocity,
            }

        ranked = (
            signal.get(
                "ranked_opportunity"
            )
            or {}
        )

        quality = max(
            0.0,
            min(
                1.0,
                cls._number(
                    ranked.get(
                        "quality_multiplier"
                    )
                ),
            ),
        )

        decision_score = min(
            0.98,
            0.45
            + min(
                0.25,
                velocity[
                    "velocity_score"
                ]
                / 100.0,
            )
            + (
                0.10
                if long_mtf_confidence
                >= 0.55
                else 0.0
            )
            + min(
                0.10,
                0.03
                * len(
                    set(
                        cached_positive
                    )
                ),
            ),
        )

        return {
            "allowed": True,
            "reason": (
                "subsecond_velocity_exploration"
            ),
            "entry_mode": (
                "velocity_sniper_probe"
            ),
            "velocity_sniper": True,
            "price": velocity["price"],
            "spread_bps": (
                velocity["spread_bps"]
            ),
            "quality": quality,
            "decision_score": (
                decision_score
            ),
            "micro_confidence": min(
                0.95,
                0.25
                + velocity[
                    "velocity_score"
                ]
                / 120.0,
            ),
            "mtf_confidence": (
                long_mtf_confidence
            ),
            "support_groups": sorted(
                set(support_groups)
            ),
            "cached_contributors": sorted(
                set(cached_positive)
            ),
            "micro_support": [],
            "mtf_support": [
                row
                for row in mtf_rows
                if (
                    cls._direction(
                        row.get("direction")
                    )
                    == "long"
                )
            ][:4],
            "velocity": velocity,
            "cost_qualified": False,
            "modeled_round_trip_cost_bps": (
                30.0
            ),
            "proven_positive_net_edge": False,
            "testnet_exploration_authority": (
                True
            ),
            "live_authority": False,
        }

    def health(self) -> dict[str, Any]:
        payload = super().health()

        payload.update(
            {
                "version": self.VERSION,
                "velocity_sniper": True,
                "precision_cadence_seconds": (
                    self.cadence_seconds
                ),
                "subsecond_detection": (
                    self.cadence_seconds
                    < 1.0
                ),
                "velocity_acceleration_routing": (
                    True
                ),
                "microstream_priority": True,
                "adaptive_fast_exit": True,
                "live_authority": False,
            }
        )

        return payload
