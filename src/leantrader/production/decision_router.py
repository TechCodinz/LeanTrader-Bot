from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


class MarketEvidenceGate:
    """Classify symbol performance from closed, net-of-cost paper outcomes."""

    VERSION = "1.0"

    def __init__(self, state_path: Path, *, minimum_samples: int = 8, rolling_window: int = 50) -> None:
        if minimum_samples < 3:
            raise ValueError("market evidence minimum_samples must be at least 3")
        if rolling_window < minimum_samples:
            raise ValueError("market evidence rolling_window cannot be smaller than minimum_samples")
        self.state_path = state_path
        self.minimum_samples = minimum_samples
        self.rolling_window = rolling_window
        self.state = self._load()

    def assess(self, symbol: str) -> dict[str, Any]:
        record = self.state.setdefault("symbols", {}).setdefault(
            symbol,
            {"outcomes": [], "assessments": 0, "last_trade": None},
        )
        record["assessments"] = int(record.get("assessments", 0)) + 1
        outcomes = [float(value) for value in record.get("outcomes", [])][-self.rolling_window :]
        samples = len(outcomes)
        wins = [value for value in outcomes if value > 0]
        losses = [value for value in outcomes if value < 0]
        expectancy = float(np.mean(outcomes)) if outcomes else 0.0
        win_rate = len(wins) / samples if samples else 0.0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else (math.inf if wins else 0.0)

        if samples < self.minimum_samples:
            classification, allowed, multiplier = "exploration", True, 0.35
        elif expectancy > 0 and profit_factor >= 1.10 and win_rate >= 0.35:
            classification, allowed, multiplier = "qualified", True, 1.0
        elif expectancy >= -0.001 and profit_factor >= 0.90:
            classification, allowed, multiplier = "probation", True, 0.25
        else:
            # A quarantined market receives one small evidence probe every 100
            # assessments; otherwise it could never demonstrate recovery.
            probe = int(record["assessments"]) % 100 == 0
            classification = "quarantined_probe" if probe else "quarantined"
            allowed, multiplier = probe, 0.10 if probe else 0.0
        record["classification"] = classification
        return {
            "classification": classification,
            "allowed": allowed,
            "size_multiplier": multiplier,
            "samples": samples,
            "expectancy": expectancy,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "rolling_window": self.rolling_window,
        }

    def record(
        self,
        symbol: str,
        realized_return: float,
        predicted_probability: float | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(realized_return):
            raise ValueError("market evidence requires a finite realized return")
        if predicted_probability is not None and (
            not math.isfinite(predicted_probability) or not 0.0 <= predicted_probability <= 1.0
        ):
            raise ValueError("predicted probability must be finite and between zero and one")
        record = self.state.setdefault("symbols", {}).setdefault(
            symbol,
            {"outcomes": [], "assessments": 0, "last_trade": None},
        )
        outcomes = [float(value) for value in record.get("outcomes", [])]
        outcomes.append(float(realized_return))
        record["outcomes"] = outcomes[-self.rolling_window :]
        if predicted_probability is not None:
            forecasts = list(record.get("forecasts", []))
            forecasts.append(
                {
                    "probability": float(predicted_probability),
                    "outcome": int(realized_return > 0),
                }
            )
            record["forecasts"] = forecasts[-self.rolling_window :]
        record["last_trade"] = dt.datetime.now(dt.UTC).isoformat()
        self._save()
        return self.assess(symbol)

    def health(self) -> dict[str, Any]:
        classifications: dict[str, int] = {}
        samples = 0
        forecasts: list[dict[str, Any]] = []
        for record in self.state.get("symbols", {}).values():
            name = str(record.get("classification", "unassessed"))
            classifications[name] = classifications.get(name, 0) + 1
            samples += len(record.get("outcomes", []))
            forecasts.extend(record.get("forecasts", []))
        calibration: dict[str, Any] = {"samples": len(forecasts), "state": "waiting_for_closed_trades"}
        if forecasts:
            probabilities = np.asarray([row["probability"] for row in forecasts], dtype=float)
            outcomes = np.asarray([row["outcome"] for row in forecasts], dtype=float)
            brier = float(np.mean((probabilities - outcomes) ** 2))
            ece = 0.0
            for lower in np.linspace(0.0, 1.0, 10, endpoint=False):
                upper = lower + 0.1
                mask = (probabilities >= lower) & (
                    probabilities < upper if upper < 1.0 else probabilities <= upper
                )
                if mask.any():
                    ece += float(mask.mean()) * abs(
                        float(probabilities[mask].mean() - outcomes[mask].mean())
                    )
            calibration = {
                "samples": len(forecasts),
                "state": "measured",
                "brier_score": brier,
                "expected_calibration_error": ece,
            }
        return {
            "persistent": True,
            "state_path": str(self.state_path),
            "minimum_samples": self.minimum_samples,
            "rolling_window": self.rolling_window,
            "symbols_observed": len(self.state.get("symbols", {})),
            "closed_trade_samples": samples,
            "classifications": classifications,
            "cold_start": "bounded_exploration",
            "quarantine_retest": "one_in_100_assessments",
            "online_calibration": calibration,
        }

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 1, "symbols": {}}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == 1 and isinstance(payload.get("symbols"), dict):
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return {"schema_version": 1, "symbols": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)


class BoundedDecisionRouter:
    """Collective paper/Testnet decision fabric over LeanTrader intelligence.

    The router preserves the legacy adaptive + ultra path while allowing
    independently costed fast-swarm, microstructure and validated evolution
    evidence to originate a bounded paper/Testnet trade when multiple
    independent groups agree.

    Sensor, derivatives, liquidation, flow, options and cross-venue evidence
    contribute to the ensemble but cannot originate a trade without at least
    one independently costed positive-edge group.

    Real-money live authority is deliberately absent.
    """

    VERSION = "1.56.0"

    def __init__(
        self,
        evidence: MarketEvidenceGate,
        *,
        minimum_advanced_confidence: float = 0.20,
        minimum_combined_score: float = 0.20,
        negative_consensus_veto: float = -0.25,
    ) -> None:
        self.evidence = evidence
        self.minimum_advanced_confidence = (
            minimum_advanced_confidence
        )
        self.minimum_combined_score = (
            minimum_combined_score
        )
        self.negative_consensus_veto = (
            negative_consensus_veto
        )

        self.routes = 0
        self.approved = 0
        self.collective_origin_approved = 0
        self.collective_inputs_seen = 0
        self.blocked: dict[str, int] = {}
        self.last_routes: dict[
            str,
            dict[str, Any],
        ] = {}

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return (
            number
            if math.isfinite(number)
            else default
        )

    @classmethod
    def _clip(
        cls,
        value: Any,
        lower: float = -1.0,
        upper: float = 1.0,
    ) -> float:
        return float(
            np.clip(
                cls._number(value),
                lower,
                upper,
            )
        )

    @staticmethod
    def _direction(value: Any) -> float:
        normalized = str(value or "").lower()
        if normalized in {
            "long",
            "buy",
            "bull",
            "bullish",
        }:
            return 1.0
        if normalized in {
            "short",
            "sell",
            "bear",
            "bearish",
        }:
            return -1.0
        return 0.0

    @classmethod
    def _alpha_multiplier(
        cls,
        name: str,
        alpha: dict[str, Any],
    ) -> float:
        normalized = str(name or "").lower()
        if not normalized or not isinstance(
            alpha,
            dict,
        ):
            return 1.0

        tokens = {
            normalized,
            f"engine:{normalized}",
            f"collective:{normalized}",
        }

        if normalized.startswith(
            "timeframe_mind_"
        ):
            timeframe = normalized.split(
                "timeframe_mind_",
                1,
            )[1]
            tokens.add(f"timeframe:{timeframe}")

        for row in alpha.get("ranking") or []:
            if not isinstance(row, dict):
                continue
            strategy = str(
                row.get("strategy") or ""
            ).lower()
            if any(
                token in strategy
                for token in tokens
            ):
                return 1.10

        for row in (
            alpha.get("rejected_after_costs")
            or []
        ):
            if not isinstance(row, dict):
                continue
            strategy = str(
                row.get("strategy") or ""
            ).lower()
            if any(
                token in strategy
                for token in tokens
            ):
                return 0.85

        return 1.0

    def _collective_summary(
        self,
        collective: dict[str, Any] | None,
    ) -> dict[str, Any]:
        collective = (
            collective
            if isinstance(collective, dict)
            else {}
        )

        alpha = (
            collective.get("alpha_tournament")
            or {}
        )

        contributors: list[
            dict[str, Any]
        ] = []

        def add(
            *,
            name: str,
            group: str,
            score: float,
            confidence: float,
            origin_eligible: bool = False,
            net_edge_bps: float = 0.0,
            source: str,
        ) -> None:
            bounded_score = self._clip(score)
            bounded_confidence = self._clip(
                confidence,
                0.0,
                1.0,
            )

            if (
                abs(bounded_score) < 0.03
                or bounded_confidence <= 0.0
            ):
                return

            alpha_multiplier = (
                self._alpha_multiplier(
                    name,
                    alpha,
                )
            )

            bounded_confidence = self._clip(
                bounded_confidence
                * alpha_multiplier,
                0.0,
                1.0,
            )

            contributors.append(
                {
                    "name": str(name)[:120],
                    "group": str(group)[:80],
                    "score": bounded_score,
                    "confidence": bounded_confidence,
                    "origin_eligible": bool(
                        origin_eligible
                    ),
                    "net_edge_bps": max(
                        0.0,
                        self._number(
                            net_edge_bps
                        ),
                    ),
                    "alpha_multiplier": (
                        alpha_multiplier
                    ),
                    "source": str(source)[:120],
                }
            )

        # --------------------------------------------------
        # Fast swarm + independently-qualified MTF minds
        # --------------------------------------------------

        fast = (
            collective.get("fast_swarm")
            or {}
        )

        if (
            isinstance(fast, dict)
            and fast.get("fresh") is True
        ):
            ranked = (
                fast.get("ranked_opportunity")
                or {}
            )

            quality = self._clip(
                ranked.get("quality_multiplier")
                if isinstance(ranked, dict)
                else 0.0,
                0.0,
                1.0,
            )

            if (
                isinstance(ranked, dict)
                and ranked.get("qualified")
                is not True
            ):
                quality *= 0.50

            assessments = (
                fast.get(
                    "timeframe_assessments"
                )
                or {}
            )

            if isinstance(
                assessments,
                dict,
            ):
                for timeframe, row in (
                    assessments.items()
                ):
                    if (
                        not isinstance(row, dict)
                        or row.get(
                            "independently_qualified"
                        )
                        is not True
                    ):
                        continue

                    direction = self._direction(
                        row.get("direction")
                    )
                    cost = max(
                        30.0,
                        self._number(
                            row.get(
                                "modeled_round_trip_cost_bps"
                            ),
                            30.0,
                        ),
                    )
                    expected = self._number(
                        row.get(
                            "expected_edge_bps"
                        )
                    )
                    net_edge = (
                        expected - cost
                    )

                    if (
                        direction == 0.0
                        or net_edge <= 0.0
                    ):
                        continue

                    magnitude = min(
                        1.0,
                        max(
                            0.15,
                            net_edge / cost,
                        ),
                    )

                    add(
                        name=(
                            f"timeframe_mind_"
                            f"{timeframe}"
                        ),
                        group="fast_mtf",
                        score=(
                            direction
                            * magnitude
                        ),
                        confidence=(
                            self._number(
                                row.get(
                                    "confidence"
                                )
                            )
                            * max(
                                0.50,
                                quality,
                            )
                        ),
                        origin_eligible=True,
                        net_edge_bps=net_edge,
                        source="fast_swarm_mtf",
                    )

            for row in (
                fast.get("micro_proposals")
                or []
            ):
                if (
                    not isinstance(row, dict)
                    or row.get(
                        "evidence_qualified"
                    )
                    is not True
                    or row.get(
                        "independently_qualified"
                    )
                    is not True
                ):
                    continue

                direction = self._direction(
                    row.get("side")
                )
                net_edge = self._number(
                    row.get(
                        "conservative_net_edge_bps"
                    )
                )
                cost = max(
                    30.0,
                    self._number(
                        row.get(
                            "modeled_round_trip_cost_bps"
                        ),
                        30.0,
                    ),
                )

                if (
                    direction == 0.0
                    or net_edge <= 0.0
                ):
                    continue

                evidence_confidence = (
                    self._clip(
                        row.get("confidence"),
                        0.0,
                        1.0,
                    )
                )
                current_confidence = (
                    self._clip(
                        row.get(
                            "current_signal_confidence"
                        ),
                        0.0,
                        1.0,
                    )
                )
                confidence = (
                    0.70
                    * evidence_confidence
                    + 0.30
                    * current_confidence
                )

                magnitude = min(
                    1.0,
                    max(
                        0.15,
                        net_edge / cost,
                    ),
                )

                add(
                    name=str(
                        row.get("specialist")
                        or "microstructure"
                    ),
                    group="microstructure",
                    score=(
                        direction
                        * magnitude
                    ),
                    confidence=(
                        confidence
                        * max(
                            0.50,
                            quality,
                        )
                    ),
                    origin_eligible=True,
                    net_edge_bps=net_edge,
                    source="micro_agent_foundry",
                )

        # --------------------------------------------------
        # Validated continuous-evolution challengers
        # --------------------------------------------------

        for row in (
            collective.get(
                "evolution_evidence"
            )
            or []
        ):
            if (
                not isinstance(row, dict)
                or str(
                    row.get("kind") or ""
                ).lower()
                != "signal"
            ):
                continue

            validated = (
                row.get("research_validated")
                is True
            )

            average = self._number(
                row.get(
                    "average_net_return"
                )
            )
            ewma = self._number(
                row.get(
                    "ewma_net_return"
                )
            )

            net_edge = (
                max(
                    0.0,
                    min(
                        average,
                        ewma,
                    )
                    * 10_000.0,
                )
                if validated
                else 0.0
            )

            add(
                name=(
                    "evolution:"
                    + str(
                        row.get("pack_id")
                        or row.get("source")
                        or "challenger"
                    )
                ),
                group="evolution",
                score=self._number(
                    row.get("score")
                ),
                confidence=self._number(
                    row.get("confidence")
                ),
                origin_eligible=(
                    validated
                    and int(
                        row.get(
                            "shadow_samples"
                        )
                        or 0
                    )
                    >= 100
                    and net_edge > 0.0
                ),
                net_edge_bps=net_edge,
                source="continuous_evolution",
            )

        # --------------------------------------------------
        # Derivatives / liquidation / options / flow sensors
        # These contribute context but cannot originate alone.
        # --------------------------------------------------

        sensors = (
            collective.get("sensor_context")
            or {}
        )

        if isinstance(sensors, dict):
            flow = (
                sensors.get(
                    "flow_intelligence"
                )
                or {}
            )
            if (
                isinstance(flow, dict)
                and flow.get("status")
                == "available"
            ):
                values = (
                    flow.get("values") or {}
                )
                add(
                    name="flow_intelligence",
                    group="onchain_flow",
                    score=self._number(
                        values.get(
                            "flow_score"
                        )
                    ),
                    confidence=self._number(
                        flow.get("confidence")
                    ),
                    source="market_sensor_fabric",
                )

            liquidations = (
                sensors.get("liquidations")
                or {}
            )
            if isinstance(
                liquidations,
                dict,
            ):
                values = (
                    liquidations.get("values")
                    or {}
                )
                events = max(
                    0,
                    int(
                        values.get("events")
                        or 0
                    ),
                )
                if events > 0:
                    event_quality = min(
                        1.0,
                        0.25
                        + events / 20.0,
                    )
                    add(
                        name="liquidation_tape",
                        group="derivatives",
                        score=self._number(
                            values.get(
                                "liquidation_imbalance"
                            )
                        ),
                        confidence=(
                            self._number(
                                liquidations.get(
                                    "confidence"
                                )
                            )
                            * event_quality
                        ),
                        source=(
                            "market_sensor_fabric"
                        ),
                    )

            derivatives = (
                sensors.get("derivatives")
                or {}
            )
            if (
                isinstance(
                    derivatives,
                    dict,
                )
                and derivatives.get(
                    "status"
                )
                == "available"
            ):
                values = (
                    derivatives.get("values")
                    or {}
                )
                skew = self._clip(
                    values.get(
                        "positioning_skew"
                    )
                )
                oi_change = self._number(
                    values.get(
                        "open_interest_change_15m_window"
                    )
                )
                intensity = (
                    0.60
                    + 0.40
                    * min(
                        1.0,
                        abs(oi_change)
                        * 10.0,
                    )
                )
                add(
                    name=(
                        "derivatives_positioning"
                    ),
                    group="derivatives",
                    score=skew * intensity,
                    confidence=(
                        self._number(
                            derivatives.get(
                                "confidence"
                            )
                        )
                        * 0.60
                    ),
                    source=(
                        "market_sensor_fabric"
                    ),
                )

            options = (
                sensors.get("options")
                or {}
            )
            if (
                isinstance(options, dict)
                and options.get("status")
                == "available"
            ):
                values = (
                    options.get("values") or {}
                )
                oi_skew = self._clip(
                    values.get(
                        "open_interest_skew"
                    )
                )
                add(
                    name="options_positioning",
                    group="derivatives",
                    score=-oi_skew,
                    confidence=(
                        self._number(
                            options.get(
                                "confidence"
                            )
                        )
                        * 0.55
                    ),
                    source=(
                        "market_sensor_fabric"
                    ),
                )

        # --------------------------------------------------
        # Cross-venue lead/lag/dislocation contribution
        # The current executor is not a two-venue arbitrage executor,
        # therefore this is support evidence, not origin authority.
        # --------------------------------------------------

        quotes = [
            row
            for row in (
                collective.get(
                    "arbitrage_quotes"
                )
                or []
            )
            if isinstance(row, dict)
        ]

        primary = str(
            collective.get(
                "primary_venue"
            )
            or "bybit"
        ).lower()

        primary_rows = [
            row
            for row in quotes
            if str(
                row.get("venue") or ""
            ).lower()
            == primary
        ]
        other_rows = [
            row
            for row in quotes
            if str(
                row.get("venue") or ""
            ).lower()
            != primary
        ]

        dislocations: list[float] = []

        for p in primary_rows:
            p_bid = self._number(
                p.get("bid")
            )
            p_ask = self._number(
                p.get("ask")
            )

            if (
                p_bid <= 0.0
                or p_ask <= 0.0
            ):
                continue

            p_cost = (
                max(
                    0.0,
                    self._number(
                        p.get("fee_bps")
                    ),
                )
                + max(
                    0.0,
                    self._number(
                        p.get(
                            "slippage_bps"
                        )
                    ),
                )
            )

            for other in other_rows:
                o_bid = self._number(
                    other.get("bid")
                )
                o_ask = self._number(
                    other.get("ask")
                )
                if (
                    o_bid <= 0.0
                    or o_ask <= 0.0
                ):
                    continue

                o_cost = (
                    max(
                        0.0,
                        self._number(
                            other.get(
                                "fee_bps"
                            )
                        ),
                    )
                    + max(
                        0.0,
                        self._number(
                            other.get(
                                "slippage_bps"
                            )
                        ),
                    )
                )

                long_net = (
                    (
                        o_bid
                        / p_ask
                        - 1.0
                    )
                    * 10_000.0
                    - p_cost
                    - o_cost
                )

                short_net = (
                    (
                        p_bid
                        / o_ask
                        - 1.0
                    )
                    * 10_000.0
                    - p_cost
                    - o_cost
                )

                if long_net > 0.0:
                    dislocations.append(
                        long_net
                    )
                if short_net > 0.0:
                    dislocations.append(
                        -short_net
                    )

        if dislocations:
            strongest = max(
                dislocations,
                key=abs,
            )
            add(
                name="cross_venue_dislocation",
                group="cross_venue",
                score=(
                    (1.0 if strongest > 0 else -1.0)
                    * min(
                        1.0,
                        max(
                            0.05,
                            abs(strongest)
                            / 30.0,
                        ),
                    )
                ),
                confidence=0.60,
                source="cross_venue_quote_collector",
            )

        # --------------------------------------------------
        # Correlation-group collapse
        # --------------------------------------------------

        grouped: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for row in contributors:
            grouped.setdefault(
                str(row["group"]),
                [],
            ).append(row)

        collapsed: list[
            dict[str, Any]
        ] = []

        for group, rows in sorted(
            grouped.items()
        ):
            total_weight = sum(
                float(row["confidence"])
                for row in rows
            )

            if total_weight <= 0.0:
                continue

            score = sum(
                float(row["score"])
                * float(
                    row["confidence"]
                )
                for row in rows
            ) / total_weight

            confidence = max(
                float(row["confidence"])
                for row in rows
            )

            aligned_origin = [
                row
                for row in rows
                if (
                    row.get(
                        "origin_eligible"
                    )
                    is True
                    and float(
                        row.get("net_edge_bps")
                        or 0.0
                    )
                    > 0.0
                    and float(row["score"])
                    * score
                    > 0.0
                )
            ]

            collapsed.append(
                {
                    "group": group,
                    "score": self._clip(
                        score
                    ),
                    "confidence": self._clip(
                        confidence,
                        0.0,
                        1.0,
                    ),
                    "members": [
                        row["name"]
                        for row in rows
                    ],
                    "origin_eligible": bool(
                        aligned_origin
                    ),
                    "net_edge_bps": max(
                        [
                            float(
                                row.get(
                                    "net_edge_bps"
                                )
                                or 0.0
                            )
                            for row in (
                                aligned_origin
                            )
                        ]
                        or [0.0]
                    ),
                }
            )

        total_weight = sum(
            row["confidence"]
            for row in collapsed
        )

        ensemble_score = (
            sum(
                row["score"]
                * row["confidence"]
                for row in collapsed
            )
            / total_weight
            if total_weight > 0.0
            else 0.0
        )

        ensemble_confidence = (
            sum(
                row["confidence"]
                for row in collapsed
            )
            / len(collapsed)
            if collapsed
            else 0.0
        )

        long_support = [
            row["group"]
            for row in collapsed
            if (
                row["score"] >= 0.10
                and row["confidence"]
                >= 0.20
            )
        ]

        short_support = [
            row["group"]
            for row in collapsed
            if (
                row["score"] <= -0.10
                and row["confidence"]
                >= 0.20
            )
        ]

        long_origin = [
            row["group"]
            for row in collapsed
            if (
                row["score"] >= 0.10
                and row.get(
                    "origin_eligible"
                )
                is True
                and float(
                    row.get(
                        "net_edge_bps"
                    )
                    or 0.0
                )
                > 0.0
            )
        ]

        short_origin = [
            row["group"]
            for row in collapsed
            if (
                row["score"] <= -0.10
                and row.get(
                    "origin_eligible"
                )
                is True
                and float(
                    row.get(
                        "net_edge_bps"
                    )
                    or 0.0
                )
                > 0.0
            )
        ]

        return {
            "version": self.VERSION,
            "contributor_count": len(
                contributors
            ),
            "group_count": len(
                collapsed
            ),
            "contributors": contributors,
            "groups": collapsed,
            "ensemble_score": self._clip(
                ensemble_score
            ),
            "ensemble_confidence": (
                self._clip(
                    ensemble_confidence,
                    0.0,
                    1.0,
                )
            ),
            "long_support_groups": (
                long_support
            ),
            "short_support_groups": (
                short_support
            ),
            "long_origin_edge_groups": (
                long_origin
            ),
            "short_origin_edge_groups": (
                short_origin
            ),
            "costed_origin_required": True,
            "sensor_only_origin_allowed": False,
            "alpha_tournament_weighting": True,
            "paper_authority": True,
            "testnet_authority": True,
            "live_authority": False,
        }

    def route(
        self,
        *,
        symbol: str,
        base_enter: bool,
        base_score: float,
        base_confidence: float,
        advanced: dict[str, Any],
        collective: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        swarm = advanced.get("swarm") or {}
        advanced_score = float(
            swarm.get("score") or 0.0
        )
        advanced_confidence = float(
            swarm.get("confidence") or 0.0
        )
        signals = list(
            advanced.get("signals") or []
        )

        contributing = [
            str(signal.get("engine"))
            for signal in signals
            if float(
                signal.get("confidence")
                or 0.0
            )
            > 0
        ]

        collective_summary = (
            self._collective_summary(
                collective
            )
        )

        if (
            collective_summary[
                "contributor_count"
            ]
            > 0
        ):
            self.collective_inputs_seen += 1

        long_support = set(
            collective_summary[
                "long_support_groups"
            ]
        )
        short_support = set(
            collective_summary[
                "short_support_groups"
            ]
        )

        if (
            float(base_score) >= 0.10
            and float(base_confidence)
            >= 0.20
        ):
            long_support.add(
                "adaptive"
            )
        elif (
            float(base_score) <= -0.10
            and float(base_confidence)
            >= 0.20
        ):
            short_support.add(
                "adaptive"
            )

        if (
            advanced_score >= 0.10
            and advanced_confidence
            >= max(
                0.10,
                self.minimum_advanced_confidence
                * 0.75,
            )
        ):
            long_support.add(
                "ultra_ensemble"
            )
        elif (
            advanced_score <= -0.10
            and advanced_confidence
            >= max(
                0.10,
                self.minimum_advanced_confidence
                * 0.75,
            )
        ):
            short_support.add(
                "ultra_ensemble"
            )

        origin_candidate = bool(
            collective_summary[
                "long_origin_edge_groups"
            ]
            and len(long_support) >= 2
            and collective_summary[
                "ensemble_confidence"
            ]
            >= 0.45
            and collective_summary[
                "ensemble_score"
            ]
            >= 0.10
            and len(short_support)
            <= len(long_support)
        )

        legacy_combined = float(
            np.clip(
                0.70 * base_score
                + 0.30 * advanced_score,
                -1.0,
                1.0,
            )
        )

        if origin_candidate:
            combined_score = float(
                np.clip(
                    0.25 * base_score
                    + 0.25
                    * advanced_score
                    + 0.50
                    * collective_summary[
                        "ensemble_score"
                    ],
                    -1.0,
                    1.0,
                )
            )
        elif (
            collective_summary[
                "contributor_count"
            ]
            > 0
        ):
            combined_score = float(
                np.clip(
                    0.45 * base_score
                    + 0.30
                    * advanced_score
                    + 0.25
                    * collective_summary[
                        "ensemble_score"
                    ],
                    -1.0,
                    1.0,
                )
            )
        else:
            combined_score = (
                legacy_combined
            )

        collective_origin_ready = bool(
            origin_candidate
            and combined_score
            >= self.minimum_combined_score
        )

        collective_support_ready = bool(
            collective_summary[
                "contributor_count"
            ]
            > 0
            and len(long_support) >= 3
            and len(short_support)
            < len(long_support)
        )

        evidence = self.evidence.assess(
            symbol
        )

        reason = "approved"
        allowed = True

        if (
            not base_enter
            and not collective_origin_ready
        ):
            allowed = False
            reason = (
                "adaptive_signal_not_ready"
                if collective_summary[
                    "contributor_count"
                ]
                == 0
                else (
                    "collective_and_adaptive_"
                    "signal_not_ready"
                )
            )

        elif advanced.get(
            "news_blackout"
        ) is True:
            allowed = False
            reason = (
                "high_impact_news_blackout"
            )

        elif (
            base_enter
            and len(contributing) < 3
            and not collective_support_ready
            and not collective_origin_ready
        ):
            allowed = False
            reason = (
                "insufficient_ultra_engine_evidence"
            )

        elif (
            base_enter
            and advanced_confidence
            < self.minimum_advanced_confidence
            and not collective_support_ready
            and not collective_origin_ready
        ):
            allowed = False
            reason = (
                "low_ultra_consensus_confidence"
            )

        elif (
            advanced_score
            <= self.negative_consensus_veto
            and not collective_origin_ready
        ):
            allowed = False
            reason = (
                "negative_ultra_consensus"
            )

        elif (
            combined_score
            < self.minimum_combined_score
        ):
            allowed = False
            reason = (
                "combined_score_below_threshold"
            )

        elif (
            not evidence["allowed"]
            and not collective_origin_ready
        ):
            allowed = False
            reason = (
                "market_evidence_quarantine"
            )

        if allowed:
            if evidence["allowed"]:
                size_multiplier = float(
                    evidence[
                        "size_multiplier"
                    ]
                )
            else:
                # A new independently costed strategy family
                # receives only a small paper/Testnet probe
                # when old symbol-level evidence is quarantined.
                size_multiplier = 0.10
                reason = (
                    "approved_collective_probe"
                )

            if (
                collective_summary[
                    "contributor_count"
                ]
                > 0
            ):
                collective_quality = (
                    0.50
                    + 0.50
                    * collective_summary[
                        "ensemble_confidence"
                    ]
                )
                # Collective intelligence may reduce the
                # existing risk budget but cannot enlarge it.
                size_multiplier *= min(
                    1.0,
                    max(
                        0.50,
                        collective_quality,
                    ),
                )

            if (
                advanced_score
                <= self.negative_consensus_veto
                and collective_origin_ready
            ):
                size_multiplier *= 0.50

            if (
                collective_origin_ready
                and reason == "approved"
            ):
                reason = (
                    "approved_collective_origin"
                )
        else:
            size_multiplier = 0.0

        entry_origin = (
            "adaptive_plus_collective"
            if (
                base_enter
                and collective_summary[
                    "contributor_count"
                ]
                > 0
            )
            else (
                "adaptive"
                if base_enter
                else (
                    "collective_profit_fabric"
                    if collective_origin_ready
                    else "none"
                )
            )
        )

        result = {
            "allowed": allowed,
            "reason": reason,
            "entry_origin": entry_origin,
            "combined_score": combined_score,
            "legacy_combined_score": (
                legacy_combined
            ),
            "predicted_probability": float(
                np.clip(
                    0.5
                    + combined_score / 2.0,
                    0.0,
                    1.0,
                )
            ),
            "base_score": base_score,
            "base_confidence": (
                base_confidence
            ),
            "advanced_score": (
                advanced_score
            ),
            "advanced_confidence": (
                advanced_confidence
            ),
            "contributing_engines": (
                contributing
            ),
            "collective_contributing_engines": [
                row["name"]
                for row in (
                    collective_summary[
                        "contributors"
                    ]
                )
            ],
            "collective_origin_ready": (
                collective_origin_ready
            ),
            "collective_support_ready": (
                collective_support_ready
            ),
            "collective_profit_fabric": (
                collective_summary
            ),
            "size_multiplier": max(
                0.0,
                min(
                    1.0,
                    size_multiplier,
                ),
            ),
            "market_evidence": evidence,
            "authority": (
                "paper_and_testnet_only"
            ),
            "paper_authority": True,
            "testnet_authority": True,
            "live_authority": False,
        }

        self.routes += 1

        if allowed:
            self.approved += 1
            if collective_origin_ready:
                self.collective_origin_approved += 1
        else:
            self.blocked[reason] = (
                self.blocked.get(
                    reason,
                    0,
                )
                + 1
            )

        self.last_routes[symbol] = result
        return result

    def record_outcome(
        self,
        symbol: str,
        realized_return: float,
        predicted_probability: float | None = None,
    ) -> dict[str, Any]:
        return self.evidence.record(
            symbol,
            realized_return,
            predicted_probability,
        )

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "routes": self.routes,
            "approved": self.approved,
            "collective_origin_approved": (
                self.collective_origin_approved
            ),
            "collective_inputs_seen": (
                self.collective_inputs_seen
            ),
            "blocked": dict(self.blocked),
            "symbols_routed": len(
                self.last_routes
            ),
            "minimum_advanced_confidence": (
                self.minimum_advanced_confidence
            ),
            "minimum_combined_score": (
                self.minimum_combined_score
            ),
            "negative_consensus_veto": (
                self.negative_consensus_veto
            ),
            # Legacy weights remain visible for
            # backward-compatible/no-collective routes.
            "adaptive_weight": 0.70,
            "ultra_weight": 0.30,
            "collective_profit_fabric": True,
            "collective_origin_requires_costed_edge": True,
            "sensor_only_origin_allowed": False,
            "collective_origin_weights": {
                "adaptive": 0.25,
                "ultra": 0.25,
                "collective": 0.50,
            },
            "paper_authority": True,
            "testnet_authority": True,
            "live_authority": False,
            "market_evidence": (
                self.evidence.health()
            ),
        }
