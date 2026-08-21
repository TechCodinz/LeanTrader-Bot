from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable


def _finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _bounded(value: Any, *, name: str, lower: float, upper: float) -> float:
    number = _finite(value, name=name)
    if not lower <= number <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return number


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def specialist_correlation_group(name: str) -> str:
    """Map known specialist families to conservative correlation buckets."""
    normalized = str(name or "").strip().lower()
    if any(token in normalized for token in ("liquidity", "orderbook", "microstructure", "imbalance")):
        return "microstructure"
    if any(token in normalized for token in ("news", "fundamental", "context", "macro")):
        return "public_context"
    if any(token in normalized for token in ("onchain", "mempool", "flow")):
        return "onchain_flow"
    if any(token in normalized for token in ("arbitrage", "cross_venue", "lead_lag")):
        return "cross_venue"
    if any(token in normalized for token in ("funding", "liquidation", "derivative", "options")):
        return "derivatives"
    return "price_technical"


def host_resource_snapshot(*, runtime_healthy: bool) -> dict[str, Any]:
    """Return a bounded local resource budget without network or private data."""
    cpu_count = max(1, int(os.cpu_count() or 1))
    try:
        load_ratio = max(0.0, float(os.getloadavg()[0]) / cpu_count)
    except (AttributeError, OSError):
        load_ratio = 1.0
    memory_available_fraction = 0.0
    try:
        rows = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            rows[key] = float(value.strip().split()[0])
        memory_available_fraction = rows.get("MemAvailable", 0.0) / max(
            rows.get("MemTotal", 0.0), 1.0
        )
    except (OSError, ValueError, IndexError):
        memory_available_fraction = 0.0
    return {
        "load_ratio": min(10.0, load_ratio),
        "memory_available_fraction": min(1.0, max(0.0, memory_available_fraction)),
        "runtime_healthy": bool(runtime_healthy),
    }


def build_specialist_evidence(
    *,
    symbol: str,
    current_regime: str,
    timeframe: str,
    base_score: float,
    base_confidence: float,
    advanced_signals: list[dict[str, Any]],
    evidence_lookup: Callable[[str, str], dict[str, Any]],
    minimum_samples: int,
    modeled_round_trip_cost_bps: float,
    calibration_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Join raw specialist signals to closed, costed observatory evidence."""
    calibration_mature = calibration_snapshot.get("evidence_mature") is True
    calibration_error = (
        _bounded(
            calibration_snapshot.get("expected_calibration_error", 1.0),
            name="expected calibration error",
            lower=0.0,
            upper=1.0,
        )
        if calibration_mature
        else 1.0
    )
    raw = [
        {
            "engine": "adaptive_ensemble",
            "score": base_score,
            "confidence": base_confidence,
        },
        *[
            signal
            for signal in advanced_signals
            if str(signal.get("engine") or "")
            not in {"swarm_hivemind", "bounded_decision_router"}
        ],
    ]
    rows: list[dict[str, Any]] = []
    for signal in raw:
        name = str(signal.get("engine") or "").strip()
        if not name:
            continue
        score = _bounded(
            signal.get("score", 0.0),
            name=f"{name} score",
            lower=-1.0,
            upper=1.0,
        )
        confidence = _bounded(
            signal.get("confidence", 0.0),
            name=f"{name} confidence",
            lower=0.0,
            upper=1.0,
        )
        observed = evidence_lookup(f"engine:{name}", symbol)
        samples = max(0, int(observed.get("samples", 0)))
        average_net_return = _finite(
            observed.get("average_net_return", 0.0),
            name=f"{name} average net return",
        )
        # A raw signal magnitude is never converted into expected PnL. Gross
        # edge becomes non-zero only after enough closed costed episodes.
        gross_edge_bps = (
            max(
                0.0,
                average_net_return * 10_000.0
                + float(modeled_round_trip_cost_bps),
            )
            if samples >= int(minimum_samples)
            else 0.0
        )
        rows.append(
            {
                "name": name,
                "correlation_group": specialist_correlation_group(name),
                "timeframe": timeframe,
                "regime": current_regime,
                "direction": score,
                "confidence": confidence,
                "calibration_error": calibration_error,
                "expected_gross_edge_bps": gross_edge_bps,
                "closed_costed_samples": samples,
                "observed_average_net_return": average_net_return,
                "evidence_authority": observed.get("authority"),
            }
        )
    return rows


class PaperOrderSimulator:
    """Deterministic expected paper fill model; it has no exchange authority."""

    VERSION = "1.41.0"
    MINIMUM_ROUND_TRIP_COST_BPS = 30.0

    def __init__(self, *, minimum_round_trip_cost_bps: float = 30.0) -> None:
        floor = _finite(minimum_round_trip_cost_bps, name="minimum round-trip cost")
        if floor < self.MINIMUM_ROUND_TRIP_COST_BPS:
            raise ValueError("paper order simulator cannot lower the 30-bps cost floor")
        self.minimum_round_trip_cost_bps = floor

    def simulate(self, execution: dict[str, Any]) -> dict[str, Any]:
        requested = max(
            0.0,
            _finite(execution.get("order_notional_usd", 0.0), name="order notional"),
        )
        liquidity = max(
            0.0,
            _finite(execution.get("liquidity_usd", 0.0), name="liquidity"),
        )
        max_participation = _bounded(
            execution.get("max_participation_rate", 0.01),
            name="max participation rate",
            lower=0.0001,
            upper=0.10,
        )
        rejection_probability = _bounded(
            execution.get("rejection_probability", 0.0),
            name="rejection probability",
            lower=0.0,
            upper=1.0,
        )
        available = liquidity * max_participation
        liquidity_fill_ratio = min(1.0, available / requested) if requested > 0.0 else 0.0
        expected_fill_ratio = liquidity_fill_ratio * (1.0 - rejection_probability)

        spread = max(0.0, _finite(execution.get("spread_bps", 0.0), name="spread"))
        fee_per_side = max(
            0.0,
            _finite(execution.get("fee_bps_per_side", 0.0), name="fee per side"),
        )
        base_slippage_per_side = max(
            0.0,
            _finite(
                execution.get("base_slippage_bps_per_side", 0.0),
                name="base slippage per side",
            ),
        )
        funding = abs(_finite(execution.get("funding_bps", 0.0), name="funding"))
        latency_ms = max(
            0.0,
            _finite(execution.get("latency_ms", 0.0), name="latency"),
        )
        volatility_bps_per_second = max(
            0.0,
            _finite(
                execution.get("volatility_bps_per_second", 0.0),
                name="latency volatility",
            ),
        )
        adverse_selection = max(
            0.0,
            _finite(
                execution.get("adverse_selection_bps", 0.0),
                name="adverse selection",
            ),
        )
        impact_coefficient = max(
            0.0,
            _finite(
                execution.get("impact_coefficient_bps", 50.0),
                name="impact coefficient",
            ),
        )
        participation = min(1.0, requested / max(liquidity, 1e-12))
        impact_per_side = impact_coefficient * math.sqrt(participation)
        latency_drag = min(250.0, latency_ms / 1_000.0 * volatility_bps_per_second)
        modeled = (
            2.0 * fee_per_side
            + spread
            + 2.0 * base_slippage_per_side
            + 2.0 * impact_per_side
            + funding
            + latency_drag
            + adverse_selection
        )
        round_trip_cost = max(self.minimum_round_trip_cost_bps, modeled)
        return {
            "requested_notional_usd": requested,
            "available_notional_at_participation_cap_usd": available,
            "expected_filled_notional_usd": requested * expected_fill_ratio,
            "liquidity_fill_ratio": liquidity_fill_ratio,
            "expected_fill_ratio": expected_fill_ratio,
            "expected_rejection_probability": rejection_probability,
            "partial_fill_expected": 0.0 < expected_fill_ratio < 1.0,
            "no_fill_expected": expected_fill_ratio <= 0.0,
            "costs_bps": {
                "fee_round_trip": 2.0 * fee_per_side,
                "spread_round_trip": spread,
                "base_slippage_round_trip": 2.0 * base_slippage_per_side,
                "market_impact_round_trip": 2.0 * impact_per_side,
                "funding": funding,
                "latency": latency_drag,
                "adverse_selection": adverse_selection,
                "modeled_total": modeled,
                "floor": self.minimum_round_trip_cost_bps,
                "round_trip_total": round_trip_cost,
            },
            "paper_simulation": True,
            "live_authority": False,
            "testnet_authority": False,
            "execution_authority": False,
        }


class UnifiedDecisionControlPlane:
    """Fail-closed v1.41 shadow control plane over existing specialist engines.

    The component deliberately has no order, sizing, paper-promotion, Testnet,
    or live authority. It produces a reproducible recommendation which can be
    compared prospectively with the existing route before any later promotion.
    """

    VERSION = "1.41.0"
    SCHEMA_VERSION = 1
    HISTORY_LIMIT = 1_000

    def __init__(
        self,
        state_path: Path,
        *,
        minimum_round_trip_cost_bps: float = 30.0,
        minimum_independent_groups: int = 2,
        minimum_ensemble_confidence: float = 0.55,
        minimum_independent_samples: int = 100,
        max_gross_exposure_fraction: float = 0.50,
        max_symbol_exposure_fraction: float = 0.15,
        max_correlation_bucket_fraction: float = 0.25,
        max_drawdown_fraction: float = 0.10,
        max_daily_loss_fraction: float = 0.03,
        max_loss_streak: int = 5,
        max_volatility_ratio: float = 3.0,
    ) -> None:
        if int(minimum_independent_groups) < 2:
            raise ValueError("at least two independent evidence groups are required")
        if int(minimum_independent_samples) < 30:
            raise ValueError("minimum independent samples cannot be below 30")
        self.state_path = state_path
        self.minimum_independent_groups = int(minimum_independent_groups)
        self.minimum_ensemble_confidence = _bounded(
            minimum_ensemble_confidence,
            name="minimum ensemble confidence",
            lower=0.50,
            upper=0.99,
        )
        self.minimum_independent_samples = int(minimum_independent_samples)
        self.max_gross_exposure_fraction = _bounded(
            max_gross_exposure_fraction,
            name="maximum gross exposure",
            lower=0.01,
            upper=1.0,
        )
        self.max_symbol_exposure_fraction = _bounded(
            max_symbol_exposure_fraction,
            name="maximum symbol exposure",
            lower=0.01,
            upper=self.max_gross_exposure_fraction,
        )
        self.max_correlation_bucket_fraction = _bounded(
            max_correlation_bucket_fraction,
            name="maximum correlation bucket exposure",
            lower=self.max_symbol_exposure_fraction,
            upper=self.max_gross_exposure_fraction,
        )
        self.max_drawdown_fraction = _bounded(
            max_drawdown_fraction,
            name="maximum drawdown",
            lower=0.01,
            upper=0.50,
        )
        self.max_daily_loss_fraction = _bounded(
            max_daily_loss_fraction,
            name="maximum daily loss",
            lower=0.005,
            upper=0.20,
        )
        if int(max_loss_streak) < 2:
            raise ValueError("maximum loss streak must be at least two")
        self.max_loss_streak = int(max_loss_streak)
        self.max_volatility_ratio = _finite(
            max_volatility_ratio, name="maximum volatility ratio"
        )
        if self.max_volatility_ratio <= 1.0:
            raise ValueError("maximum volatility ratio must exceed one")
        self.order_simulator = PaperOrderSimulator(
            minimum_round_trip_cost_bps=minimum_round_trip_cost_bps
        )
        self.last_error: str | None = None
        self.state = self._load()
        self.lineage_integrity_ok = self._verify_history(
            self.state.get("history", []),
            anchor_hash=str(self.state.get("anchor_hash") or "GENESIS"),
            anchor_sequence=int(self.state.get("anchor_sequence") or 0),
        )

    @staticmethod
    def _authority_denied() -> dict[str, bool]:
        return {
            "shadow_only": True,
            "advisory_only": True,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_modify_routes": False,
            "can_modify_orders": False,
            "can_modify_sizing": False,
            "can_increase_risk": False,
            "execution_authority": False,
        }

    def start(self) -> None:
        self.state = self._load()
        self.lineage_integrity_ok = self._verify_history(
            self.state.get("history", []),
            anchor_hash=str(self.state.get("anchor_hash") or "GENESIS"),
            anchor_sequence=int(self.state.get("anchor_sequence") or 0),
        )

    def stop(self) -> None:
        if self.lineage_integrity_ok:
            self._save()

    def _data_quality_reasons(self, market_data: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        age = _finite(market_data.get("age_seconds"), name="market-data age")
        maximum_age = _finite(
            market_data.get("max_age_seconds"), name="maximum market-data age"
        )
        quality = _bounded(
            market_data.get("quality_score"),
            name="market-data quality",
            lower=0.0,
            upper=1.0,
        )
        if maximum_age <= 0.0 or age < 0.0 or age > maximum_age:
            reasons.append("stale_market_data")
        if quality < 0.90:
            reasons.append("market_data_quality_below_floor")
        if market_data.get("future_leakage_detected") is not False:
            reasons.append("future_leakage_not_cleared")
        if market_data.get("survivorship_bias_controlled") is not True:
            reasons.append("survivorship_bias_not_controlled")
        if market_data.get("exchange_anomaly_detected") is not False:
            reasons.append("exchange_anomaly_not_cleared")
        if market_data.get("freshness_verified") is not True:
            reasons.append("freshness_not_verified")
        return reasons

    def _safety_reasons(
        self,
        system: dict[str, Any],
        portfolio: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        if str(system.get("trading_mode") or "").lower() != "paper":
            reasons.append("trading_mode_not_paper")
        if system.get("enable_live") is not False:
            reasons.append("enable_live_not_false")
        if system.get("allow_live") is not False:
            reasons.append("allow_live_not_false")
        if str(system.get("live_confirm") or "").upper() != "NO":
            reasons.append("live_confirm_not_no")
        if system.get("testnet_enabled") is not False:
            reasons.append("testnet_execution_not_disabled")
        if system.get("runtime_integrity_ok") is not True:
            reasons.append("runtime_integrity_not_verified")
        if system.get("heartbeat_fresh") is not True:
            reasons.append("heartbeat_not_fresh")

        equity = _finite(portfolio.get("equity"), name="equity")
        peak = _finite(portfolio.get("peak_equity"), name="peak equity")
        daily_pnl = _finite(portfolio.get("daily_pnl", 0.0), name="daily PnL")
        loss_streak = int(portfolio.get("loss_streak", 0))
        volatility_ratio = _finite(
            portfolio.get("volatility_ratio", 1.0), name="volatility ratio"
        )
        if equity <= 0.0 or peak <= 0.0:
            reasons.append("invalid_equity")
        else:
            drawdown = max(0.0, peak - equity) / peak
            if drawdown >= self.max_drawdown_fraction:
                reasons.append("drawdown_circuit_breaker")
            if daily_pnl / equity <= -self.max_daily_loss_fraction:
                reasons.append("daily_loss_circuit_breaker")
        if loss_streak >= self.max_loss_streak:
            reasons.append("loss_streak_circuit_breaker")
        if volatility_ratio >= self.max_volatility_ratio:
            reasons.append("volatility_shock_circuit_breaker")
        if not self.lineage_integrity_ok:
            reasons.append("lineage_integrity_failure")
        return reasons

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> int:
        unit = timeframe[-1:] if timeframe else ""
        try:
            value = int(timeframe[:-1])
        except (TypeError, ValueError):
            return 0
        return value * {"m": 60, "h": 3_600, "d": 86_400}.get(unit, 0)

    def _combine_evidence(
        self,
        evidence: list[dict[str, Any]],
        *,
        current_regime: str,
    ) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {}
        rejected: list[dict[str, str]] = []
        for index, row in enumerate(evidence):
            if not isinstance(row, dict):
                rejected.append({"name": f"row_{index}", "reason": "malformed"})
                continue
            name = str(row.get("name") or f"specialist_{index}")[:100]
            group = str(row.get("correlation_group") or "").strip()
            timeframe = str(row.get("timeframe") or "").strip()
            regime = str(row.get("regime") or "").strip()
            if not group or self._timeframe_seconds(timeframe) <= 0:
                rejected.append({"name": name, "reason": "missing_independence_metadata"})
                continue
            direction = _bounded(
                row.get("direction"), name=f"{name} direction", lower=-1.0, upper=1.0
            )
            confidence = _bounded(
                row.get("confidence"), name=f"{name} confidence", lower=0.0, upper=1.0
            )
            calibration_error = _bounded(
                row.get("calibration_error", 1.0),
                name=f"{name} calibration error",
                lower=0.0,
                upper=1.0,
            )
            gross_edge = max(
                0.0,
                _finite(row.get("expected_gross_edge_bps"), name=f"{name} gross edge"),
            )
            if regime not in {"*", "any", current_regime}:
                rejected.append({"name": name, "reason": "regime_mismatch"})
                continue
            reliability = confidence * (1.0 - calibration_error)
            groups.setdefault(group, []).append(
                {
                    "name": name,
                    "group": group,
                    "timeframe": timeframe,
                    "direction": direction,
                    "gross_edge_bps": gross_edge,
                    "reliability": reliability,
                }
            )

        collapsed: list[dict[str, Any]] = []
        for group, rows in sorted(groups.items()):
            weight = sum(row["reliability"] for row in rows)
            if weight <= 0.0:
                continue
            signed_edge = sum(
                row["direction"] * row["gross_edge_bps"] * row["reliability"]
                for row in rows
            ) / weight
            mean_direction = sum(
                row["direction"] * row["reliability"] for row in rows
            ) / weight
            collapsed.append(
                {
                    "correlation_group": group,
                    "members": [row["name"] for row in rows],
                    "member_count": len(rows),
                    "timeframes": sorted({row["timeframe"] for row in rows}),
                    "signed_edge_bps": signed_edge,
                    "direction": mean_direction,
                    # A group is capped at one unit of weight regardless of how
                    # many correlated specialists it contains.
                    "reliability": min(1.0, max(row["reliability"] for row in rows)),
                }
            )
        total_weight = sum(row["reliability"] for row in collapsed)
        signed_edge = (
            sum(row["signed_edge_bps"] * row["reliability"] for row in collapsed)
            / total_weight
            if total_weight > 0.0
            else 0.0
        )
        direction_score = (
            sum(row["direction"] * row["reliability"] for row in collapsed)
            / total_weight
            if total_weight > 0.0
            else 0.0
        )
        ensemble_confidence = (
            sum(row["reliability"] for row in collapsed) / len(collapsed)
            if collapsed
            else 0.0
        )
        timeframe_scores: dict[str, float] = {}
        for row in collapsed:
            for timeframe in row["timeframes"]:
                timeframe_scores[timeframe] = timeframe_scores.get(timeframe, 0.0) + abs(
                    row["signed_edge_bps"]
                ) * row["reliability"]
        selected_timeframe = (
            max(
                timeframe_scores,
                key=lambda item: (
                    timeframe_scores[item],
                    self._timeframe_seconds(item),
                ),
            )
            if timeframe_scores
            else None
        )
        return {
            "independent_group_count": len(collapsed),
            "raw_specialist_count": len(evidence),
            "accepted_specialist_count": sum(row["member_count"] for row in collapsed),
            "collapsed_groups": collapsed,
            "rejected": rejected,
            "signed_gross_edge_bps": signed_edge,
            "gross_edge_bps": abs(signed_edge),
            "direction": "long" if direction_score > 0.0 else "short" if direction_score < 0.0 else "flat",
            "direction_score": direction_score,
            "ensemble_confidence": ensemble_confidence,
            "selected_timeframe": selected_timeframe,
            "timeframe_scores": timeframe_scores,
        }

    def _portfolio_allocation(
        self,
        *,
        symbol: str,
        correlation_bucket: str,
        portfolio: dict[str, Any],
        requested_notional: float,
        expected_fill_ratio: float,
        confidence: float,
    ) -> dict[str, Any]:
        equity = max(0.0, _finite(portfolio.get("equity"), name="equity"))
        positions = portfolio.get("positions") or []
        gross = 0.0
        symbol_exposure = 0.0
        bucket_exposure = 0.0
        normalized_symbol = symbol.upper()
        for row in positions:
            if not isinstance(row, dict):
                continue
            notional = abs(_finite(row.get("notional_usd", 0.0), name="position notional"))
            gross += notional
            if str(row.get("symbol") or "").upper() == normalized_symbol:
                symbol_exposure += notional
            if str(row.get("correlation_bucket") or "") == correlation_bucket:
                bucket_exposure += notional
        gross_room = max(0.0, equity * self.max_gross_exposure_fraction - gross)
        symbol_room = max(
            0.0, equity * self.max_symbol_exposure_fraction - symbol_exposure
        )
        bucket_room = max(
            0.0,
            equity * self.max_correlation_bucket_fraction - bucket_exposure,
        )
        cap = min(requested_notional, gross_room, symbol_room, bucket_room)
        confidence_haircut = max(0.0, min(1.0, confidence))
        allocation = cap * expected_fill_ratio * confidence_haircut
        reasons: list[str] = []
        if gross_room <= 0.0:
            reasons.append("gross_exposure_cap")
        if symbol_room <= 0.0:
            reasons.append("symbol_exposure_cap")
        if bucket_room <= 0.0:
            reasons.append("correlation_bucket_cap")
        if expected_fill_ratio <= 0.0:
            reasons.append("no_expected_fill")
        return {
            "equity": equity,
            "existing_gross_exposure_usd": gross,
            "existing_symbol_exposure_usd": symbol_exposure,
            "existing_correlation_bucket_exposure_usd": bucket_exposure,
            "gross_room_usd": gross_room,
            "symbol_room_usd": symbol_room,
            "correlation_bucket_room_usd": bucket_room,
            "pre_confidence_cap_usd": cap,
            "recommended_shadow_notional_usd": allocation,
            "confidence_haircut": confidence_haircut,
            "reasons": reasons,
        }

    def _promotion_gate(self, validation: dict[str, Any]) -> dict[str, Any]:
        independent_samples = max(0, int(validation.get("independent_samples", 0)))
        reasons: list[str] = []
        if independent_samples < self.minimum_independent_samples:
            reasons.append("insufficient_independent_samples")
        required_true = (
            ("purged_walk_forward_passed", "purged_walk_forward_not_passed"),
            ("embargo_applied", "embargo_not_applied"),
            ("untouched_holdout_passed", "untouched_holdout_not_passed"),
            ("multiple_testing_controlled", "multiple_testing_not_controlled"),
            ("prospective_net_positive", "prospective_net_not_positive"),
            ("calibration_reliable", "calibration_not_reliable"),
            ("drift_stable", "drift_or_decay_detected"),
        )
        for field, reason in required_true:
            if validation.get(field) is not True:
                reasons.append(reason)
        pbo = _bounded(
            validation.get("probability_backtest_overfitting", 1.0),
            name="probability of backtest overfitting",
            lower=0.0,
            upper=1.0,
        )
        if pbo > 0.20:
            reasons.append("backtest_overfitting_probability_too_high")
        deflated = _finite(
            validation.get("deflated_performance_statistic", -math.inf),
            name="deflated performance statistic",
        )
        if deflated <= 0.0:
            reasons.append("deflated_performance_not_positive")
        partitions = validation.get("partitions") or {}
        for name in ("training", "validation", "prospective_paper", "untouched_holdout"):
            if name not in partitions:
                reasons.append(f"missing_{name}_partition")
        return {
            "eligible_for_human_review": not reasons,
            "automatic_promotion": False,
            "demote_to_shadow": (
                validation.get("drift_stable") is not True
                or validation.get("prospective_net_positive") is not True
            ),
            "independent_samples": independent_samples,
            "minimum_independent_samples": self.minimum_independent_samples,
            "probability_backtest_overfitting": pbo,
            "deflated_performance_statistic": deflated,
            "reasons": reasons,
            "partitions": partitions,
            **self._authority_denied(),
        }

    @staticmethod
    def _research_budget(resources: dict[str, Any]) -> dict[str, Any]:
        load_ratio = max(0.0, _finite(resources.get("load_ratio", 1.0), name="load ratio"))
        memory_available = _bounded(
            resources.get("memory_available_fraction", 0.0),
            name="available memory fraction",
            lower=0.0,
            upper=1.0,
        )
        reasons: list[str] = []
        if resources.get("runtime_healthy") is not True:
            reasons.append("runtime_not_healthy")
        if load_ratio >= 0.75:
            reasons.append("host_load_budget_exhausted")
        if memory_available < 0.25:
            reasons.append("memory_budget_exhausted")
        return {
            "research_permitted": not reasons,
            "production_runtime_priority": True,
            "bounded_hypotheses_per_cycle": 1 if not reasons else 0,
            "reasons": reasons,
        }

    @staticmethod
    def _lifecycle(
        *,
        atr_fraction: float,
        selected_timeframe: str | None,
        direction: str,
    ) -> dict[str, Any]:
        bounded_atr = min(0.20, max(0.001, atr_fraction))
        stop_distance = min(0.25, 2.0 * bounded_atr)
        take_profit = min(0.50, 3.0 * bounded_atr)
        bars = {"m": 24, "h": 12, "d": 5}.get(
            (selected_timeframe or "")[-1:],
            12,
        )
        return {
            "direction": direction,
            "initial_stop_distance_fraction": stop_distance,
            "take_profit_distance_fraction": take_profit,
            "reward_to_risk": take_profit / stop_distance,
            "trailing_activation_r": 1.0,
            "trailing_distance_atr": 1.5,
            "time_stop_bars": bars,
            "invalidation": [
                "regime_change",
                "data_quality_failure",
                "liquidity_failure",
                "confidence_decay",
                "system_integrity_failure",
            ],
            "paper_plan_only": True,
        }

    def evaluate(
        self,
        *,
        symbol: str,
        current_regime: str,
        correlation_bucket: str,
        specialist_evidence: list[dict[str, Any]],
        market_data: dict[str, Any],
        execution: dict[str, Any],
        portfolio: dict[str, Any],
        validation: dict[str, Any],
        resources: dict[str, Any],
        system: dict[str, Any],
        atr_fraction: float,
    ) -> dict[str, Any]:
        try:
            normalized_symbol = str(symbol or "").strip().upper()
            if not normalized_symbol:
                raise ValueError("symbol is required")
            if not str(current_regime or "").strip():
                raise ValueError("current regime is required")
            if not str(correlation_bucket or "").strip():
                raise ValueError("correlation bucket is required")
            data_reasons = self._data_quality_reasons(market_data)
            safety_reasons = self._safety_reasons(system, portfolio)
            ensemble = self._combine_evidence(
                specialist_evidence,
                current_regime=str(current_regime),
            )
            fill = self.order_simulator.simulate(execution)
            requested = fill["requested_notional_usd"]
            allocation = self._portfolio_allocation(
                symbol=normalized_symbol,
                correlation_bucket=str(correlation_bucket),
                portfolio=portfolio,
                requested_notional=requested,
                expected_fill_ratio=fill["expected_fill_ratio"],
                confidence=ensemble["ensemble_confidence"],
            )
            promotion = self._promotion_gate(validation)
            research_budget = self._research_budget(resources)
            net_edge = ensemble["gross_edge_bps"] - fill["costs_bps"]["round_trip_total"]
            decision_reasons = list(data_reasons) + list(safety_reasons)
            if ensemble["independent_group_count"] < self.minimum_independent_groups:
                decision_reasons.append("insufficient_independent_evidence")
            if ensemble["ensemble_confidence"] < self.minimum_ensemble_confidence:
                decision_reasons.append("ensemble_confidence_below_floor")
            if ensemble["direction"] == "flat":
                decision_reasons.append("no_directional_consensus")
            if net_edge <= 0.0:
                decision_reasons.append("non_positive_net_edge_after_costs")
            decision_reasons.extend(allocation["reasons"])
            decision_reasons = list(dict.fromkeys(decision_reasons))
            allowed = not decision_reasons
            lifecycle = self._lifecycle(
                atr_fraction=_finite(atr_fraction, name="ATR fraction"),
                selected_timeframe=ensemble["selected_timeframe"],
                direction=ensemble["direction"],
            )
            reproducibility_payload = {
                "version": self.VERSION,
                "configuration": self._configuration(),
                "inputs": {
                    "symbol": normalized_symbol,
                    "current_regime": current_regime,
                    "correlation_bucket": correlation_bucket,
                    "specialist_evidence": specialist_evidence,
                    "market_data": market_data,
                    "execution": execution,
                    "portfolio": portfolio,
                    "validation": validation,
                    "resources": resources,
                    "system": system,
                    "atr_fraction": atr_fraction,
                },
            }
            result = {
                "allowed_shadow_recommendation": allowed,
                "reason": "approved_for_shadow_comparison" if allowed else decision_reasons[0],
                "reasons": decision_reasons,
                "symbol": normalized_symbol,
                "regime": current_regime,
                "ensemble": ensemble,
                "execution_simulation": fill,
                "portfolio_allocation": allocation,
                "expected_net_edge_bps": net_edge,
                "trade_lifecycle": lifecycle,
                "promotion_gate": promotion,
                "research_budget": research_budget,
                "experiment_id": _digest(reproducibility_payload),
                "lineage_integrity_ok": self.lineage_integrity_ok,
                **self._authority_denied(),
            }
            self._append_history(
                {
                    "experiment_id": result["experiment_id"],
                    "symbol": normalized_symbol,
                    "allowed": allowed,
                    "reasons": decision_reasons,
                    "expected_net_edge_bps": net_edge,
                    "independent_group_count": ensemble["independent_group_count"],
                    "selected_timeframe": ensemble["selected_timeframe"],
                    "promotion_review_eligible": promotion["eligible_for_human_review"],
                }
            )
            self.last_error = None
            return result
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return {
                "allowed_shadow_recommendation": False,
                "reason": "control_plane_input_failure",
                "reasons": ["control_plane_input_failure"],
                "error": self.last_error,
                "lineage_integrity_ok": self.lineage_integrity_ok,
                **self._authority_denied(),
            }

    def _configuration(self) -> dict[str, Any]:
        return {
            "minimum_round_trip_cost_bps": self.order_simulator.minimum_round_trip_cost_bps,
            "minimum_independent_groups": self.minimum_independent_groups,
            "minimum_ensemble_confidence": self.minimum_ensemble_confidence,
            "minimum_independent_samples": self.minimum_independent_samples,
            "max_gross_exposure_fraction": self.max_gross_exposure_fraction,
            "max_symbol_exposure_fraction": self.max_symbol_exposure_fraction,
            "max_correlation_bucket_fraction": self.max_correlation_bucket_fraction,
            "max_drawdown_fraction": self.max_drawdown_fraction,
            "max_daily_loss_fraction": self.max_daily_loss_fraction,
            "max_loss_streak": self.max_loss_streak,
            "max_volatility_ratio": self.max_volatility_ratio,
        }

    def health(self) -> dict[str, Any]:
        history = self.state.get("history", [])
        last = history[-1]["payload"] if history else None
        return {
            "version": self.VERSION,
            "persistent": True,
            "state_path": str(self.state_path),
            "evaluations": len(history),
            "lineage_integrity_ok": self.lineage_integrity_ok,
            "last_evaluation": last,
            "last_error": self.last_error,
            "configuration": self._configuration(),
            "correlation_deduplication": "one_weight_cap_per_declared_group",
            "regime_aware": True,
            "timeframe_selection": True,
            "portfolio_risk_budgeting": True,
            "paper_execution_simulation": True,
            "lifecycle_planning": True,
            "validation_partitions_required": [
                "training",
                "validation",
                "prospective_paper",
                "untouched_holdout",
            ],
            "automatic_demotion_to_shadow": True,
            "minimum_cost_floor_bps": 30.0,
            **self._authority_denied(),
        }

    def _append_history(self, payload: dict[str, Any]) -> None:
        if not self.lineage_integrity_ok:
            return
        history = list(self.state.get("history", []))
        previous_hash = (
            history[-1]["record_hash"]
            if history
            else str(self.state.get("anchor_hash") or "GENESIS")
        )
        previous_sequence = (
            int(history[-1]["sequence"])
            if history
            else int(self.state.get("anchor_sequence") or 0)
        )
        record = {
            "sequence": previous_sequence + 1,
            "previous_hash": previous_hash,
            "observed_at": time.time(),
            "payload": payload,
        }
        record["record_hash"] = _digest(record)
        history.append(record)
        if len(history) > self.HISTORY_LIMIT:
            removed = history[: -self.HISTORY_LIMIT]
            self.state["anchor_hash"] = removed[-1]["record_hash"]
            self.state["anchor_sequence"] = int(removed[-1]["sequence"])
            history = history[-self.HISTORY_LIMIT :]
        self.state["history"] = history
        self.state["schema_version"] = self.SCHEMA_VERSION
        self._save()

    @staticmethod
    def _verify_history(
        history: Any,
        *,
        anchor_hash: str = "GENESIS",
        anchor_sequence: int = 0,
    ) -> bool:
        if not isinstance(history, list):
            return False
        previous = anchor_hash
        for expected_sequence, record in enumerate(
            history, start=int(anchor_sequence) + 1
        ):
            if not isinstance(record, dict):
                return False
            supplied_hash = str(record.get("record_hash") or "")
            unhashed = {key: value for key, value in record.items() if key != "record_hash"}
            if (
                int(record.get("sequence", 0)) != expected_sequence
                or record.get("previous_hash") != previous
                or supplied_hash != _digest(unhashed)
            ):
                return False
            previous = supplied_hash
        return True

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": self.SCHEMA_VERSION, "history": []}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") == self.SCHEMA_VERSION
                and isinstance(payload.get("history"), list)
            ):
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return {"schema_version": self.SCHEMA_VERSION, "history": "invalid"}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)
