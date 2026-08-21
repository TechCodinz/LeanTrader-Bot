from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _wilson_interval(wins: int, samples: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if samples <= 0:
        return (0.0, 1.0)
    rate = wins / samples
    z2 = z * z
    denominator = 1.0 + z2 / samples
    centre = rate + z2 / (2.0 * samples)
    spread = z * math.sqrt(
        (rate * (1.0 - rate) + z2 / (4.0 * samples)) / samples
    )
    return (
        max(0.0, (centre - spread) / denominator),
        min(1.0, (centre + spread) / denominator),
    )


class ProbabilityCalibrationLab:
    """Prospective route-probability calibration with no rewrite authority."""

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    MINIMUM_COST_FLOOR_BPS = 30.0
    OBSERVATION_LIMIT = 10_000
    SEEN_LIMIT = 25_000
    BIN_COUNT = 10

    def __init__(
        self,
        state_path: Path,
        *,
        minimum_samples: int = 100,
        minimum_regimes: int = 2,
        minimum_class_samples: int = 20,
        modeled_round_trip_cost_bps: float = 30.0,
    ) -> None:
        if int(minimum_samples) < 100:
            raise ValueError("probability calibration cannot lower the 100-sample floor")
        if int(minimum_regimes) < 2:
            raise ValueError("probability calibration requires at least two regimes")
        if int(minimum_class_samples) < 20:
            raise ValueError("probability calibration requires at least 20 outcomes per class")
        if float(modeled_round_trip_cost_bps) < self.MINIMUM_COST_FLOOR_BPS:
            raise ValueError("probability calibration cannot lower the 30-bps cost floor")
        self.state_path = state_path
        self.minimum_samples = int(minimum_samples)
        self.minimum_regimes = int(minimum_regimes)
        self.minimum_class_samples = int(minimum_class_samples)
        self.modeled_round_trip_cost_bps = float(modeled_round_trip_cost_bps)
        self.last_error: str | None = None
        self.state = self._load()

    def start(self) -> None:
        self.state = self._load()

    def stop(self) -> None:
        self._save()

    @staticmethod
    def _authority_denied() -> dict[str, bool]:
        return {
            "research_only": True,
            "advisory_only": True,
            "prospective_closed_outcomes_only": True,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_rewrite_probabilities": False,
            "can_modify_routes": False,
            "can_modify_orders": False,
            "can_modify_sizing": False,
            "can_increase_risk": False,
            "execution_authority": False,
        }

    def _observation(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if str(event.get("side") or "").lower() != "sell":
            return None
        if max(0.0, _finite(event.get("remaining_quantity"), 0.0)) > 1e-12:
            return None
        event_id = str(event.get("event_id") or "").strip()
        symbol = str(event.get("symbol") or "").strip().upper()
        metadata = event.get("position_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        route = metadata.get("decision_route")
        route = route if isinstance(route, dict) else {}
        raw_probability = _finite(route.get("predicted_probability"))
        if not event_id or not symbol or not math.isfinite(raw_probability):
            return None
        probability = max(0.001, min(0.999, raw_probability))
        net_return = _finite(
            event.get("trade_realized_return_total"),
            _finite(event.get("realized_return"), 0.0),
        )
        regime = str(metadata.get("regime") or "unknown").strip().lower()[:120]
        return {
            "event_id": event_id,
            "observed_at": time.time(),
            "event_timestamp": str(event.get("timestamp") or ""),
            "symbol": symbol,
            "regime": regime or "unknown",
            "predicted_probability": probability,
            "raw_predicted_probability": raw_probability,
            "profitable_after_costs": net_return > 0.0,
            "net_return": net_return,
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
        }

    def observe(self, *, events: list[dict[str, Any]]) -> dict[str, Any]:
        observations = self.state.setdefault("observations", [])
        seen = self.state.setdefault("seen_event_ids", {})
        recorded: list[str] = []
        partial_exits_ignored = 0
        missing_predictions_ignored = 0
        duplicates_ignored = 0

        for event in events:
            if not isinstance(event, dict):
                missing_predictions_ignored += 1
                continue
            event_id = str(event.get("event_id") or "").strip()
            if event_id and event_id in seen:
                duplicates_ignored += 1
                continue
            if (
                str(event.get("side") or "").lower() == "sell"
                and max(0.0, _finite(event.get("remaining_quantity"), 0.0)) > 1e-12
            ):
                partial_exits_ignored += 1
                continue
            row = self._observation(event)
            if row is None:
                missing_predictions_ignored += 1
                continue
            observations.append(row)
            seen[row["event_id"]] = time.time()
            recorded.append(row["event_id"])

        self.state["observations"] = observations[-self.OBSERVATION_LIMIT :]
        if len(seen) > self.SEEN_LIMIT:
            ordered = sorted(seen.items(), key=lambda item: float(item[1]))
            for key, _ in ordered[: len(seen) - self.SEEN_LIMIT]:
                seen.pop(key, None)
        self.state["cycles"] = int(self.state.get("cycles") or 0) + 1
        self.state["last"] = {
            "observed_at": time.time(),
            "recorded_event_ids": recorded,
            "partial_exits_ignored": partial_exits_ignored,
            "missing_predictions_ignored": missing_predictions_ignored,
            "duplicates_ignored": duplicates_ignored,
        }
        self._save()
        return self.snapshot()

    @staticmethod
    def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        samples = len(rows)
        if not samples:
            return {
                "samples": 0,
                "wins": 0,
                "losses": 0,
                "observed_win_rate": 0.0,
                "mean_predicted_probability": 0.0,
                "calibration_bias": 0.0,
                "brier_score": None,
                "baseline_brier_score": None,
                "brier_skill_score": None,
                "log_loss": None,
                "average_net_return": 0.0,
            }
        probabilities = [
            max(0.001, min(0.999, _finite(row.get("predicted_probability"), 0.5)))
            for row in rows
        ]
        outcomes = [1.0 if row.get("profitable_after_costs") else 0.0 for row in rows]
        returns = [_finite(row.get("net_return"), 0.0) for row in rows]
        wins = int(sum(outcomes))
        observed = wins / samples
        predicted = sum(probabilities) / samples
        brier = sum((probability - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes)) / samples
        baseline_brier = observed * (1.0 - observed)
        log_loss = -sum(
            outcome * math.log(probability)
            + (1.0 - outcome) * math.log(1.0 - probability)
            for probability, outcome in zip(probabilities, outcomes)
        ) / samples
        win_lower, win_upper = _wilson_interval(wins, samples)
        return {
            "samples": samples,
            "wins": wins,
            "losses": samples - wins,
            "observed_win_rate": observed,
            "observed_win_rate_lower_95": win_lower,
            "observed_win_rate_upper_95": win_upper,
            "mean_predicted_probability": predicted,
            "calibration_bias": predicted - observed,
            "brier_score": brier,
            "baseline_brier_score": baseline_brier,
            "brier_skill_score": (
                1.0 - brier / baseline_brier
                if baseline_brier > 0.0
                else None
            ),
            "log_loss": log_loss,
            "average_net_return": sum(returns) / samples,
        }

    def _reliability_bins(
        self,
        observations: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], float]:
        bins: list[dict[str, Any]] = []
        total = max(1, len(observations))
        expected_calibration_error = 0.0
        for index in range(self.BIN_COUNT):
            lower = index / self.BIN_COUNT
            upper = (index + 1) / self.BIN_COUNT
            rows = [
                row
                for row in observations
                if lower
                <= _finite(row.get("predicted_probability"), -1.0)
                < upper
                or (
                    index == self.BIN_COUNT - 1
                    and _finite(row.get("predicted_probability"), -1.0) == 1.0
                )
            ]
            metrics = self._metrics(rows)
            gap = abs(
                float(metrics["mean_predicted_probability"])
                - float(metrics["observed_win_rate"])
            )
            expected_calibration_error += len(rows) / total * gap
            bins.append(
                {
                    "bin": index,
                    "lower": lower,
                    "upper": upper,
                    "calibration_gap": gap,
                    **metrics,
                }
            )
        return bins, expected_calibration_error

    def snapshot(self) -> dict[str, Any]:
        observations = [
            row
            for row in list(self.state.get("observations") or [])
            if isinstance(row, dict)
        ]
        overall = self._metrics(observations)
        bins, expected_calibration_error = self._reliability_bins(observations)
        regimes = sorted(
            {str(row.get("regime") or "unknown") for row in observations}
        )
        by_regime = {
            regime: self._metrics(
                [row for row in observations if row.get("regime") == regime]
            )
            for regime in regimes
        }
        samples = int(overall["samples"])
        wins = int(overall["wins"])
        losses = int(overall["losses"])
        evidence_mature = (
            samples >= self.minimum_samples
            and len(regimes) >= self.minimum_regimes
            and wins >= self.minimum_class_samples
            and losses >= self.minimum_class_samples
        )
        if samples < self.minimum_samples:
            calibration_state = "waiting_for_samples"
        elif len(regimes) < self.minimum_regimes:
            calibration_state = "waiting_for_regimes"
        elif wins < self.minimum_class_samples or losses < self.minimum_class_samples:
            calibration_state = "waiting_for_class_balance"
        elif expected_calibration_error <= 0.05:
            calibration_state = "calibrated"
        elif expected_calibration_error <= 0.10:
            calibration_state = "calibration_caution"
        else:
            calibration_state = "miscalibrated"
        return {
            "cycles": int(self.state.get("cycles") or 0),
            "calibration_state": calibration_state,
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
            "sample_unit": "fully_closed_costed_paper_trade",
            "partial_exits_are_not_independent_samples": True,
            "minimum_samples": self.minimum_samples,
            "minimum_regimes": self.minimum_regimes,
            "minimum_class_samples": self.minimum_class_samples,
            "regime_count": len(regimes),
            "evidence_mature": evidence_mature,
            "expected_calibration_error": expected_calibration_error,
            "overall": overall,
            "reliability_bins": bins,
            "by_regime": by_regime,
            "suggested_probability_application": "none_advisory_diagnostics_only",
            "last": dict(self.state.get("last") or {}),
            **self._authority_denied(),
        }

    def health(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "state_path": str(self.state_path),
            "cycles": int(self.state.get("cycles") or 0),
            "closed_trade_samples": int(snapshot["overall"]["samples"]),
            "calibration_state": snapshot["calibration_state"],
            "minimum_samples": self.minimum_samples,
            "minimum_regimes": self.minimum_regimes,
            "minimum_class_samples": self.minimum_class_samples,
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
            **self._authority_denied(),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "cycles": 0,
            "observations": [],
            "seen_event_ids": {},
            "last": {},
        }
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) == self.SCHEMA_VERSION:
                payload.setdefault("cycles", 0)
                payload.setdefault("observations", [])
                payload.setdefault("seen_event_ids", {})
                payload.setdefault("last", {})
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return empty

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["updated_at"] = time.time()
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)
