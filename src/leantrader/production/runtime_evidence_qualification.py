from __future__ import annotations

import math
import statistics
import time
from typing import Any

from .evidence_qualification import _digest, _finite, _normal_cdf, _statistics
from .partitioned_evidence_qualification import PartitionedEvidenceQualificationEngine


class RuntimeEvidenceQualificationEngine(PartitionedEvidenceQualificationEngine):
    """Production v1.42 qualifier with strict independence and decay checks.

    The partitioned parent implements the frozen v1.42 protocol, holdout opening,
    purged walk-forward validation, PBO, and persistent qualification lineage.
    This runtime layer adds two conservative production requirements that are
    intentionally stricter than the research helper:

    * temporally overlapping episodes can never count as independent evidence;
    * deflated performance and drift are re-checked with non-normality,
      rolling-confidence, regime, calibration, and evidence-age diagnostics.

    Outcomes beyond the first required holdout remain tagged as holdout by the
    recorder for append-only continuity, but qualification opens and hashes only
    the first precommitted holdout sample count. Later observations cannot alter
    that sealed result.
    """

    VERSION = "1.42.0"
    DRIFT_HISTORY_LIMIT = 1_000
    MINIMUM_DRIFT_WINDOW = 20
    MAXIMUM_EVIDENCE_STALENESS_SECONDS = 7 * 24 * 60 * 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._suppress_parent_history = False
        self.drift_lineage_integrity_ok = self._verify_drift_history()

    def start(self) -> None:
        super().start()
        self.drift_lineage_integrity_ok = self._verify_drift_history()

    @staticmethod
    def _assignment(ordinal: int, plan: dict[str, Any]) -> dict[str, Any]:
        assignment = PartitionedEvidenceQualificationEngine._assignment(
            ordinal,
            plan,
        )
        if assignment.get("partition") == "post_holdout":
            return {
                "partition": "untouched_holdout",
                "walk_forward_fold": None,
            }
        return assignment

    @staticmethod
    def _experiments(prospective_state: dict[str, Any]) -> list[dict[str, Any]]:
        raw = prospective_state.get("experiments") or {}
        if isinstance(raw, dict):
            return [row for _, row in sorted(raw.items()) if isinstance(row, dict)]
        if isinstance(raw, list):
            return [row for row in raw if isinstance(row, dict)]
        return []

    def _independence_audit(
        self,
        prospective_state: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any]]:
        reasons: list[str] = []
        candidates: dict[str, Any] = {}
        counted_partitions = {
            "training",
            "validation",
            "prospective_paper",
            "untouched_holdout",
        }
        for experiment in self._experiments(prospective_state):
            if not self._is_v142(experiment):
                continue
            candidate_id = str(experiment.get("candidate_id") or "").strip()
            if not candidate_id:
                reasons.append("candidate_identity_missing")
                continue
            protocol = experiment.get("protocol") or {}
            try:
                protocol_cost = _finite(protocol.get("round_trip_cost_bps"))
            except ValueError:
                protocol_cost = -math.inf
            if protocol_cost < self.modeled_round_trip_cost_bps:
                reasons.append(f"{candidate_id}:round_trip_cost_floor_violated")

            intervals: list[tuple[float, float, int, str]] = []
            malformed = 0
            for raw in experiment.get("outcomes") or []:
                if not isinstance(raw, dict):
                    malformed += 1
                    continue
                partition = str(raw.get("partition") or "")
                if partition not in counted_partitions:
                    continue
                try:
                    opened = _finite(raw.get("opened_at"))
                    label_end = _finite(raw.get("label_end"))
                    ordinal = int(raw.get("episode_ordinal"))
                except (TypeError, ValueError):
                    malformed += 1
                    continue
                if label_end <= opened:
                    malformed += 1
                    continue
                intervals.append((opened, label_end, ordinal, partition))
            if malformed:
                reasons.append(f"{candidate_id}:malformed_independence_interval")

            intervals.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
            overlap_count = 0
            previous_end = -math.inf
            previous_ordinal: int | None = None
            for opened, label_end, ordinal, _ in intervals:
                if opened < previous_end:
                    overlap_count += 1
                    reasons.append(
                        f"{candidate_id}:overlapping_independent_episode:"
                        f"{previous_ordinal}:{ordinal}"
                    )
                if label_end > previous_end:
                    previous_end = label_end
                    previous_ordinal = ordinal
            candidates[candidate_id] = {
                "audited_intervals": len(intervals),
                "overlap_count": overlap_count,
                "malformed_intervals": malformed,
                "rule": "opened_at >= previous_accepted_label_end",
                "protocol_round_trip_cost_bps": protocol_cost,
            }
        return list(dict.fromkeys(reasons)), {
            "valid": not reasons,
            "rule": "non_overlapping_label_intervals",
            "candidates": candidates,
        }

    @staticmethod
    def _non_normality_deflated_performance(
        values: list[float],
        *,
        number_of_trials: int,
    ) -> dict[str, Any]:
        returns = [float(value) for value in values if math.isfinite(float(value))]
        trials = max(1, int(number_of_trials))
        stats = _statistics(returns)
        if len(returns) < 8 or stats.get("standard_deviation") in {None, 0.0}:
            return {
                "valid": False,
                "deflated_performance_statistic": -math.inf,
                "multiple_testing_adjusted_p_value": 1.0,
                "number_of_trials": trials,
                "statistics": stats,
                "reason": "insufficient_non_normality_evidence",
            }
        deviation = float(stats["standard_deviation"])
        mean = float(stats["mean"])
        centered = [value - mean for value in returns]
        second = sum(value**2 for value in centered) / len(centered)
        if second <= 0.0:
            return {
                "valid": False,
                "deflated_performance_statistic": -math.inf,
                "multiple_testing_adjusted_p_value": 1.0,
                "number_of_trials": trials,
                "statistics": stats,
                "reason": "zero_return_variance",
            }
        third = sum(value**3 for value in centered) / len(centered)
        fourth = sum(value**4 for value in centered) / len(centered)
        skewness = third / (second ** 1.5)
        kurtosis = fourth / (second**2)
        sharpe_like = mean / deviation
        variance_inflation = (
            1.0
            - skewness * sharpe_like
            + ((kurtosis - 1.0) / 4.0) * (sharpe_like**2)
        )
        if not math.isfinite(variance_inflation) or variance_inflation <= 0.0:
            return {
                "valid": False,
                "deflated_performance_statistic": -math.inf,
                "multiple_testing_adjusted_p_value": 1.0,
                "number_of_trials": trials,
                "statistics": stats,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "reason": "invalid_non_normality_adjustment",
            }
        adjusted_statistic = (
            sharpe_like
            * math.sqrt(max(1, len(returns) - 1))
            / math.sqrt(variance_inflation)
        )
        expected_null_max = math.sqrt(2.0 * math.log(max(2, trials)))
        deflated = adjusted_statistic - expected_null_max
        one_sided = 1.0 - _normal_cdf(adjusted_statistic)
        adjusted_p = min(1.0, max(0.0, one_sided) * trials)
        return {
            "valid": True,
            "deflated_performance_statistic": deflated,
            "multiple_testing_adjusted_p_value": adjusted_p,
            "number_of_trials": trials,
            "observations": len(returns),
            "sharpe_like": sharpe_like,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "variance_inflation": variance_inflation,
            "non_normality_adjusted_statistic": adjusted_statistic,
            "expected_null_max_statistic": expected_null_max,
            "statistics": stats,
            "reason": None,
        }

    def _enhanced_drift(
        self,
        rows: list[dict[str, Any]],
        *,
        base_validation: dict[str, Any],
    ) -> dict[str, Any]:
        clean: list[dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            try:
                value = _finite(raw.get("net_return"))
                recorded_at = _finite(raw.get("recorded_at"))
            except ValueError:
                continue
            clean.append(
                {
                    "net_return": value,
                    "recorded_at": recorded_at,
                    "regime": str(raw.get("regime") or "unknown"),
                }
            )
        clean.sort(key=lambda row: (row["recorded_at"], row["regime"]))
        if len(clean) < self.minimum_samples:
            return {
                "valid": False,
                "drift_stable": False,
                "decay_state": "insufficient_evidence",
                "samples": len(clean),
                "reason": "insufficient_samples",
            }
        regimes: dict[str, list[float]] = {}
        for row in clean:
            regimes.setdefault(row["regime"], []).append(float(row["net_return"]))
        if len(regimes) < self.minimum_regimes:
            return {
                "valid": False,
                "drift_stable": False,
                "decay_state": "insufficient_regimes",
                "samples": len(clean),
                "regimes": len(regimes),
                "reason": "insufficient_regimes",
            }

        window = max(
            self.MINIMUM_DRIFT_WINDOW,
            min(50, max(self.MINIMUM_DRIFT_WINDOW, len(clean) // 4)),
        )
        window = min(window, len(clean) // 2)
        historical_rows = clean[:-window]
        recent_rows = clean[-window:]
        overall = _statistics([row["net_return"] for row in clean])
        historical = _statistics([row["net_return"] for row in historical_rows])
        recent = _statistics([row["net_return"] for row in recent_rows])
        rolling: list[dict[str, Any]] = []
        for end in range(window, len(clean) + 1, window):
            chunk = clean[end - window : end]
            chunk_stats = _statistics([row["net_return"] for row in chunk])
            rolling.append(
                {
                    "start_recorded_at": chunk[0]["recorded_at"],
                    "end_recorded_at": chunk[-1]["recorded_at"],
                    "samples": chunk_stats["samples"],
                    "mean_net_return": chunk_stats["mean"],
                    "lower_95_net_return": chunk_stats["lower_95"],
                    "upper_95_net_return": chunk_stats["upper_95"],
                }
            )
        if not rolling or rolling[-1]["end_recorded_at"] != clean[-1]["recorded_at"]:
            chunk = clean[-window:]
            chunk_stats = _statistics([row["net_return"] for row in chunk])
            rolling.append(
                {
                    "start_recorded_at": chunk[0]["recorded_at"],
                    "end_recorded_at": chunk[-1]["recorded_at"],
                    "samples": chunk_stats["samples"],
                    "mean_net_return": chunk_stats["mean"],
                    "lower_95_net_return": chunk_stats["lower_95"],
                    "upper_95_net_return": chunk_stats["upper_95"],
                }
            )
        regime_statistics = {
            name: _statistics(values)
            for name, values in sorted(regimes.items())
        }
        historical_mean = float(historical["mean"])
        recent_mean = float(recent["mean"])
        retention = (
            recent_mean / historical_mean
            if historical_mean > 0.0
            else (-math.inf if recent_mean < 0.0 else 0.0)
        )

        calibration_reliable = base_validation.get("calibration_reliable")
        historical_ece = base_validation.get("calibration_ece_historical")
        recent_ece = base_validation.get("calibration_ece_recent")
        calibration_degraded = calibration_reliable is False
        calibration_state = (
            "reliable" if calibration_reliable is True else
            "unreliable" if calibration_reliable is False else
            "unavailable"
        )
        try:
            if historical_ece is not None and recent_ece is not None:
                historical_ece_value = _finite(historical_ece)
                recent_ece_value = _finite(recent_ece)
                calibration_degraded = calibration_degraded or (
                    recent_ece_value
                    > max(historical_ece_value * 1.5, historical_ece_value + 0.02)
                )
                calibration_state = (
                    "degraded" if calibration_degraded else "stable"
                )
            else:
                historical_ece_value = None
                recent_ece_value = None
        except ValueError:
            historical_ece_value = None
            recent_ece_value = None
            calibration_degraded = True
            calibration_state = "malformed"

        latest_at = float(clean[-1]["recorded_at"])
        age_seconds = max(0.0, time.time() - latest_at)
        stale = age_seconds > self.MAXIMUM_EVIDENCE_STALENESS_SECONDS
        stable = bool(
            overall.get("lower_95") is not None
            and float(overall["lower_95"]) > 0.0
            and historical_mean > 0.0
            and recent_mean > 0.0
            and retention >= 0.50
            and all(float(stats["mean"]) > 0.0 for stats in regime_statistics.values())
            and not calibration_degraded
            and not stale
        )
        if stale:
            decay_state = "stale"
            reason = "evidence_stale"
        elif calibration_degraded:
            decay_state = "calibration_degraded"
            reason = "calibration_degradation"
        elif recent_mean <= 0.0 or retention < 0.50:
            decay_state = "edge_decay"
            reason = "recent_edge_degradation"
        elif any(float(stats["mean"]) <= 0.0 for stats in regime_statistics.values()):
            decay_state = "regime_instability"
            reason = "regime_instability"
        elif overall.get("lower_95") is None or float(overall["lower_95"]) <= 0.0:
            decay_state = "confidence_not_positive"
            reason = "positive_edge_confidence_not_proven"
        else:
            decay_state = "stable"
            reason = None
        return {
            "valid": True,
            "drift_stable": stable,
            "decay_state": decay_state,
            "samples": len(clean),
            "regimes": len(regimes),
            "window_samples": window,
            "overall": overall,
            "historical": historical,
            "recent": recent,
            "edge_retention_ratio": retention,
            "rolling_expectancy": rolling,
            "regime_statistics": regime_statistics,
            "calibration": {
                "state": calibration_state,
                "reliable": calibration_reliable,
                "historical_ece": historical_ece_value,
                "recent_ece": recent_ece_value,
                "degraded": calibration_degraded,
            },
            "latest_recorded_at": latest_at,
            "evidence_age_seconds": age_seconds,
            "maximum_evidence_staleness_seconds": self.MAXIMUM_EVIDENCE_STALENESS_SECONDS,
            "reason": reason,
        }

    def _verify_drift_history(self) -> bool:
        rows = self.state.get("drift_observations", [])
        if not isinstance(rows, list):
            return False
        previous = str(self.state.get("drift_anchor_hash") or "GENESIS")
        expected = int(self.state.get("drift_anchor_sequence") or 0) + 1
        for record in rows:
            if not isinstance(record, dict):
                return False
            supplied = str(record.get("record_hash") or "")
            unhashed = {key: value for key, value in record.items() if key != "record_hash"}
            if (
                int(record.get("sequence") or 0) != expected
                or str(record.get("previous_hash") or "") != previous
                or supplied != _digest(unhashed)
            ):
                return False
            previous = supplied
            expected += 1
        return True

    def _append_drift_observation(self, payload: dict[str, Any]) -> str | None:
        if not self.drift_lineage_integrity_ok:
            return None
        rows = list(self.state.get("drift_observations", []))
        previous_hash = (
            rows[-1]["record_hash"]
            if rows
            else str(self.state.get("drift_anchor_hash") or "GENESIS")
        )
        previous_sequence = (
            int(rows[-1]["sequence"])
            if rows
            else int(self.state.get("drift_anchor_sequence") or 0)
        )
        record = {
            "sequence": previous_sequence + 1,
            "previous_hash": previous_hash,
            "observed_at": time.time(),
            "payload": payload,
        }
        record["record_hash"] = _digest(record)
        rows.append(record)
        if len(rows) > self.DRIFT_HISTORY_LIMIT:
            removed = rows[: -self.DRIFT_HISTORY_LIMIT]
            self.state["drift_anchor_hash"] = removed[-1]["record_hash"]
            self.state["drift_anchor_sequence"] = int(removed[-1]["sequence"])
            rows = rows[-self.DRIFT_HISTORY_LIMIT :]
        self.state["drift_observations"] = rows
        self._save()
        return str(record["record_hash"])

    def _append_history(self, payload: dict[str, Any]) -> None:
        if self._suppress_parent_history:
            return
        super()._append_history(payload)

    def qualify(
        self,
        prospective_state: dict[str, Any],
        *,
        base_validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        experiments = self._experiments(prospective_state)
        if not any(self._is_v142(row) for row in experiments):
            return super().qualify(
                prospective_state,
                base_validation=base_validation,
            )
        base = dict(base_validation or {})
        if not self.drift_lineage_integrity_ok:
            return self._fail_closed(base, ["drift_lineage_integrity_failure"])
        independence_reasons, independence = self._independence_audit(
            prospective_state
        )
        if independence_reasons:
            blocked = self._fail_closed(
                base,
                ["independent_sample_integrity_failure", *independence_reasons],
            )
            blocked["qualification"]["independence"] = independence
            return blocked

        self._suppress_parent_history = True
        try:
            result = super().qualify(
                prospective_state,
                base_validation=base,
            )
        finally:
            self._suppress_parent_history = False

        qualification = result.get("qualification")
        if not isinstance(qualification, dict):
            return self._fail_closed(base, ["qualification_contract_malformed"])
        qualification["independence"] = independence
        selected_candidate = str(qualification.get("selected_candidate") or "")
        selected = next(
            (
                row
                for row in experiments
                if str(row.get("candidate_id") or "") == selected_candidate
            ),
            None,
        )
        mature_trials = 0
        for experiment in experiments:
            if not self._is_v142(experiment):
                continue
            plan = (experiment.get("protocol") or {}).get("partition_plan") or {}
            required = int(plan.get("walk_forward_folds") or 0) * int(
                plan.get("validation_samples_per_fold") or 0
            )
            observed = sum(
                1
                for row in experiment.get("outcomes") or []
                if isinstance(row, dict) and row.get("partition") == "validation"
            )
            if required > 0 and observed >= required:
                mature_trials += 1

        validation_returns: list[float] = []
        prospective_rows: list[dict[str, Any]] = []
        if isinstance(selected, dict):
            for row in selected.get("outcomes") or []:
                if not isinstance(row, dict):
                    continue
                if row.get("partition") == "validation":
                    try:
                        validation_returns.append(_finite(row.get("net_return")))
                    except ValueError:
                        pass
                elif row.get("partition") == "prospective_paper":
                    prospective_rows.append(row)

        non_normality = self._non_normality_deflated_performance(
            validation_returns,
            number_of_trials=max(1, mature_trials),
        )
        enhanced_drift = self._enhanced_drift(
            prospective_rows,
            base_validation=base,
        )
        parent_deflated = _finite(
            result.get("deflated_performance_statistic"),
            default=-math.inf,
        )
        strict_deflated = min(
            parent_deflated,
            _finite(
                non_normality.get("deflated_performance_statistic"),
                default=-math.inf,
            ),
        )
        non_normality_passed = bool(
            non_normality.get("valid") is True
            and float(non_normality.get("multiple_testing_adjusted_p_value", 1.0))
            < self.P_VALUE_CEILING
            and float(non_normality.get("deflated_performance_statistic", -math.inf))
            > 0.0
        )
        result["deflated_performance_statistic"] = strict_deflated
        result["multiple_testing_controlled"] = bool(
            result.get("multiple_testing_controlled") is True
            and non_normality_passed
        )
        result["drift_stable"] = bool(
            result.get("drift_stable") is True
            and enhanced_drift.get("drift_stable") is True
        )
        qualification["non_normality_deflated_performance"] = non_normality
        qualification["drift_v1_42"] = enhanced_drift
        reasons = list(qualification.get("reasons") or [])
        if not non_normality_passed:
            reasons.append(
                str(non_normality.get("reason") or "non_normality_deflated_performance_failed")
            )
        if enhanced_drift.get("drift_stable") is not True:
            reasons.append(
                str(enhanced_drift.get("reason") or "enhanced_drift_not_stable")
            )
        qualification["reasons"] = list(dict.fromkeys(reasons))

        drift_record_hash = self._append_drift_observation(
            {
                "candidate_id": selected_candidate,
                "drift_stable": result["drift_stable"],
                "decay_state": enhanced_drift.get("decay_state"),
                "samples": enhanced_drift.get("samples", 0),
                "regimes": enhanced_drift.get("regimes", 0),
                "recent_mean": (enhanced_drift.get("recent") or {}).get("mean"),
                "historical_mean": (enhanced_drift.get("historical") or {}).get("mean"),
                "edge_retention_ratio": enhanced_drift.get("edge_retention_ratio"),
                "evidence_age_seconds": enhanced_drift.get("evidence_age_seconds"),
            }
        )
        qualification["drift_lineage_record_hash"] = drift_record_hash
        qualification["drift_lineage_integrity_ok"] = self.drift_lineage_integrity_ok

        hash_payload = {
            "version": self.VERSION,
            "protocol": self.PROTOCOL,
            "configuration": self._configuration(),
            "prospective_source_hash": _digest(prospective_state),
            "selected_candidate": selected_candidate,
            "contract": {
                key: value
                for key, value in result.items()
                if key != "evidence_reproducibility_hash"
            },
        }
        result["evidence_reproducibility_hash"] = _digest(hash_payload)
        self._append_history(
            {
                "evidence_reproducibility_hash": result["evidence_reproducibility_hash"],
                "protocol": self.PROTOCOL,
                "selected_candidate": selected_candidate,
                "independent_samples": int(result.get("independent_samples") or 0),
                "purged_walk_forward_passed": result.get("purged_walk_forward_passed") is True,
                "embargo_applied": result.get("embargo_applied") is True,
                "untouched_holdout_passed": result.get("untouched_holdout_passed") is True,
                "multiple_testing_controlled": result.get("multiple_testing_controlled") is True,
                "pbo": result.get("probability_backtest_overfitting"),
                "deflated_performance_statistic": result.get("deflated_performance_statistic"),
                "drift_stable": result.get("drift_stable") is True,
                "drift_lineage_record_hash": drift_record_hash,
            }
        )
        return result

    def health(self) -> dict[str, Any]:
        health = super().health()
        drift_rows = self.state.get("drift_observations", [])
        health.update(
            {
                "version": self.VERSION,
                "runtime_independence_enforced": True,
                "independence_rule": "non_overlapping_label_intervals",
                "non_normality_adjustment": True,
                "rolling_drift_detection": True,
                "drift_lineage_integrity_ok": self.drift_lineage_integrity_ok,
                "drift_observations": len(drift_rows) if isinstance(drift_rows, list) else 0,
                "maximum_evidence_staleness_seconds": self.MAXIMUM_EVIDENCE_STALENESS_SECONDS,
                "automatic_promotion": False,
                "paper_promotion_authority": False,
                "testnet_authority": False,
                "live_authority": False,
                "execution_authority": False,
            }
        )
        return health
