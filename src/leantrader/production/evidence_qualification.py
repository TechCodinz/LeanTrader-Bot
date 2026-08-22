from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _finite(value: Any, *, default: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        if default is None:
            raise ValueError("value must be numeric")
        return default
    if not math.isfinite(number):
        if default is None:
            raise ValueError("value must be finite")
        return default
    return number


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _statistics(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    samples = len(rows)
    if not rows:
        return {
            "samples": 0,
            "mean": 0.0,
            "standard_deviation": None,
            "standard_error": None,
            "lower_95": None,
            "upper_95": None,
            "t_statistic": 0.0,
            "one_sided_p_value": 1.0,
        }
    mean = sum(rows) / samples
    if samples > 1:
        deviation = statistics.stdev(rows)
        standard_error = deviation / math.sqrt(samples)
        if standard_error > 0.0:
            t_statistic = mean / standard_error
            one_sided = 1.0 - _normal_cdf(t_statistic)
        else:
            t_statistic = math.inf if mean > 0.0 else -math.inf if mean < 0.0 else 0.0
            one_sided = 0.0 if mean > 0.0 else 1.0
        lower = mean - 1.96 * standard_error
        upper = mean + 1.96 * standard_error
    else:
        deviation = None
        standard_error = None
        t_statistic = 0.0
        one_sided = 1.0
        lower = None
        upper = None
    return {
        "samples": samples,
        "mean": mean,
        "standard_deviation": deviation,
        "standard_error": standard_error,
        "lower_95": lower,
        "upper_95": upper,
        "t_statistic": t_statistic,
        "one_sided_p_value": max(0.0, min(1.0, one_sided)),
    }


def purged_walk_forward_validation(
    folds: list[dict[str, Any]],
    *,
    embargo_seconds: float,
) -> dict[str, Any]:
    """Validate chronological walk-forward folds with purge and embargo semantics.

    Each observation must expose feature_start, feature_end and label_end as
    monotonic numeric timestamps. Training observations must end before the
    validation window begins, and the gap must be at least embargo_seconds.
    The function validates a supplied split; it never silently drops leaking
    rows because doing so after outcomes are known would hide leakage.
    """
    embargo = _finite(embargo_seconds)
    if embargo < 0.0:
        raise ValueError("embargo_seconds cannot be negative")
    reasons: list[str] = []
    fold_results: list[dict[str, Any]] = []
    previous_validation_end: float | None = None
    for index, fold in enumerate(folds):
        train = list(fold.get("training") or []) if isinstance(fold, dict) else []
        validation = list(fold.get("validation") or []) if isinstance(fold, dict) else []
        if not train or not validation:
            reasons.append(f"fold_{index}_missing_partition")
            continue
        try:
            train_rows = [
                (
                    _finite(row.get("feature_start")),
                    _finite(row.get("feature_end")),
                    _finite(row.get("label_end")),
                )
                for row in train
                if isinstance(row, dict)
            ]
            validation_rows = [
                (
                    _finite(row.get("feature_start")),
                    _finite(row.get("feature_end")),
                    _finite(row.get("label_end")),
                )
                for row in validation
                if isinstance(row, dict)
            ]
        except ValueError:
            reasons.append(f"fold_{index}_invalid_timestamp")
            continue
        if len(train_rows) != len(train) or len(validation_rows) != len(validation):
            reasons.append(f"fold_{index}_malformed_observation")
            continue
        if any(start > end or end > label_end for start, end, label_end in train_rows + validation_rows):
            reasons.append(f"fold_{index}_invalid_interval")
            continue
        validation_start = min(row[0] for row in validation_rows)
        validation_end = max(row[2] for row in validation_rows)
        train_label_end = max(row[2] for row in train_rows)
        chronological = train_label_end < validation_start
        gap = validation_start - train_label_end
        purged = chronological and gap >= embargo
        if not chronological:
            reasons.append(f"fold_{index}_training_validation_leakage")
        elif gap < embargo:
            reasons.append(f"fold_{index}_embargo_violation")
        if previous_validation_end is not None and validation_start <= previous_validation_end:
            reasons.append(f"fold_{index}_validation_windows_overlap")
        previous_validation_end = validation_end
        fold_results.append(
            {
                "fold": index,
                "training_samples": len(train_rows),
                "validation_samples": len(validation_rows),
                "training_label_end": train_label_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "embargo_gap_seconds": gap,
                "purged": purged,
            }
        )
    passed = bool(folds) and len(fold_results) == len(folds) and not reasons
    return {
        "passed": passed,
        "purged_walk_forward_passed": passed,
        "embargo_applied": passed and embargo > 0.0,
        "embargo_seconds": embargo,
        "folds": fold_results,
        "reasons": reasons,
    }


def probability_of_backtest_overfitting(
    strategy_returns: dict[str, list[float]],
    *,
    segments: int = 8,
) -> dict[str, Any]:
    """Compute a deterministic CSCV-style Probability of Backtest Overfitting.

    All strategies must have the same number of aligned observations. The best
    strategy in each in-sample combination is ranked in its complementary
    out-of-sample set. PBO is the fraction whose out-of-sample rank falls in
    the lower half. Insufficient matrices fail closed with PBO=1.
    """
    names = sorted(strategy_returns)
    if len(names) < 2:
        return {"valid": False, "pbo": 1.0, "splits": 0, "reason": "fewer_than_two_trials"}
    lengths = {len(strategy_returns[name]) for name in names}
    if len(lengths) != 1:
        return {"valid": False, "pbo": 1.0, "splits": 0, "reason": "unaligned_trial_lengths"}
    observations = lengths.pop()
    if observations < 8:
        return {"valid": False, "pbo": 1.0, "splits": 0, "reason": "insufficient_aligned_observations"}
    segment_count = min(max(4, int(segments)), observations)
    if segment_count % 2:
        segment_count -= 1
    if segment_count < 4:
        return {"valid": False, "pbo": 1.0, "splits": 0, "reason": "insufficient_segments"}
    # Deterministic near-equal contiguous segmentation.
    boundaries = [round(index * observations / segment_count) for index in range(segment_count + 1)]
    segment_indices = [list(range(boundaries[i], boundaries[i + 1])) for i in range(segment_count)]
    half = segment_count // 2
    lambdas: list[float] = []
    for selected in itertools.combinations(range(segment_count), half):
        # Complementary pairs are symmetric; keep only one representative.
        complement = tuple(index for index in range(segment_count) if index not in selected)
        if selected > complement:
            continue
        train_indices = [idx for segment in selected for idx in segment_indices[segment]]
        validation_indices = [idx for segment in complement for idx in segment_indices[segment]]
        train_scores = {
            name: sum(strategy_returns[name][idx] for idx in train_indices) / len(train_indices)
            for name in names
        }
        winner = max(names, key=lambda name: (train_scores[name], name))
        validation_scores = {
            name: sum(strategy_returns[name][idx] for idx in validation_indices) / len(validation_indices)
            for name in names
        }
        ranked = sorted(names, key=lambda name: (validation_scores[name], name))
        rank = ranked.index(winner) + 1
        percentile = (rank - 0.5) / len(ranked)
        percentile = min(1.0 - 1e-12, max(1e-12, percentile))
        lambdas.append(math.log(percentile / (1.0 - percentile)))
    if not lambdas:
        return {"valid": False, "pbo": 1.0, "splits": 0, "reason": "no_cscv_splits"}
    pbo = sum(1 for value in lambdas if value <= 0.0) / len(lambdas)
    return {
        "valid": True,
        "pbo": pbo,
        "splits": len(lambdas),
        "segments": segment_count,
        "median_logit_rank": statistics.median(lambdas),
        "reason": None,
    }


def deflated_performance_evidence(
    returns: list[float],
    *,
    number_of_trials: int,
) -> dict[str, Any]:
    """Return a conservative multiple-testing-deflated performance statistic."""
    stats = _statistics(returns)
    trials = max(1, int(number_of_trials))
    if stats["samples"] < 2 or stats["standard_error"] in {None, 0.0}:
        return {
            "valid": False,
            "deflated_performance_statistic": -math.inf,
            "multiple_testing_adjusted_p_value": 1.0,
            "number_of_trials": trials,
            "statistics": stats,
            "reason": "insufficient_variance_or_samples",
        }
    observed = float(stats["t_statistic"])
    # Extreme-value approximation for the best statistic expected under the
    # null after trying many alternatives. This intentionally grows with the
    # number of trials and therefore cannot reward strategy proliferation.
    expected_null_max = math.sqrt(2.0 * math.log(max(2, trials)))
    deflated = observed - expected_null_max
    adjusted_p = min(1.0, float(stats["one_sided_p_value"]) * trials)
    return {
        "valid": True,
        "deflated_performance_statistic": deflated,
        "multiple_testing_adjusted_p_value": adjusted_p,
        "number_of_trials": trials,
        "expected_null_max_statistic": expected_null_max,
        "observed_statistic": observed,
        "statistics": stats,
        "reason": None,
    }


def drift_and_edge_decay(
    outcomes: list[dict[str, Any]],
    *,
    minimum_samples: int = 100,
    minimum_regimes: int = 2,
) -> dict[str, Any]:
    rows = [
        row
        for row in outcomes
        if isinstance(row, dict)
        and math.isfinite(_finite(row.get("net_return"), default=math.nan))
    ]
    rows.sort(key=lambda row: (_finite(row.get("recorded_at"), default=0.0), str(row.get("symbol") or "")))
    if len(rows) < int(minimum_samples):
        return {
            "valid": False,
            "drift_stable": False,
            "samples": len(rows),
            "regimes": 0,
            "reason": "insufficient_samples",
        }
    regime_returns: dict[str, list[float]] = {}
    for row in rows:
        regime_returns.setdefault(str(row.get("regime") or "unknown"), []).append(float(row["net_return"]))
    if len(regime_returns) < int(minimum_regimes):
        return {
            "valid": False,
            "drift_stable": False,
            "samples": len(rows),
            "regimes": len(regime_returns),
            "reason": "insufficient_regimes",
        }
    midpoint = len(rows) // 2
    early = _statistics([float(row["net_return"]) for row in rows[:midpoint]])
    recent = _statistics([float(row["net_return"]) for row in rows[midpoint:]])
    regime_means = {name: sum(values) / len(values) for name, values in sorted(regime_returns.items())}
    early_mean = float(early["mean"])
    recent_mean = float(recent["mean"])
    decay_ratio = recent_mean / early_mean if early_mean > 0.0 else (-math.inf if recent_mean < 0.0 else 0.0)
    # A mature edge is considered stable only if the recent half remains
    # positive, retains at least half of a previously positive edge, and no
    # observed regime has negative expectancy.
    stable = (
        early_mean > 0.0
        and recent_mean > 0.0
        and decay_ratio >= 0.50
        and all(value > 0.0 for value in regime_means.values())
    )
    return {
        "valid": True,
        "drift_stable": stable,
        "samples": len(rows),
        "regimes": len(regime_returns),
        "early_mean": early_mean,
        "recent_mean": recent_mean,
        "edge_retention_ratio": decay_ratio,
        "regime_means": regime_means,
        "reason": None if stable else "edge_decay_or_regime_instability",
    }


class EvidenceQualificationEngine:
    """Persistent v1.42 research-evidence qualifier with no trading authority."""

    VERSION = "1.42.0"
    SCHEMA_VERSION = 1
    MINIMUM_SAMPLE_FLOOR = 100
    MINIMUM_REGIMES_FLOOR = 2
    MINIMUM_COST_FLOOR_BPS = 30.0
    HISTORY_LIMIT = 1_000

    def __init__(
        self,
        state_path: Path,
        *,
        prospective_state_path: Path | None = None,
        minimum_samples: int = 100,
        minimum_regimes: int = 2,
        modeled_round_trip_cost_bps: float = 30.0,
        embargo_seconds: float = 1.0,
    ) -> None:
        if int(minimum_samples) < self.MINIMUM_SAMPLE_FLOOR:
            raise ValueError("evidence qualification cannot lower the 100-sample floor")
        if int(minimum_regimes) < self.MINIMUM_REGIMES_FLOOR:
            raise ValueError("evidence qualification requires at least two regimes")
        if float(modeled_round_trip_cost_bps) < self.MINIMUM_COST_FLOOR_BPS:
            raise ValueError("evidence qualification cannot lower the 30-bps cost floor")
        if float(embargo_seconds) <= 0.0:
            raise ValueError("embargo must be positive")
        self.state_path = Path(state_path)
        self.prospective_state_path = Path(prospective_state_path) if prospective_state_path else None
        self.minimum_samples = int(minimum_samples)
        self.minimum_regimes = int(minimum_regimes)
        self.modeled_round_trip_cost_bps = float(modeled_round_trip_cost_bps)
        self.embargo_seconds = float(embargo_seconds)
        self.last_error: str | None = None
        self.state = self._load()
        self.lineage_integrity_ok = self._verify_history()

    @staticmethod
    def _authority_denied() -> dict[str, bool]:
        return {
            "research_only": True,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "execution_authority": False,
            "automatic_promotion": False,
        }

    def start(self) -> None:
        self.state = self._load()
        self.lineage_integrity_ok = self._verify_history()

    def stop(self) -> None:
        if self.lineage_integrity_ok:
            self._save()

    def _manifest_payload(self, experiment: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_id": str(experiment.get("candidate_id") or ""),
            "base_strategy": str(experiment.get("base_strategy") or ""),
            "manifest_sha256": str(experiment.get("manifest_sha256") or ""),
            "protocol": experiment.get("protocol") or {},
            "selection_evidence": experiment.get("selection_evidence") or {},
        }

    @staticmethod
    def _outcome_payload(outcome: dict[str, Any]) -> dict[str, Any]:
        return {
            "recorded_at": _finite(outcome.get("recorded_at"), default=0.0),
            "strategy": str(outcome.get("strategy") or ""),
            "symbol": str(outcome.get("symbol") or ""),
            "regime": str(outcome.get("regime") or "unknown"),
            "net_return": _finite(outcome.get("net_return"), default=math.nan),
            "evidence_authority": str(outcome.get("evidence_authority") or ""),
        }

    def _freeze_or_validate(
        self,
        experiment: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        candidate_id = str(experiment.get("candidate_id") or "").strip()
        if not candidate_id:
            return {}, [], [], ["candidate_identity_missing"]
        outcomes = [self._outcome_payload(row) for row in list(experiment.get("outcomes") or []) if isinstance(row, dict)]
        if any(not math.isfinite(float(row["net_return"])) for row in outcomes):
            return {}, [], [], [f"{candidate_id}:non_finite_outcome"]
        freezes = self.state.setdefault("holdout_freezes", {})
        manifest_hash = _digest(self._manifest_payload(experiment))
        freeze = freezes.get(candidate_id)
        if not isinstance(freeze, dict):
            prefix_hash = _digest(outcomes)
            freeze = {
                "candidate_id": candidate_id,
                "frozen_at": time.time(),
                "frozen_outcome_count": len(outcomes),
                "manifest_hash": manifest_hash,
                "prospective_prefix_hash": prefix_hash,
                "freeze_hash": "",
            }
            freeze["freeze_hash"] = _digest({key: value for key, value in freeze.items() if key != "freeze_hash"})
            freezes[candidate_id] = freeze
            self._save()
            return freeze, outcomes, [], []
        reasons: list[str] = []
        unhashed = {key: value for key, value in freeze.items() if key != "freeze_hash"}
        if str(freeze.get("freeze_hash") or "") != _digest(unhashed):
            reasons.append(f"{candidate_id}:freeze_record_corrupted")
        if str(freeze.get("manifest_hash") or "") != manifest_hash:
            reasons.append(f"{candidate_id}:manifest_changed_after_freeze")
        count = max(0, int(freeze.get("frozen_outcome_count") or 0))
        if len(outcomes) < count:
            reasons.append(f"{candidate_id}:outcomes_removed_after_freeze")
            return freeze, outcomes, [], reasons
        prefix = outcomes[:count]
        if str(freeze.get("prospective_prefix_hash") or "") != _digest(prefix):
            reasons.append(f"{candidate_id}:prefreeze_outcomes_mutated")
        holdout = outcomes[count:]
        frozen_at = _finite(freeze.get("frozen_at"), default=math.inf)
        if any(float(row["recorded_at"]) < frozen_at for row in holdout):
            reasons.append(f"{candidate_id}:holdout_outcome_predates_freeze")
        return freeze, prefix, holdout, reasons

    @staticmethod
    def _selection_walk_forward(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for experiment in experiments:
            selection = experiment.get("selection_evidence") or {}
            if not isinstance(selection, dict):
                continue
            folds = selection.get("purged_walk_forward_folds")
            if isinstance(folds, list) and folds:
                return folds
        return []

    @staticmethod
    def _selection_return_matrix(experiments: list[dict[str, Any]]) -> dict[str, list[float]]:
        matrix: dict[str, list[float]] = {}
        for experiment in experiments:
            selection = experiment.get("selection_evidence") or {}
            if not isinstance(selection, dict):
                continue
            candidate_id = str(experiment.get("candidate_id") or "").strip()
            values = selection.get("aligned_validation_returns")
            if candidate_id and isinstance(values, list):
                finite = [_finite(value, default=math.nan) for value in values]
                if finite and all(math.isfinite(value) for value in finite):
                    matrix[candidate_id] = finite
        return matrix

    def qualify(
        self,
        prospective_state: dict[str, Any],
        *,
        base_validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = dict(base_validation or {})
        if not self.lineage_integrity_ok:
            return self._fail_closed(base, ["evidence_lineage_integrity_failure"])
        try:
            experiments_map = prospective_state.get("experiments") or {}
            if isinstance(experiments_map, dict):
                experiments = [row for _, row in sorted(experiments_map.items()) if isinstance(row, dict)]
            elif isinstance(experiments_map, list):
                experiments = [row for row in experiments_map if isinstance(row, dict)]
            else:
                experiments = []
            if not experiments:
                return self._fail_closed(base, ["no_prospective_experiments"])

            prospective_by_candidate: dict[str, list[dict[str, Any]]] = {}
            holdout_by_candidate: dict[str, list[dict[str, Any]]] = {}
            freeze_rows: dict[str, Any] = {}
            integrity_reasons: list[str] = []
            for experiment in experiments:
                freeze, prospective_rows, holdout_rows, reasons = self._freeze_or_validate(experiment)
                candidate_id = str(experiment.get("candidate_id") or "").strip()
                if candidate_id:
                    prospective_by_candidate[candidate_id] = prospective_rows
                    holdout_by_candidate[candidate_id] = holdout_rows
                    freeze_rows[candidate_id] = freeze
                integrity_reasons.extend(reasons)

            selected_candidate = max(
                prospective_by_candidate,
                key=lambda name: (len(prospective_by_candidate[name]), name),
                default="",
            )
            prospective_rows = prospective_by_candidate.get(selected_candidate, [])
            holdout_rows = holdout_by_candidate.get(selected_candidate, [])
            prospective_stats = _statistics([float(row["net_return"]) for row in prospective_rows])
            holdout_stats = _statistics([float(row["net_return"]) for row in holdout_rows])
            prospective_regimes = len({str(row.get("regime") or "unknown") for row in prospective_rows})
            holdout_regimes = len({str(row.get("regime") or "unknown") for row in holdout_rows})

            folds = self._selection_walk_forward(experiments)
            walk_forward = purged_walk_forward_validation(folds, embargo_seconds=self.embargo_seconds) if folds else {
                "passed": False,
                "purged_walk_forward_passed": False,
                "embargo_applied": False,
                "embargo_seconds": self.embargo_seconds,
                "folds": [],
                "reasons": ["explicit_purged_walk_forward_folds_missing"],
            }
            matrix = self._selection_return_matrix(experiments)
            pbo = probability_of_backtest_overfitting(matrix)
            deflated = deflated_performance_evidence(
                [float(row["net_return"]) for row in prospective_rows],
                number_of_trials=max(1, len(experiments)),
            )
            drift = drift_and_edge_decay(
                prospective_rows + holdout_rows,
                minimum_samples=self.minimum_samples,
                minimum_regimes=self.minimum_regimes,
            )
            prospective_positive = (
                int(prospective_stats["samples"]) >= self.minimum_samples
                and prospective_regimes >= self.minimum_regimes
                and float(prospective_stats["mean"]) > 0.0
                and prospective_stats["lower_95"] is not None
                and float(prospective_stats["lower_95"]) > 0.0
            )
            untouched_holdout_passed = (
                not integrity_reasons
                and int(holdout_stats["samples"]) >= self.minimum_samples
                and holdout_regimes >= self.minimum_regimes
                and float(holdout_stats["mean"]) > 0.0
                and holdout_stats["lower_95"] is not None
                and float(holdout_stats["lower_95"]) > 0.0
            )
            multiple_testing_controlled = (
                pbo.get("valid") is True
                and deflated.get("valid") is True
                and float(deflated.get("multiple_testing_adjusted_p_value", 1.0)) < 0.05
            )
            reasons = list(integrity_reasons)
            reasons.extend(walk_forward.get("reasons") or [])
            if pbo.get("valid") is not True:
                reasons.append(str(pbo.get("reason") or "pbo_invalid"))
            if deflated.get("valid") is not True:
                reasons.append(str(deflated.get("reason") or "deflated_statistic_invalid"))
            if not prospective_positive:
                reasons.append("prospective_positive_lower_bound_not_proven")
            if not untouched_holdout_passed:
                reasons.append("untouched_holdout_not_mature_or_positive")
            if drift.get("drift_stable") is not True:
                reasons.append(str(drift.get("reason") or "drift_not_stable"))

            contract = {
                **base,
                "independent_samples": int(prospective_stats["samples"]),
                "purged_walk_forward_passed": walk_forward["purged_walk_forward_passed"] is True,
                "embargo_applied": walk_forward["embargo_applied"] is True,
                "untouched_holdout_passed": untouched_holdout_passed,
                "multiple_testing_controlled": multiple_testing_controlled,
                "prospective_net_positive": prospective_positive,
                "drift_stable": drift.get("drift_stable") is True,
                "probability_backtest_overfitting": float(pbo.get("pbo", 1.0)),
                "deflated_performance_statistic": float(deflated.get("deflated_performance_statistic", -math.inf)),
                "partitions": {
                    "training": {
                        "status": "explicit_selection_partition" if folds else "missing_explicit_selection_partition",
                        "folds": len(folds),
                    },
                    "validation": {
                        "status": "purged_embargo_verified" if walk_forward["passed"] else "purge_embargo_not_verified",
                        "folds": len(walk_forward.get("folds") or []),
                    },
                    "prospective_paper": {
                        "candidate_id": selected_candidate,
                        "samples": int(prospective_stats["samples"]),
                        "regimes": prospective_regimes,
                        "mean_net_return": prospective_stats["mean"],
                        "lower_95_net_return": prospective_stats["lower_95"],
                    },
                    "untouched_holdout": {
                        "candidate_id": selected_candidate,
                        "samples": int(holdout_stats["samples"]),
                        "regimes": holdout_regimes,
                        "mean_net_return": holdout_stats["mean"],
                        "lower_95_net_return": holdout_stats["lower_95"],
                        "freeze": freeze_rows.get(selected_candidate) or {},
                    },
                },
                "qualification": {
                    "version": self.VERSION,
                    "valid": not integrity_reasons,
                    "reasons": list(dict.fromkeys(reasons)),
                    "walk_forward": walk_forward,
                    "pbo": pbo,
                    "deflated_performance": deflated,
                    "drift": drift,
                    "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
                    "minimum_samples": self.minimum_samples,
                    "minimum_regimes": self.minimum_regimes,
                },
            }
            reproducibility_payload = {
                "version": self.VERSION,
                "configuration": self._configuration(),
                "prospective_source_hash": _digest(prospective_state),
                "contract": contract,
            }
            contract["evidence_reproducibility_hash"] = _digest(reproducibility_payload)
            self._append_history(
                {
                    "evidence_reproducibility_hash": contract["evidence_reproducibility_hash"],
                    "selected_candidate": selected_candidate,
                    "independent_samples": contract["independent_samples"],
                    "holdout_samples": contract["partitions"]["untouched_holdout"]["samples"],
                    "purged_walk_forward_passed": contract["purged_walk_forward_passed"],
                    "untouched_holdout_passed": contract["untouched_holdout_passed"],
                    "pbo": contract["probability_backtest_overfitting"],
                    "deflated_performance_statistic": contract["deflated_performance_statistic"],
                    "drift_stable": contract["drift_stable"],
                }
            )
            self.last_error = None
            return {**contract, **self._authority_denied()}
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self._fail_closed(base, ["evidence_qualification_input_failure", self.last_error])

    def qualify_from_disk(self, *, base_validation: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.prospective_state_path is None:
            return self._fail_closed(dict(base_validation or {}), ["prospective_state_path_not_configured"])
        try:
            payload = json.loads(self.prospective_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self._fail_closed(dict(base_validation or {}), ["prospective_state_unavailable"])
        return self.qualify(payload, base_validation=base_validation)

    def _fail_closed(self, base: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
        contract = {
            **base,
            "independent_samples": 0,
            "purged_walk_forward_passed": False,
            "embargo_applied": False,
            "untouched_holdout_passed": False,
            "multiple_testing_controlled": False,
            "prospective_net_positive": False,
            "drift_stable": False,
            "probability_backtest_overfitting": 1.0,
            "deflated_performance_statistic": -math.inf,
            "partitions": {
                "training": {"status": "unavailable"},
                "validation": {"status": "unavailable"},
                "prospective_paper": {"samples": 0, "status": "unavailable"},
                "untouched_holdout": {"samples": 0, "status": "unavailable"},
            },
            "qualification": {
                "version": self.VERSION,
                "valid": False,
                "reasons": reasons,
                "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
                "minimum_samples": self.minimum_samples,
                "minimum_regimes": self.minimum_regimes,
            },
            "evidence_reproducibility_hash": None,
        }
        return {**contract, **self._authority_denied()}

    def _configuration(self) -> dict[str, Any]:
        return {
            "minimum_samples": self.minimum_samples,
            "minimum_regimes": self.minimum_regimes,
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
            "embargo_seconds": self.embargo_seconds,
        }

    def health(self) -> dict[str, Any]:
        history = self.state.get("history", [])
        return {
            "version": self.VERSION,
            "persistent": True,
            "state_path": str(self.state_path),
            "prospective_state_path": str(self.prospective_state_path) if self.prospective_state_path else None,
            "lineage_integrity_ok": self.lineage_integrity_ok,
            "records": len(history) if isinstance(history, list) else 0,
            "holdout_freezes": len(self.state.get("holdout_freezes", {})) if isinstance(self.state.get("holdout_freezes"), dict) else 0,
            "last_error": self.last_error,
            "configuration": self._configuration(),
            **self._authority_denied(),
        }

    def _append_history(self, payload: dict[str, Any]) -> None:
        if not self.lineage_integrity_ok:
            return
        history = list(self.state.get("history", []))
        previous_hash = history[-1]["record_hash"] if history else str(self.state.get("anchor_hash") or "GENESIS")
        previous_sequence = int(history[-1]["sequence"]) if history else int(self.state.get("anchor_sequence") or 0)
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

    def _verify_history(self) -> bool:
        history = self.state.get("history", [])
        if not isinstance(history, list):
            return False
        previous = str(self.state.get("anchor_hash") or "GENESIS")
        expected = int(self.state.get("anchor_sequence") or 0) + 1
        for record in history:
            if not isinstance(record, dict):
                return False
            supplied = str(record.get("record_hash") or "")
            unhashed = {key: value for key, value in record.items() if key != "record_hash"}
            if int(record.get("sequence") or 0) != expected or record.get("previous_hash") != previous or supplied != _digest(unhashed):
                return False
            previous = supplied
            expected += 1
        return True

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": self.SCHEMA_VERSION, "history": [], "holdout_freezes": {}}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == self.SCHEMA_VERSION and isinstance(payload.get("history"), list) and isinstance(payload.get("holdout_freezes", {}), dict):
                return payload
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return {"schema_version": self.SCHEMA_VERSION, "history": "invalid", "holdout_freezes": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
