from __future__ import annotations

import math
import time
from typing import Any

from .evidence_qualification import (
    EvidenceQualificationEngine,
    _digest,
    _finite,
    _statistics,
    deflated_performance_evidence,
    drift_and_edge_decay,
    probability_of_backtest_overfitting,
    purged_walk_forward_validation,
)


class PartitionedEvidenceQualificationEngine(EvidenceQualificationEngine):
    """v1.42 qualifier for precommitted, non-overlapping evidence partitions.

    Legacy v1.40/v1.41 experiments are delegated to the conservative compatibility
    qualifier. Once v1.42 manifests exist, only their precommitted partitioned
    evidence can satisfy the Unified Decision Control Plane promotion contract.
    """

    VERSION = "1.42.0"
    PROTOCOL = "v1.42_partitioned_evidence_v1"
    PBO_CEILING = 0.20
    P_VALUE_CEILING = 0.05
    PBO_BUCKETS = 16

    @staticmethod
    def _is_v142(experiment: dict[str, Any]) -> bool:
        protocol = experiment.get("protocol") or {}
        plan = protocol.get("partition_plan") or {}
        return bool(
            isinstance(protocol, dict)
            and str(protocol.get("evidence_protocol_version") or "") == "1.42"
            and isinstance(plan, dict)
            and str(plan.get("protocol") or "")
            == PartitionedEvidenceQualificationEngine.PROTOCOL
        )

    @staticmethod
    def _assignment(ordinal: int, plan: dict[str, Any]) -> dict[str, Any]:
        initial = int(plan.get("initial_training_samples") or 0)
        folds = int(plan.get("walk_forward_folds") or 0)
        validation_per_fold = int(plan.get("validation_samples_per_fold") or 0)
        embargo_per_fold = int(plan.get("embargo_samples_per_fold") or 0)
        prospective = int(plan.get("prospective_paper_samples") or 0)
        holdout = int(plan.get("untouched_holdout_samples") or 0)
        cursor = 0
        if ordinal < initial:
            return {"partition": "training", "walk_forward_fold": None}
        cursor += initial
        for fold in range(folds):
            if ordinal < cursor + embargo_per_fold:
                return {"partition": "embargo", "walk_forward_fold": fold}
            cursor += embargo_per_fold
            if ordinal < cursor + validation_per_fold:
                return {"partition": "validation", "walk_forward_fold": fold}
            cursor += validation_per_fold
        if ordinal < cursor + prospective:
            return {"partition": "prospective_paper", "walk_forward_fold": None}
        cursor += prospective
        if ordinal < cursor + holdout:
            return {"partition": "untouched_holdout", "walk_forward_fold": None}
        return {"partition": "post_holdout", "walk_forward_fold": None}

    @staticmethod
    def _interval(row: dict[str, Any]) -> dict[str, float]:
        return {
            "feature_start": _finite(row.get("feature_start")),
            "feature_end": _finite(row.get("feature_end")),
            "label_end": _finite(row.get("label_end")),
        }

    def _validated_rows(
        self,
        experiment: dict[str, Any],
        freeze: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        candidate_id = str(experiment.get("candidate_id") or "")
        plan = (experiment.get("protocol") or {}).get("partition_plan") or {}
        registered_at = _finite(experiment.get("registered_at"), default=math.inf)
        frozen_at = _finite(freeze.get("frozen_at"), default=math.inf)
        rows: list[dict[str, Any]] = []
        reasons: list[str] = []
        raw_outcomes = experiment.get("outcomes") or []
        if not isinstance(raw_outcomes, list):
            return [], [f"{candidate_id}:outcomes_not_a_list"]
        for expected_ordinal, raw in enumerate(raw_outcomes):
            if not isinstance(raw, dict):
                reasons.append(f"{candidate_id}:malformed_outcome")
                continue
            try:
                ordinal = int(raw.get("episode_ordinal"))
                opened_at = _finite(raw.get("opened_at"))
                closed_at = _finite(raw.get("closed_at"))
                recorded_at = _finite(raw.get("recorded_at"))
                net_return = _finite(raw.get("net_return"))
                feature_start = _finite(raw.get("feature_start"))
                feature_end = _finite(raw.get("feature_end"))
                label_end = _finite(raw.get("label_end"))
            except (TypeError, ValueError):
                reasons.append(f"{candidate_id}:invalid_outcome_numeric")
                continue
            expected = self._assignment(expected_ordinal, plan)
            supplied_partition = str(raw.get("partition") or "")
            supplied_fold = raw.get("walk_forward_fold")
            if ordinal != expected_ordinal:
                reasons.append(f"{candidate_id}:outcome_ordinal_not_contiguous")
            if supplied_partition != expected["partition"]:
                reasons.append(f"{candidate_id}:partition_assignment_mutated")
            if supplied_fold != expected["walk_forward_fold"]:
                reasons.append(f"{candidate_id}:walk_forward_fold_assignment_mutated")
            if raw.get("evidence_interval_complete") is not True:
                reasons.append(f"{candidate_id}:incomplete_evidence_interval")
            if not (
                registered_at <= opened_at <= feature_start <= feature_end <= label_end
                and opened_at <= closed_at
                and label_end == closed_at
                and recorded_at == closed_at
            ):
                reasons.append(f"{candidate_id}:invalid_or_prefreeze_interval")
            if opened_at < frozen_at:
                reasons.append(f"{candidate_id}:outcome_signal_predates_qualification_freeze")
            if str(raw.get("evidence_authority") or "") != "costed_shadow_episode_v2":
                reasons.append(f"{candidate_id}:evidence_authority_mismatch")
            rows.append(
                {
                    "episode_ordinal": ordinal,
                    "partition": supplied_partition,
                    "walk_forward_fold": supplied_fold,
                    "opened_at": opened_at,
                    "closed_at": closed_at,
                    "recorded_at": recorded_at,
                    "feature_start": feature_start,
                    "feature_end": feature_end,
                    "label_end": label_end,
                    "net_return": net_return,
                    "strategy": str(raw.get("strategy") or ""),
                    "symbol": str(raw.get("symbol") or ""),
                    "regime": str(raw.get("regime") or "unknown"),
                    "evidence_authority": "costed_shadow_episode_v2",
                }
            )
        return rows, reasons

    def _walk_forward(
        self,
        rows: list[dict[str, Any]],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        initial_required = int(plan.get("initial_training_samples") or 0)
        folds_required = int(plan.get("walk_forward_folds") or 0)
        validation_required = int(plan.get("validation_samples_per_fold") or 0)
        embargo_required = int(plan.get("embargo_samples_per_fold") or 0)
        training = [row for row in rows if row["partition"] == "training"]
        validations = {
            fold: [
                row
                for row in rows
                if row["partition"] == "validation"
                and row.get("walk_forward_fold") == fold
            ]
            for fold in range(folds_required)
        }
        embargoes = {
            fold: [
                row
                for row in rows
                if row["partition"] == "embargo"
                and row.get("walk_forward_fold") == fold
            ]
            for fold in range(folds_required)
        }
        if len(training) < initial_required:
            return {
                "passed": False,
                "purged_walk_forward_passed": False,
                "embargo_applied": False,
                "folds": [],
                "reasons": ["initial_training_partition_immature"],
            }
        folds: list[dict[str, Any]] = []
        expanding = list(training[:initial_required])
        for fold in range(folds_required):
            if len(embargoes[fold]) < embargo_required:
                return {
                    "passed": False,
                    "purged_walk_forward_passed": False,
                    "embargo_applied": False,
                    "folds": [],
                    "reasons": [f"walk_forward_fold_{fold}_embargo_immature"],
                }
            if len(validations[fold]) < validation_required:
                return {
                    "passed": False,
                    "purged_walk_forward_passed": False,
                    "embargo_applied": False,
                    "folds": [],
                    "reasons": [f"walk_forward_fold_{fold}_validation_immature"],
                }
            current_validation = validations[fold][:validation_required]
            folds.append(
                {
                    "training": [self._interval(row) for row in expanding],
                    "validation": [self._interval(row) for row in current_validation],
                }
            )
            expanding.extend(current_validation)
        return purged_walk_forward_validation(
            folds,
            embargo_seconds=self.embargo_seconds,
        )

    @classmethod
    def _aligned_validation_matrix(
        cls,
        experiment_rows: dict[str, list[dict[str, Any]]],
        plans: dict[str, dict[str, Any]],
    ) -> dict[str, list[float]]:
        mature: dict[str, list[dict[str, Any]]] = {}
        all_rows: list[dict[str, Any]] = []
        for candidate_id, rows in experiment_rows.items():
            plan = plans[candidate_id]
            required = int(plan.get("walk_forward_folds") or 0) * int(
                plan.get("validation_samples_per_fold") or 0
            )
            validation = [row for row in rows if row["partition"] == "validation"]
            if len(validation) >= required and required > 0:
                selected = validation[:required]
                mature[candidate_id] = selected
                all_rows.extend(selected)
        if len(mature) < 2 or not all_rows:
            return {}
        start = min(float(row["feature_start"]) for row in all_rows)
        end = max(float(row["label_end"]) for row in all_rows)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            return {}
        width = (end - start) / cls.PBO_BUCKETS
        if width <= 0.0:
            return {}
        matrix: dict[str, list[float]] = {}
        for candidate_id, rows in mature.items():
            series = [0.0 for _ in range(cls.PBO_BUCKETS)]
            for row in rows:
                index = min(
                    cls.PBO_BUCKETS - 1,
                    max(0, int((float(row["label_end"]) - start) / width)),
                )
                series[index] += float(row["net_return"])
            matrix[candidate_id] = series
        return matrix

    def _open_holdout_once(
        self,
        candidate_id: str,
        rows: list[dict[str, Any]],
        required: int,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        holdout = [row for row in rows if row["partition"] == "untouched_holdout"]
        if len(holdout) < required or required < self.minimum_samples:
            return None, []
        sealed = holdout[:required]
        payload = [
            {
                "episode_ordinal": row["episode_ordinal"],
                "opened_at": row["opened_at"],
                "closed_at": row["closed_at"],
                "net_return": row["net_return"],
                "symbol": row["symbol"],
                "regime": row["regime"],
            }
            for row in sealed
        ]
        digest = _digest(payload)
        openings = self.state.setdefault("holdout_openings", {})
        opening = openings.get(candidate_id)
        if not isinstance(opening, dict):
            opening = {
                "candidate_id": candidate_id,
                "opened_at": time.time(),
                "sample_count": required,
                "sealed_outcomes_sha256": digest,
            }
            opening["opening_hash"] = _digest(opening)
            openings[candidate_id] = opening
            self._save()
            return opening, []
        reasons: list[str] = []
        unhashed = {key: value for key, value in opening.items() if key != "opening_hash"}
        if str(opening.get("opening_hash") or "") != _digest(unhashed):
            reasons.append(f"{candidate_id}:holdout_opening_record_corrupted")
        if int(opening.get("sample_count") or 0) != required:
            reasons.append(f"{candidate_id}:holdout_opening_sample_count_changed")
        if str(opening.get("sealed_outcomes_sha256") or "") != digest:
            reasons.append(f"{candidate_id}:opened_holdout_mutated")
        return opening, reasons

    def qualify(
        self,
        prospective_state: dict[str, Any],
        *,
        base_validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        experiments_map = prospective_state.get("experiments") or {}
        if isinstance(experiments_map, dict):
            all_experiments = [
                row
                for _, row in sorted(experiments_map.items())
                if isinstance(row, dict)
            ]
        elif isinstance(experiments_map, list):
            all_experiments = [row for row in experiments_map if isinstance(row, dict)]
        else:
            all_experiments = []
        experiments = [row for row in all_experiments if self._is_v142(row)]
        if not experiments:
            return super().qualify(
                prospective_state,
                base_validation=base_validation,
            )
        base = dict(base_validation or {})
        if not self.lineage_integrity_ok:
            return self._fail_closed(base, ["evidence_lineage_integrity_failure"])

        integrity_reasons: list[str] = []
        rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
        plans: dict[str, dict[str, Any]] = {}
        freezes: dict[str, dict[str, Any]] = {}
        for experiment in experiments:
            candidate_id = str(experiment.get("candidate_id") or "").strip()
            raw_outcomes = experiment.get("outcomes") or []
            existing_freeze = (self.state.get("holdout_freezes") or {}).get(candidate_id)
            if not isinstance(existing_freeze, dict) and raw_outcomes:
                integrity_reasons.append(
                    f"{candidate_id}:qualification_freeze_missing_before_outcomes"
                )
                continue
            freeze, _, _, freeze_reasons = self._freeze_or_validate(experiment)
            freezes[candidate_id] = freeze
            integrity_reasons.extend(freeze_reasons)
            rows, row_reasons = self._validated_rows(experiment, freeze)
            rows_by_candidate[candidate_id] = rows
            plans[candidate_id] = dict(
                (experiment.get("protocol") or {}).get("partition_plan") or {}
            )
            integrity_reasons.extend(row_reasons)

        def selection_score(experiment: dict[str, Any]) -> tuple[float, str]:
            selection = experiment.get("selection_evidence") or {}
            return (
                _finite(selection.get("conservative_score"), default=-math.inf),
                str(experiment.get("candidate_id") or ""),
            )

        selected_experiment = max(experiments, key=selection_score)
        selected_candidate = str(selected_experiment.get("candidate_id") or "")
        selected_rows = rows_by_candidate.get(selected_candidate, [])
        selected_plan = plans.get(selected_candidate, {})

        walk_forward = self._walk_forward(selected_rows, selected_plan)
        matrix = self._aligned_validation_matrix(rows_by_candidate, plans)
        pbo = probability_of_backtest_overfitting(matrix, segments=8) if matrix else {
            "valid": False,
            "pbo": 1.0,
            "splits": 0,
            "reason": "aligned_validation_family_immature",
        }
        selected_validation_series = matrix.get(selected_candidate, [])
        deflated = deflated_performance_evidence(
            selected_validation_series,
            number_of_trials=max(1, len(matrix)),
        )

        prospective_rows = [
            row for row in selected_rows if row["partition"] == "prospective_paper"
        ]
        prospective_required = max(
            self.minimum_samples,
            int(selected_plan.get("prospective_paper_samples") or 0),
        )
        prospective_used = prospective_rows[:prospective_required]
        prospective_stats = _statistics(
            [float(row["net_return"]) for row in prospective_used]
        )
        prospective_regimes = len(
            {str(row.get("regime") or "unknown") for row in prospective_used}
        )
        prospective_positive = bool(
            int(prospective_stats["samples"]) >= prospective_required
            and prospective_regimes >= self.minimum_regimes
            and float(prospective_stats["mean"]) > 0.0
            and prospective_stats["lower_95"] is not None
            and float(prospective_stats["lower_95"]) > 0.0
        )
        drift = drift_and_edge_decay(
            prospective_used,
            minimum_samples=prospective_required,
            minimum_regimes=self.minimum_regimes,
        )

        holdout_required = max(
            self.minimum_samples,
            int(selected_plan.get("untouched_holdout_samples") or 0),
        )
        holdout_rows = [
            row for row in selected_rows if row["partition"] == "untouched_holdout"
        ]
        opening, opening_reasons = self._open_holdout_once(
            selected_candidate,
            selected_rows,
            holdout_required,
        )
        integrity_reasons.extend(opening_reasons)
        if opening is None:
            holdout_stats = {
                "samples": min(len(holdout_rows), holdout_required),
                "mean": None,
                "lower_95": None,
            }
            holdout_regimes = 0
            holdout_opened = False
        else:
            sealed_holdout = holdout_rows[:holdout_required]
            holdout_stats = _statistics(
                [float(row["net_return"]) for row in sealed_holdout]
            )
            holdout_regimes = len(
                {str(row.get("regime") or "unknown") for row in sealed_holdout}
            )
            holdout_opened = True
        untouched_holdout_passed = bool(
            holdout_opened
            and not integrity_reasons
            and int(holdout_stats["samples"]) >= holdout_required
            and holdout_regimes >= self.minimum_regimes
            and holdout_stats["mean"] is not None
            and float(holdout_stats["mean"]) > 0.0
            and holdout_stats["lower_95"] is not None
            and float(holdout_stats["lower_95"]) > 0.0
        )

        multiple_testing_controlled = bool(
            pbo.get("valid") is True
            and float(pbo.get("pbo", 1.0)) <= self.PBO_CEILING
            and deflated.get("valid") is True
            and float(deflated.get("multiple_testing_adjusted_p_value", 1.0))
            < self.P_VALUE_CEILING
        )

        reasons = list(integrity_reasons)
        reasons.extend(walk_forward.get("reasons") or [])
        if pbo.get("valid") is not True:
            reasons.append(str(pbo.get("reason") or "pbo_invalid"))
        elif float(pbo.get("pbo", 1.0)) > self.PBO_CEILING:
            reasons.append("pbo_above_ceiling")
        if deflated.get("valid") is not True:
            reasons.append(str(deflated.get("reason") or "deflated_statistic_invalid"))
        elif float(deflated.get("deflated_performance_statistic", -math.inf)) <= 0.0:
            reasons.append("deflated_performance_not_positive")
        if not prospective_positive:
            reasons.append("prospective_positive_lower_bound_not_proven")
        if drift.get("drift_stable") is not True:
            reasons.append(str(drift.get("reason") or "drift_not_stable"))
        if not holdout_opened:
            reasons.append("untouched_holdout_sealed_until_minimum_samples")
        elif not untouched_holdout_passed:
            reasons.append("untouched_holdout_not_positive_with_confidence")

        partition_counts = {
            name: sum(1 for row in selected_rows if row["partition"] == name)
            for name in (
                "training",
                "embargo",
                "validation",
                "prospective_paper",
                "untouched_holdout",
                "post_holdout",
            )
        }
        contract = {
            **base,
            "independent_samples": int(prospective_stats["samples"]),
            "purged_walk_forward_passed": walk_forward.get("purged_walk_forward_passed") is True,
            "embargo_applied": walk_forward.get("embargo_applied") is True,
            "untouched_holdout_passed": untouched_holdout_passed,
            "multiple_testing_controlled": multiple_testing_controlled,
            "prospective_net_positive": prospective_positive,
            "drift_stable": drift.get("drift_stable") is True,
            "probability_backtest_overfitting": float(pbo.get("pbo", 1.0)),
            "deflated_performance_statistic": float(
                deflated.get("deflated_performance_statistic", -math.inf)
            ),
            "partitions": {
                "training": {
                    "status": "precommitted",
                    "samples": partition_counts["training"],
                    "required": int(selected_plan.get("initial_training_samples") or 0),
                },
                "validation": {
                    "status": "purged_embargo_verified" if walk_forward.get("passed") else "collecting_or_failed",
                    "samples": partition_counts["validation"],
                    "embargo_samples": partition_counts["embargo"],
                    "folds": len(walk_forward.get("folds") or []),
                },
                "prospective_paper": {
                    "candidate_id": selected_candidate,
                    "samples": int(prospective_stats["samples"]),
                    "required": prospective_required,
                    "regimes": prospective_regimes,
                    "mean_net_return": prospective_stats["mean"],
                    "lower_95_net_return": prospective_stats["lower_95"],
                },
                "untouched_holdout": {
                    "candidate_id": selected_candidate,
                    "samples_collected": min(len(holdout_rows), holdout_required),
                    "required": holdout_required,
                    "sealed": not holdout_opened,
                    "opened_once": holdout_opened,
                    "opening": opening or {},
                    "regimes": holdout_regimes if holdout_opened else None,
                    "mean_net_return": holdout_stats.get("mean") if holdout_opened else None,
                    "lower_95_net_return": holdout_stats.get("lower_95") if holdout_opened else None,
                },
            },
            "qualification": {
                "version": self.VERSION,
                "protocol": self.PROTOCOL,
                "valid": not integrity_reasons,
                "selected_candidate": selected_candidate,
                "selection_basis": "frozen_pre_outcome_conservative_score",
                "reasons": list(dict.fromkeys(reasons)),
                "walk_forward": walk_forward,
                "pbo": pbo,
                "deflated_performance": deflated,
                "drift": drift,
                "partition_counts": partition_counts,
                "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
                "minimum_samples": self.minimum_samples,
                "minimum_regimes": self.minimum_regimes,
            },
        }
        reproducibility_payload = {
            "version": self.VERSION,
            "protocol": self.PROTOCOL,
            "configuration": self._configuration(),
            "prospective_source_hash": _digest(prospective_state),
            "selected_candidate": selected_candidate,
            "contract": contract,
        }
        contract["evidence_reproducibility_hash"] = _digest(reproducibility_payload)
        self._append_history(
            {
                "evidence_reproducibility_hash": contract["evidence_reproducibility_hash"],
                "protocol": self.PROTOCOL,
                "selected_candidate": selected_candidate,
                "partition_counts": partition_counts,
                "independent_samples": contract["independent_samples"],
                "purged_walk_forward_passed": contract["purged_walk_forward_passed"],
                "untouched_holdout_passed": contract["untouched_holdout_passed"],
                "pbo": contract["probability_backtest_overfitting"],
                "deflated_performance_statistic": contract["deflated_performance_statistic"],
                "drift_stable": contract["drift_stable"],
            }
        )
        self.last_error = None
        return {**contract, **self._authority_denied()}

    def health(self) -> dict[str, Any]:
        health = super().health()
        health.update(
            {
                "version": self.VERSION,
                "partition_protocol": self.PROTOCOL,
                "holdout_openings": len(self.state.get("holdout_openings") or {}),
                "pbo_ceiling": self.PBO_CEILING,
                "automatic_promotion": False,
                "paper_promotion_authority": False,
                "testnet_authority": False,
                "live_authority": False,
                "execution_authority": False,
            }
        )
        return health
