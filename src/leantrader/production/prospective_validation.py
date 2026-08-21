from __future__ import annotations

import math
import time
from typing import Any

from .prospective_validation_v141 import *  # noqa: F401,F403
from .prospective_validation_v141 import ProspectiveValidationLab as _V141ProspectiveValidationLab
from .prospective_validation_v141 import _finite


class ProspectiveValidationLab(_V141ProspectiveValidationLab):
    """v1.42 partitioned prospective evidence recorder.

    v1.41 experiments remain readable and continue collecting under their
    original contract. New v1.42 manifests precommit their partition schedule
    at registration; only episodes whose signal opened after that registration
    are accepted into the v1.42 evidence partitions.
    """

    VERSION = "1.42.0"
    PARTITION_REJECTION_LIMIT = 500

    @staticmethod
    def _v142_protocol(experiment_or_manifest: dict[str, Any]) -> bool:
        protocol = (
            experiment_or_manifest.get("protocol")
            if "protocol" in experiment_or_manifest
            else experiment_or_manifest.get("research_protocol")
        )
        return isinstance(protocol, dict) and str(
            protocol.get("evidence_protocol_version") or ""
        ) == "1.42"

    def _validate_manifest(self, manifest: dict[str, Any]) -> tuple[bool, str]:
        valid, reason = super()._validate_manifest(manifest)
        if not valid or not self._v142_protocol(manifest):
            return valid, reason
        protocol = manifest.get("research_protocol") or {}
        plan = protocol.get("partition_plan") or {}
        if not isinstance(plan, dict):
            return False, "v1.42 partition plan is required"
        minimums = {
            "initial_training_samples": 60,
            "walk_forward_folds": 3,
            "validation_samples_per_fold": 20,
            "embargo_samples_per_fold": 1,
            "prospective_paper_samples": self.minimum_samples,
            "untouched_holdout_samples": self.minimum_samples,
        }
        for key, minimum in minimums.items():
            try:
                observed = int(plan.get(key) or 0)
            except (TypeError, ValueError):
                return False, f"v1.42 partition plan has invalid {key}"
            if observed < int(minimum):
                return False, f"v1.42 partition plan lowers {key}"
        required_true = (
            "purged_walk_forward_required",
            "embargo_required",
            "pbo_required",
            "deflated_performance_required",
            "drift_detection_required",
            "untouched_holdout_required",
            "freeze_before_outcome_required",
        )
        for key in required_true:
            if protocol.get(key) is not True:
                return False, f"v1.42 protocol requires {key}=true"
        return True, "accepted"

    @staticmethod
    def _partition_assignment(
        ordinal: int,
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        initial = int(plan.get("initial_training_samples") or 0)
        folds = int(plan.get("walk_forward_folds") or 0)
        validation_per_fold = int(plan.get("validation_samples_per_fold") or 0)
        embargo_per_fold = int(plan.get("embargo_samples_per_fold") or 0)
        prospective = int(plan.get("prospective_paper_samples") or 0)
        holdout = int(plan.get("untouched_holdout_samples") or 0)
        cursor = 0
        if ordinal < initial:
            return {
                "partition": "training",
                "walk_forward_fold": None,
                "partition_index": ordinal,
            }
        cursor += initial
        for fold in range(folds):
            if ordinal < cursor + embargo_per_fold:
                return {
                    "partition": "embargo",
                    "walk_forward_fold": fold,
                    "partition_index": ordinal - cursor,
                }
            cursor += embargo_per_fold
            if ordinal < cursor + validation_per_fold:
                return {
                    "partition": "validation",
                    "walk_forward_fold": fold,
                    "partition_index": ordinal - cursor,
                }
            cursor += validation_per_fold
        if ordinal < cursor + prospective:
            return {
                "partition": "prospective_paper",
                "walk_forward_fold": None,
                "partition_index": ordinal - cursor,
            }
        cursor += prospective
        if ordinal < cursor + holdout:
            return {
                "partition": "untouched_holdout",
                "walk_forward_fold": None,
                "partition_index": ordinal - cursor,
            }
        return {
            "partition": "untouched_holdout",
            "walk_forward_fold": None,
            "partition_index": ordinal - cursor,
            "beyond_minimum_holdout": True,
        }

    def _reject_partition_episode(self, candidate_id: str, reason: str) -> None:
        rows = self.state.setdefault("partition_rejections", [])
        rows.append(
            {
                "candidate_id": str(candidate_id)[:120],
                "reason": str(reason)[:240],
                "rejected_at": time.time(),
            }
        )
        self.state["partition_rejections"] = rows[-self.PARTITION_REJECTION_LIMIT :]

    def _record_strategy_episodes(
        self,
        *,
        existing_experiment_ids: set[str],
        episodes: list[dict[str, Any]],
        contract_valid: bool,
    ) -> int:
        if not contract_valid:
            return 0
        experiments = self.state.setdefault("experiments", {})
        legacy_ids = {
            candidate_id
            for candidate_id in existing_experiment_ids
            if isinstance(experiments.get(candidate_id), dict)
            and not self._v142_protocol(experiments[candidate_id])
        }
        recorded = super()._record_strategy_episodes(
            existing_experiment_ids=legacy_ids,
            episodes=episodes,
            contract_valid=contract_valid,
        )
        v142_ids = sorted(existing_experiment_ids - legacy_ids)
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            if episode.get("evidence_authority") != self.EVIDENCE_AUTHORITY:
                continue
            strategy = str(episode.get("strategy") or "").strip()
            net_return = _finite(episode.get("net_return"), math.nan)
            opened_at = _finite(episode.get("opened_at"), math.nan)
            closed_at = _finite(episode.get("closed_at"), math.nan)
            interval_complete = episode.get("evidence_interval_complete") is True
            if (
                not strategy
                or not math.isfinite(net_return)
                or not math.isfinite(opened_at)
                or not math.isfinite(closed_at)
                or closed_at < opened_at
                or not interval_complete
            ):
                continue
            symbol = str(episode.get("symbol") or "UNKNOWN")[:80]
            regime = str(episode.get("regime") or "unknown")[:80]
            for candidate_id in v142_ids:
                experiment = experiments.get(candidate_id)
                if not isinstance(experiment, dict):
                    continue
                if str(experiment.get("base_strategy") or "") != strategy:
                    continue
                registered_at = _finite(experiment.get("registered_at"), math.inf)
                if opened_at < registered_at:
                    self._reject_partition_episode(
                        candidate_id,
                        "episode signal opened before v1.42 manifest registration",
                    )
                    continue
                protocol = experiment.get("protocol") or {}
                plan = protocol.get("partition_plan") or {}
                ordinal = int(experiment.get("partition_episode_ordinal") or 0)
                assignment = self._partition_assignment(ordinal, plan)
                outcomes = experiment.setdefault("outcomes", [])
                outcomes.append(
                    {
                        "recorded_at": closed_at,
                        "opened_at": opened_at,
                        "closed_at": closed_at,
                        "feature_start": opened_at,
                        "feature_end": opened_at,
                        "label_end": closed_at,
                        "evidence_interval_complete": True,
                        "episode_ordinal": ordinal,
                        **assignment,
                        "strategy": strategy,
                        "symbol": symbol,
                        "regime": regime,
                        "net_return": net_return,
                        "evidence_authority": self.EVIDENCE_AUTHORITY,
                    }
                )
                experiment["outcomes"] = outcomes[-self.RETURN_LIMIT :]
                experiment["partition_episode_ordinal"] = ordinal + 1
                regime_returns = experiment.setdefault("regime_returns", {})
                values = regime_returns.setdefault(regime, [])
                values.append(net_return)
                regime_returns[regime] = values[-self.RETURN_LIMIT :]
                recorded += 1
        return recorded

    def _experiment_snapshots(self) -> list[dict[str, Any]]:
        snapshots = super()._experiment_snapshots()
        experiments = self.state.setdefault("experiments", {})
        for snapshot in snapshots:
            candidate_id = str(snapshot.get("candidate_id") or "")
            experiment = experiments.get(candidate_id)
            if not isinstance(experiment, dict) or not self._v142_protocol(experiment):
                continue
            counts = {
                "training": 0,
                "embargo": 0,
                "validation": 0,
                "prospective_paper": 0,
                "untouched_holdout": 0,
            }
            for outcome in experiment.get("outcomes") or []:
                if isinstance(outcome, dict):
                    partition = str(outcome.get("partition") or "")
                    if partition in counts:
                        counts[partition] += 1
            plan = dict((experiment.get("protocol") or {}).get("partition_plan") or {})
            ready = bool(
                counts["training"] >= int(plan.get("initial_training_samples") or 0)
                and counts["validation"]
                >= int(plan.get("walk_forward_folds") or 0)
                * int(plan.get("validation_samples_per_fold") or 0)
                and counts["prospective_paper"]
                >= int(plan.get("prospective_paper_samples") or 0)
                and counts["untouched_holdout"]
                >= int(plan.get("untouched_holdout_samples") or 0)
            )
            status = (
                "ready_for_v1_42_evidence_qualification"
                if ready
                else "collecting_partitioned_evidence"
            )
            experiment["status"] = status
            experiment["partition_counts"] = counts
            snapshot.update(
                {
                    "status": status,
                    "evidence_protocol_version": "1.42",
                    "partition_counts": counts,
                    "partition_plan": plan,
                    "walk_forward": "measured_by_v1_42_evidence_qualification",
                    "untouched_holdout": True,
                    "freeze_before_outcome": True,
                    "automatic_promotion": False,
                    "paper_promotion_authority": False,
                    "testnet_authority": False,
                    "live_authority": False,
                    "execution_authority": False,
                }
            )
        return snapshots

    def health(self) -> dict[str, Any]:
        health = super().health()
        health.update(
            {
                "version": self.VERSION,
                "evidence_protocol_version": "1.42",
                "partitioned_evidence": True,
                "partition_rejections": len(self.state.get("partition_rejections") or []),
                "automatic_promotion": False,
                "paper_promotion_authority": False,
                "testnet_authority": False,
                "live_authority": False,
                "execution_authority": False,
            }
        )
        return health
