from __future__ import annotations

import hashlib
from typing import Any

from .alpha_tournament_v141 import *  # noqa: F401,F403
from .alpha_tournament_v141 import AlphaTournament as _V141AlphaTournament
from .alpha_tournament_v141 import StrategyFoundry as _V141StrategyFoundry


class StrategyFoundry(_V141StrategyFoundry):
    """v1.42 research manifests with a precommitted evidence partition plan."""

    VERSION = "1.42.0"
    INITIAL_TRAINING_SAMPLES = 60
    WALK_FORWARD_FOLDS = 3
    VALIDATION_SAMPLES_PER_FOLD = 20
    EMBARGO_SAMPLES_PER_FOLD = 1
    PROSPECTIVE_PAPER_SAMPLES = 100
    UNTOUCHED_HOLDOUT_SAMPLES = 100

    @classmethod
    def partition_plan(cls) -> dict[str, Any]:
        return {
            "protocol": "v1.42_partitioned_evidence_v1",
            "initial_training_samples": cls.INITIAL_TRAINING_SAMPLES,
            "walk_forward_folds": cls.WALK_FORWARD_FOLDS,
            "validation_samples_per_fold": cls.VALIDATION_SAMPLES_PER_FOLD,
            "embargo_samples_per_fold": cls.EMBARGO_SAMPLES_PER_FOLD,
            "prospective_paper_samples": cls.PROSPECTIVE_PAPER_SAMPLES,
            "untouched_holdout_samples": cls.UNTOUCHED_HOLDOUT_SAMPLES,
            "assignment": "candidate_episode_ordinal_frozen_at_registration",
            "post_holdout_assignment": "untouched_holdout",
        }

    def forge(
        self,
        ranking: list[dict[str, Any]],
        hypothesis_agenda: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        manifests = super().forge(ranking, hypothesis_agenda)
        plan = self.partition_plan()
        for manifest in manifests:
            strategy = str(manifest.get("base_strategy") or "")
            digest = hashlib.sha256(
                (
                    f"v1.42|{strategy}|{self.minimum_samples}|"
                    f"{self.round_trip_cost_bps:.8f}|partitioned-prospective"
                ).encode("utf-8")
            ).hexdigest()[:20]
            manifest["candidate_id"] = f"v142-foundry-{digest}"
            selection = dict(manifest.get("selection_evidence") or {})
            selection.update(
                {
                    "evidence_protocol_version": "1.42",
                    "selection_frozen_before_partitioned_outcomes": True,
                }
            )
            manifest["selection_evidence"] = selection
            protocol = dict(manifest.get("research_protocol") or {})
            protocol.update(
                {
                    "evidence_protocol_version": "1.42",
                    "purged_walk_forward_required": True,
                    "embargo_required": True,
                    "pbo_required": True,
                    "deflated_performance_required": True,
                    "drift_detection_required": True,
                    "untouched_holdout_required": True,
                    "freeze_before_outcome_required": True,
                    "partition_plan": dict(plan),
                }
            )
            manifest["research_protocol"] = protocol
        return manifests


class AlphaTournament(_V141AlphaTournament):
    """v1.42 Alpha Tournament with immutable partitioned evidence manifests."""

    VERSION = "1.42.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.foundry = StrategyFoundry(
            minimum_samples=self.minimum_samples,
            round_trip_cost_bps=self.expected_round_trip_cost_bps,
        )

    def health(self) -> dict[str, Any]:
        health = super().health()
        health.update(
            {
                "version": self.VERSION,
                "evidence_protocol_version": "1.42",
                "partition_plan": self.foundry.partition_plan(),
                "automatic_promotion": False,
                "paper_promotion_authority": False,
                "testnet_authority": False,
                "live_authority": False,
                "execution_authority": False,
            }
        )
        return health
