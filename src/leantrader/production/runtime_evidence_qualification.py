from __future__ import annotations

from typing import Any

from .partitioned_evidence_qualification import PartitionedEvidenceQualificationEngine


class RuntimeEvidenceQualificationEngine(PartitionedEvidenceQualificationEngine):
    """Production v1.42 qualifier honoring the Foundry's post-minimum plan.

    Outcomes beyond the first required holdout remain tagged as holdout by the
    recorder for append-only continuity, but qualification opens and hashes only
    the first precommitted holdout sample count. Later observations cannot alter
    that sealed result.
    """

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
