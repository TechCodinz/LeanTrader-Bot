from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .runtime_evidence_qualification import RuntimeEvidenceQualificationEngine
from .unified_control_plane_v141 import *  # noqa: F401,F403
from .unified_control_plane_v141 import UnifiedDecisionControlPlane as _V141UnifiedDecisionControlPlane


class UnifiedDecisionControlPlane(_V141UnifiedDecisionControlPlane):
    """v1.42 evidence-qualified wrapper over the exact v1.41 control plane.

    The v1.41 decision, execution-costing, portfolio, safety and authority logic
    is preserved byte-for-byte in ``unified_control_plane_v141.py``. v1.42 only
    replaces the intentionally closed validation placeholders with persisted,
    reproducible measurements from the precommitted partition protocol.

    Missing, corrupt or immature measured evidence remains fail-closed. This
    wrapper has no authority to promote paper routes, enable Testnet, enable
    live trading, mutate risk limits, or place orders.
    """

    VERSION = "1.42.0"

    def __init__(self, state_path: Path, *args: Any, **kwargs: Any) -> None:
        super().__init__(state_path, *args, **kwargs)
        self._prospective_state_path = Path(state_path).with_name(
            "vps_prospective_validation.json"
        )
        self.evidence_qualification = RuntimeEvidenceQualificationEngine(
            Path(state_path).with_name("vps_evidence_qualification_v142.json"),
            prospective_state_path=self._prospective_state_path,
            minimum_samples=self.minimum_independent_samples,
            minimum_regimes=2,
            modeled_round_trip_cost_bps=self.order_simulator.minimum_round_trip_cost_bps,
            embargo_seconds=1.0,
        )
        self._last_measured_validation: dict[str, Any] | None = None

    def start(self) -> None:
        super().start()
        self.evidence_qualification.start()

    def stop(self) -> None:
        self.evidence_qualification.stop()
        super().stop()

    def _measured_validation(self, supplied: dict[str, Any]) -> dict[str, Any]:
        # Unit callers may supply an already-qualified contract without a
        # runtime evidence store. Production always has the sibling prospective
        # state file; when it exists, measured evidence is authoritative.
        if not self._prospective_state_path.exists():
            self._last_measured_validation = None
            return supplied
        measured = self.evidence_qualification.qualify_from_disk(
            base_validation=supplied
        )
        # v1.41's promotion parser deliberately requires finite numerics. The
        # qualifier uses non-finite values internally to denote unavailable
        # statistics, so normalize only at this compatibility boundary while
        # preserving explicit failure reasons in qualification metadata.
        try:
            deflated = float(measured.get("deflated_performance_statistic", -1.0))
        except (TypeError, ValueError):
            deflated = -1.0
        if not math.isfinite(deflated):
            measured["deflated_performance_statistic"] = -1.0
        self._last_measured_validation = measured
        return measured

    def evaluate(
        self,
        *,
        validation: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        measured = self._measured_validation(dict(validation))
        result = super().evaluate(validation=measured, **kwargs)
        if not isinstance(result, dict):
            return result
        result["evidence_qualification"] = {
            "version": self.evidence_qualification.VERSION,
            "source_available": self._prospective_state_path.exists(),
            "lineage_integrity_ok": self.evidence_qualification.lineage_integrity_ok,
            "measured_contract_applied": self._last_measured_validation is not None,
            "evidence_reproducibility_hash": (
                measured.get("evidence_reproducibility_hash")
                if isinstance(measured, dict)
                else None
            ),
            "qualification": (
                measured.get("qualification")
                if isinstance(measured, dict)
                else None
            ),
            "partitions": measured.get("partitions", {}) if isinstance(measured, dict) else {},
            "automatic_promotion": False,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "execution_authority": False,
        }
        if isinstance(result.get("promotion_gate"), dict):
            result["promotion_gate"]["evidence_reproducibility_hash"] = (
                measured.get("evidence_reproducibility_hash")
                if isinstance(measured, dict)
                else None
            )
            result["promotion_gate"]["evidence_qualification_version"] = (
                self.evidence_qualification.VERSION
            )
        return result

    def health(self) -> dict[str, Any]:
        health = super().health()
        health.update(
            {
                "version": self.VERSION,
                "legacy_control_plane_version": "1.41.0",
                "v1_42_evidence_qualification": True,
                "evidence_qualification": self.evidence_qualification.health(),
                "measured_validation_applied": self._last_measured_validation is not None,
                "measured_validation": self._last_measured_validation,
                "automatic_promotion": False,
                "paper_promotion_authority": False,
                "testnet_authority": False,
                "live_authority": False,
                "execution_authority": False,
            }
        )
        return health
