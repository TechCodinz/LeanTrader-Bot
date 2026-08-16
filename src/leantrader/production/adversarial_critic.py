from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class AdversarialCritic:
    """Independent red-team review of LeanTrader's own market belief.

    The critic never creates risk.  It attempts to falsify strong conclusions,
    identifies missing evidence and outputs a confidence/risk haircut that is
    visible to research and meta-cognition.  v12.6 keeps it shadow-only so it
    cannot disturb the established v12.5 execution experiment.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    SAVE_INTERVAL = 10

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.last_error: str | None = None
        self.state = self._load()
        self.reviews = int(self.state.get("reviews") or 0)

    def start(self) -> None:
        self.state = self._load()
        self.reviews = int(self.state.get("reviews") or 0)

    def stop(self) -> None:
        self._save()

    def review(
        self,
        *,
        symbol: str,
        council: dict[str, Any],
        world: dict[str, Any],
        self_model: dict[str, Any],
        memory: dict[str, Any],
        route: dict[str, Any],
        brain: dict[str, Any],
        public_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        objections: list[dict[str, Any]] = []

        def object_to(reason: str, severity: float, evidence: str) -> None:
            objections.append(
                {
                    "reason": reason,
                    "severity": _clip(severity),
                    "evidence": evidence,
                }
            )

        senses = world.get("senses") or {}
        disagreement = _clip(_finite(council.get("disagreement")))
        uncertainty = _clip(_finite(council.get("uncertainty")))
        novelty = _clip(_finite(senses.get("novelty")))
        liquidity_stress = _clip(_finite(senses.get("liquidity_stress")))
        volatility_shock = _clip(_finite(senses.get("volatility_shock")))
        timeframe_fracture = _clip(_finite(senses.get("timeframe_fracture")))
        state_confidence = _clip(_finite(world.get("state_confidence")))
        memory_support = _clip(_finite(memory.get("support")))
        memory_samples = int(memory.get("contextual_samples") or memory.get("samples") or 0)
        council_conf = _clip(_finite(council.get("confidence")))
        council_score = max(-1.0, min(1.0, _finite(council.get("consensus_score"))))

        if disagreement >= 0.45:
            object_to("specialist_disagreement", disagreement, "council specialists materially disagree")
        if novelty >= 0.65:
            object_to("out_of_distribution_risk", novelty, "current market state has weak historical analogue")
        if liquidity_stress >= 0.50:
            object_to("liquidity_execution_risk", liquidity_stress, "spread/depth imbalance increases execution uncertainty")
        if volatility_shock >= 0.60:
            object_to("volatility_regime_instability", volatility_shock, "short volatility materially exceeds baseline")
        if timeframe_fracture >= 0.50:
            object_to("timeframe_structure_conflict", timeframe_fracture, "multi-timeframe evidence is structurally dispersed")
        if memory_samples < 4 or memory_support < 0.30:
            object_to("weak_historical_grounding", 0.45, "few sufficiently similar closed outcomes")
        if state_confidence < 0.30:
            object_to("weak_world_state_confidence", 1.0 - state_confidence, "market world state is poorly identified")
        if self_model.get("contradictions"):
            object_to(
                "internal_contradictions",
                min(1.0, 0.25 * len(self_model.get("contradictions") or [])),
                ",".join(self_model.get("contradictions") or []),
            )
        context = public_context or {}
        if not context.get("available"):
            object_to("public_context_gap", 0.25, "fresh public context is unavailable")
        if brain.get("strategy_quarantined"):
            object_to("strategy_quarantine", 1.0, "Brain has authoritative negative evidence")
        elif "negative_strategy_evidence" in (brain.get("reasons") or []):
            object_to("negative_strategy_evidence", 0.70, "costed strategy evidence is negative")
        if route.get("allowed") is True and uncertainty >= 0.65:
            object_to("approval_under_high_uncertainty", uncertainty, "route approval exists despite high research uncertainty")

        combined_severity = 1.0
        for row in objections:
            combined_severity *= 1.0 - 0.55 * _clip(_finite(row.get("severity")))
        combined_severity = 1.0 - combined_severity
        confidence_haircut = _clip(combined_severity)
        adjusted_confidence = _clip(council_conf * (1.0 - 0.75 * confidence_haircut))
        adjusted_score = council_score * adjusted_confidence
        risk_guidance_multiplier = _clip(1.0 - 0.85 * confidence_haircut)

        falsification_questions: list[str] = []
        if abs(council_score) >= 0.35:
            falsification_questions.append("What market evidence would invalidate the current council direction within the next horizon?")
        if novelty >= 0.65:
            falsification_questions.append("Is the apparent edge stable outside the nearest historical analogues, or is this a regime-specific artifact?")
        if liquidity_stress >= 0.40:
            falsification_questions.append("Does the signal survive realistic spread, slippage and depth constraints during the observed liquidity state?")
        if disagreement >= 0.40:
            falsification_questions.append("Which specialist has historically been best calibrated in this exact regime and why are the others disagreeing?")
        if not falsification_questions:
            falsification_questions.append("Which independent observation would most strongly disconfirm this market hypothesis?")

        result = {
            "symbol": symbol.upper(),
            "original_score": council_score,
            "original_confidence": council_conf,
            "adjusted_score": adjusted_score,
            "adjusted_confidence": adjusted_confidence,
            "confidence_haircut": confidence_haircut,
            "risk_guidance_multiplier": risk_guidance_multiplier,
            "objections": objections,
            "falsification_questions": falsification_questions,
            "shadow_only": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "can_enable_live": False,
            "reviewed_at": time.time(),
        }
        self.state.setdefault("latest", {})[symbol.upper()] = result
        self.reviews += 1
        self.state["reviews"] = self.reviews
        if self.reviews % self.SAVE_INTERVAL == 0:
            self._save()
        return dict(result)

    def health(self) -> dict[str, Any]:
        latest = self.state.get("latest") or {}
        objections = sum(len(row.get("objections") or []) for row in latest.values())
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "reviews": self.reviews,
            "tracked_symbols": len(latest),
            "current_objections": objections,
            "shadow_only": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "can_enable_live": False,
            "state_path": str(self.state_path),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {"schema_version": self.SCHEMA_VERSION, "reviews": 0, "latest": {}}
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) == self.SCHEMA_VERSION:
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return empty

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["reviews"] = self.reviews
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
