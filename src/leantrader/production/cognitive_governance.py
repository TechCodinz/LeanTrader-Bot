from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _clip(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(low, min(high, number))


def _signed(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(-1.0, min(1.0, number))


class CognitiveGovernanceBridge:
    """Fail-safe bridge from higher-order research intelligence to the Brain.

    The bridge does not create trading authority or alpha claims.  It converts
    world-model, self-model, council, critic, hypothesis, sensor and tail-risk
    evidence into one normalized safety envelope.  The envelope can only leave
    an upstream decision unchanged, reduce confidence/risk, or veto a new entry.
    It can never enlarge upstream risk, enable testnet/live, add credentials,
    rewrite source, or deploy itself.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    SAVE_INTERVAL = 10
    HISTORY_LIMIT = 2_000

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.last_error: str | None = None
        self.state = self._load()
        self.assessments = int(self.state.get("assessments") or 0)
        self.vetoes = int(self.state.get("vetoes") or 0)
        self.downsizes = int(self.state.get("downsizes") or 0)

    def start(self) -> None:
        self.state = self._load()
        self.assessments = int(self.state.get("assessments") or 0)
        self.vetoes = int(self.state.get("vetoes") or 0)
        self.downsizes = int(self.state.get("downsizes") or 0)

    def stop(self) -> None:
        self._save()

    @staticmethod
    def _sensor_diversity(sensor_context: dict[str, Any]) -> tuple[float, int, int]:
        available = 0
        relevant = 0
        for key, row in sensor_context.items():
            if key in {"execution_authority", "read_only", "symbol"} or not isinstance(row, dict):
                continue
            status = str(row.get("status") or "").lower()
            if status in {"", "not_applicable"}:
                continue
            relevant += 1
            if status == "available":
                available += 1
        if relevant <= 0:
            return 0.0, 0, 0
        return available / relevant, available, relevant

    @staticmethod
    def _active_hypotheses(hypothesis: dict[str, Any]) -> int:
        rows = hypothesis.get("active_for_symbol") or hypothesis.get("active") or []
        if isinstance(rows, dict):
            return len(rows)
        if isinstance(rows, list):
            return len(rows)
        return 0

    def evaluate(
        self,
        *,
        symbol: str,
        upstream_allowed: bool,
        world: dict[str, Any],
        self_model: dict[str, Any],
        council: dict[str, Any],
        critic: dict[str, Any],
        hypothesis: dict[str, Any],
        tail_risk: dict[str, Any],
        research_plan: dict[str, Any],
        memory: dict[str, Any],
        public_context: dict[str, Any] | None = None,
        sensor_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        public_context = public_context or {}
        sensor_context = sensor_context or {}

        senses = world.get("senses") or {}
        world_confidence = _clip(world.get("state_confidence"))
        data_quality = _clip(world.get("data_quality"))
        novelty = _clip(senses.get("novelty"))
        knowledge_state = str(world.get("knowledge_state") or "unknown")
        world_unknowns = [str(value) for value in (world.get("unknowns") or [])]

        self_uncertainty = _clip(self_model.get("uncertainty"))
        self_confidence = _clip(self_model.get("meta_confidence"))
        contradictions = [str(value) for value in (self_model.get("contradictions") or [])]

        council_score = _signed(council.get("consensus_score"))
        council_confidence = _clip(council.get("confidence"))
        council_uncertainty = _clip(council.get("uncertainty"))
        council_disagreement = _clip(council.get("disagreement"))

        critic_score = _signed(
            critic.get("adjusted_score")
            if critic.get("adjusted_score") is not None
            else council_score
        )
        critic_confidence = _clip(
            critic.get("adjusted_confidence")
            if critic.get("adjusted_confidence") is not None
            else council_confidence
        )
        critic_haircut = _clip(critic.get("confidence_haircut"))
        critic_risk = _clip(critic.get("risk_guidance_multiplier"), default=1.0)
        critic_objections = [
            str(row.get("reason"))
            for row in (critic.get("objections") or [])
            if isinstance(row, dict) and row.get("reason")
        ]

        tail_state = str(tail_risk.get("state") or "normal")
        tail_severity = _clip(tail_risk.get("severity"))
        tail_risk_multiplier = _clip(tail_risk.get("risk_guidance_multiplier"), default=1.0)

        memory_support = _clip(memory.get("support"))
        memory_samples = int(memory.get("contextual_samples") or memory.get("samples") or 0)
        historical_support = memory_support if memory_samples >= 4 else memory_support * min(1.0, memory_samples / 4.0)

        sensor_diversity, available_sensors, relevant_sensors = self._sensor_diversity(sensor_context)
        public_available = bool(public_context.get("available"))
        public_fresh = bool(public_context.get("fresh", public_available))
        evidence_freshness = 0.0
        freshness_parts = 0
        if relevant_sensors:
            evidence_freshness += sensor_diversity
            freshness_parts += 1
        if public_context:
            evidence_freshness += 1.0 if (public_available and public_fresh) else 0.0
            freshness_parts += 1
        if freshness_parts:
            evidence_freshness /= freshness_parts

        ood_probability = novelty
        if knowledge_state == "out_of_distribution":
            ood_probability = max(ood_probability, 0.75)
        elif knowledge_state in {"unknown", "poorly_identified"}:
            ood_probability = max(ood_probability, 1.0 - world_confidence)

        uncertainty = max(
            council_uncertainty,
            self_uncertainty,
            1.0 - world_confidence,
            0.50 * critic_haircut,
        )
        risk_pressure = _clip(
            0.35 * tail_severity
            + 0.25 * critic_haircut
            + 0.15 * council_disagreement
            + 0.15 * self_uncertainty
            + 0.10 * ood_probability
        )

        risk_multiplier = 1.0
        confidence_multiplier = 1.0
        reasons: list[str] = []
        allow_entry = bool(upstream_allowed)

        if not upstream_allowed:
            allow_entry = False
            risk_multiplier = 0.0
            reasons.append("upstream_not_allowed")

        # Hard veto authority is deliberately narrow and defensive: only a
        # compound severe/extreme Tail-Risk state can block an otherwise valid
        # upstream entry on its own.
        if tail_state in {"severe", "extreme"} or tail_severity >= 0.85:
            allow_entry = False
            risk_multiplier = 0.0
            reasons.append(f"tail_risk_{tail_state if tail_state else 'extreme'}")
        elif tail_state == "elevated" or tail_severity >= 0.45:
            risk_multiplier = min(risk_multiplier, max(0.0, tail_risk_multiplier))
            reasons.append("elevated_tail_risk")

        # The adversarial critic and uncertainty stack may only reduce risk.
        if critic_risk < 0.999:
            risk_multiplier = min(risk_multiplier, critic_risk)
            reasons.append("adversarial_critic_haircut")
        if critic_haircut >= 0.75:
            confidence_multiplier = min(confidence_multiplier, 0.45)
            risk_multiplier = min(risk_multiplier, 0.35)
            reasons.append("high_critic_objection_severity")
        elif critic_haircut >= 0.45:
            confidence_multiplier = min(confidence_multiplier, 0.70)
            risk_multiplier = min(risk_multiplier, 0.65)
            reasons.append("material_critic_objections")

        if council_disagreement >= 0.75 or council_uncertainty >= 0.80:
            confidence_multiplier = min(confidence_multiplier, 0.55)
            risk_multiplier = min(risk_multiplier, 0.45)
            reasons.append("high_council_uncertainty")
        elif council_disagreement >= 0.50 or council_uncertainty >= 0.60:
            confidence_multiplier = min(confidence_multiplier, 0.75)
            risk_multiplier = min(risk_multiplier, 0.70)
            reasons.append("council_disagreement")

        if len(contradictions) >= 2:
            confidence_multiplier = min(confidence_multiplier, 0.60)
            risk_multiplier = min(risk_multiplier, 0.50)
            reasons.append("multiple_internal_contradictions")
        elif contradictions:
            confidence_multiplier = min(confidence_multiplier, 0.80)
            risk_multiplier = min(risk_multiplier, 0.80)
            reasons.append("internal_contradiction")

        if ood_probability >= 0.85 and uncertainty >= 0.65:
            confidence_multiplier = min(confidence_multiplier, 0.50)
            risk_multiplier = min(risk_multiplier, 0.30)
            reasons.append("high_ood_uncertainty")
        elif ood_probability >= 0.70:
            confidence_multiplier = min(confidence_multiplier, 0.75)
            risk_multiplier = min(risk_multiplier, 0.65)
            reasons.append("out_of_distribution_market_state")

        if world_confidence < 0.20:
            confidence_multiplier = min(confidence_multiplier, 0.45)
            risk_multiplier = min(risk_multiplier, 0.40)
            reasons.append("weak_world_state_confidence")
        elif world_confidence < 0.35:
            confidence_multiplier = min(confidence_multiplier, 0.70)
            risk_multiplier = min(risk_multiplier, 0.70)
            reasons.append("developing_world_state_confidence")

        if data_quality and data_quality < 0.50:
            confidence_multiplier = min(confidence_multiplier, 0.65)
            risk_multiplier = min(risk_multiplier, 0.60)
            reasons.append("world_data_quality_low")

        if relevant_sensors >= 3 and sensor_diversity < 0.25:
            risk_multiplier = min(risk_multiplier, 0.65)
            reasons.append("low_sensor_diversity")
        if public_context and not (public_available and public_fresh):
            confidence_multiplier = min(confidence_multiplier, 0.85)
            reasons.append("public_context_not_fresh")

        # Research backlog and unresolved hypotheses are visible but are not
        # treated as proof against an entry.  They can only make uncertainty
        # explicit and gently reduce confidence when evidence is immature.
        active_hypotheses = self._active_hypotheses(hypothesis)
        missing_adapters = [str(value) for value in (research_plan.get("missing_adapters") or [])]
        degraded_sources = [str(value) for value in (research_plan.get("degraded_sources") or [])]
        if active_hypotheses and historical_support < 0.25:
            confidence_multiplier = min(confidence_multiplier, 0.85)
            reasons.append("unresolved_hypotheses_with_weak_history")
        if len(degraded_sources) >= 3:
            confidence_multiplier = min(confidence_multiplier, 0.85)
            reasons.append("multiple_degraded_research_sources")

        risk_multiplier = _clip(risk_multiplier, default=1.0)
        confidence_multiplier = _clip(confidence_multiplier, default=1.0)
        if allow_entry and risk_multiplier <= 0.0:
            allow_entry = False
        if not allow_entry and upstream_allowed:
            self.vetoes += 1
        elif upstream_allowed and risk_multiplier < 0.999:
            self.downsizes += 1

        action = "veto" if not allow_entry and upstream_allowed else "reduce" if risk_multiplier < 0.999 else "allow"
        result = {
            "symbol": symbol,
            "directional_bias": critic_score,
            "confidence": critic_confidence,
            "uncertainty": uncertainty,
            "risk_pressure": risk_pressure,
            "novelty": novelty,
            "ood_probability": ood_probability,
            "contradictions": contradictions,
            "critic_objections": critic_objections,
            "tail_risk": {
                "state": tail_state,
                "severity": tail_severity,
                "risk_guidance_multiplier": tail_risk_multiplier,
            },
            "evidence_freshness": evidence_freshness,
            "source_diversity": sensor_diversity,
            "available_sensors": available_sensors,
            "relevant_sensors": relevant_sensors,
            "historical_support": historical_support,
            "memory_samples": memory_samples,
            "world_unknowns": world_unknowns,
            "active_hypotheses": active_hypotheses,
            "falsification_status": "active_research" if active_hypotheses else "no_active_hypothesis",
            "missing_adapters": missing_adapters,
            "degraded_sources": degraded_sources,
            "action": action,
            "allow_entry": allow_entry,
            "risk_multiplier": risk_multiplier,
            "confidence_multiplier": confidence_multiplier,
            "reasons": list(dict.fromkeys(reasons)),
            "evidence_authority": "bounded_cognitive_safety_v1",
            "execution_authority": False,
            "paper_execution_authority": False,
            "testnet_execution_authority": False,
            "live_execution_authority": False,
            "can_increase_upstream_risk": False,
            "can_enable_live": False,
            "can_add_credentials": False,
            "can_rewrite_or_deploy": False,
            "assessed_at": time.time(),
        }

        self.state.setdefault("latest", {})[symbol] = result
        history = self.state.setdefault("history", [])
        history.append(
            {
                "symbol": symbol,
                "action": action,
                "risk_multiplier": risk_multiplier,
                "confidence_multiplier": confidence_multiplier,
                "tail_state": tail_state,
                "uncertainty": uncertainty,
                "ood_probability": ood_probability,
                "reasons": result["reasons"],
                "timestamp": result["assessed_at"],
            }
        )
        self.state["history"] = history[-self.HISTORY_LIMIT :]
        self.assessments += 1
        self.state["assessments"] = self.assessments
        self.state["vetoes"] = self.vetoes
        self.state["downsizes"] = self.downsizes
        if self.assessments % self.SAVE_INTERVAL == 0 or action == "veto":
            self._save()
        return dict(result)

    def health(self) -> dict[str, Any]:
        latest = self.state.get("latest") or {}
        actions = {"allow": 0, "reduce": 0, "veto": 0}
        for row in latest.values():
            action = str((row or {}).get("action") or "allow")
            actions[action] = actions.get(action, 0) + 1
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "assessments": self.assessments,
            "tracked_symbols": len(latest),
            "vetoes": self.vetoes,
            "downsizes": self.downsizes,
            "current_actions": actions,
            "evidence_authority": "bounded_cognitive_safety_v1",
            "fail_closed_on_runtime_integration_error": True,
            "execution_authority": False,
            "can_only_preserve_reduce_or_veto": True,
            "can_increase_upstream_risk": False,
            "can_enable_live": False,
            "can_add_credentials": False,
            "can_rewrite_or_deploy": False,
            "state_path": str(self.state_path),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "assessments": 0,
            "vetoes": 0,
            "downsizes": 0,
            "latest": {},
            "history": [],
        }
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
        self.state["assessments"] = self.assessments
        self.state["vetoes"] = self.vetoes
        self.state["downsizes"] = self.downsizes
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
