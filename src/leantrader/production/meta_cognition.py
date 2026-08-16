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


def _sign(value: float, epsilon: float = 1e-12) -> int:
    if value > epsilon:
        return 1
    if value < -epsilon:
        return -1
    return 0


class MetaCognitiveSelfModel:
    """Persistent operational self-model for LeanTrader.

    This is machine self-awareness in the engineering sense: the system tracks
    what evidence it has, where it is uncertain, which internal specialists have
    historically aligned with realized outcomes, and where its own subsystems
    contradict one another.  It has no execution, deployment, credential or
    risk-increase authority.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    SAVE_INTERVAL = 10
    TRUST_PRIOR = 2.0

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.last_error: str | None = None
        self.state = self._load()
        self.assessments = int(self.state.get("assessments") or 0)
        self.closed_outcomes = int(self.state.get("closed_outcomes") or 0)
        self.system_observations = int(self.state.get("system_observations") or 0)

    def start(self) -> None:
        self.state = self._load()
        self.assessments = int(self.state.get("assessments") or 0)
        self.closed_outcomes = int(self.state.get("closed_outcomes") or 0)
        self.system_observations = int(self.state.get("system_observations") or 0)

    def stop(self) -> None:
        self._save()

    def specialist_trust(self, specialist: str) -> dict[str, Any]:
        row = dict((self.state.get("specialist_trust") or {}).get(specialist, {}))
        wins = float(row.get("aligned") or 0.0)
        losses = float(row.get("misaligned") or 0.0)
        samples = int(row.get("samples") or 0)
        reliability = (wins + self.TRUST_PRIOR) / (wins + losses + 2.0 * self.TRUST_PRIOR)
        return {
            "specialist": specialist,
            "samples": samples,
            "aligned": int(wins),
            "misaligned": int(losses),
            "reliability": reliability,
            "maturity": "measured" if samples >= 20 else "developing" if samples >= 5 else "prior_only",
            "ewma_signed_value": _finite(row.get("ewma_signed_value")),
        }

    def _update_trust(self, specialist: str, score: float, realized_return: float) -> None:
        direction = _sign(score)
        outcome_direction = _sign(realized_return)
        if direction == 0 or outcome_direction == 0:
            return
        row = self.state.setdefault("specialist_trust", {}).setdefault(
            specialist,
            {
                "samples": 0,
                "aligned": 0,
                "misaligned": 0,
                "ewma_signed_value": 0.0,
                "last_score": None,
                "last_outcome": None,
            },
        )
        row["samples"] = int(row.get("samples", 0)) + 1
        aligned = direction == outcome_direction
        row["aligned"] = int(row.get("aligned", 0)) + int(aligned)
        row["misaligned"] = int(row.get("misaligned", 0)) + int(not aligned)
        signed_value = score * realized_return
        previous = _finite(row.get("ewma_signed_value"))
        row["ewma_signed_value"] = signed_value if row["samples"] == 1 else 0.90 * previous + 0.10 * signed_value
        row["last_score"] = score
        row["last_outcome"] = realized_return
        row["updated_at"] = time.time()

    def record_outcome(self, metadata: dict[str, Any], realized_return: float) -> dict[str, Any]:
        if not math.isfinite(float(realized_return)):
            raise ValueError("self model outcome must be finite")
        specialists: dict[str, float] = {}
        component_scores = metadata.get("component_scores") or {}
        weights = metadata.get("weights") or {}
        for name, value in component_scores.items():
            specialists[f"adaptive_component:{name}"] = _finite(value)
        if component_scores:
            specialists["adaptive_ensemble"] = sum(
                _finite(value) * _finite(weights.get(name), 0.0)
                for name, value in component_scores.items()
            )
        for name, value in (metadata.get("advanced_feature_vector") or {}).items():
            specialists[f"advanced:{name}"] = _finite(value)
        advanced = metadata.get("advanced_shadow") or {}
        for row in advanced.get("signals") or []:
            if isinstance(row, dict) and row.get("engine"):
                specialists[f"advanced:{row['engine']}"] = _finite(row.get("score"))
        swarm = advanced.get("swarm") or {}
        if swarm:
            specialists["advanced:swarm_hivemind"] = _finite(swarm.get("score"))
        route = metadata.get("decision_route") or {}
        specialists["bounded_decision_router"] = _finite(route.get("combined_score"))
        council = metadata.get("intelligence_council") or {}
        if council:
            specialists["intelligence_council"] = _finite(council.get("consensus_score"))
        critic = metadata.get("adversarial_critic") or {}
        if critic:
            specialists["critic_adjusted_council"] = _finite(critic.get("adjusted_score"))
        world = metadata.get("market_world_model") or {}
        flow_sensor = (world.get("external_sensors") or {}).get("flow_intelligence") or {}
        if flow_sensor.get("status") == "available":
            specialists["onchain_flow_intelligence"] = _finite(
                (world.get("specialist_scores") or {}).get("onchain_flow_intelligence")
            )

        for name, score in specialists.items():
            self._update_trust(name, score, float(realized_return))

        self.closed_outcomes += 1
        self.state["closed_outcomes"] = self.closed_outcomes
        outcomes = self.state.setdefault("outcome_history", [])
        outcomes.append(
            {
                "timestamp": time.time(),
                "symbol": str(metadata.get("symbol") or ""),
                "realized_return": float(realized_return),
                "specialists_scored": len([value for value in specialists.values() if _sign(value) != 0]),
            }
        )
        self.state["outcome_history"] = outcomes[-500:]
        self._save()
        return {
            "recorded": True,
            "closed_outcomes": self.closed_outcomes,
            "specialists_updated": len(specialists),
            "execution_authority": False,
        }

    def assess_symbol(
        self,
        *,
        symbol: str,
        world: dict[str, Any],
        cns: dict[str, Any],
        memory: dict[str, Any],
        route: dict[str, Any],
        brain: dict[str, Any],
        strategy_evidence: dict[str, Any] | None,
        engine_health: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        senses = world.get("senses") or {}
        state_confidence = _clip(_finite(world.get("state_confidence")))
        novelty = _clip(_finite(senses.get("novelty")))
        disagreement = _clip(_finite(senses.get("model_disagreement")))
        coherence = _clip(_finite(cns.get("signal_coherence")))
        risk_pressure = _clip(_finite(cns.get("risk_pressure")))
        memory_support = _clip(_finite(memory.get("support")))
        memory_samples = int(memory.get("contextual_samples") or memory.get("samples") or 0)
        evidence = strategy_evidence or {}
        strategy_samples = int(evidence.get("samples") or 0)
        strategy_expectancy = _finite(
            evidence.get("ewma_net_return")
            if evidence.get("ewma_net_return") is not None
            else evidence.get("average_net_return")
        )

        unknowns = set(str(value) for value in (world.get("unknowns") or []))
        if memory_samples < 4:
            unknowns.add("similar_closed_outcomes")
        if strategy_samples < 8:
            unknowns.add("costed_strategy_evidence")
        if self.closed_outcomes < 8:
            unknowns.add("self_calibration_outcomes")
        if not engine_health:
            unknowns.add("engine_health_snapshot")

        contradictions: list[str] = []
        adaptive_score = _finite((world.get("adaptive") or {}).get("score"))
        swarm_score = _finite((world.get("swarm") or {}).get("score"))
        if _sign(adaptive_score) and _sign(swarm_score) and _sign(adaptive_score) != _sign(swarm_score):
            contradictions.append("adaptive_swarm_direction_conflict")
        if bool(route.get("router_allowed_pre_brain")) and brain.get("allow_entry") is False:
            contradictions.append("router_brain_authority_conflict")
        if state_confidence >= 0.60 and novelty >= 0.70:
            contradictions.append("high_confidence_in_novel_state")
        if coherence < 0.25 and max(
            _clip(_finite((world.get("adaptive") or {}).get("confidence"))),
            _clip(_finite((world.get("swarm") or {}).get("confidence"))),
        ) >= 0.60:
            contradictions.append("high_component_confidence_low_coherence")
        if strategy_samples >= 20 and strategy_expectancy < 0 and route.get("allowed") is True:
            contradictions.append("negative_strategy_evidence_but_route_allowed")

        trust_rows = [self.specialist_trust(name) for name in (self.state.get("specialist_trust") or {})]
        measured_trust = [float(row["reliability"]) for row in trust_rows if row["samples"] >= 5]
        trust_confidence = float(sum(measured_trust) / len(measured_trust)) if measured_trust else 0.50
        evidence_maturity = _clip((min(strategy_samples, 50) / 50.0 + min(memory_samples, 20) / 20.0) / 2.0)
        self_calibration_maturity = _clip(self.closed_outcomes / 50.0)
        uncertainty = _clip(
            0.30 * (1.0 - state_confidence)
            + 0.20 * novelty
            + 0.20 * disagreement
            + 0.10 * (1.0 - coherence)
            + 0.10 * (1.0 - evidence_maturity)
            + 0.10 * min(1.0, len(unknowns) / 5.0)
        )
        meta_confidence = _clip(
            (1.0 - uncertainty)
            * (0.55 + 0.25 * trust_confidence + 0.20 * self_calibration_maturity)
        )

        if risk_pressure >= 0.70 or novelty >= 0.85:
            stance = "defensive_observation"
        elif uncertainty >= 0.65:
            stance = "research_before_risk"
        elif evidence_maturity < 0.25:
            stance = "bounded_exploration"
        else:
            stance = "measured_operation"

        result = {
            "symbol": symbol,
            "knowledge_state": world.get("knowledge_state", "unknown"),
            "stance": stance,
            "meta_confidence": meta_confidence,
            "uncertainty": uncertainty,
            "unknowns": sorted(unknowns),
            "contradictions": sorted(set(contradictions)),
            "awareness_profile": {
                "market_state_confidence": state_confidence,
                "evidence_maturity": evidence_maturity,
                "memory_grounding": memory_support,
                "cross_engine_coherence": coherence,
                "specialist_trust_confidence": trust_confidence,
                "self_calibration_maturity": self_calibration_maturity,
            },
            "specialist_trust": {
                name: self.specialist_trust(name)
                for name in sorted((self.state.get("specialist_trust") or {}).keys())
            },
            "risk_guidance_multiplier": max(0.0, min(1.0, 1.0 - 0.65 * uncertainty - 0.35 * risk_pressure)),
            "can_only_inform_or_reduce_risk": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "can_enable_live": False,
            "can_modify_code": False,
            "can_deploy": False,
            "assessed_at": time.time(),
        }
        self.state.setdefault("latest", {})[symbol] = result
        self.assessments += 1
        self.state["assessments"] = self.assessments
        if contradictions:
            records = self.state.setdefault("contradiction_history", [])
            records.append(
                {
                    "timestamp": result["assessed_at"],
                    "symbol": symbol,
                    "contradictions": result["contradictions"],
                }
            )
            self.state["contradiction_history"] = records[-500:]
        if self.assessments % self.SAVE_INTERVAL == 0:
            self._save()
        return dict(result)

    def observe_system(
        self,
        *,
        engines: dict[str, Any],
        execution_funnel: dict[str, Any],
        errors: dict[str, Any],
        capital: dict[str, Any],
        world_health: dict[str, Any],
    ) -> dict[str, Any]:
        required = [row for row in engines.values() if isinstance(row, dict) and row.get("required")]
        healthy_required = sum(1 for row in required if row.get("healthy") is True)
        required_health = healthy_required / len(required) if required else 1.0
        approved = int(execution_funnel.get("final_route_allowed") or 0)
        attempts = int(execution_funnel.get("entry_attempts") or 0)
        positions = int(execution_funnel.get("open_positions") or 0)
        capacity_constrained = approved > 0 and attempts == 0 and positions > 0
        system_unknowns: list[str] = []
        if errors:
            system_unknowns.append("active_runtime_errors")
        if required_health < 1.0:
            system_unknowns.append("required_engine_degradation")
        if int(world_health.get("out_of_distribution_symbols") or 0) > 0:
            system_unknowns.append("out_of_distribution_market_states")
        if self.closed_outcomes < 8:
            system_unknowns.append("insufficient_self_calibration_outcomes")

        status = {
            "timestamp": time.time(),
            "required_engine_health": required_health,
            "active_errors": len(errors),
            "execution_capacity_constrained": capacity_constrained,
            "capital_state": str(capital.get("state") or "unknown"),
            "capital_can_open_entries": bool(capital.get("new_entries_allowed", False)),
            "unknowns": system_unknowns,
            "awareness_maturity": {
                "operational": "measured" if required_health == 1.0 else "degraded",
                "market_world_model": "measured" if int(world_health.get("observations") or 0) >= 100 else "developing",
                "outcome_calibration": "measured" if self.closed_outcomes >= 20 else "developing",
                "specialist_trust": "measured" if self.closed_outcomes >= 20 else "developing",
            },
            "execution_authority": False,
        }
        self.state["system"] = status
        self.system_observations += 1
        self.state["system_observations"] = self.system_observations
        self._save()
        return dict(status)

    def health(self) -> dict[str, Any]:
        latest = self.state.get("latest") or {}
        contradictions = sum(len(row.get("contradictions") or []) for row in latest.values())
        unknowns = sum(len(row.get("unknowns") or []) for row in latest.values())
        trust = {
            name: self.specialist_trust(name)
            for name in sorted((self.state.get("specialist_trust") or {}).keys())
        }
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "assessments": self.assessments,
            "closed_outcomes": self.closed_outcomes,
            "system_observations": self.system_observations,
            "tracked_symbols": len(latest),
            "current_unknowns": unknowns,
            "current_contradictions": contradictions,
            "specialist_trust": trust,
            "system": dict(self.state.get("system") or {}),
            "machine_self_model": True,
            "consciousness_claim": False,
            "execution_authority": False,
            "can_increase_risk": False,
            "can_enable_live": False,
            "can_modify_code": False,
            "can_deploy": False,
            "state_path": str(self.state_path),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "assessments": 0,
            "closed_outcomes": 0,
            "system_observations": 0,
            "latest": {},
            "system": {},
            "specialist_trust": {},
            "outcome_history": [],
            "contradiction_history": [],
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
        self.state["closed_outcomes"] = self.closed_outcomes
        self.state["system_observations"] = self.system_observations
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
