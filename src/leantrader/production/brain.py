from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class TradingBrain:
    """Bounded meta-controller over CNS, memory and measured strategy evidence.

    The Brain has no execution authority. It can only preserve an upstream
    approval unchanged, reduce risk/confidence, or veto a new entry. It cannot
    raise the upstream risk budget, add leverage, enable live trading, rewrite
    code, or deploy itself.
    """

    VERSION = "2.2"
    SAVE_INTERVAL = 10

    def __init__(
        self,
        state_path: Path,
        *,
        min_strategy_samples: int = 50,
        negative_expectancy_floor: float = -0.001,
        quarantine_min_samples: int = 100,
        quarantine_expectancy_floor: float = -0.004,
        recovery_expectancy_floor: float = 0.0005,
    ) -> None:
        if min_strategy_samples < 1:
            raise ValueError("brain minimum strategy samples must be positive")
        if quarantine_min_samples < min_strategy_samples:
            raise ValueError("brain quarantine samples cannot be below minimum strategy samples")
        if quarantine_expectancy_floor > negative_expectancy_floor:
            raise ValueError("brain quarantine floor must be no greater than negative evidence floor")
        if recovery_expectancy_floor < 0:
            raise ValueError("brain recovery expectancy floor cannot be negative")
        self.state_path = state_path
        self.min_strategy_samples = min_strategy_samples
        self.negative_expectancy_floor = float(negative_expectancy_floor)
        self.quarantine_min_samples = quarantine_min_samples
        self.quarantine_expectancy_floor = float(quarantine_expectancy_floor)
        self.recovery_expectancy_floor = float(recovery_expectancy_floor)
        self.evaluations = 0
        self.vetoes = 0
        self.downsizes = 0
        self.last: dict[str, dict[str, Any]] = {}
        self.quarantined_strategies: dict[str, dict[str, Any]] = {}
        self.legacy_quarantined_strategies: dict[str, dict[str, Any]] = {}
        self.last_error: str | None = None
        self._load()

    def start(self) -> None:
        self._load()

    def stop(self) -> None:
        self._save()

    @staticmethod
    def strategy_expectancy(strategy_evidence: dict[str, Any] | None) -> tuple[int, float, str]:
        evidence = strategy_evidence or {}
        authority = str(evidence.get("authority") or "untrusted_or_legacy")
        if authority not in {"costed_shadow_episode_v2", "closed_trade"}:
            return 0, 0.0, authority
        samples = int(evidence.get("samples") or 0)
        if samples <= 0:
            return 0, 0.0, authority
        if evidence.get("ewma_net_return") is not None:
            return samples, float(evidence["ewma_net_return"]), authority
        if evidence.get("average_net_return") is not None:
            return samples, float(evidence["average_net_return"]), authority
        cumulative = float(evidence.get("cumulative_net_return") or 0.0)
        return samples, cumulative / samples, authority

    def evaluate(
        self,
        *,
        symbol: str,
        cns: dict[str, Any],
        memory: dict[str, Any],
        strategy_evidence: dict[str, Any] | None = None,
        upstream_allowed: bool,
        strategy_name: str = "bounded_decision_router",
    ) -> dict[str, Any]:
        coherence = _clip(float(cns.get("signal_coherence") or 0.0))
        risk_pressure = _clip(float(cns.get("risk_pressure") or 0.0))
        memory_support = _clip(float(memory.get("support") or 0.0))
        memory_return = float(memory.get("weighted_net_return") or 0.0)
        if "contextual_samples" in memory or "samples" in memory:
            memory_samples = int(memory.get("contextual_samples") or memory.get("samples") or 0)
        else:
            # Backward-compatible input contract for callers that supplied only
            # support/return before v12.2. Runtime v12.2 always sends samples.
            memory_samples = 4 if memory_support >= 0.50 else 0
        memory_source = str(memory.get("source") or "none")
        samples, expectancy, evidence_authority = self.strategy_expectancy(strategy_evidence)

        reasons: list[str] = []
        allow_entry = bool(upstream_allowed)
        risk_multiplier = 1.0
        confidence_multiplier = 1.0

        if not upstream_allowed:
            reasons.append("upstream_router_rejected")
            allow_entry = False
            risk_multiplier = 0.0

        safety_blocks = [str(value) for value in (cns.get("safety_blocks") or [])]
        if safety_blocks:
            reasons.extend(safety_blocks)
            if "cns_high_risk_pressure" in safety_blocks:
                allow_entry = False
                risk_multiplier = 0.0

        if risk_pressure > 0.25:
            risk_multiplier *= max(0.20, 1.0 - risk_pressure)
        if coherence < 0.35:
            confidence_multiplier *= max(0.50, coherence / 0.35)
            risk_multiplier *= max(0.35, coherence / 0.35)
            reasons.append("low_cross_engine_coherence")

        quarantine = self.quarantined_strategies.get(strategy_name)
        if (
            samples >= self.quarantine_min_samples
            and expectancy <= self.quarantine_expectancy_floor
        ):
            quarantine = {
                "strategy": strategy_name,
                "samples": samples,
                "expectancy": expectancy,
                "reason": "persistent_negative_expectancy",
                "evidence_authority": evidence_authority,
                "quarantined_at": time.time(),
            }
            self.quarantined_strategies[strategy_name] = quarantine
        elif (
            quarantine is not None
            and samples >= self.min_strategy_samples
            and expectancy >= self.recovery_expectancy_floor
        ):
            self.quarantined_strategies.pop(strategy_name, None)
            quarantine = None
            reasons.append("strategy_recovered_from_quarantine")

        if quarantine is not None:
            allow_entry = False
            risk_multiplier = 0.0
            confidence_multiplier = min(confidence_multiplier, 0.50)
            reasons.append("strategy_quarantined")
        elif samples >= self.min_strategy_samples and expectancy < self.negative_expectancy_floor:
            # Preserve shadow exploration in the observatory while sharply
            # reducing actual paper/Testnet exposure to sufficiently negative evidence.
            risk_multiplier = min(risk_multiplier, 0.20)
            confidence_multiplier = min(confidence_multiplier, 0.65)
            reasons.append("negative_strategy_evidence")

        # Only contextual v12 closed outcomes may materially alter risk. A
        # legacy fill prior can inform CNS context at very low weight, but it
        # cannot veto/downsize until at least four fingerprinted outcomes exist.
        if memory_samples >= 4 and memory_support >= 0.50 and memory_return < self.negative_expectancy_floor:
            risk_multiplier = min(risk_multiplier, 0.25)
            reasons.append("negative_similar_memory")
        elif memory_samples >= 4 and memory_support >= 0.50 and memory_return > 0:
            # Positive memory may recover confidence but can never enlarge the
            # upstream risk allocation.
            confidence_multiplier = min(1.0, confidence_multiplier + 0.10 * memory_support)

        risk_multiplier = _clip(risk_multiplier)
        confidence_multiplier = _clip(confidence_multiplier)
        if allow_entry and risk_multiplier <= 0.0:
            allow_entry = False
        if not allow_entry:
            self.vetoes += 1
        elif risk_multiplier < 0.999:
            self.downsizes += 1

        preferred_action = "hold"
        action_bias = float(cns.get("action_bias") or 0.0)
        if allow_entry and action_bias > 0.10:
            preferred_action = "long_candidate"
        elif action_bias < -0.10:
            preferred_action = "exit_or_avoid"

        result = {
            "symbol": symbol.upper(),
            "strategy": strategy_name,
            "allow_entry": allow_entry,
            "risk_multiplier": risk_multiplier,
            "confidence_multiplier": confidence_multiplier,
            "preferred_action": preferred_action,
            "reasons": list(dict.fromkeys(reasons)),
            "memory_support": memory_support,
            "memory_samples": memory_samples,
            "memory_source": memory_source,
            "memory_weighted_net_return": memory_return,
            "strategy_samples": samples,
            "strategy_expectancy": expectancy,
            "strategy_evidence_authority": evidence_authority,
            "strategy_quarantined": quarantine is not None,
            "execution_authority": False,
            "can_increase_upstream_risk": False,
            "can_enable_live": False,
            "can_rewrite_or_deploy": False,
            "evaluated_at": time.time(),
        }
        self.last[symbol.upper()] = result
        self.evaluations += 1
        if self.evaluations % self.SAVE_INTERVAL == 0:
            self._save()
        return dict(result)

    def health(self) -> dict[str, Any]:
        return {
            "healthy": self.last_error is None,
            "evaluations": self.evaluations,
            "vetoes": self.vetoes,
            "downsizes": self.downsizes,
            "tracked_symbols": len(self.last),
            "quarantined_strategies": len(self.quarantined_strategies),
            "quarantine": dict(self.quarantined_strategies),
            "legacy_quarantines_released": len(self.legacy_quarantined_strategies),
            "accepted_strategy_evidence_authorities": ["costed_shadow_episode_v2", "closed_trade"],
            "minimum_strategy_samples": self.min_strategy_samples,
            "negative_expectancy_floor": self.negative_expectancy_floor,
            "quarantine_min_samples": self.quarantine_min_samples,
            "quarantine_expectancy_floor": self.quarantine_expectancy_floor,
            "recovery_expectancy_floor": self.recovery_expectancy_floor,
            "execution_authority": False,
            "safety_governor_only": True,
            "can_increase_upstream_risk": False,
            "can_enable_live": False,
            "can_rewrite_or_deploy": False,
            "error": self.last_error,
        }

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.evaluations = int(payload.get("evaluations", 0))
            self.vetoes = int(payload.get("vetoes", 0))
            self.downsizes = int(payload.get("downsizes", 0))
            self.last = dict(payload.get("last") or {})
            loaded_quarantine = dict(payload.get("quarantined_strategies") or {})
            self.legacy_quarantined_strategies = dict(payload.get("legacy_quarantined_strategies") or {})
            if str(payload.get("version") or "") == self.VERSION:
                self.quarantined_strategies = loaded_quarantine
            else:
                # v12.4/v2.1 quarantines were allowed to depend on per-poll
                # observatory pseudo-P&L. Preserve them for audit but do not let
                # contaminated legacy evidence veto new paper/Testnet entries.
                self.legacy_quarantined_strategies.update(loaded_quarantine)
                self.quarantined_strategies = {}
            self.last_error = None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last = {}
            self.quarantined_strategies = {}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = {
            "version": self.VERSION,
            "evaluations": self.evaluations,
            "vetoes": self.vetoes,
            "downsizes": self.downsizes,
            "last": self.last,
            "quarantined_strategies": self.quarantined_strategies,
            "legacy_quarantined_strategies": self.legacy_quarantined_strategies,
            "updated_at": time.time(),
        }
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
