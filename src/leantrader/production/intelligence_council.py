from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class IntelligenceCouncil:
    """Trust-aware council of independent market specialists.

    Council output is research/meta-cognitive evidence only in v12.6.  It cannot
    place orders or increase any existing risk budget.  Persisted activity makes
    it possible to measure whether a seemingly sophisticated specialist is
    actually contributing over time.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    SAVE_INTERVAL = 10

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.last_error: str | None = None
        self.state = self._load()
        self.deliberations = int(self.state.get("deliberations") or 0)

    def start(self) -> None:
        self.state = self._load()
        self.deliberations = int(self.state.get("deliberations") or 0)

    def stop(self) -> None:
        self._save()

    @staticmethod
    def _trust(self_model: dict[str, Any], key: str) -> float:
        trust = self_model.get("specialist_trust") or {}
        row = trust.get(key) or {}
        samples = int(row.get("samples") or 0)
        reliability = _finite(row.get("reliability"), 0.50)
        # Shrink immature historical reliability toward a neutral prior.
        maturity = min(1.0, samples / 20.0)
        return 0.50 * (1.0 - maturity) + max(0.20, min(0.80, reliability)) * maturity

    def deliberate(
        self,
        *,
        symbol: str,
        adaptive: dict[str, Any],
        advanced: dict[str, Any],
        world: dict[str, Any],
        self_model: dict[str, Any],
        memory: dict[str, Any],
        public_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        members: list[dict[str, Any]] = []

        def add(name: str, score: Any, confidence: Any, trust_key: str, base_weight: float = 1.0) -> None:
            parsed_score = _clip(_finite(score))
            parsed_conf = max(0.0, min(1.0, _finite(confidence)))
            trust = self._trust(self_model, trust_key)
            weight = max(0.0, base_weight * parsed_conf * (0.50 + trust))
            members.append(
                {
                    "name": name,
                    "score": parsed_score,
                    "confidence": parsed_conf,
                    "historical_trust": trust,
                    "weight": weight,
                }
            )

        add(
            "adaptive_ensemble",
            adaptive.get("score"),
            adaptive.get("confidence"),
            "bounded_decision_router",
            1.20,
        )
        for row in advanced.get("signals") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("engine") or "").strip()
            if not name:
                continue
            add(
                name,
                row.get("score"),
                row.get("confidence"),
                f"advanced:{name}",
                1.0,
            )
        swarm = advanced.get("swarm") or {}
        add(
            "swarm_hivemind",
            swarm.get("score"),
            swarm.get("confidence"),
            "advanced:swarm_hivemind",
            0.75,
        )
        if public_context and public_context.get("available"):
            add(
                "public_market_context",
                public_context.get("score"),
                public_context.get("confidence"),
                "public_market_context",
                0.65,
            )
        memory_support = max(0.0, min(1.0, _finite(memory.get("support"))))
        memory_return = _finite(memory.get("weighted_net_return"))
        if memory_support > 0:
            add(
                "episodic_memory",
                math.tanh(memory_return * 50.0),
                memory_support,
                "episodic_memory",
                0.80,
            )

        eligible = [member for member in members if member["weight"] > 0.0]
        if eligible:
            scores = np.asarray([member["score"] for member in eligible], dtype=float)
            weights = np.asarray([member["weight"] for member in eligible], dtype=float)
            consensus = float(np.average(scores, weights=weights))
            disagreement = float(np.sqrt(np.average((scores - consensus) ** 2, weights=weights)))
            raw_confidence = float(np.average([member["confidence"] for member in eligible], weights=weights))
        else:
            consensus = 0.0
            disagreement = 1.0
            raw_confidence = 0.0

        world_quality = max(0.0, min(1.0, _finite(world.get("data_quality"))))
        meta_conf = max(0.0, min(1.0, _finite(self_model.get("meta_confidence"))))
        council_confidence = max(
            0.0,
            min(1.0, raw_confidence * (1.0 - min(1.0, disagreement)) * (0.60 + 0.20 * world_quality + 0.20 * meta_conf)),
        )
        uncertainty = max(
            0.0,
            min(
                1.0,
                0.45 * disagreement
                + 0.25 * (1.0 - world_quality)
                + 0.20 * _finite(self_model.get("uncertainty"))
                + 0.10 * _finite((world.get("senses") or {}).get("novelty")),
            ),
        )
        if consensus > 0.15:
            direction = "bullish"
        elif consensus < -0.15:
            direction = "bearish"
        else:
            direction = "indeterminate"

        rare_scope = _finite((world.get("senses") or {}).get("rare_scope_score"))
        result = {
            "symbol": symbol.upper(),
            "consensus_score": _clip(consensus),
            "confidence": council_confidence,
            "uncertainty": uncertainty,
            "disagreement": min(1.0, max(0.0, disagreement)),
            "direction": direction,
            "members": members,
            "members_eligible": len(eligible),
            "rare_scope_research_candidate": rare_scope >= 0.60,
            "rare_scope_score": rare_scope,
            "knowledge_state": world.get("knowledge_state", "unknown"),
            "execution_authority": False,
            "paper_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_only_inform_research": True,
            "deliberated_at": time.time(),
        }
        self.state.setdefault("latest", {})[symbol.upper()] = result
        self.deliberations += 1
        self.state["deliberations"] = self.deliberations
        if self.deliberations % self.SAVE_INTERVAL == 0:
            self._save()
        return dict(result)

    def health(self) -> dict[str, Any]:
        latest = self.state.get("latest") or {}
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "deliberations": self.deliberations,
            "tracked_symbols": len(latest),
            "execution_authority": False,
            "paper_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "trust_aware": True,
            "disagreement_visible": True,
            "state_path": str(self.state_path),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {"schema_version": self.SCHEMA_VERSION, "deliberations": 0, "latest": {}}
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
        self.state["deliberations"] = self.deliberations
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
