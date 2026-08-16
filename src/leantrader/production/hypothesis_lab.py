from __future__ import annotations

import hashlib
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


class HypothesisLab:
    """Persistent falsifiable-hypothesis engine for rare/novel market structure.

    Novelty is treated as a research trigger, never as a trading edge.  Each
    hypothesis stores the evidence that created it and is later resolved against
    observed market outcomes.  This creates a causal-research journal rather than
    a collection of impressive but untestable narratives.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    SAVE_INTERVAL = 5
    ACTIVE_LIMIT = 1_000
    RESOLVED_LIMIT = 5_000

    def __init__(self, state_path: Path, *, horizon_observations: int = 8) -> None:
        if horizon_observations < 2:
            raise ValueError("hypothesis horizon must be at least two observations")
        self.state_path = state_path
        self.horizon_observations = horizon_observations
        self.state = self._load()
        self.observations = int(self.state.get("observations") or 0)
        self.generated = int(self.state.get("generated") or 0)
        self.resolved = int(self.state.get("resolved_count") or 0)
        self.last_error: str | None = None

    def start(self) -> None:
        self.state = self._load()
        self.observations = int(self.state.get("observations") or 0)
        self.generated = int(self.state.get("generated") or 0)
        self.resolved = int(self.state.get("resolved_count") or 0)

    def stop(self) -> None:
        self._save()

    @staticmethod
    def _statement(pattern: str) -> str:
        return {
            "volume_without_price_followthrough": "Participation is rising without immediate price follow-through; test for delayed volatility expansion rather than assume direction.",
            "liquidity_price_divergence": "Order-book imbalance and recent price direction disagree; test whether price subsequently converges toward the liquidity imbalance.",
            "cross_model_disagreement": "Independent market models disagree materially; test whether this disagreement predicts regime transition or elevated realized surprise.",
            "multi_timeframe_phase_fracture": "Fast/tactical/strategic horizons are structurally misaligned; test for a regime transition or volatility expansion before trusting a directional thesis.",
            "compression_with_participation_anomaly": "Volatility compression coexists with unusual participation; test whether realized volatility expands after the anomaly.",
            "volatility_liquidity_coupling": "Volatility and liquidity stress are rising together; test persistence and tail-risk rather than chase the move.",
            "out_of_distribution_market_state": "The current state is unlike recent history; test whether standard model confidence is overestimated in this region.",
            "narrative_price_divergence": "Public-context direction and price direction disagree; test whether price converges toward context or the narrative signal decays.",
        }.get(pattern, "A low-frequency conjunction of market sensors appeared; test whether it has repeatable forward information after costs and regime controls.")

    def _fingerprint(self, symbol: str, pattern: str, regime: str) -> str:
        raw = f"{symbol}|{pattern}|{regime}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    def _resolve(self, row: dict[str, Any], world: dict[str, Any]) -> dict[str, Any] | None:
        observations = int(row.get("observations") or 0) + 1
        row["observations"] = observations
        if observations < int(row.get("horizon_observations") or self.horizon_observations):
            return None
        entry_price = _finite(row.get("entry_price"))
        price = _finite(world.get("price"))
        if entry_price <= 0 or price <= 0:
            return None
        realized_return = price / entry_price - 1.0
        pattern = str(row.get("pattern") or "")
        trigger = row.get("trigger") or {}
        entry_vol = max(_finite(trigger.get("vol_short")), 1e-6)
        move_threshold = max(0.001, entry_vol * math.sqrt(max(1, observations)) * 0.8)
        success = False
        criterion = "absolute_move_exceeds_state_baseline"
        if pattern == "liquidity_price_divergence":
            success = _sign(realized_return) == _sign(_finite(trigger.get("liquidity_imbalance"))) and _sign(realized_return) != 0
            criterion = "price_converges_toward_prior_liquidity_imbalance"
        elif pattern == "narrative_price_divergence":
            success = _sign(realized_return) == _sign(_finite(trigger.get("context_score"))) and _sign(realized_return) != 0
            criterion = "price_converges_toward_public_context"
        elif pattern in {
            "volume_without_price_followthrough",
            "compression_with_participation_anomaly",
            "multi_timeframe_phase_fracture",
            "cross_model_disagreement",
        }:
            success = abs(realized_return) >= move_threshold or bool(world.get("regime_changed"))
            criterion = "volatility_expansion_or_regime_transition"
        elif pattern == "volatility_liquidity_coupling":
            senses = world.get("senses") or {}
            success = _finite(senses.get("volatility_shock")) >= 0.40 or _finite(senses.get("liquidity_stress")) >= 0.40
            criterion = "tail_risk_state_persists"
        elif pattern == "out_of_distribution_market_state":
            surprise = abs(_finite((world.get("features") or {}).get("price_surprise")))
            success = surprise >= 1.0 or bool(world.get("regime_changed"))
            criterion = "novel_state_produces_surprise_or_regime_change"
        else:
            success = abs(realized_return) >= move_threshold

        return {
            **row,
            "status": "supported" if success else "not_supported",
            "resolved_at": time.time(),
            "exit_price": price,
            "realized_return": realized_return,
            "criterion": criterion,
            "criterion_threshold": move_threshold,
            "supported": bool(success),
        }

    def observe(
        self,
        *,
        symbol: str,
        world: dict[str, Any],
        council: dict[str, Any],
        critic: dict[str, Any],
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        active = self.state.setdefault("active", {})
        resolved_rows = self.state.setdefault("resolved", [])
        resolved_now: list[dict[str, Any]] = []

        for key, row in list(active.items()):
            if str(row.get("symbol") or "").upper() != symbol:
                continue
            resolution = self._resolve(row, world)
            if resolution is not None:
                resolved_now.append(resolution)
                resolved_rows.append(resolution)
                active.pop(key, None)
                self.resolved += 1
        self.state["resolved"] = resolved_rows[-self.RESOLVED_LIMIT :]

        generated_now: list[dict[str, Any]] = []
        regime = str(world.get("regime") or "unknown")
        price = _finite(world.get("price"))
        senses = world.get("senses") or {}
        features = world.get("features") or {}
        patterns = list(world.get("latent_patterns") or [])
        if _finite(senses.get("rare_scope_score")) >= 0.60 and not patterns:
            patterns.append("rare_scope_conjunction")

        for pattern in patterns:
            key = self._fingerprint(symbol, str(pattern), regime)
            if key in active:
                continue
            # Avoid immediately recreating the same hypothesis after resolution.
            recent_same = any(
                row.get("hypothesis_id") == key
                for row in self.state.get("resolved", [])[-100:]
            )
            if recent_same:
                continue
            confidence = _clip(
                0.20
                + 0.35 * _finite(senses.get("rare_scope_score"))
                + 0.20 * _finite(council.get("confidence"))
                - 0.25 * _finite(critic.get("confidence_haircut"))
            )
            record = {
                "hypothesis_id": key,
                "symbol": symbol,
                "pattern": str(pattern),
                "statement": self._statement(str(pattern)),
                "status": "observing",
                "created_at": time.time(),
                "horizon_observations": self.horizon_observations,
                "observations": 0,
                "entry_price": price,
                "entry_regime": regime,
                "confidence": confidence,
                "trigger": {
                    "rare_scope_score": _finite(senses.get("rare_scope_score")),
                    "novelty": _finite(senses.get("novelty")),
                    "volatility_shock": _finite(senses.get("volatility_shock")),
                    "volume_shock": _finite(senses.get("volume_shock")),
                    "liquidity_stress": _finite(senses.get("liquidity_stress")),
                    "model_disagreement": _finite(senses.get("model_disagreement")),
                    "timeframe_fracture": _finite(senses.get("timeframe_fracture")),
                    "liquidity_imbalance": _finite(features.get("liquidity_imbalance")),
                    "context_score": _finite(features.get("context_score")),
                    "vol_short": _finite(features.get("vol_short")),
                },
                "falsification_questions": list(critic.get("falsification_questions") or []),
                "execution_authority": False,
                "research_only": True,
            }
            active[key] = record
            generated_now.append(record)
            self.generated += 1

        if len(active) > self.ACTIVE_LIMIT:
            ordered = sorted(active.items(), key=lambda item: float(item[1].get("created_at") or 0.0))
            for key, _ in ordered[: len(active) - self.ACTIVE_LIMIT]:
                active.pop(key, None)

        self.observations += 1
        self.state["observations"] = self.observations
        self.state["generated"] = self.generated
        self.state["resolved_count"] = self.resolved
        if self.observations % self.SAVE_INTERVAL == 0 or generated_now or resolved_now:
            self._save()
        return {
            "symbol": symbol,
            "generated": generated_now,
            "resolved": resolved_now,
            "active_for_symbol": [
                row for row in active.values() if str(row.get("symbol") or "").upper() == symbol
            ],
            "execution_authority": False,
            "research_only": True,
        }

    def agenda(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = list((self.state.get("active") or {}).values())
        rows.sort(key=lambda row: (float(row.get("confidence") or 0.0), float(row.get("created_at") or 0.0)), reverse=True)
        return [dict(row) for row in rows[: max(0, int(limit))]]

    def health(self) -> dict[str, Any]:
        resolved = self.state.get("resolved") or []
        supported = sum(1 for row in resolved if row.get("supported") is True)
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "observations": self.observations,
            "generated": self.generated,
            "active": len(self.state.get("active") or {}),
            "resolved": self.resolved,
            "supported": supported,
            "support_rate": supported / len(resolved) if resolved else 0.0,
            "falsifiable": True,
            "novelty_is_not_edge": True,
            "research_only": True,
            "execution_authority": False,
            "state_path": str(self.state_path),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "observations": 0,
            "generated": 0,
            "resolved_count": 0,
            "active": {},
            "resolved": [],
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
        self.state["observations"] = self.observations
        self.state["generated"] = self.generated
        self.state["resolved_count"] = self.resolved
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
