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


class TailRiskSentinel:
    """Modernized Black-Swan/Tail-Risk sentinel.

    It watches for compound discontinuities rather than attempting to predict a
    mythical black swan from one indicator.  The sentinel is shadow-only in
    v12.6.  It can recommend less risk, never more risk, and cannot halt or trade
    without a separately validated governance integration.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    SAVE_INTERVAL = 5
    EVENT_LIMIT = 1_000

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.last_error: str | None = None
        self.state = self._load()
        self.assessments = int(self.state.get("assessments") or 0)

    def start(self) -> None:
        self.state = self._load()
        self.assessments = int(self.state.get("assessments") or 0)

    def stop(self) -> None:
        self._save()

    def assess(
        self,
        *,
        symbol: str,
        world: dict[str, Any],
        market_world: dict[str, Any] | None = None,
        advanced: dict[str, Any] | None = None,
        runtime_errors: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        senses = world.get("senses") or {}
        market_world = market_world or {}
        advanced = advanced or {}
        runtime_errors = runtime_errors or {}

        price_shock = _clip(_finite(senses.get("price_shock")))
        volatility_shock = _clip(_finite(senses.get("volatility_shock")))
        liquidity_stress = _clip(_finite(senses.get("liquidity_stress")))
        novelty = _clip(_finite(senses.get("novelty")))
        disagreement = _clip(_finite(senses.get("model_disagreement")))
        correlation_fracture = _clip(_finite(market_world.get("correlation_fracture")) / 0.40)
        dispersion = _clip(_finite(market_world.get("cross_sectional_dispersion")) / 0.05)
        error_pressure = _clip(len(runtime_errors) / 3.0)
        news_blackout = bool(advanced.get("news_blackout"))

        components = {
            "price_discontinuity": price_shock,
            "volatility_discontinuity": volatility_shock,
            "liquidity_dislocation": liquidity_stress,
            "novelty": novelty,
            "model_disagreement": disagreement,
            "correlation_fracture": correlation_fracture,
            "cross_sectional_dispersion": dispersion,
            "runtime_error_pressure": error_pressure,
            "news_blackout": 1.0 if news_blackout else 0.0,
        }
        severity = _clip(
            0.17 * price_shock
            + 0.17 * volatility_shock
            + 0.17 * liquidity_stress
            + 0.12 * novelty
            + 0.10 * disagreement
            + 0.10 * correlation_fracture
            + 0.07 * dispersion
            + 0.06 * error_pressure
            + 0.04 * (1.0 if news_blackout else 0.0)
        )
        compound_extreme_count = sum(1 for value in components.values() if value >= 0.70)
        if compound_extreme_count >= 3:
            severity = max(severity, 0.85)
        elif compound_extreme_count >= 2:
            severity = max(severity, 0.70)

        if severity >= 0.85:
            state = "extreme"
            recommendation = "governance_review_and_defensive_halt_candidate"
            risk_guidance = 0.0
        elif severity >= 0.70:
            state = "severe"
            recommendation = "block_new_risk_candidate_and_reduce_exposure"
            risk_guidance = 0.20
        elif severity >= 0.45:
            state = "elevated"
            recommendation = "reduce_risk_and_increase_research"
            risk_guidance = 0.55
        else:
            state = "normal"
            recommendation = "normal_governance"
            risk_guidance = 1.0

        result = {
            "symbol": symbol.upper(),
            "state": state,
            "severity": severity,
            "components": components,
            "compound_extreme_count": compound_extreme_count,
            "recommendation": recommendation,
            "risk_guidance_multiplier": risk_guidance,
            "shadow_only": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "can_halt_execution": False,
            "assessed_at": time.time(),
        }
        self.state.setdefault("latest", {})[symbol.upper()] = result
        if state != "normal":
            events = self.state.setdefault("events", [])
            events.append(result)
            self.state["events"] = events[-self.EVENT_LIMIT :]
        self.assessments += 1
        self.state["assessments"] = self.assessments
        if self.assessments % self.SAVE_INTERVAL == 0 or state in {"severe", "extreme"}:
            self._save()
        return dict(result)

    def health(self) -> dict[str, Any]:
        latest = self.state.get("latest") or {}
        counts = {"normal": 0, "elevated": 0, "severe": 0, "extreme": 0}
        for row in latest.values():
            state = str(row.get("state") or "normal")
            counts[state] = counts.get(state, 0) + 1
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "assessments": self.assessments,
            "tracked_symbols": len(latest),
            "states": counts,
            "events": len(self.state.get("events") or []),
            "legacy_black_swan_code_loaded": False,
            "compound_tail_risk_detection": True,
            "shadow_only": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "can_halt_execution": False,
            "state_path": str(self.state_path),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {"schema_version": self.SCHEMA_VERSION, "assessments": 0, "latest": {}, "events": []}
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
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
