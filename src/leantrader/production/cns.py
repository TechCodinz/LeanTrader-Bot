from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class CentralNervousSystem:
    """Fuse measured LeanTrader sensor streams into a bounded state packet.

    The CNS has no order authority. It exposes coherence, memory support,
    liquidity/risk pressure and safety blocks to the Trading Brain.
    """

    VERSION = "2.0"
    SAVE_INTERVAL = 10

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.latest: dict[str, dict[str, Any]] = {}
        self.integrations = 0
        self.last_error: str | None = None
        self._load()

    def start(self) -> None:
        self._load()

    def stop(self) -> None:
        self._save()

    def integrate(
        self,
        *,
        symbol: str,
        adaptive: dict[str, Any] | None,
        advanced: dict[str, Any] | None,
        routed: dict[str, Any] | None,
        memory_summary: dict[str, Any] | None,
        exchange_protection: dict[str, Any] | None = None,
        runtime_errors: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        adaptive = adaptive or {}
        advanced = advanced or {}
        routed = routed or {}
        memory_summary = memory_summary or {}
        exchange_protection = exchange_protection or {}
        runtime_errors = runtime_errors or {}

        adaptive_score = float(adaptive.get("score") or 0.0)
        adaptive_conf = float(adaptive.get("confidence") or 0.0)
        swarm = advanced.get("swarm") or {}
        ultra_score = float(swarm.get("score") or 0.0)
        ultra_conf = float(swarm.get("confidence") or 0.0)
        liquidity = advanced.get("liquidity") or {}
        spread_bps = float(liquidity.get("spread_bps") or 0.0)
        imbalance = float(liquidity.get("imbalance") or 0.0)

        agreement = 1.0 - min(1.0, abs(adaptive_score - ultra_score) / 2.0)
        confidence = _clip((adaptive_conf + ultra_conf) / 2.0, 0.0, 1.0)
        coherence = _clip(agreement * confidence, 0.0, 1.0)

        memory_support = _clip(float(memory_summary.get("support") or 0.0), 0.0, 1.0)
        memory_return = float(memory_summary.get("weighted_net_return") or 0.0)
        memory_bias = _clip(math.tanh(memory_return * 50.0))

        spread_pressure = _clip(math.tanh(max(0.0, spread_bps) / 30.0), 0.0, 1.0)
        protection_blocked = bool(exchange_protection) and exchange_protection.get("allowed") is False
        error_pressure = min(1.0, len(runtime_errors) / 3.0)
        risk_pressure = _clip(
            0.45 * spread_pressure
            + 0.35 * error_pressure
            + 0.20 * (1.0 if protection_blocked else 0.0),
            0.0,
            1.0,
        )

        route_allowed = bool(routed.get("allowed", False))
        route_score = float(routed.get("score") or routed.get("combined_score") or 0.0)
        action_bias = _clip(
            0.45 * adaptive_score
            + 0.35 * ultra_score
            + 0.10 * memory_bias
            + 0.10 * route_score
        )
        blocks: list[str] = []
        if risk_pressure >= 0.70:
            blocks.append("cns_high_risk_pressure")
        if coherence < 0.10 and max(adaptive_conf, ultra_conf) >= 0.50:
            blocks.append("cns_signal_disagreement")
        if not route_allowed:
            blocks.append("bounded_router_not_approved")
        if protection_blocked:
            blocks.append("exchange_protection_block")

        packet = {
            "symbol": symbol.upper(),
            "signal_coherence": coherence,
            "risk_pressure": risk_pressure,
            "memory_support": memory_support,
            "memory_bias": memory_bias,
            "action_bias": action_bias,
            "adaptive_score": adaptive_score,
            "adaptive_confidence": adaptive_conf,
            "ultra_score": ultra_score,
            "ultra_confidence": ultra_conf,
            "spread_bps": spread_bps,
            "liquidity_imbalance": imbalance,
            "route_allowed": route_allowed,
            "safety_blocks": blocks,
            "execution_authority": False,
            "integrated_at": time.time(),
        }
        self.latest[symbol.upper()] = packet
        self.integrations += 1
        if self.integrations % self.SAVE_INTERVAL == 0:
            self._save()
        return dict(packet)

    def health(self) -> dict[str, Any]:
        rows = list(self.latest.values())
        average_coherence = (
            sum(float(row.get("signal_coherence") or 0.0) for row in rows) / len(rows)
            if rows
            else 0.0
        )
        average_risk_pressure = (
            sum(float(row.get("risk_pressure") or 0.0) for row in rows) / len(rows)
            if rows
            else 0.0
        )
        return {
            "healthy": self.last_error is None,
            "integrations": self.integrations,
            "symbols": len(self.latest),
            "average_signal_coherence": average_coherence,
            "average_risk_pressure": average_risk_pressure,
            "execution_authority": False,
            "can_only_inform_or_restrict": True,
            "error": self.last_error,
        }

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.latest = dict(payload.get("latest") or {})
            self.integrations = int(payload.get("integrations", 0))
            self.last_error = None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.latest = {}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = {
            "version": self.VERSION,
            "latest": self.latest,
            "integrations": self.integrations,
            "updated_at": time.time(),
        }
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
