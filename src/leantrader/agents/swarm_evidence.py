from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable


EVIDENCE_AUTHORITY = "costed_shadow_episode_v2"
SWARM_RESEARCH_STRATEGIES = (
    "swarm_scalp_1m",
    "swarm_momentum_5m",
    "swarm_momentum_15m",
    "swarm_trend_1h",
    "swarm_trend_4h",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def strategy_name(role: str, timeframe: str) -> str:
    role = str(role).strip().lower()
    timeframe = str(timeframe).strip().lower()
    return f"swarm_{role}_{timeframe}".replace("-", "_")


def build_v142_swarm_manifests(
    *,
    minimum_samples: int,
    round_trip_cost_bps: float,
    strategies: Iterable[str] = SWARM_RESEARCH_STRATEGIES,
) -> list[dict[str, Any]]:
    minimum_samples = max(100, int(minimum_samples))
    round_trip_cost_bps = max(30.0, float(round_trip_cost_bps))
    partition_plan = {
        "protocol": "v1.42_partitioned_evidence_v1",
        "initial_training_samples": 60,
        "walk_forward_folds": 3,
        "validation_samples_per_fold": 20,
        "embargo_samples_per_fold": 1,
        "prospective_paper_samples": minimum_samples,
        "untouched_holdout_samples": minimum_samples,
        "assignment": "candidate_episode_ordinal_frozen_at_registration",
        "post_holdout_assignment": "untouched_holdout",
    }
    manifests: list[dict[str, Any]] = []
    for strategy in sorted(set(str(value) for value in strategies if value)):
        digest = hashlib.sha256(
            f"v1.43|{strategy}|{minimum_samples}|{round_trip_cost_bps:.8f}|swarm-shadow".encode("utf-8")
        ).hexdigest()[:20]
        manifests.append(
            {
                "candidate_id": f"v143-swarm-{digest}",
                "candidate_kind": "shadow_research_manifest",
                "base_strategy": strategy,
                "status": "awaiting_independent_validation",
                "selection_evidence": {
                    "authority": EVIDENCE_AUTHORITY,
                    "samples": 0,
                    "average_net_return": 0.0,
                    "ewma_net_return": 0.0,
                    "win_rate": 0.0,
                    "conservative_score": 0.0,
                    "evidence_protocol_version": "1.42",
                    "selection_frozen_before_partitioned_outcomes": True,
                },
                "research_protocol": {
                    "prospective_only": True,
                    "minimum_additional_shadow_samples": minimum_samples,
                    "round_trip_cost_bps": round_trip_cost_bps,
                    "walk_forward_required": True,
                    "out_of_sample_required": True,
                    "multiple_testing_correction_required": True,
                    "paper_review_required": True,
                    "automatic_promotion": False,
                    "evidence_protocol_version": "1.42",
                    "purged_walk_forward_required": True,
                    "embargo_required": True,
                    "pbo_required": True,
                    "deflated_performance_required": True,
                    "drift_detection_required": True,
                    "untouched_holdout_required": True,
                    "freeze_before_outcome_required": True,
                    "partition_plan": dict(partition_plan),
                },
                "hypothesis_context": [
                    "Measure whether cost-clearing multi-timeframe swarm tranches retain positive net expectancy prospectively."
                ],
                "executable_code": None,
                "parameter_mutation_authority": False,
                "paper_promotion_authority": False,
                "testnet_authority": False,
                "live_authority": False,
                "can_increase_risk": False,
                "execution_authority": False,
            }
        )
    return manifests


class SwarmOutcomeJournal:
    """Thread-safe persistent closed-outcome queue for swarm shadow evidence."""

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    MAX_RETAINED = 20_000

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self._lock = threading.RLock()
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        empty = {"schema_version": self.SCHEMA_VERSION, "pending": [], "acknowledged": 0}
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) == self.SCHEMA_VERSION:
                payload.setdefault("pending", [])
                payload.setdefault("acknowledged", 0)
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return empty

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def append(self, episode: dict[str, Any]) -> str:
        row = dict(episode)
        opened_at = _finite(row.get("opened_at"), math.nan)
        closed_at = _finite(row.get("closed_at"), math.nan)
        net_return = _finite(row.get("net_return"), math.nan)
        strategy = str(row.get("strategy") or "").strip()
        if not strategy:
            raise ValueError("swarm evidence requires strategy")
        if not math.isfinite(opened_at) or not math.isfinite(closed_at) or closed_at <= opened_at:
            raise ValueError("swarm evidence requires a complete time interval")
        if not math.isfinite(net_return):
            raise ValueError("swarm evidence requires finite net_return")
        row["evidence_authority"] = EVIDENCE_AUTHORITY
        row["evidence_interval_complete"] = True
        row["automatic_promotion"] = False
        row["paper_promotion_authority"] = False
        row["testnet_authority"] = False
        row["live_authority"] = False
        row["execution_authority"] = False
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        episode_id = str(row.get("episode_id") or f"swarm-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}")
        row["episode_id"] = episode_id
        with self._lock:
            existing = {str(item.get("episode_id")) for item in self.state.get("pending") or [] if isinstance(item, dict)}
            if episode_id not in existing:
                rows = list(self.state.get("pending") or [])
                rows.append(row)
                self.state["pending"] = rows[-self.MAX_RETAINED :]
                self._save()
        return episode_id

    def pending(self, limit: int = 500) -> list[dict[str, Any]]:
        limit = max(1, min(5_000, int(limit)))
        with self._lock:
            return [dict(row) for row in list(self.state.get("pending") or [])[:limit] if isinstance(row, dict)]

    def acknowledge(self, episode_ids: Iterable[str]) -> int:
        ids = {str(value) for value in episode_ids if value}
        if not ids:
            return 0
        with self._lock:
            before = list(self.state.get("pending") or [])
            after = [row for row in before if str((row or {}).get("episode_id")) not in ids]
            removed = len(before) - len(after)
            self.state["pending"] = after
            self.state["acknowledged"] = int(self.state.get("acknowledged") or 0) + removed
            self._save()
            return removed

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self.VERSION,
                "pending_closed_outcomes": len(self.state.get("pending") or []),
                "acknowledged_closed_outcomes": int(self.state.get("acknowledged") or 0),
                "evidence_authority": EVIDENCE_AUTHORITY,
                "closed_outcomes_only": True,
                "automatic_promotion": False,
                "execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            }
