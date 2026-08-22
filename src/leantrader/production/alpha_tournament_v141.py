from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class StrategyFoundry:
    """Build deterministic research manifests; never code, orders, or authority."""

    VERSION = "1.0"
    MAX_MANIFESTS = 10

    def __init__(
        self,
        *,
        minimum_samples: int,
        round_trip_cost_bps: float,
    ) -> None:
        self.minimum_samples = int(minimum_samples)
        self.round_trip_cost_bps = float(round_trip_cost_bps)

    @staticmethod
    def _bounded_hypotheses(rows: list[dict[str, Any]]) -> list[str]:
        questions: list[str] = []
        for row in rows[:10]:
            value = row.get("question") or row.get("statement") or row.get("pattern")
            text = str(value or "").strip()
            if text and text not in questions:
                questions.append(text[:300])
            if len(questions) >= 3:
                break
        return questions

    def forge(
        self,
        ranking: list[dict[str, Any]],
        hypothesis_agenda: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        context = self._bounded_hypotheses(hypothesis_agenda)
        manifests: list[dict[str, Any]] = []
        for row in ranking[: self.MAX_MANIFESTS]:
            strategy = str(row["strategy"])
            digest = hashlib.sha256(
                (
                    f"{strategy}|{self.minimum_samples}|"
                    f"{self.round_trip_cost_bps:.8f}|prospective"
                ).encode("utf-8")
            ).hexdigest()[:20]
            manifests.append(
                {
                    "candidate_id": f"foundry-{digest}",
                    "candidate_kind": "shadow_research_manifest",
                    "base_strategy": strategy,
                    "status": "awaiting_independent_validation",
                    "selection_evidence": {
                        "authority": "costed_shadow_episode_v2",
                        "samples": int(row["samples"]),
                        "average_net_return": float(row["average_net_return"]),
                        "ewma_net_return": float(row["ewma_net_return"]),
                        "win_rate": float(row["win_rate"]),
                        "conservative_score": float(row["conservative_score"]),
                    },
                    "research_protocol": {
                        "prospective_only": True,
                        "minimum_additional_shadow_samples": self.minimum_samples,
                        "round_trip_cost_bps": self.round_trip_cost_bps,
                        "walk_forward_required": True,
                        "out_of_sample_required": True,
                        "multiple_testing_correction_required": True,
                        "paper_review_required": True,
                        "automatic_promotion": False,
                    },
                    "hypothesis_context": list(context),
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


class AlphaTournament:
    """Rank costed shadow evidence without creating execution authority.

    This component is intentionally downstream of StrategyObservatory. It cannot
    alter signals, routes, sizing, credentials, trading mode, or risk limits.
    Eligibility means only that a strategy may enter another prospective research
    round; it never means automatic paper, testnet, or live promotion.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    EVIDENCE_AUTHORITY = "costed_shadow_episode_v2"
    MINIMUM_SAMPLE_FLOOR = 100
    MINIMUM_COST_FLOOR_BPS = 30.0
    HISTORY_LIMIT = 200

    def __init__(
        self,
        state_path: Path,
        *,
        minimum_samples: int = 100,
        expected_round_trip_cost_bps: float = 30.0,
        minimum_win_rate: float = 0.50,
    ) -> None:
        if int(minimum_samples) < self.MINIMUM_SAMPLE_FLOOR:
            raise ValueError("alpha tournament cannot lower the 100-sample research floor")
        if float(expected_round_trip_cost_bps) < self.MINIMUM_COST_FLOOR_BPS:
            raise ValueError("alpha tournament cannot lower the 30-bps cost floor")
        if not 0.50 <= float(minimum_win_rate) <= 1.0:
            raise ValueError("alpha tournament win-rate floor must be between 0.50 and 1.0")
        self.state_path = state_path
        self.minimum_samples = int(minimum_samples)
        self.expected_round_trip_cost_bps = float(expected_round_trip_cost_bps)
        self.minimum_win_rate = float(minimum_win_rate)
        self.last_error: str | None = None
        self.state = self._load()
        self.foundry = StrategyFoundry(
            minimum_samples=self.minimum_samples,
            round_trip_cost_bps=self.expected_round_trip_cost_bps,
        )

    def start(self) -> None:
        self.state = self._load()

    def stop(self) -> None:
        self._save()

    @staticmethod
    def _strategy_row(name: str, record: dict[str, Any]) -> dict[str, Any]:
        samples = max(0, int(record.get("samples") or 0))
        wins = max(0, min(samples, int(record.get("wins") or 0)))
        win_rate = _finite(record.get("win_rate"), wins / samples if samples else 0.0)
        average = _finite(
            record.get("average_net_return"),
            _finite(record.get("average_return")),
        )
        ewma = _finite(record.get("ewma_net_return"))
        return {
            "strategy": name,
            "samples": samples,
            "wins": wins,
            "win_rate": max(0.0, min(1.0, win_rate)),
            "average_net_return": average,
            "ewma_net_return": ewma,
            "negative_streak": max(0, int(record.get("negative_streak") or 0)),
        }

    def evaluate(
        self,
        *,
        strategy_health: dict[str, Any],
        hypothesis_agenda: list[dict[str, Any]] | None = None,
        evolution_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        authority = str(strategy_health.get("evidence_authority") or "")
        observed_cost = _finite(strategy_health.get("round_trip_cost_bps"), -1.0)
        evidence_contract_valid = (
            authority == self.EVIDENCE_AUTHORITY
            and observed_cost >= self.expected_round_trip_cost_bps
        )

        eligible: list[dict[str, Any]] = []
        awaiting: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        rows = strategy_health.get("strategies")
        if not isinstance(rows, dict):
            rows = {}

        if evidence_contract_valid:
            for name in sorted(rows):
                record = rows.get(name)
                if not isinstance(record, dict):
                    continue
                row = self._strategy_row(str(name), record)
                if row["samples"] < self.minimum_samples:
                    row["status"] = "awaiting_samples"
                    row["samples_remaining"] = self.minimum_samples - row["samples"]
                    awaiting.append(row)
                    continue
                if (
                    row["average_net_return"] <= 0.0
                    or row["ewma_net_return"] <= 0.0
                    or row["win_rate"] < self.minimum_win_rate
                ):
                    row["status"] = "rejected_after_costs"
                    row["reasons"] = [
                        reason
                        for condition, reason in (
                            (row["average_net_return"] <= 0.0, "non_positive_average_net_return"),
                            (row["ewma_net_return"] <= 0.0, "non_positive_ewma_net_return"),
                            (row["win_rate"] < self.minimum_win_rate, "win_rate_below_floor"),
                        )
                        if condition
                    ]
                    rejected.append(row)
                    continue
                row["status"] = "eligible_for_more_shadow_research"
                row["conservative_score"] = (
                    min(row["average_net_return"], row["ewma_net_return"])
                    * math.sqrt(row["samples"])
                    * row["win_rate"]
                )
                eligible.append(row)

        eligible.sort(
            key=lambda row: (
                float(row["conservative_score"]),
                int(row["samples"]),
                str(row["strategy"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(eligible, start=1):
            row["rank"] = rank
            row["research_only"] = True
            row["execution_authority"] = False

        manifests = self.foundry.forge(eligible, list(hypothesis_agenda or []))
        external_metrics = (evolution_snapshot or {}).get("shadow_metrics")
        if not isinstance(external_metrics, dict):
            external_metrics = {}
        external_validated = sum(
            1
            for row in external_metrics.values()
            if isinstance(row, dict) and row.get("research_validated") is True
        )

        result = {
            "evaluated_at": now,
            "status": (
                "ranked_shadow_research"
                if evidence_contract_valid
                else "blocked_evidence_contract"
            ),
            "evidence_contract_valid": evidence_contract_valid,
            "required_evidence_authority": self.EVIDENCE_AUTHORITY,
            "observed_evidence_authority": authority,
            "minimum_samples": self.minimum_samples,
            "required_round_trip_cost_bps": self.expected_round_trip_cost_bps,
            "observed_round_trip_cost_bps": observed_cost,
            "minimum_win_rate": self.minimum_win_rate,
            "strategies_seen": len(rows),
            "eligible_count": len(eligible),
            "awaiting_count": len(awaiting),
            "rejected_count": len(rejected),
            "ranking": eligible,
            "awaiting_samples": awaiting,
            "rejected_after_costs": rejected,
            "foundry_manifests": manifests,
            "external_shadow_validated_not_promoted": external_validated,
            "selection_is_not_promotion": True,
            "prospective_validation_required": True,
            "research_only": True,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_increase_risk": False,
            "execution_authority": False,
        }
        self.state["evaluations"] = int(self.state.get("evaluations") or 0) + 1
        self.state["latest"] = result
        self.state["manifests"] = {
            row["candidate_id"]: row for row in manifests
        }
        history = self.state.setdefault("history", [])
        history.append(
            {
                "evaluated_at": now,
                "status": result["status"],
                "eligible_count": len(eligible),
                "awaiting_count": len(awaiting),
                "rejected_count": len(rejected),
                "top_strategy": eligible[0]["strategy"] if eligible else None,
            }
        )
        self.state["history"] = history[-self.HISTORY_LIMIT :]
        self._save()
        return result

    def health(self) -> dict[str, Any]:
        latest = self.state.get("latest")
        if not isinstance(latest, dict):
            latest = {}
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "state_path": str(self.state_path),
            "evaluations": int(self.state.get("evaluations") or 0),
            "status": str(latest.get("status") or "waiting_for_evidence"),
            "evidence_contract_valid": bool(latest.get("evidence_contract_valid")),
            "minimum_samples": self.minimum_samples,
            "round_trip_cost_bps": self.expected_round_trip_cost_bps,
            "eligible_count": int(latest.get("eligible_count") or 0),
            "foundry_manifest_count": len(self.state.get("manifests") or {}),
            "research_only": True,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_increase_risk": False,
            "execution_authority": False,
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "evaluations": 0,
            "latest": {},
            "history": [],
            "manifests": {},
        }
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) == self.SCHEMA_VERSION:
                payload.setdefault("evaluations", 0)
                payload.setdefault("latest", {})
                payload.setdefault("history", [])
                payload.setdefault("manifests", {})
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        return empty

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["updated_at"] = time.time()
        temporary.write_text(
            json.dumps(self.state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)
