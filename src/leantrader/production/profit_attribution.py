from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _slug(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text[:120] if text else fallback


class NetProfitAttribution:
    """Costed closed-trade attribution with no trading or promotion authority."""

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    MINIMUM_COST_FLOOR_BPS = 30.0
    OBSERVATION_LIMIT = 10_000
    SEEN_LIMIT = 25_000
    CONFIDENCE_LEVEL = 0.95

    def __init__(
        self,
        state_path: Path,
        *,
        minimum_samples: int = 100,
        minimum_regimes: int = 2,
        modeled_round_trip_cost_bps: float = 30.0,
    ) -> None:
        if int(minimum_samples) < 100:
            raise ValueError("profit attribution cannot lower the 100-sample evidence floor")
        if int(minimum_regimes) < 2:
            raise ValueError("profit attribution requires at least two regimes")
        if float(modeled_round_trip_cost_bps) < self.MINIMUM_COST_FLOOR_BPS:
            raise ValueError("profit attribution cannot lower the 30-bps cost floor")
        self.state_path = state_path
        self.minimum_samples = int(minimum_samples)
        self.minimum_regimes = int(minimum_regimes)
        self.modeled_round_trip_cost_bps = float(modeled_round_trip_cost_bps)
        self.last_error: str | None = None
        self.state = self._load()

    def start(self) -> None:
        self.state = self._load()

    def stop(self) -> None:
        self._save()

    @staticmethod
    def _authority_denied() -> dict[str, bool]:
        return {
            "research_only": True,
            "advisory_only": True,
            "observational_not_causal": True,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_modify_routes": False,
            "can_modify_orders": False,
            "can_modify_sizing": False,
            "can_increase_risk": False,
            "execution_authority": False,
        }

    @staticmethod
    def _confidence_band(confidence: float) -> str:
        bounded = max(0.0, min(1.0, confidence))
        lower = min(0.9, math.floor(bounded * 10.0) / 10.0)
        upper = min(1.0, lower + 0.1)
        return f"{lower:.1f}-{upper:.1f}"

    @staticmethod
    def _component_band(score: float) -> str:
        if score >= 0.15:
            return "positive"
        if score <= -0.15:
            return "negative"
        return "neutral"

    def _observation(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if str(event.get("side") or "").lower() != "sell":
            return None
        if max(0.0, _finite(event.get("remaining_quantity"))) > 1e-12:
            return None
        event_id = str(event.get("event_id") or "").strip()
        symbol = str(event.get("symbol") or "").strip().upper()
        if not event_id or not symbol:
            return None
        metadata = event.get("position_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        route = metadata.get("decision_route")
        route = route if isinstance(route, dict) else {}
        component_scores = metadata.get("component_scores")
        component_scores = (
            component_scores if isinstance(component_scores, dict) else {}
        )
        components = {
            str(name)[:80]: _finite(score)
            for name, score in component_scores.items()
            if str(name).strip() and math.isfinite(_finite(score, math.nan))
        }
        net_return = _finite(
            event.get("trade_realized_return_total"),
            _finite(event.get("realized_return")),
        )
        realized_pnl = _finite(
            event.get("trade_realized_pnl_total"),
            _finite(event.get("realized_pnl")),
        )
        confidence = max(0.0, min(1.0, _finite(metadata.get("confidence"))))
        return {
            "event_id": event_id,
            "observed_at": time.time(),
            "event_timestamp": str(event.get("timestamp") or ""),
            "symbol": symbol,
            "regime": _slug(metadata.get("regime")),
            "exit_reason": _slug(event.get("reason")),
            "route_reason": _slug(route.get("reason")),
            "confidence": confidence,
            "confidence_band": self._confidence_band(confidence),
            "net_return": net_return,
            "realized_pnl": realized_pnl,
            "win": net_return > 0.0,
            "component_scores": components,
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
        }

    def observe(self, *, events: list[dict[str, Any]]) -> dict[str, Any]:
        observations = self.state.setdefault("observations", [])
        seen = self.state.setdefault("seen_event_ids", {})
        recorded: list[str] = []
        partial_exits_ignored = 0
        invalid_events_ignored = 0
        duplicates_ignored = 0

        for event in events:
            if not isinstance(event, dict):
                invalid_events_ignored += 1
                continue
            event_id = str(event.get("event_id") or "").strip()
            if event_id and event_id in seen:
                duplicates_ignored += 1
                continue
            if (
                str(event.get("side") or "").lower() == "sell"
                and max(0.0, _finite(event.get("remaining_quantity"))) > 1e-12
            ):
                partial_exits_ignored += 1
                continue
            row = self._observation(event)
            if row is None:
                invalid_events_ignored += 1
                continue
            observations.append(row)
            seen[row["event_id"]] = time.time()
            recorded.append(row["event_id"])

        self.state["observations"] = observations[-self.OBSERVATION_LIMIT :]
        if len(seen) > self.SEEN_LIMIT:
            ordered = sorted(seen.items(), key=lambda item: float(item[1]))
            for key, _ in ordered[: len(seen) - self.SEEN_LIMIT]:
                seen.pop(key, None)
        self.state["cycles"] = int(self.state.get("cycles") or 0) + 1
        self.state["last"] = {
            "observed_at": time.time(),
            "recorded_event_ids": recorded,
            "partial_exits_ignored": partial_exits_ignored,
            "invalid_events_ignored": invalid_events_ignored,
            "duplicates_ignored": duplicates_ignored,
        }
        self._save()
        return self.snapshot()

    @staticmethod
    def _cohort_rows(
        observations: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        cohorts: dict[str, list[dict[str, Any]]] = {"overall:all": []}
        for row in observations:
            cohorts["overall:all"].append(row)
            keys = (
                f"symbol:{_slug(row.get('symbol'))}",
                f"regime:{_slug(row.get('regime'))}",
                f"exit_reason:{_slug(row.get('exit_reason'))}",
                f"route_reason:{_slug(row.get('route_reason'))}",
                f"confidence:{_slug(row.get('confidence_band'))}",
            )
            for key in keys:
                cohorts.setdefault(key, []).append(row)
            components = row.get("component_scores")
            if isinstance(components, dict):
                for name, score in components.items():
                    key = (
                        f"component:{_slug(name)}:"
                        f"{NetProfitAttribution._component_band(_finite(score))}"
                    )
                    cohorts.setdefault(key, []).append(row)
        return cohorts

    def _statistics(
        self,
        rows: list[dict[str, Any]],
        *,
        family_size: int,
    ) -> dict[str, Any]:
        returns = [_finite(row.get("net_return")) for row in rows]
        pnls = [_finite(row.get("realized_pnl")) for row in rows]
        samples = len(returns)
        regimes = sorted({_slug(row.get("regime")) for row in rows})
        wins = sum(1 for value in returns if value > 0.0)
        mean = sum(returns) / samples if samples else 0.0
        standard_error = (
            statistics.stdev(returns) / math.sqrt(samples)
            if samples >= 2
            else 0.0
        )
        adjusted_alpha = 0.05 / max(1, family_size)
        z_score = NormalDist().inv_cdf(1.0 - adjusted_alpha / 2.0)
        margin = z_score * standard_error
        lower = mean - margin
        upper = mean + margin
        gross_profit = sum(value for value in pnls if value > 0.0)
        gross_loss = abs(sum(value for value in pnls if value < 0.0))
        mature = samples >= self.minimum_samples and len(regimes) >= self.minimum_regimes
        positive_edge = mature and lower > 0.0
        negative_edge = mature and upper < 0.0
        return {
            "samples": samples,
            "regime_count": len(regimes),
            "regimes": regimes,
            "wins": wins,
            "win_rate": wins / samples if samples else 0.0,
            "average_net_return": mean,
            "median_net_return": statistics.median(returns) if returns else None,
            "cumulative_net_return": sum(returns),
            "realized_pnl": sum(pnls),
            "profit_factor": (
                gross_profit / gross_loss if gross_loss > 0.0 else None
            ),
            "standard_error": standard_error,
            "confidence_level": self.CONFIDENCE_LEVEL,
            "bonferroni_family_size": max(1, family_size),
            "bonferroni_adjusted_alpha": adjusted_alpha,
            "mean_return_lower_bound": lower,
            "mean_return_upper_bound": upper,
            "minimum_samples": self.minimum_samples,
            "minimum_regimes": self.minimum_regimes,
            "evidence_mature": mature,
            "positive_edge_after_costs": positive_edge,
            "negative_edge_after_costs": negative_edge,
            "insufficient_evidence": not mature,
            **self._authority_denied(),
        }

    def snapshot(self) -> dict[str, Any]:
        observations = [
            row
            for row in list(self.state.get("observations") or [])
            if isinstance(row, dict)
        ]
        cohorts = self._cohort_rows(observations)
        family_size = len(cohorts)
        attribution = {
            key: self._statistics(rows, family_size=family_size)
            for key, rows in sorted(cohorts.items())
        }
        positive = [
            key
            for key, row in attribution.items()
            if row["positive_edge_after_costs"]
        ]
        negative = [
            key
            for key, row in attribution.items()
            if row["negative_edge_after_costs"]
        ]
        overall = attribution.get(
            "overall:all",
            self._statistics([], family_size=family_size),
        )
        return {
            "cycles": int(self.state.get("cycles") or 0),
            "closed_trade_samples": len(observations),
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
            "sample_unit": "fully_closed_costed_paper_trade",
            "partial_exits_are_not_independent_samples": True,
            "multiple_testing_correction": "bonferroni",
            "overall": overall,
            "cohorts": attribution,
            "positive_edge_candidates": positive,
            "negative_edge_candidates": negative,
            "profitability_claim_allowed": bool(
                overall["positive_edge_after_costs"]
            ),
            "last": dict(self.state.get("last") or {}),
            **self._authority_denied(),
        }

    def health(self) -> dict[str, Any]:
        samples = len(self.state.get("observations") or [])
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "state_path": str(self.state_path),
            "cycles": int(self.state.get("cycles") or 0),
            "closed_trade_samples": samples,
            "minimum_samples": self.minimum_samples,
            "minimum_regimes": self.minimum_regimes,
            "modeled_round_trip_cost_bps": self.modeled_round_trip_cost_bps,
            **self._authority_denied(),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "cycles": 0,
            "observations": [],
            "seen_event_ids": {},
            "last": {},
        }
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) == self.SCHEMA_VERSION:
                payload.setdefault("cycles", 0)
                payload.setdefault("observations", [])
                payload.setdefault("seen_event_ids", {})
                payload.setdefault("last", {})
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
