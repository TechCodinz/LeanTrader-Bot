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


class ProspectiveValidationLab:
    """Prospective, costed research validation with no execution authority.

    The lab freezes Strategy Foundry manifests before accepting future outcomes,
    applies uncertainty and multiple-testing controls to those holdouts, and
    records non-overlapping counterfactual outcomes for downstream entry gates.
    None of its outputs are consumed by the execution path.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    EVIDENCE_AUTHORITY = "costed_shadow_episode_v2"
    COUNTERFACTUAL_AUTHORITY = "non_overlapping_route_counterfactual_v1"
    MINIMUM_SAMPLE_FLOOR = 100
    MINIMUM_COST_FLOOR_BPS = 30.0
    RETURN_LIMIT = 5_000
    RESOLVED_TRIAL_LIMIT = 5_000
    REJECTION_LIMIT = 500

    def __init__(
        self,
        state_path: Path,
        *,
        minimum_samples: int = 100,
        round_trip_cost_bps: float = 30.0,
        counterfactual_horizon_observations: int = 4,
    ) -> None:
        if int(minimum_samples) < self.MINIMUM_SAMPLE_FLOOR:
            raise ValueError("validation lab cannot lower the 100-sample research floor")
        if float(round_trip_cost_bps) < self.MINIMUM_COST_FLOOR_BPS:
            raise ValueError("validation lab cannot lower the 30-bps cost floor")
        if int(counterfactual_horizon_observations) < 2:
            raise ValueError("counterfactual horizon must be at least two observations")
        self.state_path = state_path
        self.minimum_samples = int(minimum_samples)
        self.round_trip_cost_bps = float(round_trip_cost_bps)
        self.counterfactual_horizon_observations = int(
            counterfactual_horizon_observations
        )
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
            "selection_is_not_promotion": True,
            "parameter_mutation_authority": False,
            "paper_promotion_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_increase_risk": False,
            "execution_authority": False,
        }

    @staticmethod
    def _statistics(values: list[float]) -> dict[str, Any]:
        finite_values = [float(value) for value in values if math.isfinite(float(value))]
        samples = len(finite_values)
        if not samples:
            return {
                "samples": 0,
                "wins": 0,
                "win_rate": 0.0,
                "cumulative_net_return": 0.0,
                "average_net_return": 0.0,
                "standard_deviation": None,
                "standard_error": None,
                "lower_95_net_return": None,
                "upper_95_net_return": None,
                "one_sided_p_value": 1.0,
                "profit_factor": None,
                "max_additive_drawdown": 0.0,
            }
        cumulative = sum(finite_values)
        average = cumulative / samples
        wins = sum(1 for value in finite_values if value > 0.0)
        if samples > 1:
            variance = sum((value - average) ** 2 for value in finite_values) / (
                samples - 1
            )
            deviation = math.sqrt(max(0.0, variance))
            standard_error = deviation / math.sqrt(samples)
            lower = average - 1.96 * standard_error
            upper = average + 1.96 * standard_error
            if standard_error > 0.0:
                z_score = average / standard_error
                p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
            else:
                p_value = 0.0 if average > 0.0 else 1.0
        else:
            deviation = None
            standard_error = None
            lower = None
            upper = None
            p_value = 1.0

        gross_profit = sum(value for value in finite_values if value > 0.0)
        gross_loss = abs(sum(value for value in finite_values if value < 0.0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else None

        curve = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for value in finite_values:
            curve += value
            peak = max(peak, curve)
            max_drawdown = max(max_drawdown, peak - curve)

        return {
            "samples": samples,
            "wins": wins,
            "win_rate": wins / samples,
            "cumulative_net_return": cumulative,
            "average_net_return": average,
            "standard_deviation": deviation,
            "standard_error": standard_error,
            "lower_95_net_return": lower,
            "upper_95_net_return": upper,
            "one_sided_p_value": max(0.0, min(1.0, p_value)),
            "profit_factor": profit_factor,
            "max_additive_drawdown": max_drawdown,
        }

    def _manifest_rejection(self, candidate_id: str, reason: str) -> None:
        rows = self.state.setdefault("manifest_rejections", [])
        rows.append(
            {
                "candidate_id": candidate_id[:120],
                "reason": reason[:240],
                "rejected_at": time.time(),
            }
        )
        self.state["manifest_rejections"] = rows[-self.REJECTION_LIMIT :]

    def _validate_manifest(self, manifest: dict[str, Any]) -> tuple[bool, str]:
        candidate_id = str(manifest.get("candidate_id") or "").strip()
        strategy = str(manifest.get("base_strategy") or "").strip()
        selection = manifest.get("selection_evidence")
        protocol = manifest.get("research_protocol")
        if not candidate_id or not strategy:
            return False, "candidate identity is incomplete"
        if manifest.get("candidate_kind") != "shadow_research_manifest":
            return False, "candidate kind is not an allowlisted shadow manifest"
        if not isinstance(selection, dict) or selection.get("authority") != self.EVIDENCE_AUTHORITY:
            return False, "selection evidence authority mismatch"
        if not isinstance(protocol, dict) or protocol.get("prospective_only") is not True:
            return False, "prospective-only protocol is required"
        if int(protocol.get("minimum_additional_shadow_samples") or 0) < self.minimum_samples:
            return False, "manifest attempts to lower the sample floor"
        if _finite(protocol.get("round_trip_cost_bps"), -1.0) < self.round_trip_cost_bps:
            return False, "manifest attempts to lower the cost floor"
        if protocol.get("walk_forward_required") is not True:
            return False, "walk-forward validation is required"
        if protocol.get("out_of_sample_required") is not True:
            return False, "out-of-sample validation is required"
        if protocol.get("multiple_testing_correction_required") is not True:
            return False, "multiple-testing correction is required"
        if protocol.get("automatic_promotion") is not False:
            return False, "automatic promotion must be disabled"
        if manifest.get("executable_code") is not None:
            return False, "executable candidate content is forbidden"
        for field in (
            "parameter_mutation_authority",
            "paper_promotion_authority",
            "testnet_authority",
            "live_authority",
            "can_increase_risk",
            "execution_authority",
        ):
            if manifest.get(field) is not False:
                return False, f"{field} must be explicitly false"
        return True, "accepted"

    def _register_manifests(
        self,
        manifests: list[dict[str, Any]],
        *,
        contract_valid: bool,
    ) -> list[str]:
        registered: list[str] = []
        if not contract_valid:
            return registered
        experiments = self.state.setdefault("experiments", {})
        for manifest in manifests:
            if not isinstance(manifest, dict):
                self._manifest_rejection("", "manifest must be an object")
                continue
            candidate_id = str(manifest.get("candidate_id") or "").strip()
            if candidate_id in experiments:
                continue
            valid, reason = self._validate_manifest(manifest)
            if not valid:
                self._manifest_rejection(candidate_id, reason)
                continue
            frozen = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            experiments[candidate_id] = {
                "candidate_id": candidate_id,
                "base_strategy": str(manifest["base_strategy"]),
                "registered_at": time.time(),
                "manifest_sha256": hashlib.sha256(frozen.encode("utf-8")).hexdigest(),
                "protocol": dict(manifest["research_protocol"]),
                "selection_evidence": dict(manifest["selection_evidence"]),
                "outcomes": [],
                "regime_returns": {},
                "status": "collecting_prospective_evidence",
                **self._authority_denied(),
            }
            registered.append(candidate_id)
        return registered

    @staticmethod
    def _gate_name(*, allowed: bool, reason: str) -> str:
        if allowed:
            return "approved"
        normalized = str(reason or "unspecified_block").strip().lower()
        prefix = normalized.split(":", 1)[0]
        if "cognitive" in prefix:
            return "cognitive_governance"
        if prefix.startswith("brain"):
            return "trading_brain"
        if "capital" in prefix:
            return "capital_governor"
        if "temporal" in prefix or "session" in prefix or "clock" in prefix:
            return "temporal_guard"
        if "exchange" in prefix:
            return "exchange_protection"
        if "router" in prefix or "evidence" in prefix or "advanced" in prefix:
            return "decision_router"
        return prefix[:80] or "unspecified_block"

    def _append_metric(
        self,
        section: str,
        key: str,
        value: float,
    ) -> None:
        metrics = self.state.setdefault("counterfactual", {}).setdefault(section, {})
        row = metrics.setdefault(key, {"returns": []})
        values = row.setdefault("returns", [])
        values.append(float(value))
        row["returns"] = values[-self.RETURN_LIMIT :]

    def _counterfactual_step(
        self,
        market_rows: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        counterfactual = self.state.setdefault("counterfactual", {})
        pending = counterfactual.setdefault("pending", {})
        resolved = counterfactual.setdefault("resolved", [])
        resolved_now: list[dict[str, Any]] = []
        opened_now: list[str] = []

        for symbol in sorted(market_rows):
            row = market_rows[symbol]
            if not isinstance(row, dict):
                continue
            price = _finite(row.get("price"))
            if price <= 0.0:
                continue
            active = pending.get(symbol)
            if isinstance(active, dict):
                active["observations"] = int(active.get("observations") or 1) + 1
                active["last_price"] = price
                if int(active["observations"]) >= self.counterfactual_horizon_observations:
                    entry_price = _finite(active.get("entry_price"))
                    if entry_price > 0.0:
                        net_return = (
                            price / entry_price
                            - 1.0
                            - self.round_trip_cost_bps / 10_000.0
                        )
                        outcome = {
                            **active,
                            "exit_price": price,
                            "closed_at": time.time(),
                            "net_return": net_return,
                            "profitable_after_cost": net_return > 0.0,
                            "evidence_authority": self.COUNTERFACTUAL_AUTHORITY,
                            **self._authority_denied(),
                        }
                        resolved.append(outcome)
                        resolved_now.append(outcome)
                        self._append_metric(
                            "gates",
                            str(active.get("gate") or "unknown"),
                            net_return,
                        )
                        self._append_metric(
                            "regimes",
                            str(active.get("regime") or "unknown"),
                            net_return,
                        )
                    pending.pop(symbol, None)

            if (
                symbol not in pending
                and row.get("base_enter_candidate") is True
            ):
                allowed = row.get("final_allowed") is True
                reason = str(row.get("route_reason") or "unspecified_block")
                pending[symbol] = {
                    "trial_id": hashlib.sha256(
                        f"{symbol}|{time.time_ns()}".encode("utf-8")
                    ).hexdigest()[:24],
                    "symbol": symbol,
                    "entry_price": price,
                    "opened_at": time.time(),
                    "observations": 1,
                    "horizon_observations": self.counterfactual_horizon_observations,
                    "gate": self._gate_name(allowed=allowed, reason=reason),
                    "route_reason": reason[:240],
                    "final_allowed": allowed,
                    "regime": str(row.get("regime") or "unknown")[:80],
                    "confidence": max(0.0, min(1.0, _finite(row.get("confidence")))),
                    "quality_score": max(0.0, min(1.0, _finite(row.get("quality_score")))),
                    "round_trip_cost_bps": self.round_trip_cost_bps,
                    "non_overlapping": True,
                    **self._authority_denied(),
                }
                opened_now.append(symbol)

        counterfactual["resolved"] = resolved[-self.RESOLVED_TRIAL_LIMIT :]
        return resolved_now, opened_now

    def _record_strategy_episodes(
        self,
        *,
        existing_experiment_ids: set[str],
        episodes: list[dict[str, Any]],
        contract_valid: bool,
    ) -> int:
        if not contract_valid:
            return 0
        experiments = self.state.setdefault("experiments", {})
        recorded = 0
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            if episode.get("evidence_authority") != self.EVIDENCE_AUTHORITY:
                continue
            strategy = str(episode.get("strategy") or "").strip()
            net_return = _finite(episode.get("net_return"), math.nan)
            if not strategy or not math.isfinite(net_return):
                continue
            symbol = str(episode.get("symbol") or "UNKNOWN")[:80]
            regime = str(episode.get("regime") or "unknown")[:80]
            for candidate_id in sorted(existing_experiment_ids):
                experiment = experiments.get(candidate_id)
                if not isinstance(experiment, dict):
                    continue
                if str(experiment.get("base_strategy") or "") != strategy:
                    continue
                outcomes = experiment.setdefault("outcomes", [])
                outcomes.append(
                    {
                        "recorded_at": time.time(),
                        "strategy": strategy,
                        "symbol": symbol,
                        "regime": regime,
                        "net_return": net_return,
                        "evidence_authority": self.EVIDENCE_AUTHORITY,
                    }
                )
                experiment["outcomes"] = outcomes[-self.RETURN_LIMIT :]
                regime_returns = experiment.setdefault("regime_returns", {})
                values = regime_returns.setdefault(regime, [])
                values.append(net_return)
                regime_returns[regime] = values[-self.RETURN_LIMIT :]
                recorded += 1
        return recorded

    def _experiment_snapshots(self) -> list[dict[str, Any]]:
        experiments = self.state.setdefault("experiments", {})
        family_size = max(1, len(experiments))
        snapshots: list[dict[str, Any]] = []
        for candidate_id in sorted(experiments):
            experiment = experiments[candidate_id]
            outcomes = experiment.get("outcomes") or []
            values = [
                _finite(row.get("net_return"), math.nan)
                for row in outcomes
                if isinstance(row, dict)
            ]
            values = [value for value in values if math.isfinite(value)]
            statistics = self._statistics(values)
            adjusted_p = min(
                1.0,
                float(statistics["one_sided_p_value"]) * family_size,
            )
            regimes = {
                regime: self._statistics(
                    [
                        _finite(value, math.nan)
                        for value in regime_values
                        if math.isfinite(_finite(value, math.nan))
                    ]
                )
                for regime, regime_values in (experiment.get("regime_returns") or {}).items()
                if isinstance(regime_values, list)
            }
            observed_regimes = sum(
                1 for row in regimes.values() if int(row.get("samples") or 0) > 0
            )
            required_samples = max(
                self.minimum_samples,
                int(
                    (experiment.get("protocol") or {}).get(
                        "minimum_additional_shadow_samples"
                    )
                    or 0
                ),
            )
            supported = bool(
                int(statistics["samples"]) >= required_samples
                and float(statistics["average_net_return"]) > 0.0
                and statistics["lower_95_net_return"] is not None
                and float(statistics["lower_95_net_return"]) > 0.0
                and adjusted_p < 0.05
                and observed_regimes >= 2
            )
            enough_samples = int(statistics["samples"]) >= required_samples
            status = (
                "research_supported_holdout"
                if supported
                else "not_supported_holdout"
                if enough_samples
                else "collecting_prospective_evidence"
            )
            experiment["status"] = status
            experiment["latest_statistics"] = statistics
            experiment["adjusted_p_value"] = adjusted_p
            experiment["observed_regimes"] = observed_regimes
            snapshots.append(
                {
                    "candidate_id": candidate_id,
                    "base_strategy": experiment.get("base_strategy"),
                    "registered_at": experiment.get("registered_at"),
                    "manifest_sha256": experiment.get("manifest_sha256"),
                    "status": status,
                    "required_samples": required_samples,
                    "samples_remaining": max(
                        0, required_samples - int(statistics["samples"])
                    ),
                    "statistics": statistics,
                    "bonferroni_family_size": family_size,
                    "adjusted_p_value": adjusted_p,
                    "observed_regimes": observed_regimes,
                    "regime_statistics": regimes,
                    "walk_forward": True,
                    "out_of_sample": True,
                    "prospective_only": True,
                    **self._authority_denied(),
                }
            )
        return snapshots

    def _counterfactual_snapshot(self) -> dict[str, Any]:
        counterfactual = self.state.setdefault("counterfactual", {})
        gates = {
            key: self._statistics(list(row.get("returns") or []))
            for key, row in (counterfactual.get("gates") or {}).items()
            if isinstance(row, dict)
        }
        regimes = {
            key: self._statistics(list(row.get("returns") or []))
            for key, row in (counterfactual.get("regimes") or {}).items()
            if isinstance(row, dict)
        }
        return {
            "authority": self.COUNTERFACTUAL_AUTHORITY,
            "horizon_observations": self.counterfactual_horizon_observations,
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "pending_non_overlapping_trials": len(counterfactual.get("pending") or {}),
            "resolved_trials_retained": len(counterfactual.get("resolved") or []),
            "gate_statistics": gates,
            "regime_statistics": regimes,
            "observational_not_causal": True,
            "cannot_replace_strategy_holdout": True,
            **self._authority_denied(),
        }

    def observe_cycle(
        self,
        *,
        observatory_authority: str,
        observed_round_trip_cost_bps: float,
        strategy_episodes: list[dict[str, Any]],
        foundry_manifests: list[dict[str, Any]],
        market_rows: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        contract_valid = bool(
            observatory_authority == self.EVIDENCE_AUTHORITY
            and _finite(observed_round_trip_cost_bps, -1.0)
            >= self.round_trip_cost_bps
        )
        existing_ids = set((self.state.get("experiments") or {}).keys())
        resolved_counterfactuals, opened_counterfactuals = self._counterfactual_step(
            market_rows
        )
        recorded_episodes = self._record_strategy_episodes(
            existing_experiment_ids=existing_ids,
            episodes=strategy_episodes,
            contract_valid=contract_valid,
        )
        registered = self._register_manifests(
            foundry_manifests,
            contract_valid=contract_valid,
        )
        experiments = self._experiment_snapshots()
        counterfactual = self._counterfactual_snapshot()
        self.state["cycles"] = int(self.state.get("cycles") or 0) + 1
        self.state["latest"] = {
            "observed_at": time.time(),
            "strategy_contract_valid": contract_valid,
            "episodes_recorded": recorded_episodes,
            "experiments_registered": registered,
            "counterfactuals_opened": opened_counterfactuals,
            "counterfactuals_resolved": len(resolved_counterfactuals),
        }
        self._save()
        return {
            **self.state["latest"],
            "minimum_samples": self.minimum_samples,
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "experiments": experiments,
            "counterfactual": counterfactual,
            "manifest_rejections_retained": len(
                self.state.get("manifest_rejections") or []
            ),
            "automatic_promotion": False,
            **self._authority_denied(),
        }

    def health(self) -> dict[str, Any]:
        latest = self.state.get("latest")
        if not isinstance(latest, dict):
            latest = {}
        counterfactual = self.state.get("counterfactual") or {}
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "state_path": str(self.state_path),
            "cycles": int(self.state.get("cycles") or 0),
            "strategy_contract_valid": bool(
                latest.get("strategy_contract_valid")
            ),
            "experiments": len(self.state.get("experiments") or {}),
            "pending_counterfactuals": len(counterfactual.get("pending") or {}),
            "resolved_counterfactuals": len(counterfactual.get("resolved") or []),
            "minimum_samples": self.minimum_samples,
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "walk_forward": True,
            "multiple_testing_correction": "bonferroni",
            "uncertainty_intervals": True,
            "observational_gate_attribution": True,
            **self._authority_denied(),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "cycles": 0,
            "experiments": {},
            "counterfactual": {
                "pending": {},
                "resolved": [],
                "gates": {},
                "regimes": {},
            },
            "manifest_rejections": [],
            "latest": {},
        }
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) == self.SCHEMA_VERSION:
                payload.setdefault("cycles", 0)
                payload.setdefault("experiments", {})
                payload.setdefault("manifest_rejections", [])
                payload.setdefault("latest", {})
                counterfactual = payload.setdefault("counterfactual", {})
                counterfactual.setdefault("pending", {})
                counterfactual.setdefault("resolved", [])
                counterfactual.setdefault("gates", {})
                counterfactual.setdefault("regimes", {})
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
