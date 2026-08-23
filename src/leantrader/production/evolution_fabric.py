from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any


class EvolutionFabric:
    """Hot-ingests bounded external capability packs without executing their code.

    The fabric is intentionally data-only. New research/sensor implementations can
    run out-of-process (for example, as a sidecar container) and publish signed-by-
    provenance JSON observations into the shared runtime volume. LeanTrader can
    discover those packs at cycle boundaries without restarting its ledger, Brain,
    memory, or market-state process.

    A pack can provide research evidence and shadow challenger signals. It can
    never place orders, enable live mode, add credentials, rewrite core code, or
    increase upstream risk.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    PACK_SCHEMA_VERSION = 1
    MAX_PACKS = 250
    MAX_OBSERVATIONS_PER_PACK = 500
    MAX_PENDING_EPISODES = 10_000
    MAX_RESOLVED_EPISODES = 10_000
    ALLOWED_KINDS = {"context", "risk", "signal", "relationship", "anomaly"}

    def __init__(
        self,
        state_path: Path,
        inbox_path: Path,
        *,
        enabled: bool = True,
        max_pack_age_seconds: int = 3_600,
        minimum_shadow_samples: int = 100,
        round_trip_cost_bps: float = 30.0,
    ) -> None:
        self.state_path = state_path
        self.inbox_path = inbox_path
        self.requests_path = inbox_path.parent / "requests.json"
        self.enabled = bool(enabled)
        self.max_pack_age_seconds = max(60, int(max_pack_age_seconds))
        self.minimum_shadow_samples = max(10, int(minimum_shadow_samples))
        self.round_trip_cost_bps = max(0.0, float(round_trip_cost_bps))
        self.last_error: str | None = None
        self.state = self._load()

    def start(self) -> None:
        self.inbox_path.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def stop(self) -> None:
        self._save()

    def refresh(
        self,
        *,
        prices: dict[str, float] | None = None,
        engine_health: dict[str, Any] | None = None,
        world_market: dict[str, Any] | None = None,
        strategy_health: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        prices = {str(k).upper(): float(v) for k, v in (prices or {}).items() if self._finite(v) and float(v) > 0}
        self.state["cycles"] = int(self.state.get("cycles") or 0) + 1
        self.state["last_refresh"] = now

        if not self.enabled:
            self.state["enabled"] = False
            self._save()
            return self.snapshot()

        self.state["enabled"] = True
        accepted_this_cycle = 0
        quarantined_this_cycle = 0
        self.inbox_path.mkdir(parents=True, exist_ok=True)

        for path in sorted(self.inbox_path.glob("*.json")):
            try:
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if digest in (self.state.get("seen_digests") or {}):
                    continue
                payload = json.loads(raw.decode("utf-8"))
                normalized = self._validate_pack(payload, now=now)
                normalized["digest"] = digest
                normalized["file_name"] = path.name
                normalized["accepted_at"] = now
                self._accept_pack(normalized, prices=prices)
                self.state.setdefault("seen_digests", {})[digest] = {
                    "pack_id": normalized["pack_id"],
                    "accepted_at": now,
                    "file_name": path.name,
                }
                accepted_this_cycle += 1
            except Exception as exc:  # noqa: BLE001 - untrusted pack must be isolated
                digest = self._safe_digest(path)
                key = digest or f"file:{path.name}:{int(path.stat().st_mtime) if path.exists() else 0}"
                if key not in (self.state.get("quarantine") or {}):
                    self.state.setdefault("quarantine", {})[key] = {
                        "file_name": path.name,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "quarantined_at": now,
                    }
                    quarantined_this_cycle += 1
                if digest:
                    self.state.setdefault("seen_digests", {})[digest] = {
                        "quarantined_at": now,
                        "file_name": path.name,
                    }

        resolved = self._resolve_shadow_episodes(prices=prices, now=now)
        self._expire_packs(now=now)
        self._trim_state()
        self.state["latest_system_context"] = {
            "engine_health": self._bounded_summary(engine_health or {}),
            "world_market": self._bounded_summary(world_market or {}),
            "strategy_health": self._bounded_summary(strategy_health or {}),
        }
        self.state["last_cycle"] = {
            "accepted": accepted_this_cycle,
            "quarantined": quarantined_this_cycle,
            "shadow_episodes_resolved": resolved,
            "timestamp": now,
        }
        self.last_error = None
        self._save()
        return self.snapshot()

    def sync_research_demand(
        self,
        *,
        adapter_backlog: list[dict[str, Any]] | None = None,
        research_agenda: list[dict[str, Any]] | None = None,
        world_market: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        capabilities = self.capability_status()
        backlog: list[dict[str, Any]] = []
        for row in (adapter_backlog or [])[:100]:
            source = str(row.get("source") or "").strip()
            if not source:
                continue
            available = source in capabilities and capabilities[source].get("status") == "available_external_shadow"
            backlog.append(
                {
                    "capability": source,
                    "requests": int(row.get("requests") or 0),
                    "max_priority": float(row.get("max_priority") or 0.0),
                    "description": row.get("description"),
                    "satisfied_by_hot_pack": available,
                }
            )
        backlog.sort(key=lambda row: (not row["satisfied_by_hot_pack"], row["max_priority"], row["requests"]), reverse=True)

        requests = {
            "schema_version": 1,
            "generated_at": now,
            "execution_authority": False,
            "can_add_credentials": False,
            "can_enable_live": False,
            "arbitrary_code_execution": False,
            "desired_capabilities": [row for row in backlog if not row["satisfied_by_hot_pack"]][:50],
            "active_external_capabilities": capabilities,
            "research_agenda": [self._bounded_summary(row) for row in (research_agenda or [])[:50]],
            "rare_scope": self._bounded_summary(world_market or {}),
            "sidecar_contract": {
                "publish_directory": str(self.inbox_path),
                "pack_schema_version": self.PACK_SCHEMA_VERSION,
                "hot_reload": True,
                "core_restart_required": False,
                "data_only": True,
            },
        }
        self.requests_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.requests_path.with_suffix(self.requests_path.suffix + ".tmp")
        temporary.write_text(json.dumps(requests, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.requests_path)
        self.state["research_demand"] = {
            "generated_at": now,
            "desired_capabilities": requests["desired_capabilities"],
            "active_external_capabilities": capabilities,
        }
        self._save()
        return requests

    def capability_status(self) -> dict[str, dict[str, Any]]:
        now = time.time()
        output: dict[str, dict[str, Any]] = {}
        for pack_id, pack in (self.state.get("packs") or {}).items():
            if pack.get("status") != "active":
                continue
            expires_at = float(pack.get("expires_at") or 0.0)
            if expires_at and expires_at < now:
                continue
            for capability in pack.get("capabilities") or []:
                current = output.get(capability)
                candidate = {
                    "status": "available_external_shadow",
                    "pack_id": pack_id,
                    "version": pack.get("version"),
                    "producer": pack.get("producer"),
                    "confidence": float(pack.get("pack_confidence") or 0.0),
                    "accepted_at": pack.get("accepted_at"),
                    "expires_at": pack.get("expires_at"),
                    "execution_authority": False,
                }
                if current is None or candidate["confidence"] > float(current.get("confidence") or 0.0):
                    output[capability] = candidate
        return output

    def evidence_for(self, symbol: str) -> list[dict[str, Any]]:
        symbol = str(symbol).upper()
        now = time.time()
        rows: list[dict[str, Any]] = []
        metrics = self.state.get("shadow_metrics") or {}

        for pack_id, pack in (
            self.state.get("packs") or {}
        ).items():
            if (
                pack.get("status") != "active"
                or float(pack.get("expires_at") or 0.0)
                < now
            ):
                continue

            metric = metrics.get(pack_id) or {}
            if not isinstance(metric, dict):
                metric = {}

            for observation in (
                pack.get("observations") or []
            ):
                if observation.get("symbol") not in {
                    symbol,
                    "GLOBAL",
                }:
                    continue

                row = dict(observation)
                row.update(
                    {
                        "pack_id": pack_id,
                        "pack_version": pack.get(
                            "version"
                        ),
                        "producer": pack.get(
                            "producer"
                        ),
                        "research_validated": (
                            metric.get(
                                "research_validated"
                            )
                            is True
                        ),
                        "shadow_samples": int(
                            metric.get("samples") or 0
                        ),
                        "shadow_win_rate": float(
                            metric.get("win_rate")
                            or 0.0
                        ),
                        "average_net_return": float(
                            metric.get(
                                "average_net_return"
                            )
                            or 0.0
                        ),
                        "ewma_net_return": float(
                            metric.get(
                                "ewma_net_return"
                            )
                            or 0.0
                        ),
                    }
                )
                rows.append(row)

        rows.sort(
            key=lambda row: (
                row.get("research_validated") is True,
                float(
                    row.get("confidence")
                    or 0.0
                ),
                float(
                    row.get("observed_at")
                    or 0.0
                ),
            ),
            reverse=True,
        )
        return rows[:50]

    def snapshot(self) -> dict[str, Any]:
        capabilities = self.capability_status()
        metrics = self.state.get("shadow_metrics") or {}
        validated = {
            pack_id: dict(row)
            for pack_id, row in metrics.items()
            if row.get("research_validated") is True
        }
        return {
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "enabled": self.enabled,
            "cycles": int(self.state.get("cycles") or 0),
            "active_packs": sum(1 for row in (self.state.get("packs") or {}).values() if row.get("status") == "active"),
            "quarantined_packs": len(self.state.get("quarantine") or {}),
            "capabilities": capabilities,
            "shadow_metrics": metrics,
            "research_validated_packs": validated,
            "pending_shadow_episodes": len(self.state.get("pending_shadow") or {}),
            "resolved_shadow_episodes": len(self.state.get("resolved_shadow") or []),
            "last_cycle": dict(self.state.get("last_cycle") or {}),
            "hot_reload_supported": True,
            "core_restart_required_for_new_pack": False,
            "sidecar_candidate_supported": True,
            "arbitrary_code_execution": False,
            "execution_authority": False,
            "can_enable_live": False,
            "can_add_credentials": False,
            "can_increase_upstream_risk": False,
            "state_path": str(self.state_path),
            "inbox_path": str(self.inbox_path),
            "requests_path": str(self.requests_path),
        }

    def health(self) -> dict[str, Any]:
        return {
            "healthy": self.last_error is None,
            "state": "running" if self.enabled else "disabled",
            "last_error": self.last_error,
            **self.snapshot(),
        }

    def _validate_pack(self, payload: Any, *, now: float) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("capability pack must be a JSON object")
        if int(payload.get("schema_version") or 0) != self.PACK_SCHEMA_VERSION:
            raise ValueError("unsupported capability pack schema_version")
        if payload.get("execution_authority") not in {None, False}:
            raise ValueError("capability pack cannot request execution authority")
        if payload.get("can_enable_live") not in {None, False}:
            raise ValueError("capability pack cannot enable live trading")
        if payload.get("can_increase_risk") not in {None, False}:
            raise ValueError("capability pack cannot increase risk")
        if payload.get("can_add_credentials") not in {None, False}:
            raise ValueError("capability pack cannot add credentials")

        pack_id = str(payload.get("pack_id") or "").strip()
        version = str(payload.get("version") or "").strip()
        producer = str(payload.get("producer") or "").strip()
        if not pack_id or len(pack_id) > 120 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in pack_id):
            raise ValueError("invalid pack_id")
        if not version or len(version) > 64:
            raise ValueError("invalid version")
        if not producer or len(producer) > 160:
            raise ValueError("producer is required")

        generated_at = float(payload.get("generated_at") or payload.get("observed_at") or 0.0)
        if not self._finite(generated_at) or generated_at <= 0:
            raise ValueError("generated_at must be a unix timestamp")
        age = now - generated_at
        if age < -300:
            raise ValueError("capability pack timestamp is in the future")
        if age > self.max_pack_age_seconds:
            raise ValueError("capability pack is stale")
        expires_at = float(payload.get("expires_at") or (generated_at + self.max_pack_age_seconds))
        if not self._finite(expires_at) or expires_at <= now:
            raise ValueError("capability pack is already expired")
        expires_at = min(expires_at, generated_at + self.max_pack_age_seconds)

        capabilities = []
        for value in payload.get("capabilities") or []:
            capability = str(value).strip()
            if capability and len(capability) <= 120 and capability.replace("_", "").replace("-", "").isalnum():
                capabilities.append(capability)
        capabilities = list(dict.fromkeys(capabilities))[:100]
        if not capabilities:
            raise ValueError("at least one capability is required")

        observations_raw = payload.get("observations") or []
        if not isinstance(observations_raw, list):
            raise ValueError("observations must be a list")
        if len(observations_raw) > self.MAX_OBSERVATIONS_PER_PACK:
            raise ValueError("too many observations in one capability pack")
        observations = [self._validate_observation(row, generated_at=generated_at) for row in observations_raw]
        if not observations:
            raise ValueError("at least one observation is required")
        confidence = sum(float(row["confidence"]) for row in observations) / len(observations)
        return {
            "pack_id": pack_id,
            "version": version,
            "producer": producer,
            "generated_at": generated_at,
            "expires_at": expires_at,
            "capabilities": capabilities,
            "observations": observations,
            "pack_confidence": confidence,
            "read_only": True,
            "execution_authority": False,
            "can_enable_live": False,
            "can_increase_risk": False,
        }

    def _validate_observation(self, row: Any, *, generated_at: float) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise ValueError("observation must be an object")
        kind = str(row.get("kind") or "").strip().lower()
        if kind not in self.ALLOWED_KINDS:
            raise ValueError(f"unsupported observation kind: {kind!r}")
        symbol = str(row.get("symbol") or "GLOBAL").strip().upper()
        if not symbol or len(symbol) > 40:
            raise ValueError("invalid observation symbol")
        score = float(row.get("score") or 0.0)
        confidence = float(row.get("confidence") or 0.0)
        if not self._finite(score) or not -1.0 <= score <= 1.0:
            raise ValueError("observation score must be in [-1, 1]")
        if not self._finite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("observation confidence must be in [0, 1]")
        source = str(row.get("source") or "").strip()
        provenance = str(row.get("provenance") or "").strip()
        if not source or not provenance:
            raise ValueError("observation source and provenance are required")
        observed_at = float(row.get("observed_at") or generated_at)
        if not self._finite(observed_at) or abs(observed_at - generated_at) > self.max_pack_age_seconds:
            raise ValueError("observation timestamp is outside pack freshness window")
        horizon_seconds = int(row.get("horizon_seconds") or 900)
        horizon_seconds = min(max(horizon_seconds, 60), 604_800)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        return {
            "symbol": symbol,
            "kind": kind,
            "score": score,
            "confidence": confidence,
            "source": source[:240],
            "provenance": provenance[:500],
            "observed_at": observed_at,
            "horizon_seconds": horizon_seconds,
            "metadata": self._bounded_summary(metadata),
            "execution_authority": False,
        }

    def _accept_pack(self, pack: dict[str, Any], *, prices: dict[str, float]) -> None:
        pack_id = pack["pack_id"]
        existing = (self.state.get("packs") or {}).get(pack_id)
        generation = int(existing.get("generation") or 0) + 1 if isinstance(existing, dict) else 1
        pack["generation"] = generation
        pack["status"] = "active"
        pack["hot_reloaded"] = existing is not None
        self.state.setdefault("packs", {})[pack_id] = pack
        self.state["generation"] = int(self.state.get("generation") or 0) + 1
        self.state["last_hot_reload"] = time.time()
        self._open_shadow_episodes(pack, prices=prices)

    def _open_shadow_episodes(self, pack: dict[str, Any], *, prices: dict[str, float]) -> None:
        pending = self.state.setdefault("pending_shadow", {})
        for index, row in enumerate(pack.get("observations") or []):
            if row.get("kind") != "signal" or abs(float(row.get("score") or 0.0)) < 0.05:
                continue
            symbol = str(row.get("symbol") or "").upper()
            if symbol == "GLOBAL" or symbol not in prices:
                continue
            episode_id = hashlib.sha256(f"{pack['digest']}|{index}|{symbol}".encode("utf-8")).hexdigest()[:24]
            if episode_id in pending:
                continue
            pending[episode_id] = {
                "episode_id": episode_id,
                "pack_id": pack["pack_id"],
                "pack_version": pack["version"],
                "symbol": symbol,
                "signal": float(row["score"]),
                "confidence": float(row["confidence"]),
                "entry_price": float(prices[symbol]),
                "opened_at": float(row.get("observed_at") or time.time()),
                "due_at": float(row.get("observed_at") or time.time()) + int(row.get("horizon_seconds") or 900),
                "execution_authority": False,
            }

    def _resolve_shadow_episodes(self, *, prices: dict[str, float], now: float) -> int:
        pending = self.state.setdefault("pending_shadow", {})
        resolved_rows = self.state.setdefault("resolved_shadow", [])
        metrics = self.state.setdefault("shadow_metrics", {})
        resolved = 0
        for episode_id, row in list(pending.items()):
            if now < float(row.get("due_at") or 0.0):
                continue
            symbol = str(row.get("symbol") or "").upper()
            if symbol not in prices:
                if now - float(row.get("due_at") or now) > self.max_pack_age_seconds:
                    pending.pop(episode_id, None)
                continue
            entry = float(row.get("entry_price") or 0.0)
            exit_price = float(prices[symbol])
            if entry <= 0 or exit_price <= 0:
                pending.pop(episode_id, None)
                continue
            direction = 1.0 if float(row.get("signal") or 0.0) > 0 else -1.0
            gross = direction * (exit_price / entry - 1.0)
            net = gross - self.round_trip_cost_bps / 10_000.0
            outcome = {
                **row,
                "closed_at": now,
                "exit_price": exit_price,
                "gross_return": gross,
                "net_return": net,
                "profitable_after_cost": net > 0,
                "evidence_authority": "shadow_candidate_only",
            }
            resolved_rows.append(outcome)
            pending.pop(episode_id, None)
            self._update_metrics(metrics, row["pack_id"], net)
            resolved += 1
        return resolved

    def _update_metrics(self, metrics: dict[str, Any], pack_id: str, net: float) -> None:
        row = metrics.setdefault(
            pack_id,
            {
                "samples": 0,
                "wins": 0,
                "cumulative_net_return": 0.0,
                "average_net_return": 0.0,
                "ewma_net_return": 0.0,
                "negative_streak": 0,
                "research_validated": False,
                "execution_authority": False,
            },
        )
        samples = int(row.get("samples") or 0) + 1
        row["samples"] = samples
        row["wins"] = int(row.get("wins") or 0) + int(net > 0)
        row["cumulative_net_return"] = float(row.get("cumulative_net_return") or 0.0) + net
        row["average_net_return"] = row["cumulative_net_return"] / samples
        prior_ewma = float(row.get("ewma_net_return") or 0.0)
        row["ewma_net_return"] = net if samples == 1 else 0.10 * net + 0.90 * prior_ewma
        row["negative_streak"] = 0 if net > 0 else int(row.get("negative_streak") or 0) + 1
        row["win_rate"] = int(row["wins"]) / samples
        row["research_validated"] = bool(
            samples >= self.minimum_shadow_samples
            and float(row["average_net_return"]) > 0.0
            and float(row["ewma_net_return"]) > 0.0
        )
        row["can_enable_live"] = False
        row["can_increase_risk"] = False
        row["execution_authority"] = False

    def _expire_packs(self, *, now: float) -> None:
        for row in (self.state.get("packs") or {}).values():
            if row.get("status") == "active" and float(row.get("expires_at") or 0.0) < now:
                row["status"] = "expired"
                row["expired_at"] = now

    def _trim_state(self) -> None:
        packs = self.state.setdefault("packs", {})
        if len(packs) > self.MAX_PACKS:
            ordered = sorted(packs.items(), key=lambda item: float(item[1].get("accepted_at") or 0.0))
            for key, _ in ordered[: len(packs) - self.MAX_PACKS]:
                packs.pop(key, None)
        seen = self.state.setdefault("seen_digests", {})
        if len(seen) > self.MAX_PACKS * 10:
            ordered = sorted(seen.items(), key=lambda item: max(float(item[1].get("accepted_at") or 0.0), float(item[1].get("quarantined_at") or 0.0)))
            for key, _ in ordered[: len(seen) - self.MAX_PACKS * 10]:
                seen.pop(key, None)
        quarantine = self.state.setdefault("quarantine", {})
        if len(quarantine) > self.MAX_PACKS:
            ordered = sorted(quarantine.items(), key=lambda item: float(item[1].get("quarantined_at") or 0.0))
            for key, _ in ordered[: len(quarantine) - self.MAX_PACKS]:
                quarantine.pop(key, None)
        pending = self.state.setdefault("pending_shadow", {})
        if len(pending) > self.MAX_PENDING_EPISODES:
            ordered = sorted(pending.items(), key=lambda item: float(item[1].get("opened_at") or 0.0))
            for key, _ in ordered[: len(pending) - self.MAX_PENDING_EPISODES]:
                pending.pop(key, None)
        resolved = self.state.setdefault("resolved_shadow", [])
        if len(resolved) > self.MAX_RESOLVED_EPISODES:
            self.state["resolved_shadow"] = resolved[-self.MAX_RESOLVED_EPISODES :]

    @staticmethod
    def _finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _safe_digest(path: Path) -> str | None:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None

    @classmethod
    def _bounded_summary(cls, value: Any, depth: int = 0) -> Any:
        if depth >= 4:
            return str(value)[:500]
        if isinstance(value, dict):
            output: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 100:
                    break
                output[str(key)[:120]] = cls._bounded_summary(item, depth + 1)
            return output
        if isinstance(value, list):
            return [cls._bounded_summary(item, depth + 1) for item in value[:100]]
        if isinstance(value, tuple):
            return [cls._bounded_summary(item, depth + 1) for item in value[:100]]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value if not isinstance(value, str) else value[:1000]
        return str(value)[:1000]

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "enabled": self.enabled,
            "cycles": 0,
            "generation": 0,
            "packs": {},
            "seen_digests": {},
            "quarantine": {},
            "pending_shadow": {},
            "resolved_shadow": [],
            "shadow_metrics": {},
            "research_demand": {},
        }
        if not self.state_path.exists():
            return empty
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) != self.SCHEMA_VERSION:
                return empty
            for key, default in empty.items():
                payload.setdefault(key, default)
            return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return empty

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        self.state["schema_version"] = self.SCHEMA_VERSION
        self.state["updated_at"] = time.time()
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
