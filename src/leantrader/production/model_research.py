from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar


class ModelResearchError(RuntimeError):
    """Raised when an external research model violates the bounded contract."""


class StructuredResearchProvider:
    """Small provider adapter for structured, non-executing research proposals."""

    PROVIDERS: ClassVar[set[str]] = {"openai", "anthropic", "gemini"}

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key_path: Path,
        endpoint: str = "",
        http_post: Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.provider = provider.strip().lower()
        self.model = model.strip()
        self.api_key_path = api_key_path
        self.endpoint = endpoint.strip()
        self.http_post = http_post or self._http_post
        if self.provider not in self.PROVIDERS:
            raise ValueError(f"unsupported model research provider: {self.provider}")
        if not self.model:
            raise ValueError("model research requires a model identifier")

    def propose(self, evidence: dict[str, Any]) -> dict[str, Any]:
        key = self._read_key()
        system = (
            "You are a trading research scientist. Return one JSON object only. "
            "You may propose bounded research parameters, but never orders, credentials, leverage, "
            "profit guarantees, or live execution. Use only the supplied evidence."
        )
        prompt = json.dumps(
            {
                "task": "propose one falsifiable paper-trading challenger",
                "required_schema": {
                    "hypothesis": "string",
                    "confidence": "number 0..1",
                    "evidence_refs": ["string"],
                    "risk_flags": ["string"],
                    "candidate": {
                        "component_weight_deltas": {
                            "trend": "-0.10..0.10",
                            "momentum": "-0.10..0.10",
                            "mean_reversion": "-0.10..0.10",
                            "bollinger_breakout": "-0.10..0.10",
                        },
                        "timeframe_group_deltas": {
                            "fast": "-0.10..0.10",
                            "tactical": "-0.10..0.10",
                            "strategic": "-0.10..0.10",
                        },
                        "router_threshold_delta": "-0.10..0.10",
                        "risk_size_multiplier": "0.25..1.00",
                    },
                },
                "evidence": evidence,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if self.provider == "openai":
            payload = self.http_post(
                self.endpoint or "https://api.openai.com/v1/responses",
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                {"model": self.model, "input": f"{system}\n\n{prompt}"},
            )
            text = self._openai_text(payload)
        elif self.provider == "anthropic":
            payload = self.http_post(
                self.endpoint or "https://api.anthropic.com/v1/messages",
                {
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                {
                    "model": self.model,
                    "max_tokens": 1_200,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            text = self._anthropic_text(payload)
        else:
            endpoint = self.endpoint or (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:generateContent"
            )
            payload = self.http_post(
                endpoint,
                {"Content-Type": "application/json", "x-goog-api-key": key},
                {
                    "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{prompt}"}]}],
                    "generationConfig": {"responseMimeType": "application/json"},
                },
            )
            text = self._gemini_text(payload)
        try:
            return json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelResearchError("research provider did not return valid JSON") from exc

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "configured": self.api_key_path.exists(),
            "api_key_path": str(self.api_key_path),
            "credentials_exposed": False,
            "execution_authority": False,
        }

    def _read_key(self) -> str:
        try:
            key = self.api_key_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ModelResearchError("model research API key file is unavailable") from exc
        if len(key) < 8:
            raise ModelResearchError("model research API key is invalid")
        return key

    @staticmethod
    def _openai_text(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str):
            return str(payload["output_text"])
        for item in payload.get("output", []):
            for content in item.get("content", []):
                if isinstance(content.get("text"), str):
                    return str(content["text"])
        raise ModelResearchError("OpenAI response contained no output text")

    @staticmethod
    def _anthropic_text(payload: dict[str, Any]) -> str:
        for content in payload.get("content", []):
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                return str(content["text"])
        raise ModelResearchError("Anthropic response contained no text")

    @staticmethod
    def _gemini_text(payload: dict[str, Any]) -> str:
        try:
            return str(payload["candidates"][0]["content"]["parts"][0]["text"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResearchError("Gemini response contained no text") from exc

    @staticmethod
    def _http_post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))


class ModelResearchEngine:
    """Schedules, validates and journals model-proposed paper challengers."""

    VERSION = "1.0"
    COMPONENTS: ClassVar[set[str]] = {
        "trend",
        "momentum",
        "mean_reversion",
        "bollinger_breakout",
    }
    TIMEFRAME_GROUPS: ClassVar[set[str]] = {"fast", "tactical", "strategic"}
    CANDIDATE_KEYS: ClassVar[set[str]] = {
        "component_weight_deltas",
        "timeframe_group_deltas",
        "router_threshold_delta",
        "risk_size_multiplier",
    }

    def __init__(
        self,
        state_path: Path,
        *,
        enabled: bool = False,
        interval_cycles: int = 60,
        provider: StructuredResearchProvider | None = None,
    ) -> None:
        if interval_cycles < 10:
            raise ValueError("model research interval must be at least 10 cycles")
        if enabled and provider is None:
            raise ValueError("enabled model research requires a provider")
        self.state_path = state_path
        self.enabled = enabled
        self.interval_cycles = interval_cycles
        self.provider = provider
        self.state = self._load()
        self.cycles = 0
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None

    def observe(self, evidence: dict[str, Any]) -> dict[str, Any]:
        self.cycles += 1
        if not self.enabled:
            return {"requested": False, "reason": "disabled", **self.health()}
        if self.cycles != 1 and self.cycles % self.interval_cycles:
            return {"requested": False, "reason": "interval_not_due", **self.health()}
        self.calls += 1
        try:
            raw = self.provider.propose(evidence) if self.provider is not None else {}
            proposal = self._validate(raw)
            canonical = json.dumps(proposal, sort_keys=True, separators=(",", ":"))
            fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if any(row.get("fingerprint") == fingerprint for row in self.state["proposals"]):
                return {"requested": True, "accepted": False, "reason": "duplicate", **self.health()}
            record = {
                **proposal,
                "fingerprint": fingerprint,
                "provider": self.provider.provider if self.provider else None,
                "model": self.provider.model if self.provider else None,
                "created_at": dt.datetime.now(dt.UTC).isoformat(),
                "status": "pending_causal_replay",
                "paper_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            }
            self.state["proposals"].append(record)
            self.state["proposals"] = self.state["proposals"][-500:]
            self._save()
            self.last_error = None
            return {"requested": True, "accepted": True, "proposal": record, **self.health()}
        except Exception as exc:
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise

    def record_validation(
        self,
        fingerprint: str,
        *,
        windows: int,
        net_return: float,
        max_drawdown: float,
        brier_score: float,
    ) -> bool:
        record = next(
            (row for row in self.state["proposals"] if row.get("fingerprint") == fingerprint),
            None,
        )
        if record is None:
            raise KeyError("unknown model research proposal")
        values = (net_return, max_drawdown, brier_score)
        if not all(math.isfinite(float(value)) for value in values) or windows < 1:
            raise ValueError("valid causal validation metrics required")
        eligible = windows >= 5 and net_return > 0 and max_drawdown <= 0.10 and brier_score <= 0.25
        record["validation"] = {
            "windows": windows,
            "net_return": net_return,
            "max_drawdown": max_drawdown,
            "brier_score": brier_score,
        }
        record["status"] = "eligible_paper_challenger" if eligible else "rejected_by_evidence"
        record["paper_authority"] = eligible
        self._save()
        return eligible

    def health(self) -> dict[str, Any]:
        proposals = self.state.get("proposals", [])
        return {
            "enabled": self.enabled,
            "configured": self.provider is not None and self.provider.health()["configured"],
            "provider": self.provider.health() if self.provider is not None else None,
            "interval_cycles": self.interval_cycles,
            "cycles": self.cycles,
            "calls": self.calls,
            "failures": self.failures,
            "last_error": self.last_error,
            "proposals": len(proposals),
            "pending_replay": sum(row.get("status") == "pending_causal_replay" for row in proposals),
            "eligible_paper_challengers": sum(
                row.get("status") == "eligible_paper_challenger" for row in proposals
            ),
            "structured_output_validated": True,
            "automatic_live_promotion": False,
            "execution_authority": False,
            "state_path": str(self.state_path),
        }

    def _validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ModelResearchError("proposal must be a JSON object")
        hypothesis = str(raw.get("hypothesis", "")).strip()
        if not hypothesis or len(hypothesis) > 2_000:
            raise ModelResearchError("proposal hypothesis is missing or too long")
        confidence = float(raw.get("confidence", -1))
        if not 0 <= confidence <= 1:
            raise ModelResearchError("proposal confidence must be in [0, 1]")
        candidate = raw.get("candidate")
        if not isinstance(candidate, dict) or set(candidate) - self.CANDIDATE_KEYS:
            raise ModelResearchError("proposal contains unsupported candidate controls")
        component_deltas = self._bounded_map(
            candidate.get("component_weight_deltas", {}), self.COMPONENTS, -0.10, 0.10
        )
        timeframe_deltas = self._bounded_map(
            candidate.get("timeframe_group_deltas", {}), self.TIMEFRAME_GROUPS, -0.10, 0.10
        )
        threshold_delta = float(candidate.get("router_threshold_delta", 0.0))
        risk_multiplier = float(candidate.get("risk_size_multiplier", 1.0))
        if not -0.10 <= threshold_delta <= 0.10:
            raise ModelResearchError("router threshold delta exceeds research bounds")
        if not 0.25 <= risk_multiplier <= 1.0:
            raise ModelResearchError("risk multiplier exceeds research bounds")
        return {
            "schema_version": 1,
            "hypothesis": hypothesis,
            "confidence": confidence,
            "evidence_refs": [str(value)[:200] for value in raw.get("evidence_refs", [])][:50],
            "risk_flags": [str(value)[:200] for value in raw.get("risk_flags", [])][:50],
            "candidate": {
                "component_weight_deltas": component_deltas,
                "timeframe_group_deltas": timeframe_deltas,
                "router_threshold_delta": threshold_delta,
                "risk_size_multiplier": risk_multiplier,
            },
        }

    @staticmethod
    def _bounded_map(raw: Any, allowed: set[str], lower: float, upper: float) -> dict[str, float]:
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ModelResearchError("proposal contains unsupported parameter names")
        output = {str(key): float(value) for key, value in raw.items()}
        if not all(math.isfinite(value) and lower <= value <= upper for value in output.values()):
            raise ModelResearchError("proposal parameter exceeds research bounds")
        return output

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 1, "proposals": []}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == 1 and isinstance(payload.get("proposals"), list):
                return payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return {"schema_version": 1, "proposals": []}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
