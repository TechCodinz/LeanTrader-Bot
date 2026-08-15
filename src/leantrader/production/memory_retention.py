from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


@dataclass(frozen=True)
class MarketFingerprint:
    regime: str = "unknown"
    volatility: float = 0.0
    trend: float = 0.0
    momentum: float = 0.0
    spread_bps: float = 0.0
    liquidity_imbalance: float = 0.0
    ultra_score: float = 0.0
    ultra_confidence: float = 0.0

    def vector(self) -> tuple[float, ...]:
        # Bounded transforms prevent one noisy sensor from dominating recall.
        return (
            _clip(math.tanh(self.volatility * 50.0)),
            _clip(self.trend),
            _clip(self.momentum),
            _clip(math.tanh(self.spread_bps / 25.0)),
            _clip(self.liquidity_imbalance),
            _clip(self.ultra_score),
            _clip(self.ultra_confidence, 0.0, 1.0),
        )


class MemoryRetentionEngine:
    """Causal working, episodic and semantic trading memory.

    Pending decisions stay in working memory. Only closed outcomes are promoted
    to long-term episodic/semantic memory. Wins and losses are both retained so
    the Brain cannot learn from survivorship-biased examples.
    """

    VERSION = "2.0"

    def __init__(
        self,
        state_path: Path,
        *,
        max_episodes: int = 5_000,
        half_life_hours: float = 720.0,
    ) -> None:
        if max_episodes < 100:
            raise ValueError("memory retention requires at least 100 episode slots")
        if half_life_hours <= 0:
            raise ValueError("memory half-life must be positive")
        self.state_path = state_path
        self.max_episodes = max_episodes
        self.half_life_seconds = half_life_hours * 3_600.0
        self.pending: dict[str, dict[str, Any]] = {}
        self.episodes: list[dict[str, Any]] = []
        self.semantic: dict[str, dict[str, Any]] = {}
        self.closed_outcomes = 0
        self.recalls = 0
        self.last_error: str | None = None
        self._load()

    def start(self) -> None:
        self._load()

    def stop(self) -> None:
        self._save()

    def remember_decision(
        self,
        decision_id: str,
        *,
        symbol: str,
        strategy: str,
        fingerprint: MarketFingerprint,
        confidence: float,
        metadata: dict[str, Any] | None = None,
        observed_at: float | None = None,
    ) -> dict[str, Any]:
        if not decision_id:
            raise ValueError("decision_id is required")
        existing = self.pending.get(decision_id)
        if existing is not None:
            return dict(existing)
        closed = self._episode_for_decision(decision_id)
        if closed is not None:
            return dict(closed)
        record = {
            "decision_id": decision_id,
            "symbol": symbol.upper(),
            "strategy": strategy,
            "fingerprint": asdict(fingerprint),
            "confidence": _clip(confidence, 0.0, 1.0),
            "metadata": dict(metadata or {}),
            "observed_at": float(observed_at or time.time()),
        }
        self.pending[decision_id] = record
        self._save()
        return dict(record)

    def close_decision(
        self,
        decision_id: str,
        *,
        net_return: float,
        closed_at: float | None = None,
    ) -> dict[str, Any]:
        already_closed = self._episode_for_decision(decision_id)
        if already_closed is not None:
            return dict(already_closed)
        pending = self.pending.pop(decision_id, None)
        if pending is None:
            raise KeyError(f"unknown pending decision: {decision_id}")
        closed_epoch = float(closed_at or time.time())
        fp = MarketFingerprint(**pending["fingerprint"])
        similarity = self._max_similarity(pending["symbol"], fp)
        novelty = 1.0 - similarity
        importance = min(1.0, abs(float(net_return)) / 0.02)
        retention_score = _clip(
            0.35 * importance
            + 0.25 * novelty
            + 0.20 * float(pending["confidence"])
            + 0.20,
            0.0,
            1.0,
        )
        episode = {
            **pending,
            "closed_at": closed_epoch,
            "net_return": float(net_return),
            "win": bool(net_return > 0),
            "retention_score": retention_score,
            "causal_closed_outcome": True,
        }
        self.episodes.append(episode)
        self.closed_outcomes += 1
        self._update_semantic(episode)
        self._prune(closed_epoch)
        self._save()
        return dict(episode)

    def record_closed_observation(
        self,
        *,
        observation_id: str,
        symbol: str,
        strategy: str,
        fingerprint: MarketFingerprint,
        confidence: float,
        net_return: float,
        metadata: dict[str, Any] | None = None,
        observed_at: float | None = None,
        closed_at: float | None = None,
    ) -> dict[str, Any]:
        existing = self._episode_for_decision(observation_id)
        if existing is not None:
            return dict(existing)
        self.remember_decision(
            observation_id,
            symbol=symbol,
            strategy=strategy,
            fingerprint=fingerprint,
            confidence=confidence,
            metadata=metadata,
            observed_at=observed_at,
        )
        return self.close_decision(observation_id, net_return=net_return, closed_at=closed_at)

    def recall(
        self,
        *,
        symbol: str,
        fingerprint: MarketFingerprint,
        limit: int = 8,
        min_similarity: float = 0.55,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        self.recalls += 1
        epoch = float(now or time.time())
        rows: list[tuple[float, dict[str, Any]]] = []
        for episode in self.episodes:
            candidate = MarketFingerprint(**episode["fingerprint"])
            similarity = self._similarity(fingerprint, candidate)
            if episode["symbol"] == symbol.upper():
                similarity = min(1.0, similarity + 0.08)
            if candidate.regime == fingerprint.regime and fingerprint.regime != "unknown":
                similarity = min(1.0, similarity + 0.08)
            if similarity < min_similarity:
                continue
            age = max(0.0, epoch - float(episode["closed_at"]))
            recency = math.exp(-math.log(2.0) * age / self.half_life_seconds)
            rank_score = similarity * (0.65 + 0.35 * recency) * (
                0.75 + 0.25 * float(episode.get("retention_score", 0.5))
            )
            rows.append((rank_score, episode))
        rows.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "similarity_score": score,
                **{key: value for key, value in episode.items() if key != "metadata"},
            }
            for score, episode in rows[:limit]
        ]

    def summarize(
        self,
        *,
        symbol: str,
        fingerprint: MarketFingerprint,
        limit: int = 8,
    ) -> dict[str, Any]:
        recalled = self.recall(symbol=symbol, fingerprint=fingerprint, limit=limit)
        if not recalled:
            return {
                "samples": 0,
                "weighted_net_return": 0.0,
                "win_rate": 0.0,
                "support": 0.0,
                "similar_episodes": [],
            }
        weights = [max(1e-9, float(row["similarity_score"])) for row in recalled]
        total = sum(weights)
        weighted_net = sum(w * float(row["net_return"]) for w, row in zip(weights, recalled)) / total
        win_rate = sum(w * float(bool(row["win"])) for w, row in zip(weights, recalled)) / total
        support = min(1.0, math.log1p(len(recalled)) / math.log1p(max(2, limit)))
        return {
            "samples": len(recalled),
            "weighted_net_return": weighted_net,
            "win_rate": win_rate,
            "support": support,
            "similar_episodes": recalled,
        }

    def semantic_evidence(self, *, symbol: str, regime: str, strategy: str) -> dict[str, Any]:
        key = self._semantic_key(symbol.upper(), regime, strategy)
        return dict(self.semantic.get(key, {}))

    def health(self) -> dict[str, Any]:
        return {
            "healthy": self.last_error is None,
            "pending_working_memory": len(self.pending),
            "retained_episodes": len(self.episodes),
            "semantic_buckets": len(self.semantic),
            "closed_outcomes": self.closed_outcomes,
            "recalls": self.recalls,
            "max_episodes": self.max_episodes,
            "half_life_hours": self.half_life_seconds / 3_600.0,
            "causal_closed_outcomes_only": True,
            "survivorship_bias_guard": True,
            "execution_authority": False,
            "error": self.last_error,
        }

    def _episode_for_decision(self, decision_id: str) -> dict[str, Any] | None:
        for episode in reversed(self.episodes):
            if str(episode.get("decision_id")) == decision_id:
                return episode
        return None

    def _update_semantic(self, episode: dict[str, Any]) -> None:
        fp = MarketFingerprint(**episode["fingerprint"])
        key = self._semantic_key(episode["symbol"], fp.regime, episode["strategy"])
        bucket = self.semantic.setdefault(
            key,
            {
                "symbol": episode["symbol"],
                "regime": fp.regime,
                "strategy": episode["strategy"],
                "samples": 0,
                "wins": 0,
                "cumulative_net_return": 0.0,
                "ewma_net_return": 0.0,
                "last_updated": None,
            },
        )
        bucket["samples"] = int(bucket.get("samples", 0)) + 1
        bucket["wins"] = int(bucket.get("wins", 0)) + int(bool(episode["win"]))
        bucket["cumulative_net_return"] = float(bucket.get("cumulative_net_return", 0.0)) + float(
            episode["net_return"]
        )
        alpha = 0.10
        if bucket["samples"] == 1:
            bucket["ewma_net_return"] = float(episode["net_return"])
        else:
            bucket["ewma_net_return"] = (
                (1.0 - alpha) * float(bucket.get("ewma_net_return", 0.0))
                + alpha * float(episode["net_return"])
            )
        bucket["last_updated"] = episode["closed_at"]

    def _max_similarity(self, symbol: str, fingerprint: MarketFingerprint) -> float:
        if not self.episodes:
            return 0.0
        values = []
        for episode in self.episodes[-256:]:
            candidate = MarketFingerprint(**episode["fingerprint"])
            sim = self._similarity(fingerprint, candidate)
            if episode["symbol"] == symbol.upper():
                sim = min(1.0, sim + 0.05)
            values.append(sim)
        return max(values, default=0.0)

    @staticmethod
    def _similarity(left: MarketFingerprint, right: MarketFingerprint) -> float:
        a = left.vector()
        b = right.vector()
        distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))
        return max(0.0, 1.0 - distance / 2.0)

    def _prune(self, now: float) -> None:
        if len(self.episodes) <= self.max_episodes:
            return
        ranked: list[tuple[float, dict[str, Any]]] = []
        for episode in self.episodes:
            age = max(0.0, now - float(episode["closed_at"]))
            recency = math.exp(-math.log(2.0) * age / self.half_life_seconds)
            score = 0.75 * float(episode.get("retention_score", 0.5)) + 0.25 * recency
            ranked.append((score, episode))
        ranked.sort(key=lambda item: item[0], reverse=True)
        self.episodes = [episode for _, episode in ranked[: self.max_episodes]]

    @staticmethod
    def _semantic_key(symbol: str, regime: str, strategy: str) -> str:
        return f"{symbol}|{regime}|{strategy}"

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.pending = dict(payload.get("pending") or {})
            self.episodes = list(payload.get("episodes") or [])
            self.semantic = dict(payload.get("semantic") or {})
            self.closed_outcomes = int(payload.get("closed_outcomes", len(self.episodes)))
            self.recalls = int(payload.get("recalls", 0))
            self.last_error = None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.pending = {}
            self.episodes = []
            self.semantic = {}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "pending": self.pending,
            "episodes": self.episodes,
            "semantic": self.semantic,
            "closed_outcomes": self.closed_outcomes,
            "recalls": self.recalls,
            "updated_at": time.time(),
        }
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
