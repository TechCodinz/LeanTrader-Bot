from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


class MarketWorldModel:
    """Persistent, non-executing model of market state and market-state transitions.

    The world model is deliberately descriptive rather than authoritative.  It
    combines price, volume, multi-timeframe, liquidity, public-context and
    cross-engine evidence into a persistent state representation.  It also
    surfaces low-frequency conjunctions ("rare scope" candidates) so research
    engines can investigate them without turning novelty itself into a trade.
    """

    VERSION = "1.0"
    SCHEMA_VERSION = 1
    HISTORY_LIMIT = 256
    ANOMALY_LIMIT = 1_000
    SAVE_INTERVAL = 10

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.last_error: str | None = None
        self.state = self._load()
        self.observations = int(self.state.get("observations") or 0)
        self.market_snapshots = int(self.state.get("market_snapshots") or 0)

    def start(self) -> None:
        self.state = self._load()
        self.observations = int(self.state.get("observations") or 0)
        self.market_snapshots = int(self.state.get("market_snapshots") or 0)

    def stop(self) -> None:
        self._save()

    @staticmethod
    def _series(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame:
            return pd.Series(dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    @staticmethod
    def _return(close: pd.Series, periods: int) -> float:
        if len(close) <= periods or periods <= 0:
            return 0.0
        start = float(close.iloc[-periods - 1])
        end = float(close.iloc[-1])
        if start <= 0 or end <= 0:
            return 0.0
        return end / start - 1.0

    @staticmethod
    def _entropy(returns: pd.Series) -> float:
        values = returns.tail(96).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 12:
            return 0.0
        hist, _ = np.histogram(values, bins=min(12, max(4, int(math.sqrt(len(values))))))
        counts = hist[hist > 0].astype(float)
        if not len(counts):
            return 0.0
        probs = counts / counts.sum()
        raw = float(-(probs * np.log(probs)).sum())
        maximum = math.log(len(counts)) if len(counts) > 1 else 1.0
        return _clip(raw / maximum)

    @staticmethod
    def _autocorrelation(returns: pd.Series) -> float:
        values = returns.tail(96)
        if len(values) < 12:
            return 0.0
        corr = values.autocorr(lag=1)
        return max(-1.0, min(1.0, _finite(corr)))

    @staticmethod
    def _volume_z(volume: pd.Series) -> float:
        values = volume.tail(96)
        if len(values) < 12:
            return 0.0
        baseline = values.iloc[:-1]
        scale = float(baseline.std(ddof=0))
        last = float(values.iloc[-1])
        mean = float(baseline.mean())
        if not math.isfinite(scale) or scale <= 1e-12:
            if abs(last - mean) <= max(1e-12, abs(mean) * 1e-9):
                return 0.0
            return 6.0 if last > mean else -6.0
        return _finite((last - mean) / scale)

    @staticmethod
    def _price_surprise(returns: pd.Series) -> float:
        values = returns.tail(96)
        if len(values) < 12:
            return 0.0
        baseline = values.iloc[:-1]
        scale = float(baseline.std(ddof=0))
        if not math.isfinite(scale) or scale <= 1e-12:
            return 0.0
        return _finite(float(values.iloc[-1]) / scale)

    @staticmethod
    def _signal_map(advanced: dict[str, Any]) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for row in advanced.get("signals") or []:
            if not isinstance(row, dict):
                continue
            engine = str(row.get("engine") or "").strip()
            if engine:
                output[engine] = row
        return output

    def _novelty_score(self, symbol: str, features: dict[str, float]) -> float:
        history = (self.state.get("history") or {}).get(symbol, [])
        rows = [row.get("features") or {} for row in history[-64:] if isinstance(row, dict)]
        if len(rows) < 12:
            return 1.0
        distances: list[float] = []
        for key, value in features.items():
            values = np.asarray([_finite(row.get(key)) for row in rows], dtype=float)
            if not len(values):
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            scale = max(1.4826 * mad, float(np.std(values)), 1e-6)
            distances.append(abs(value - median) / scale)
        if not distances:
            return 0.0
        robust_distance = float(np.mean(np.clip(np.asarray(distances), 0.0, 6.0)))
        return _clip(robust_distance / 3.0)

    def observe_symbol(
        self,
        symbol: str,
        frame: pd.DataFrame,
        *,
        adaptive: dict[str, Any],
        advanced: dict[str, Any],
        public_context: dict[str, Any] | None = None,
        timeframe_signals: dict[str, float] | None = None,
        timeframe_coverage: float = 0.0,
    ) -> dict[str, Any]:
        symbol = symbol.upper()
        close = self._series(frame, "close")
        volume = self._series(frame, "volume")
        if len(close) < 32 or (close <= 0).any():
            raise ValueError("world model requires at least 32 positive closes")

        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        short = returns.tail(16)
        long = returns.tail(64)
        vol_short = _finite(short.std(ddof=0)) if len(short) >= 4 else 0.0
        vol_long = _finite(long.std(ddof=0)) if len(long) >= 8 else vol_short
        vol_ratio = vol_short / max(vol_long, 1e-12) if vol_long > 0 else 1.0
        volume_z = self._volume_z(volume) if len(volume) else 0.0
        price_surprise = self._price_surprise(returns)
        momentum_4 = self._return(close, 4)
        momentum_16 = self._return(close, 16)
        momentum_64 = self._return(close, min(64, len(close) - 1))
        entropy = self._entropy(returns)
        autocorrelation = self._autocorrelation(returns)

        adaptive_score = _finite(adaptive.get("score"))
        adaptive_confidence = _clip(_finite(adaptive.get("confidence")))
        regime = str(adaptive.get("regime") or "unknown")
        swarm = advanced.get("swarm") or {}
        swarm_score = _finite(swarm.get("score"))
        swarm_confidence = _clip(_finite(swarm.get("confidence")))
        liquidity = advanced.get("liquidity") or {}
        spread_bps = max(0.0, _finite(liquidity.get("spread_bps")))
        liquidity_imbalance = max(-1.0, min(1.0, _finite(liquidity.get("imbalance"))))
        context = public_context or {}
        context_available = bool(context.get("available"))
        context_score = _finite(context.get("score")) if context_available else 0.0
        context_confidence = _clip(_finite(context.get("confidence"))) if context_available else 0.0

        tf_values = [
            _finite(value)
            for value in (timeframe_signals or {}).values()
            if math.isfinite(_finite(value))
        ]
        timeframe_dispersion = float(np.std(tf_values)) if len(tf_values) >= 2 else 0.0
        timeframe_direction_balance = float(np.mean([_sign(value) for value in tf_values])) if tf_values else 0.0

        signals = self._signal_map(advanced)
        spectral = signals.get("spectral_harmonics") or {}
        technical = signals.get("technical_structure") or {}
        scalping = signals.get("smart_scalping") or {}
        pattern_memory = signals.get("pattern_memory") or {}

        features = {
            "momentum_4": momentum_4,
            "momentum_16": momentum_16,
            "momentum_64": momentum_64,
            "vol_short": vol_short,
            "vol_ratio": vol_ratio,
            "volume_z": volume_z,
            "price_surprise": price_surprise,
            "entropy": entropy,
            "autocorrelation": autocorrelation,
            "spread_bps": spread_bps,
            "liquidity_imbalance": liquidity_imbalance,
            "adaptive_score": adaptive_score,
            "swarm_score": swarm_score,
            "context_score": context_score,
            "timeframe_dispersion": timeframe_dispersion,
        }
        novelty = self._novelty_score(symbol, features)
        model_disagreement = _clip(abs(adaptive_score - swarm_score) / 2.0)
        volatility_shock = _clip(max(0.0, vol_ratio - 1.0) / 2.0)
        volume_shock = _clip(max(0.0, abs(volume_z) - 1.0) / 3.0)
        price_shock = _clip(max(0.0, abs(price_surprise) - 1.0) / 4.0)
        liquidity_stress = _clip(
            0.55 * _clip(spread_bps / 50.0)
            + 0.45 * _clip(abs(liquidity_imbalance))
        ) if liquidity else 0.0
        timeframe_fracture = _clip(timeframe_dispersion / 0.75)
        information_disorder = _clip(entropy * (1.0 - abs(autocorrelation)))

        latent_patterns: list[str] = []
        if volume_shock >= 0.50 and abs(momentum_4) < max(vol_long, 1e-6) * 2.0:
            latent_patterns.append("volume_without_price_followthrough")
        if abs(liquidity_imbalance) >= 0.65 and _sign(liquidity_imbalance) != _sign(momentum_4):
            latent_patterns.append("liquidity_price_divergence")
        if model_disagreement >= 0.55:
            latent_patterns.append("cross_model_disagreement")
        if timeframe_fracture >= 0.50:
            latent_patterns.append("multi_timeframe_phase_fracture")
        if vol_ratio <= 0.55 and abs(volume_z) >= 1.5:
            latent_patterns.append("compression_with_participation_anomaly")
        if volatility_shock >= 0.60 and liquidity_stress >= 0.50:
            latent_patterns.append("volatility_liquidity_coupling")
        if novelty >= 0.70:
            latent_patterns.append("out_of_distribution_market_state")
        if abs(context_score) >= 0.45 and _sign(context_score) != _sign(momentum_16):
            latent_patterns.append("narrative_price_divergence")

        rare_scope_score = _clip(
            0.26 * novelty
            + 0.16 * model_disagreement
            + 0.14 * volatility_shock
            + 0.12 * volume_shock
            + 0.12 * liquidity_stress
            + 0.10 * timeframe_fracture
            + 0.10 * price_shock
        )
        data_quality = _clip(
            0.45
            + 0.20 * _clip(timeframe_coverage)
            + 0.15 * float(bool(liquidity))
            + 0.10 * float(context_available)
            + 0.10 * min(1.0, len(close) / 320.0)
        )
        state_confidence = _clip(
            data_quality
            * (
                0.40 * adaptive_confidence
                + 0.35 * swarm_confidence
                + 0.15 * context_confidence
                + 0.10 * (1.0 - model_disagreement)
            )
        )

        unknowns: list[str] = []
        if not liquidity:
            unknowns.append("order_book_liquidity")
        if not context_available:
            unknowns.append("fresh_public_context")
        if timeframe_coverage < 0.50:
            unknowns.append("broad_timeframe_context")
        if novelty >= 0.70:
            unknowns.append("historical_analogue")
        if model_disagreement >= 0.55:
            unknowns.append("model_consensus")
        if state_confidence < 0.25:
            unknowns.append("state_confidence")

        previous = dict((self.state.get("current") or {}).get(symbol, {}))
        previous_regime = str(previous.get("regime") or "unknown")
        regime_changed = bool(previous) and previous_regime != regime
        transition_key = f"{previous_regime}->{regime}" if regime_changed else None
        if transition_key:
            transitions = self.state.setdefault("transitions", {})
            transitions[transition_key] = int(transitions.get(transition_key, 0)) + 1

        now = time.time()
        state = {
            "symbol": symbol,
            "timestamp": now,
            "regime": regime,
            "previous_regime": previous_regime if previous else None,
            "regime_changed": regime_changed,
            "price": float(close.iloc[-1]),
            "features": features,
            "adaptive": {"score": adaptive_score, "confidence": adaptive_confidence},
            "swarm": {"score": swarm_score, "confidence": swarm_confidence},
            "specialist_scores": {
                "technical_structure": _finite(technical.get("score")),
                "smart_scalping": _finite(scalping.get("score")),
                "spectral_harmonics": _finite(spectral.get("score")),
                "pattern_memory": _finite(pattern_memory.get("score")),
                "public_context": context_score,
            },
            "senses": {
                "novelty": novelty,
                "volatility_shock": volatility_shock,
                "volume_shock": volume_shock,
                "price_shock": price_shock,
                "liquidity_stress": liquidity_stress,
                "model_disagreement": model_disagreement,
                "timeframe_fracture": timeframe_fracture,
                "information_disorder": information_disorder,
                "rare_scope_score": rare_scope_score,
            },
            "timeframe_direction_balance": timeframe_direction_balance,
            "timeframe_coverage": _clip(timeframe_coverage),
            "data_quality": data_quality,
            "state_confidence": state_confidence,
            "knowledge_state": (
                "out_of_distribution"
                if novelty >= 0.70
                else "uncertain"
                if state_confidence < 0.35 or model_disagreement >= 0.55
                else "measured"
            ),
            "latent_patterns": latent_patterns,
            "unknowns": sorted(set(unknowns)),
            "execution_authority": False,
        }

        self.state.setdefault("current", {})[symbol] = state
        history = self.state.setdefault("history", {}).setdefault(symbol, [])
        history.append(
            {
                "timestamp": now,
                "regime": regime,
                "features": features,
                "senses": state["senses"],
                "latent_patterns": latent_patterns,
            }
        )
        self.state["history"][symbol] = history[-self.HISTORY_LIMIT :]
        if rare_scope_score >= 0.60 or latent_patterns:
            anomalies = self.state.setdefault("rare_scope_events", [])
            anomalies.append(
                {
                    "timestamp": now,
                    "symbol": symbol,
                    "score": rare_scope_score,
                    "patterns": latent_patterns,
                    "regime": regime,
                    "novelty": novelty,
                }
            )
            self.state["rare_scope_events"] = anomalies[-self.ANOMALY_LIMIT :]

        self.observations += 1
        self.state["observations"] = self.observations
        self.state["updated_at"] = now
        if self.observations % self.SAVE_INTERVAL == 0:
            self._save()
        return dict(state)

    def observe_market(self, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        returns: dict[str, pd.Series] = {}
        for symbol, frame in frames.items():
            close = self._series(frame, "close")
            if len(close) >= 32:
                series = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(64).reset_index(drop=True)
                if len(series) >= 16:
                    returns[symbol] = series

        average_abs_correlation = 0.0
        correlation_pairs: list[dict[str, Any]] = []
        lead_lag_candidates: list[dict[str, Any]] = []
        if len(returns) >= 2:
            matrix = pd.DataFrame(returns).corr(min_periods=12)
            symbols = list(matrix.columns)
            values: list[float] = []
            for i, left in enumerate(symbols):
                for right in symbols[i + 1 :]:
                    value = _finite(matrix.loc[left, right])
                    values.append(abs(value))
                    if abs(value) <= 0.20 or abs(value) >= 0.85:
                        correlation_pairs.append({"left": left, "right": right, "correlation": value})

                    left_values = returns[left].to_numpy(dtype=float)
                    right_values = returns[right].to_numpy(dtype=float)
                    length = min(len(left_values), len(right_values))
                    if length < 24:
                        continue
                    left_values = left_values[-length:]
                    right_values = right_values[-length:]
                    best: dict[str, Any] | None = None
                    for lag in range(1, min(5, length - 12)):
                        for leader, follower, leader_values, follower_values in (
                            (left, right, left_values, right_values),
                            (right, left, right_values, left_values),
                        ):
                            x = leader_values[:-lag]
                            y = follower_values[lag:]
                            if len(x) < 12 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
                                continue
                            lag_corr = _finite(np.corrcoef(x, y)[0, 1])
                            incremental = abs(lag_corr) - abs(value)
                            candidate = {
                                "leader": leader,
                                "follower": follower,
                                "lag_steps": lag,
                                "correlation": lag_corr,
                                "contemporaneous_correlation": value,
                                "incremental_strength": incremental,
                            }
                            if best is None or abs(lag_corr) > abs(float(best["correlation"])):
                                best = candidate
                    # This is a research anomaly only, never an execution signal.
                    # Require meaningful lagged association and material improvement
                    # over same-time correlation to avoid relabeling ordinary beta.
                    if (
                        best is not None
                        and abs(float(best["correlation"])) >= 0.45
                        and float(best["incremental_strength"]) >= 0.15
                    ):
                        lead_lag_candidates.append(best)
            average_abs_correlation = float(np.mean(values)) if values else 0.0
            correlation_pairs.sort(key=lambda row: abs(float(row["correlation"])), reverse=True)
            lead_lag_candidates.sort(
                key=lambda row: abs(float(row["correlation"])) + max(0.0, float(row["incremental_strength"])),
                reverse=True,
            )

        current_states = list((self.state.get("current") or {}).values())
        rare_scores = [_finite((row.get("senses") or {}).get("rare_scope_score")) for row in current_states]
        dispersion_inputs = [_finite((row.get("features") or {}).get("momentum_16")) for row in current_states]
        cross_sectional_dispersion = float(np.std(dispersion_inputs)) if len(dispersion_inputs) >= 2 else 0.0
        previous = dict(self.state.get("market") or {})
        previous_corr = _finite(previous.get("average_abs_correlation"))
        correlation_fracture = abs(average_abs_correlation - previous_corr) if previous else 0.0
        market = {
            "timestamp": time.time(),
            "symbols_modeled": len(current_states),
            "correlation_symbols": len(returns),
            "average_abs_correlation": average_abs_correlation,
            "correlation_fracture": correlation_fracture,
            "cross_sectional_dispersion": cross_sectional_dispersion,
            "rare_scope_symbols": sum(1 for score in rare_scores if score >= 0.60),
            "maximum_rare_scope_score": max(rare_scores, default=0.0),
            "extreme_correlation_pairs": correlation_pairs[:20],
            "lead_lag_research_candidates": lead_lag_candidates[:20],
            "rare_relationship_count": len(lead_lag_candidates),
            "relationship_discovery_is_not_trade_authority": True,
            "execution_authority": False,
        }
        self.market_snapshots += 1
        self.state["market_snapshots"] = self.market_snapshots
        self.state["market"] = market
        self.state["updated_at"] = market["timestamp"]
        self._save()
        return dict(market)

    def current(self, symbol: str) -> dict[str, Any]:
        return dict((self.state.get("current") or {}).get(symbol.upper(), {}))

    def research_candidates(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = list((self.state.get("current") or {}).values())
        rows.sort(key=lambda row: float((row.get("senses") or {}).get("rare_scope_score") or 0.0), reverse=True)
        return [
            {
                "symbol": row.get("symbol"),
                "regime": row.get("regime"),
                "knowledge_state": row.get("knowledge_state"),
                "rare_scope_score": (row.get("senses") or {}).get("rare_scope_score"),
                "latent_patterns": list(row.get("latent_patterns") or []),
                "unknowns": list(row.get("unknowns") or []),
            }
            for row in rows[: max(0, int(limit))]
        ]

    def health(self) -> dict[str, Any]:
        current = self.state.get("current") or {}
        rows = list(current.values())
        unknowns = sum(len(row.get("unknowns") or []) for row in rows)
        ood = sum(1 for row in rows if row.get("knowledge_state") == "out_of_distribution")
        return {
            "healthy": self.last_error is None,
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "observations": self.observations,
            "market_snapshots": self.market_snapshots,
            "symbols": len(current),
            "unknowns": unknowns,
            "out_of_distribution_symbols": ood,
            "rare_scope_events": len(self.state.get("rare_scope_events") or []),
            "market": dict(self.state.get("market") or {}),
            "execution_authority": False,
            "novelty_is_not_trade_authority": True,
            "state_path": str(self.state_path),
            "error": self.last_error,
        }

    def _load(self) -> dict[str, Any]:
        empty = {
            "schema_version": self.SCHEMA_VERSION,
            "observations": 0,
            "market_snapshots": 0,
            "current": {},
            "history": {},
            "transitions": {},
            "rare_scope_events": [],
            "market": {},
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
        self.state["market_snapshots"] = self.market_snapshots
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.state_path)
