from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineSignal:
    engine: str
    score: float
    confidence: float
    rationale: str


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


def _zscore(value: float, values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    std = float(np.std(array))
    return 0.0 if std <= 1e-12 else float((value - np.mean(array)) / std)


class SmartScalpingEngine:
    """Fast technical signal with explicit spread and liquidity penalties."""

    VERSION = "1.0"

    def evaluate(self, frame: pd.DataFrame, spread_bps: float = 0.0) -> EngineSignal:
        if len(frame) < 60:
            raise ValueError("smart scalping requires at least 60 candles")
        close = pd.to_numeric(frame["close"], errors="raise")
        volume = pd.to_numeric(frame["volume"], errors="raise")
        ema_fast = close.ewm(span=5, adjust=False).mean()
        ema_slow = close.ewm(span=13, adjust=False).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rsi = float(100 - 100 / (1 + gain.iloc[-1] / max(float(loss.iloc[-1]), 1e-12)))
        trend = float(np.tanh((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / max(close.iloc[-1] * 0.001, 1e-9)))
        rsi_score = _clip((rsi - 50.0) / 25.0)
        volume_ratio = float(volume.iloc[-1] / max(float(volume.rolling(30).median().iloc[-1]), 1e-9))
        volume_score = _clip((volume_ratio - 1.0) / 2.0, 0.0, 1.0)
        spread_penalty = _clip(spread_bps / 30.0, 0.0, 1.0)
        score = _clip(0.55 * trend + 0.25 * rsi_score + 0.20 * volume_score)
        confidence = _clip((0.45 + 0.35 * abs(score) + 0.20 * volume_score) * (1 - spread_penalty), 0.0, 1.0)
        return EngineSignal(
            "smart_scalping",
            score,
            confidence,
            f"rsi={rsi:.1f},volume_ratio={volume_ratio:.2f},spread_bps={spread_bps:.1f}",
        )

    def health(self) -> dict[str, Any]:
        return {"deterministic": True, "spread_aware": True}


class TechnicalStructureEngine:
    """Deterministic MACD/ADX/Stochastic/OBV and liquidity-sweep confirmation."""

    VERSION = "1.0"

    def evaluate(self, frame: pd.DataFrame) -> EngineSignal:
        if len(frame) < 60:
            raise ValueError("technical structure requires at least 60 candles")
        close = pd.to_numeric(frame["close"], errors="raise")
        high = pd.to_numeric(frame["high"], errors="raise")
        low = pd.to_numeric(frame["low"], errors="raise")
        volume = pd.to_numeric(frame["volume"], errors="raise")

        previous_close = close.shift(1)
        true_range = pd.concat(
            [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
        ).max(axis=1)
        atr = true_range.ewm(alpha=1 / 14, adjust=False).mean()
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = float(dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        adx = adx if math.isfinite(adx) else 0.0

        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        macd_histogram = float((macd - macd.ewm(span=9, adjust=False).mean()).iloc[-1])
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        stochastic = float(100 * (close.iloc[-1] - low_14.iloc[-1]) / max(high_14.iloc[-1] - low_14.iloc[-1], 1e-12))
        obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
        obv_change = float(obv.iloc[-1] - obv.iloc[-10]) / max(float(volume.tail(10).sum()), 1e-12)

        prior_high = float(high.shift(1).rolling(20).max().iloc[-1])
        prior_low = float(low.shift(1).rolling(20).min().iloc[-1])
        high_sweep = bool(high.iloc[-1] > prior_high and close.iloc[-1] < prior_high)
        low_sweep = bool(low.iloc[-1] < prior_low and close.iloc[-1] > prior_low)
        plus_value = float(plus_di.iloc[-1])
        minus_value = float(minus_di.iloc[-1])
        plus_value = plus_value if math.isfinite(plus_value) else 0.0
        minus_value = minus_value if math.isfinite(minus_value) else 0.0
        direction = _clip((plus_value - minus_value) / 50.0)
        trend_strength = _clip(adx / 35.0, 0.0, 1.0)
        macd_score = float(np.tanh(macd_histogram / max(float(atr.iloc[-1]), 1e-12)))
        stochastic_score = _clip((stochastic - 50.0) / 50.0)
        sweep_score = 1.0 if low_sweep else (-1.0 if high_sweep else 0.0)
        score = _clip(
            0.35 * macd_score
            + 0.30 * direction * trend_strength
            + 0.15 * stochastic_score
            + 0.10 * _clip(obv_change * 4.0)
            + 0.10 * sweep_score
        )
        confidence = _clip(0.35 + 0.35 * trend_strength + 0.20 * abs(score) + 0.10 * abs(sweep_score), 0.0, 1.0)
        return EngineSignal(
            "technical_structure",
            score,
            confidence,
            (
                f"adx={adx:.1f},stochastic={stochastic:.1f},obv_change={obv_change:.3f},"
                f"high_sweep={high_sweep},low_sweep={low_sweep}"
            ),
        )

    def health(self) -> dict[str, Any]:
        return {
            "indicators": ["macd", "adx", "stochastic", "obv", "liquidity_sweeps"],
            "lookahead": False,
            "execution_authority": False,
        }


class SpectralHarmonicsEngine:
    """Measured cycle detector replacing fictional frequency/ultrasonic signals."""

    VERSION = "1.0"

    def evaluate(self, frame: pd.DataFrame) -> EngineSignal:
        if len(frame) < 128:
            raise ValueError("spectral analysis requires at least 128 candles")
        close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=float)
        log_price = np.log(close[-256:])
        x = log_price - np.linspace(log_price[0], log_price[-1], len(log_price))
        spectrum = np.fft.rfft(x)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(len(x))
        valid = (frequencies > 0) & (frequencies <= 0.2)
        if not valid.any() or float(power[valid].sum()) <= 1e-12:
            return EngineSignal("spectral_harmonics", 0.0, 0.0, "no stable cycle")
        valid_indices = np.where(valid)[0]
        peak_index = int(valid_indices[np.argmax(power[valid])])
        period = float(1.0 / frequencies[peak_index])
        concentration = float(power[peak_index] / power[valid].sum())
        recent_return = float(close[-1] / close[-min(len(close), max(2, int(period / 4)))] - 1.0)
        score = _clip(np.tanh(recent_return * 100.0) * concentration * 4.0)
        confidence = _clip(concentration * 4.0, 0.0, 1.0)
        return EngineSignal(
            "spectral_harmonics", score, confidence, f"dominant_period={period:.1f},concentration={concentration:.3f}"
        )

    def health(self) -> dict[str, Any]:
        return {"method": "detrended_fft", "random_inputs": False}


class LiquidityFluidEngine:
    """Order-book pressure, spread, fill impact, and safe-size calculations."""

    VERSION = "1.0"

    def evaluate(self, order_book: dict[str, Any], desired_qty: float = 0.0) -> dict[str, float]:
        bids = [(float(p), float(q)) for p, q, *_ in order_book.get("bids", []) if float(p) > 0 and float(q) > 0]
        asks = [(float(p), float(q)) for p, q, *_ in order_book.get("asks", []) if float(p) > 0 and float(q) > 0]
        if not bids or not asks:
            raise ValueError("two-sided order book required")
        best_bid, best_ask = max(p for p, _ in bids), min(p for p, _ in asks)
        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 10_000
        bid_depth = sum(q for _, q in sorted(bids, reverse=True)[:10])
        ask_depth = sum(q for _, q in sorted(asks)[:10])
        imbalance = (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-12)
        impact_bps = self._walk_impact(sorted(asks), desired_qty, mid) if desired_qty > 0 else 0.0
        safe_qty = self._safe_qty(sorted(asks), mid, 30.0)
        return {
            "mid": mid,
            "spread_bps": spread_bps,
            "imbalance": imbalance,
            "buy_impact_bps": impact_bps,
            "safe_buy_qty_30bps": safe_qty,
        }

    @staticmethod
    def _walk_impact(levels: list[tuple[float, float]], quantity: float, mid: float) -> float:
        remaining, cost, filled = quantity, 0.0, 0.0
        for price, available in levels:
            take = min(remaining, available)
            cost += take * price
            filled += take
            remaining -= take
            if remaining <= 1e-12:
                break
        if filled <= 0 or remaining > 1e-12:
            return math.inf
        return (cost / filled - mid) / mid * 10_000

    @staticmethod
    def _safe_qty(levels: list[tuple[float, float]], mid: float, cap_bps: float) -> float:
        quantity = 0.0
        for price, available in levels:
            if (price - mid) / mid * 10_000 > cap_bps:
                break
            quantity += available
        return quantity

    def health(self) -> dict[str, Any]:
        return {"depth_levels": 10, "impact_cap_bps": 30.0}


class NewsAwarenessEngine:
    """Local, auditable news/event state with sentiment decay and blackout gates."""

    VERSION = "1.0"
    POSITIVE: ClassVar[set[str]] = {
        "beat",
        "bullish",
        "approval",
        "upgrade",
        "growth",
        "surge",
        "adoption",
        "cut",
    }
    NEGATIVE: ClassVar[set[str]] = {
        "miss",
        "bearish",
        "ban",
        "hack",
        "downgrade",
        "crash",
        "lawsuit",
        "hike",
    }

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path

    def evaluate(self, symbol: str, now: dt.datetime | None = None) -> dict[str, Any]:
        current = now or dt.datetime.now(dt.UTC)
        score, weight, blackout = 0.0, 0.0, False
        matched = 0
        for item in self._items():
            try:
                timestamp = dt.datetime.fromisoformat(str(item["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=dt.UTC)
            tokens = set(re.findall(r"[a-z0-9]+", str(item.get("title", "")).lower()))
            symbols = {str(value).upper().replace("/", "") for value in item.get("symbols", [])}
            normalized = symbol.upper().replace("/", "").replace("_", "")
            if symbols and normalized not in symbols and normalized[:3] not in symbols:
                continue
            age_minutes = (current - timestamp).total_seconds() / 60.0
            if -30 <= age_minutes <= 30 and str(item.get("impact", "")).lower() == "high":
                blackout = True
            if age_minutes < 0 or age_minutes > 1_440:
                continue
            sentiment = (len(tokens & self.POSITIVE) - len(tokens & self.NEGATIVE)) / max(len(tokens), 1)
            decay = math.exp(-age_minutes / 180.0)
            score += sentiment * decay
            weight += decay
            matched += 1
        return {
            "sentiment": _clip(score / weight) if weight else 0.0,
            "confidence": _clip(weight / 3.0, 0.0, 1.0),
            "blackout": blackout,
            "matched_items": matched,
        }

    def ingest(self, items: list[dict[str, Any]], max_items: int = 5_000) -> int:
        """Validate, deduplicate, and atomically persist items from external collectors."""
        merged = self._items()
        known = {
            (str(item.get("timestamp", "")), str(item.get("title", "")), str(item.get("source", ""))) for item in merged
        }
        added = 0
        for item in items:
            try:
                dt.datetime.fromisoformat(str(item["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            clean = {
                "timestamp": str(item["timestamp"]),
                "title": title,
                "source": str(item.get("source", "unknown")),
                "symbols": [str(value).upper().replace("/", "") for value in item.get("symbols", [])],
                "impact": str(item.get("impact", "unknown")).lower(),
            }
            key = (clean["timestamp"], clean["title"], clean["source"])
            if key in known:
                continue
            known.add(key)
            merged.append(clean)
            added += 1
        merged = sorted(merged, key=lambda row: str(row.get("timestamp", "")))[-max_items:]
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps({"schema_version": 1, "items": merged}, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_path)
        return added

    def _items(self) -> list[dict[str, Any]]:
        if not self.state_path.exists():
            return []
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else list(payload.get("items", []))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []

    def health(self) -> dict[str, Any]:
        return {"state_path": str(self.state_path), "items": len(self._items()), "outbound_network": False}


class PatternMemoryEngine:
    """Outcome-labelled nearest-pattern memory; never invents observations."""

    VERSION = "1.0"

    def __init__(self, state_path: Path, max_records: int = 5_000) -> None:
        self.state_path = state_path
        self.max_records = max_records
        self.records = self._load()

    def remember(self, features: dict[str, float], outcome: float, label: str = "") -> None:
        clean = {key: float(value) for key, value in sorted(features.items()) if math.isfinite(float(value))}
        if not clean:
            return
        self.records.append({"features": clean, "outcome": _clip(float(outcome)), "label": label})
        self.records = self.records[-self.max_records :]
        self._save()

    def recall(self, features: dict[str, float], neighbors: int = 20) -> EngineSignal:
        clean = {key: float(value) for key, value in features.items() if math.isfinite(float(value))}
        candidates: list[tuple[float, float]] = []
        for record in self.records:
            stored = record.get("features", {})
            shared = sorted(set(clean) & set(stored))
            if not shared:
                continue
            distance = math.sqrt(sum((clean[key] - float(stored[key])) ** 2 for key in shared) / len(shared))
            candidates.append((distance, float(record.get("outcome", 0.0))))
        nearest = sorted(candidates)[:neighbors]
        if len(nearest) < 5:
            return EngineSignal("pattern_memory", 0.0, 0.0, f"insufficient_neighbors={len(nearest)}")
        weights = [1.0 / (distance + 1e-6) for distance, _ in nearest]
        score = sum(weight * outcome for weight, (_, outcome) in zip(weights, nearest, strict=True)) / sum(weights)
        confidence = _clip(len(nearest) / neighbors, 0.0, 1.0)
        return EngineSignal("pattern_memory", _clip(score), confidence, f"neighbors={len(nearest)}")

    def _load(self) -> list[dict[str, Any]]:
        if not self.state_path.exists():
            return []
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return list(payload.get("records", []))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return []

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"schema_version": 1, "records": self.records}, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, self.state_path)

    def health(self) -> dict[str, Any]:
        return {"records": len(self.records), "max_records": self.max_records, "invented_samples": False}


class SwarmConsensusEngine:
    """Confidence-weighted consensus with a disagreement penalty."""

    VERSION = "1.0"

    def combine(self, signals: list[EngineSignal]) -> EngineSignal:
        eligible = [signal for signal in signals if signal.confidence > 0]
        if not eligible:
            return EngineSignal("swarm_hivemind", 0.0, 0.0, "no eligible engines")
        weights = np.asarray([signal.confidence for signal in eligible], dtype=float)
        scores = np.asarray([signal.score for signal in eligible], dtype=float)
        consensus = float(np.average(scores, weights=weights))
        disagreement = float(np.sqrt(np.average((scores - consensus) ** 2, weights=weights)))
        confidence = _clip(float(np.mean(weights)) * (1.0 - min(1.0, disagreement)), 0.0, 1.0)
        return EngineSignal(
            "swarm_hivemind", _clip(consensus), confidence, f"engines={len(eligible)},disagreement={disagreement:.3f}"
        )

    def health(self) -> dict[str, Any]:
        return {"consensus": "confidence_weighted", "disagreement_penalty": True}


class MoonScoutEngine:
    """Cross-sectional anomaly ranking from real price and volume frames."""

    VERSION = "1.0"

    def rank(self, frames: dict[str, pd.DataFrame]) -> list[dict[str, float | str]]:
        raw: dict[str, dict[str, float]] = {}
        for symbol, frame in frames.items():
            if len(frame) < 60:
                continue
            close = pd.to_numeric(frame["close"], errors="coerce")
            volume = pd.to_numeric(frame["volume"], errors="coerce")
            if close.isna().any() or volume.isna().any():
                continue
            raw[symbol] = {
                "momentum": float(close.iloc[-1] / close.iloc[-25] - 1.0),
                "volume_spike": float(volume.iloc[-1] / max(float(volume.rolling(50).median().iloc[-1]), 1e-9)),
                "volatility": float(close.pct_change().tail(50).std(ddof=0)),
                "liquidity": float(close.iloc[-1] * volume.tail(20).median()),
            }
        if not raw:
            return []
        fields = tuple(next(iter(raw.values())))
        ranked = []
        for symbol, values in raw.items():
            z = {field: _zscore(values[field], [row[field] for row in raw.values()]) for field in fields}
            score = (
                0.40 * z["momentum"]
                + 0.35 * z["volume_spike"]
                + 0.20 * z["liquidity"]
                - 0.25 * max(0.0, z["volatility"])
            )
            ranked.append({"symbol": symbol, "score": float(score), **values})
        return sorted(ranked, key=lambda row: float(row["score"]), reverse=True)

    def health(self) -> dict[str, Any]:
        return {"cross_sectional": True, "random_opportunities": False}


class PortfolioRiskEngine:
    """Correlation, concentration, and parametric portfolio VaR."""

    VERSION = "1.0"

    def analyze(self, frames: dict[str, pd.DataFrame], notionals: dict[str, float]) -> dict[str, Any]:
        symbols = [symbol for symbol in notionals if symbol in frames and len(frames[symbol]) >= 30]
        total = sum(abs(notionals[symbol]) for symbol in symbols)
        if not symbols or total <= 0:
            return {"var_95_usd": 0.0, "max_correlation": 0.0, "concentration": 0.0, "symbols": []}
        returns = pd.DataFrame(
            {symbol: pd.to_numeric(frames[symbol]["close"], errors="coerce").pct_change() for symbol in symbols}
        ).dropna()
        weights = np.asarray([notionals[symbol] / total for symbol in symbols], dtype=float)
        covariance = returns.cov().to_numpy(dtype=float)
        variance = max(0.0, float(weights @ covariance @ weights))
        correlation = returns.corr().to_numpy(dtype=float)
        off_diagonal = correlation[~np.eye(len(symbols), dtype=bool)]
        return {
            "var_95_usd": float(1.645 * math.sqrt(variance) * total),
            "max_correlation": float(np.max(np.abs(off_diagonal))) if off_diagonal.size else 0.0,
            "concentration": float(np.max(np.abs(weights))),
            "symbols": symbols,
        }

    def health(self) -> dict[str, Any]:
        return {"var_confidence": 0.95, "correlation_aware": True}


class ArbitrageEngine:
    """Net executable spread detector; consumes observed venue quotes only."""

    VERSION = "1.0"

    def scan(self, quotes: list[dict[str, Any]], minimum_net_bps: float = 5.0) -> list[dict[str, Any]]:
        opportunities: list[dict[str, Any]] = []
        for symbol in sorted({str(quote["symbol"]) for quote in quotes}):
            rows = [quote for quote in quotes if str(quote["symbol"]) == symbol]
            for buy in rows:
                for sell in rows:
                    if buy["venue"] == sell["venue"]:
                        continue
                    ask, bid = float(buy["ask"]), float(sell["bid"])
                    if ask <= 0 or bid <= ask:
                        continue
                    gross_bps = (bid - ask) / ask * 10_000
                    costs = (
                        float(buy.get("fee_bps", 0))
                        + float(sell.get("fee_bps", 0))
                        + float(buy.get("slippage_bps", 0))
                        + float(sell.get("slippage_bps", 0))
                    )
                    net_bps = gross_bps - costs
                    if net_bps >= minimum_net_bps:
                        opportunities.append(
                            {
                                "symbol": symbol,
                                "buy_venue": buy["venue"],
                                "sell_venue": sell["venue"],
                                "net_bps": net_bps,
                                "max_quantity": min(
                                    float(buy.get("ask_quantity", 0)), float(sell.get("bid_quantity", 0))
                                ),
                            }
                        )
        return sorted(opportunities, key=lambda row: row["net_bps"], reverse=True)

    def health(self) -> dict[str, Any]:
        return {"cost_adjusted": True, "execution_authority": False}


class BusinessPerformanceEngine:
    """Operational KPIs from realized events, separate from signal generation."""

    VERSION = "1.0"

    def summarize(self, events: list[dict[str, Any]]) -> dict[str, float]:
        pnls = [float(event["realized_pnl"]) for event in events if "realized_pnl" in event]
        wins = [value for value in pnls if value > 0]
        losses = [value for value in pnls if value < 0]
        return {
            "closed_trades": float(len(pnls)),
            "win_rate": len(wins) / len(pnls) if pnls else 0.0,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else (math.inf if wins else 0.0),
            "expectancy": float(np.mean(pnls)) if pnls else 0.0,
        }

    def health(self) -> dict[str, Any]:
        return {"signal_authority": False, "profit_claims": False}


class UltraEngineSuite:
    """Canonical, deterministic replacements for the legacy ultra concepts."""

    VERSION = "2.0"

    def __init__(self, memory_path: Path, news_path: Path) -> None:
        self.scalping = SmartScalpingEngine()
        self.technical_structure = TechnicalStructureEngine()
        self.spectral = SpectralHarmonicsEngine()
        self.liquidity = LiquidityFluidEngine()
        self.news = NewsAwarenessEngine(news_path)
        self.memory = PatternMemoryEngine(memory_path)
        self.swarm = SwarmConsensusEngine()
        self.moon_scout = MoonScoutEngine()
        self.portfolio = PortfolioRiskEngine()
        self.arbitrage = ArbitrageEngine()
        self.business = BusinessPerformanceEngine()

    def evaluate_symbol(self, symbol: str, frame: pd.DataFrame) -> dict[str, Any]:
        signals = [
            self.scalping.evaluate(frame),
            self.technical_structure.evaluate(frame),
            self.spectral.evaluate(frame),
        ]
        news = self.news.evaluate(symbol)
        signals.append(
            EngineSignal(
                "news_awareness",
                news["sentiment"],
                news["confidence"],
                f"items={news['matched_items']},blackout={news['blackout']}",
            )
        )
        feature_vector = {signal.engine: signal.score for signal in signals}
        signals.append(self.memory.recall(feature_vector))
        swarm = self.swarm.combine(signals)
        return {
            "signals": [asdict(signal) for signal in signals],
            "swarm": asdict(swarm),
            "news_blackout": news["blackout"],
            "feature_vector": feature_vector,
        }

    def learn(self, metadata: dict[str, Any], realized_return: float, label: str = "") -> None:
        features = metadata.get("advanced_feature_vector")
        if isinstance(features, dict):
            self.memory.remember(features, realized_return / 0.02, label)

    def market_snapshot(
        self,
        frames: dict[str, pd.DataFrame],
        notionals: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "moon_scout_ranking": self.moon_scout.rank(frames),
            "portfolio_risk": self.portfolio.analyze(frames, notionals),
        }

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "legacy_random_engines_loaded": False,
            "capabilities": {
                "smart_scalping": self.scalping.health(),
                "technical_structure": self.technical_structure.health(),
                "frequency_harmonics_ultrasonic": self.spectral.health(),
                "fluid_liquidity": self.liquidity.health(),
                "news_awareness": self.news.health(),
                "photographic_pattern_memory": self.memory.health(),
                "swarm_hivemind": self.swarm.health(),
                "moon_scout_dynamic_scanner": self.moon_scout.health(),
                "portfolio_hedge_fund_arsenal": self.portfolio.health(),
                "arbitrage": self.arbitrage.health(),
                "business_performance": self.business.health(),
            },
        }
