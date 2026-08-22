from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Iterable, Mapping

import pandas as pd


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


@dataclass(frozen=True)
class MicrostructureFeatures:
    symbol: str
    timestamp: float
    midpoint: float
    spread_bps: float
    bid_depth_usd: float
    ask_depth_usd: float
    depth_imbalance: float
    microprice_shift_bps: float
    trade_imbalance: float
    trade_intensity_per_second: float
    short_momentum_bps: float
    realized_volatility_bps_1m: float
    q90_abs_move_bps: float
    cross_venue_basis_bps: float
    cross_venue_pressure: float
    liquidity_vacuum_score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MicroPathAssessment:
    symbol: str
    horizon_seconds: int
    direction: str
    specialist: str
    probability_favorable_first: float
    probability_adverse_first: float
    confidence: float
    path_budget_bps: float
    expected_edge_bps: float
    modeled_round_trip_cost_bps: float
    independently_qualified: bool
    reason: str
    pressure_score: float
    regime: str
    research_hypothesis: bool = True
    automatic_promotion: bool = False
    execution_authority: bool = False
    testnet_authority: bool = False
    live_authority: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class UltraMicrostructureSniper:
    VERSION = "1.45.0"
    HORIZONS = (5, 15, 30, 60)

    def __init__(
        self,
        *,
        minimum_modeled_round_trip_cost_bps: float = 30.0,
        minimum_confidence: float = 0.62,
        minimum_depth_usd: float = 10_000.0,
        maximum_spread_bps: float = 25.0,
        minimum_edge_buffer_bps: float = 2.0,
    ) -> None:
        if minimum_modeled_round_trip_cost_bps < 30.0:
            raise ValueError("microstructure cost floor cannot be below 30 bps")
        self.minimum_modeled_round_trip_cost_bps = float(minimum_modeled_round_trip_cost_bps)
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_depth_usd = float(minimum_depth_usd)
        self.maximum_spread_bps = float(maximum_spread_bps)
        self.minimum_edge_buffer_bps = float(minimum_edge_buffer_bps)
        self.assessments = 0
        self.qualified = 0
        self.rejected = 0

    @staticmethod
    def _depth(rows: Any, midpoint: float, levels: int = 10) -> float:
        total = 0.0
        for raw in list(rows or [])[:levels]:
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                continue
            price = _finite(raw[0])
            amount = _finite(raw[1])
            if price > 0 and amount > 0:
                total += price * amount
        return total

    @staticmethod
    def _trade_features(trades: Iterable[Mapping[str, Any]], now: float) -> tuple[float, float]:
        buy = sell = 0.0
        earliest = now
        count = 0
        for row in trades:
            ts = _finite(row.get("timestamp"))
            ts = ts / 1000.0 if ts > 10_000_000_000 else ts
            if ts > 0:
                earliest = min(earliest, ts)
            notional = max(0.0, _finite(row.get("price")) * _finite(row.get("amount")))
            side = str(row.get("side") or "").lower()
            if side == "buy":
                buy += notional
            elif side == "sell":
                sell += notional
            count += 1
        total = buy + sell
        imbalance = 0.0 if total <= 0 else (buy - sell) / total
        return _signed(imbalance), count / max(1.0, now - earliest)

    @staticmethod
    def _candle_features(frame: pd.DataFrame) -> tuple[float, float, float]:
        if not isinstance(frame, pd.DataFrame) or "close" not in frame.columns:
            return 0.0, 0.0, 0.0
        close = pd.to_numeric(frame["close"], errors="coerce")
        close = close[close.notna() & (close > 0)].astype(float)
        if len(close) < 3:
            return 0.0, 0.0, 0.0
        returns = close.pct_change().dropna() * 10_000.0
        returns = returns[returns.map(math.isfinite)]
        if returns.empty:
            return 0.0, 0.0, 0.0
        recent = returns.iloc[-min(30, len(returns)):]
        return (
            float(recent.iloc[-min(3, len(recent)):].sum()),
            float(recent.std(ddof=0)),
            float(recent.abs().quantile(0.90)),
        )

    def extract(
        self,
        *,
        symbol: str,
        order_book: Mapping[str, Any],
        trades: Iterable[Mapping[str, Any]],
        candles: pd.DataFrame,
        reference_order_book: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> MicrostructureFeatures:
        now = time.time() if now is None else float(now)
        bids = list(order_book.get("bids") or [])
        asks = list(order_book.get("asks") or [])
        if not bids or not asks:
            raise ValueError("microstructure requires non-empty book")
        bid = _finite(bids[0][0])
        ask = _finite(asks[0][0])
        if bid <= 0 or ask < bid:
            raise ValueError("invalid top of book")
        midpoint = (bid + ask) / 2.0
        spread = (ask - bid) / midpoint * 10_000.0

        bid_depth = self._depth(bids, midpoint)
        ask_depth = self._depth(asks, midpoint)
        total_depth = bid_depth + ask_depth
        depth_imbalance = 0.0 if total_depth <= 0 else (bid_depth - ask_depth) / total_depth

        bid_amount = _finite(bids[0][1])
        ask_amount = _finite(asks[0][1])
        top_total = bid_amount + ask_amount
        microprice = midpoint if top_total <= 0 else (ask * bid_amount + bid * ask_amount) / top_total
        micro_shift = (microprice - midpoint) / midpoint * 10_000.0

        trade_imbalance, intensity = self._trade_features(trades, now)
        momentum, vol, q90 = self._candle_features(candles)

        basis = reference_pressure = 0.0
        if reference_order_book:
            rb = list(reference_order_book.get("bids") or [])
            ra = list(reference_order_book.get("asks") or [])
            if rb and ra:
                ref_bid = _finite(rb[0][0])
                ref_ask = _finite(ra[0][0])
                if ref_bid > 0 and ref_ask >= ref_bid:
                    ref_mid = (ref_bid + ref_ask) / 2.0
                    basis = (midpoint - ref_mid) / midpoint * 10_000.0
                    rbd = self._depth(rb, ref_mid)
                    rad = self._depth(ra, ref_mid)
                    rt = rbd + rad
                    reference_pressure = 0.0 if rt <= 0 else (rbd - rad) / rt

        depth_quality = min(1.0, total_depth / max(1.0, self.minimum_depth_usd * 4.0))
        vacuum = _unit((1.0 - depth_quality) * (0.5 + 0.5 * abs(depth_imbalance)))

        return MicrostructureFeatures(
            symbol=str(symbol).upper(),
            timestamp=now,
            midpoint=midpoint,
            spread_bps=spread,
            bid_depth_usd=bid_depth,
            ask_depth_usd=ask_depth,
            depth_imbalance=_signed(depth_imbalance),
            microprice_shift_bps=micro_shift,
            trade_imbalance=trade_imbalance,
            trade_intensity_per_second=max(0.0, intensity),
            short_momentum_bps=momentum,
            realized_volatility_bps_1m=max(0.0, vol),
            q90_abs_move_bps=max(0.0, q90),
            cross_venue_basis_bps=basis,
            cross_venue_pressure=_signed(reference_pressure),
            liquidity_vacuum_score=vacuum,
        )

    @staticmethod
    def _logistic(x: float) -> float:
        x = max(-20.0, min(20.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def assess(self, features: MicrostructureFeatures, *, modeled_round_trip_cost_bps: float) -> list[MicroPathAssessment]:
        cost = max(self.minimum_modeled_round_trip_cost_bps, float(modeled_round_trip_cost_bps))
        q90 = features.q90_abs_move_bps
        vol = features.realized_volatility_bps_1m
        momentum = _signed(features.short_momentum_bps / max(30.0, q90 * 2.0, vol * 2.0, 1.0))
        micro = _signed(features.microprice_shift_bps / 8.0)
        basis = _signed(features.cross_venue_basis_bps / 20.0)

        pressure = _signed(
            0.34 * features.depth_imbalance
            + 0.28 * features.trade_imbalance
            + 0.14 * micro
            + 0.12 * momentum
            + 0.08 * features.cross_venue_pressure
            - 0.04 * basis
        )

        rows = []
        for horizon in self.HORIZONS:
            scale = math.sqrt(horizon / 60.0)
            path_budget = min(q90, max(abs(features.microprice_shift_bps) * 2.0, vol * scale, q90 * scale))
            strength = abs(pressure)
            favorable = max(0.50, min(0.95, self._logistic(3.0 * strength * math.sqrt(horizon / 15.0))))
            confidence = _unit((favorable - 0.50) * 2.0)
            expected_edge = path_budget * strength * confidence
            direction = "long" if pressure > 0 else ("short" if pressure < 0 else "flat")

            specialist = "orderflow_pressure"
            if features.liquidity_vacuum_score >= 0.65 and strength >= 0.35:
                specialist = "liquidity_vacuum_sniper"
            elif features.trade_intensity_per_second >= 1.5 and strength >= 0.45:
                specialist = "micro_burst_hunter"
            elif momentum * pressure < -0.12 and strength >= 0.35:
                specialist = "reversal_snapper"
            elif abs(features.cross_venue_pressure) >= 0.35:
                specialist = "cross_venue_leadlag"

            regime = "micro_trend" if momentum * pressure > 0.10 else "micro_reversal" if momentum * pressure < -0.10 else "micro_balanced"

            qualified = True
            reason = "qualified"
            if direction == "flat":
                qualified, reason = False, "flat_micro_pressure"
            elif features.spread_bps > self.maximum_spread_bps:
                qualified, reason = False, "spread_too_wide"
            elif features.bid_depth_usd + features.ask_depth_usd < self.minimum_depth_usd:
                qualified, reason = False, "insufficient_depth"
            elif confidence < self.minimum_confidence:
                qualified, reason = False, "micro_confidence_below_threshold"
            elif expected_edge <= cost + self.minimum_edge_buffer_bps:
                qualified, reason = False, "micro_edge_does_not_clear_cost_buffer"

            row = MicroPathAssessment(
                symbol=features.symbol,
                horizon_seconds=horizon,
                direction=direction,
                specialist=specialist,
                probability_favorable_first=favorable,
                probability_adverse_first=1.0 - favorable,
                confidence=confidence,
                path_budget_bps=path_budget,
                expected_edge_bps=expected_edge,
                modeled_round_trip_cost_bps=cost,
                independently_qualified=qualified,
                reason=reason,
                pressure_score=pressure,
                regime=regime,
            )
            self.assessments += 1
            if qualified:
                self.qualified += 1
            else:
                self.rejected += 1
            rows.append(row)
        return rows

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "horizons_seconds": list(self.HORIZONS),
            "assessments": self.assessments,
            "qualified": self.qualified,
            "rejected": self.rejected,
            "minimum_modeled_round_trip_cost_bps": self.minimum_modeled_round_trip_cost_bps,
            "short_path_distribution": True,
            "order_book_pressure": True,
            "public_trade_pressure": True,
            "cross_venue_reference": True,
            "predictive_profit_claim": False,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }


class MicroAgentFoundry:
    VERSION = "1.0"

    def __init__(self, maximum_candidates_per_symbol: int = 2) -> None:
        self.maximum_candidates_per_symbol = max(1, int(maximum_candidates_per_symbol))
        self.proposals = 0

    def propose(self, assessments: Iterable[MicroPathAssessment]) -> list[dict[str, Any]]:
        rows = [row for row in assessments if row.independently_qualified]
        rows.sort(
            key=lambda row: (
                row.expected_edge_bps - row.modeled_round_trip_cost_bps,
                row.confidence,
                -row.horizon_seconds,
            ),
            reverse=True,
        )
        output = []
        for row in rows[: self.maximum_candidates_per_symbol]:
            self.proposals += 1
            output.append(
                {
                    "candidate_kind": "microstructure_shadow_specialist",
                    "specialist": row.specialist,
                    "symbol": row.symbol,
                    "timeframe": f"micro-{row.horizon_seconds}s-{row.specialist}",
                    "horizon_seconds": row.horizon_seconds,
                    "side": row.direction,
                    "confidence": row.confidence,
                    "expected_edge_bps": row.expected_edge_bps,
                    "modeled_round_trip_cost_bps": row.modeled_round_trip_cost_bps,
                    "regime": row.regime,
                    "independently_qualified": True,
                    "automatic_promotion": False,
                    "execution_authority": False,
                    "testnet_authority": False,
                    "live_authority": False,
                    "can_increase_risk": False,
                }
            )
        return output

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "proposals": self.proposals,
            "maximum_candidates_per_symbol": self.maximum_candidates_per_symbol,
            "bounded_specialist_generation": True,
            "executable_code_generation": False,
            "automatic_promotion": False,
            "parameter_mutation_authority": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "can_increase_risk": False,
        }
