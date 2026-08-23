from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import math
import threading
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
    depth_imbalance_velocity: float
    microprice_velocity_bps_per_second: float
    spread_velocity_bps_per_second: float
    trade_imbalance_velocity: float
    pressure_persistence: float
    temporal_samples: int
    midpoint_velocity_bps_per_second: float = 0.0
    midpoint_acceleration_bps_per_second2: float = 0.0
    total_depth_change_fraction_per_second: float = 0.0
    recent_midpoint_range_bps_5s: float = 0.0
    recent_midpoint_trend_bps_5s: float = 0.0

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
    VERSION = "1.54.0"
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
        self.rejection_reasons: dict[str, int] = {}
        self._history: dict[str, deque[dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=64)
        )
        self._history_lock = threading.RLock()

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

    def _temporal_features(
        self,
        *,
        symbol: str,
        timestamp: float,
        midpoint: float,
        spread_bps: float,
        total_depth_usd: float,
        depth_imbalance: float,
        microprice_shift_bps: float,
        trade_imbalance: float,
        trade_observed: bool = True,
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        int,
        float,
        float,
        float,
        float,
        float,
    ]:
        key = str(symbol).upper()

        with self._history_lock:
            history = self._history[key]

            while history:
                oldest = float(
                    history[0].get("timestamp") or 0.0
                )
                if timestamp - oldest <= 90.0:
                    break
                history.popleft()

            depth_velocity = 0.0
            micro_velocity = 0.0
            spread_velocity = 0.0
            trade_velocity = 0.0

            midpoint_velocity = 0.0
            midpoint_acceleration = 0.0
            depth_change_fraction = 0.0

            if history:
                previous = history[-1]
                dt = max(
                    0.25,
                    timestamp
                    - float(previous["timestamp"]),
                )

                depth_velocity = (
                    depth_imbalance
                    - float(previous["depth_imbalance"])
                ) / dt

                micro_velocity = (
                    microprice_shift_bps
                    - float(
                        previous[
                            "microprice_shift_bps"
                        ]
                    )
                ) / dt

                spread_velocity = (
                    spread_bps
                    - float(previous["spread_bps"])
                ) / dt

                previous_midpoint = float(
                    previous.get("midpoint") or 0.0
                )

                if previous_midpoint > 0:
                    midpoint_return_bps = (
                        midpoint
                        / previous_midpoint
                        - 1.0
                    ) * 10_000.0

                    midpoint_velocity = (
                        midpoint_return_bps / dt
                    )

                previous_velocity = float(
                    previous.get(
                        "midpoint_velocity_bps_per_second",
                        0.0,
                    )
                    or 0.0
                )

                midpoint_acceleration = (
                    midpoint_velocity
                    - previous_velocity
                ) / dt

                previous_depth = float(
                    previous.get(
                        "total_depth_usd",
                        0.0,
                    )
                    or 0.0
                )

                if previous_depth > 0:
                    depth_change_fraction = (
                        (
                            total_depth_usd
                            - previous_depth
                        )
                        / previous_depth
                        / dt
                    )

            effective_trade_imbalance = (
                trade_imbalance
            )

            if not trade_observed and history:
                effective_trade_imbalance = float(
                    history[-1].get(
                        "trade_imbalance",
                        trade_imbalance,
                    )
                )

            if trade_observed:
                previous_trade = next(
                    (
                        row
                        for row in reversed(history)
                        if row.get(
                            "trade_observed",
                            True,
                        ) is True
                    ),
                    None,
                )

                if previous_trade is not None:
                    trade_dt = max(
                        0.25,
                        timestamp
                        - float(
                            previous_trade[
                                "timestamp"
                            ]
                        ),
                    )

                    trade_velocity = (
                        effective_trade_imbalance
                        - float(
                            previous_trade[
                                "trade_imbalance"
                            ]
                        )
                    ) / trade_dt

            current_pressure = _signed(
                0.45 * depth_imbalance
                + 0.35
                * effective_trade_imbalance
                + 0.20
                * _signed(
                    microprice_shift_bps / 8.0
                )
            )

            recent_pressures = [
                float(row["pressure"])
                for row in list(history)[-5:]
            ]

            if recent_pressures:
                same_sign = [
                    1.0
                    if value * current_pressure > 0
                    else 0.0
                    for value in recent_pressures
                    if abs(value) > 1e-9
                ]

                persistence = (
                    sum(same_sign)
                    / len(same_sign)
                    if same_sign
                    else 0.0
                )
            else:
                persistence = 0.0

            recent_rows = [
                row
                for row in history
                if (
                    timestamp
                    - float(
                        row.get("timestamp")
                        or 0.0
                    )
                    <= 5.0
                )
            ]

            recent_midpoints = [
                float(row.get("midpoint") or 0.0)
                for row in recent_rows
                if float(
                    row.get("midpoint") or 0.0
                ) > 0
            ]
            recent_midpoints.append(midpoint)

            recent_range_bps = 0.0
            recent_trend_bps = 0.0

            if len(recent_midpoints) >= 2:
                recent_range_bps = (
                    (
                        max(recent_midpoints)
                        - min(recent_midpoints)
                    )
                    / midpoint
                    * 10_000.0
                )

                first_midpoint = (
                    recent_midpoints[0]
                )

                if first_midpoint > 0:
                    recent_trend_bps = (
                        midpoint
                        / first_midpoint
                        - 1.0
                    ) * 10_000.0

            history.append(
                {
                    "timestamp": timestamp,
                    "midpoint": midpoint,
                    "spread_bps": spread_bps,
                    "total_depth_usd": (
                        total_depth_usd
                    ),
                    "depth_imbalance": (
                        depth_imbalance
                    ),
                    "microprice_shift_bps": (
                        microprice_shift_bps
                    ),
                    "midpoint_velocity_bps_per_second": (
                        midpoint_velocity
                    ),
                    "trade_imbalance": (
                        effective_trade_imbalance
                    ),
                    "trade_observed": (
                        trade_observed
                    ),
                    "pressure": current_pressure,
                }
            )

            return (
                depth_velocity,
                micro_velocity,
                spread_velocity,
                trade_velocity,
                persistence,
                len(history),
                midpoint_velocity,
                midpoint_acceleration,
                depth_change_fraction,
                recent_range_bps,
                recent_trend_bps,
            )

    def observe_snapshot(
        self,
        *,
        symbol: str,
        order_book: Mapping[str, Any],
        trades: Iterable[Mapping[str, Any]] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Update temporal microstructure state without creating a signal.

        This is read-only research telemetry. It never creates execution,
        Testnet, live, paper-ledger, or automatic-promotion authority.
        """
        now = time.time() if now is None else float(now)

        bids = list(order_book.get("bids") or [])
        asks = list(order_book.get("asks") or [])

        if not bids or not asks:
            raise ValueError(
                "microstream requires non-empty order book"
            )

        bid = _finite(bids[0][0])
        ask = _finite(asks[0][0])

        if bid <= 0 or ask < bid:
            raise ValueError("invalid microstream top of book")

        midpoint = (bid + ask) / 2.0
        spread_bps = (
            (ask - bid) / midpoint * 10_000.0
        )

        bid_depth = self._depth(bids, midpoint)
        ask_depth = self._depth(asks, midpoint)
        total_depth = bid_depth + ask_depth

        depth_imbalance = (
            0.0
            if total_depth <= 0
            else (bid_depth - ask_depth) / total_depth
        )

        bid_amount = _finite(bids[0][1])
        ask_amount = _finite(asks[0][1])
        top_total = bid_amount + ask_amount

        microprice = (
            midpoint
            if top_total <= 0
            else (
                ask * bid_amount
                + bid * ask_amount
            ) / top_total
        )

        micro_shift = (
            (microprice - midpoint)
            / midpoint
            * 10_000.0
        )

        trade_observed = trades is not None

        if trade_observed:
            trade_imbalance, intensity = (
                self._trade_features(
                    trades or (),
                    now,
                )
            )
        else:
            # The high-frequency book lane intentionally has no
            # trade REST dependency. Carry the latest genuine trade
            # state rather than fabricating a zero observation.
            with self._history_lock:
                history = self._history[
                    str(symbol).upper()
                ]
                trade_imbalance = (
                    float(
                        history[-1].get(
                            "trade_imbalance",
                            0.0,
                        )
                    )
                    if history
                    else 0.0
                )
            intensity = 0.0

        (
            depth_velocity,
            micro_velocity,
            spread_velocity,
            trade_velocity,
            persistence,
            temporal_samples,
            midpoint_velocity,
            midpoint_acceleration,
            depth_change_fraction,
            recent_range_bps,
            recent_trend_bps,
        ) = self._temporal_features(
            symbol=str(symbol).upper(),
            timestamp=now,
            midpoint=midpoint,
            spread_bps=spread_bps,
            total_depth_usd=total_depth,
            depth_imbalance=depth_imbalance,
            microprice_shift_bps=micro_shift,
            trade_imbalance=trade_imbalance,
            trade_observed=trade_observed,
        )

        return {
            "symbol": str(symbol).upper(),
            "timestamp": now,
            "midpoint": midpoint,
            "spread_bps": spread_bps,
            "bid_depth_usd": bid_depth,
            "ask_depth_usd": ask_depth,
            "depth_imbalance": _signed(
                depth_imbalance
            ),
            "microprice_shift_bps": micro_shift,
            "trade_imbalance": trade_imbalance,
            "trade_intensity_per_second": max(
                0.0,
                intensity,
            ),
            "trade_context_observed": trade_observed,
            "depth_imbalance_velocity": depth_velocity,
            "microprice_velocity_bps_per_second": (
                micro_velocity
            ),
            "spread_velocity_bps_per_second": (
                spread_velocity
            ),
            "trade_imbalance_velocity": trade_velocity,
            "pressure_persistence": persistence,
            "temporal_samples": temporal_samples,
            "midpoint_velocity_bps_per_second": (
                midpoint_velocity
            ),
            "midpoint_acceleration_bps_per_second2": (
                midpoint_acceleration
            ),
            "total_depth_change_fraction_per_second": (
                depth_change_fraction
            ),
            "recent_midpoint_range_bps_5s": (
                recent_range_bps
            ),
            "recent_midpoint_trend_bps_5s": (
                recent_trend_bps
            ),
            "research_only": True,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

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

        depth_quality = min(
            1.0,
            total_depth
            / max(1.0, self.minimum_depth_usd * 4.0),
        )
        vacuum = _unit(
            (1.0 - depth_quality)
            * (0.5 + 0.5 * abs(depth_imbalance))
        )

        (
            depth_velocity,
            micro_velocity,
            spread_velocity,
            trade_velocity,
            persistence,
            temporal_samples,
            midpoint_velocity,
            midpoint_acceleration,
            depth_change_fraction,
            recent_range_bps,
            recent_trend_bps,
        ) = self._temporal_features(
            symbol=str(symbol).upper(),
            timestamp=now,
            midpoint=midpoint,
            spread_bps=spread,
            total_depth_usd=total_depth,
            depth_imbalance=depth_imbalance,
            microprice_shift_bps=micro_shift,
            trade_imbalance=trade_imbalance,
        )

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
            depth_imbalance_velocity=depth_velocity,
            microprice_velocity_bps_per_second=micro_velocity,
            spread_velocity_bps_per_second=spread_velocity,
            trade_imbalance_velocity=trade_velocity,
            pressure_persistence=persistence,
            temporal_samples=temporal_samples,
            midpoint_velocity_bps_per_second=(
                midpoint_velocity
            ),
            midpoint_acceleration_bps_per_second2=(
                midpoint_acceleration
            ),
            total_depth_change_fraction_per_second=(
                depth_change_fraction
            ),
            recent_midpoint_range_bps_5s=(
                recent_range_bps
            ),
            recent_midpoint_trend_bps_5s=(
                recent_trend_bps
            ),
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

        snapshot_pressure = _signed(
            0.30 * features.depth_imbalance
            + 0.24 * features.trade_imbalance
            + 0.12 * micro
            + 0.10 * momentum
            + 0.14 * features.cross_venue_pressure
            - 0.04 * basis
        )

        temporal_pressure = _signed(
            0.30 * _signed(
                features.depth_imbalance_velocity / 0.08
            )
            + 0.30 * _signed(
                features.microprice_velocity_bps_per_second / 0.75
            )
            + 0.24 * _signed(
                features.trade_imbalance_velocity / 0.08
            )
            - 0.16 * _signed(
                features.spread_velocity_bps_per_second / 0.50
            )
        )

        temporal_ready = features.temporal_samples >= 3

        persistence_multiplier = (
            0.50 + 0.50 * features.pressure_persistence
        )

        base_pressure = _signed(
            (
                0.58 * snapshot_pressure
                + 0.42 * temporal_pressure
            )
            * persistence_multiplier
        )

        dynamics = _unit(
            0.25 * min(
                1.0,
                abs(features.depth_imbalance_velocity) / 0.08,
            )
            + 0.25 * min(
                1.0,
                abs(
                    features.microprice_velocity_bps_per_second
                ) / 0.75,
            )
            + 0.20 * min(
                1.0,
                abs(features.trade_imbalance_velocity) / 0.08,
            )
            + 0.15 * features.pressure_persistence
            + 0.15 * features.liquidity_vacuum_score
        )

        midpoint_velocity_signal = _signed(
            features.midpoint_velocity_bps_per_second
            / 2.0
        )
        midpoint_acceleration_signal = _signed(
            features.midpoint_acceleration_bps_per_second2
            / 2.0
        )
        midpoint_trend_signal = _signed(
            features.recent_midpoint_trend_bps_5s
            / 6.0
        )
        depth_velocity_signal = _signed(
            features.depth_imbalance_velocity
            / 0.08
        )

        kinematic_direction = _signed(
            0.42 * midpoint_velocity_signal
            + 0.28 * midpoint_trend_signal
            + 0.16 * midpoint_acceleration_signal
            + 0.14 * depth_velocity_signal
        )

        pressure = _signed(
            0.55 * base_pressure
            + 0.45
            * kinematic_direction
            * (
                0.50
                + 0.50 * features.pressure_persistence
            )
        )

        velocity_score = _unit(
            abs(
                features.midpoint_velocity_bps_per_second
            ) / 2.0
        )
        acceleration_score = _unit(
            abs(
                features.midpoint_acceleration_bps_per_second2
            ) / 2.0
        )
        range_score = _unit(
            features.recent_midpoint_range_bps_5s
            / max(
                4.0,
                features.spread_bps * 2.0,
            )
        )
        trend_score = _unit(
            abs(
                features.recent_midpoint_trend_bps_5s
            ) / 6.0
        )
        depletion_score = _unit(
            max(
                0.0,
                -features.total_depth_change_fraction_per_second,
            ) / 0.12
        )

        kinematic_event_score = _unit(
            0.26 * velocity_score
            + 0.18 * acceleration_score
            + 0.24 * range_score
            + 0.14 * trend_score
            + 0.10 * depletion_score
            + 0.08 * features.pressure_persistence
        )

        rows = []
        for horizon in self.HORIZONS:
            scale = math.sqrt(horizon / 60.0)
            strength = abs(pressure)

            velocity_budget = (
                abs(
                    features.microprice_velocity_bps_per_second
                )
                * horizon
                * 0.35
            )

            volatility_budget = max(
                vol * scale,
                q90 * scale,
            )

            burst_multiplier = (
                1.0 + 2.5 * dynamics
            )

            kinematic_motion_budget = max(
                features.recent_midpoint_range_bps_5s
                * 1.50,
                abs(
                    features.recent_midpoint_trend_bps_5s
                )
                * 1.35,
                abs(
                    features.midpoint_velocity_bps_per_second
                )
                * min(float(horizon), 8.0)
                * 0.80,
            )

            event_motion_floor_bps = max(
                4.0,
                min(
                    8.0,
                    cost * 0.20,
                ),
            )

            kinematic_event_detected = bool(
                temporal_ready
                and kinematic_event_score >= 0.60
                and kinematic_motion_budget
                >= event_motion_floor_bps
                and abs(kinematic_direction) >= 0.20
            )

            predicted_magnitude_bps = max(
                abs(features.microprice_shift_bps) * 2.0,
                velocity_budget,
                volatility_budget * burst_multiplier,
                (
                    kinematic_motion_budget
                    if kinematic_event_detected
                    else 0.0
                ),
            )

            # Kinematic events may legitimately exceed the historical
            # one-minute range, but only by an amount already supported
            # by observed sub-second/5-second motion.
            path_ceiling = max(
                10.0,
                q90 * 8.0,
                vol * 8.0,
                (
                    kinematic_motion_budget * 1.25
                    if kinematic_event_detected
                    else 0.0
                ),
            )

            path_budget = min(
                predicted_magnitude_bps,
                path_ceiling,
            )

            favorable = max(
                0.50,
                min(
                    0.95,
                    self._logistic(
                        3.2
                        * strength
                        * (0.65 + 0.70 * dynamics)
                        * math.sqrt(horizon / 15.0)
                    ),
                ),
            )

            confidence = _unit(
                (favorable - 0.50)
                * 2.0
                * (
                    0.60
                    + 0.40
                    * features.pressure_persistence
                )
            )

            expected_edge = (
                path_budget
                * strength
                * confidence
                * (0.50 + 0.50 * dynamics)
            )
            direction = "long" if pressure > 0 else ("short" if pressure < 0 else "flat")

            specialist = "temporal_orderflow"

            if kinematic_event_detected:
                velocity_acceleration_reversal = bool(
                    midpoint_velocity_signal
                    * midpoint_acceleration_signal
                    < -0.10
                )

                if (
                    depletion_score >= 0.50
                    and kinematic_direction
                    * base_pressure > 0.05
                ):
                    specialist = (
                        "kinematic_liquidity_sweep_v154"
                    )
                elif (
                    acceleration_score >= 0.55
                    and velocity_acceleration_reversal
                    and range_score >= 0.50
                ):
                    specialist = (
                        "kinematic_exhaustion_reversal_v154"
                    )
                else:
                    specialist = (
                        "kinematic_momentum_burst_v154"
                    )

            elif (
                temporal_ready
                and dynamics >= 0.72
                and features.pressure_persistence >= 0.60
            ):
                specialist = "micro_sweep_continuation"
            elif (
                temporal_ready
                and dynamics >= 0.62
                and snapshot_pressure * temporal_pressure < -0.15
            ):
                specialist = "micro_exhaustion_reversal"
            elif (
                features.liquidity_vacuum_score >= 0.65
                and strength >= 0.35
            ):
                specialist = "liquidity_vacuum_sniper"
            elif (
                features.trade_intensity_per_second >= 1.5
                and strength >= 0.45
            ):
                specialist = "micro_burst_hunter"
            elif (
                momentum * pressure < -0.12
                and strength >= 0.35
            ):
                specialist = "reversal_snapper"
            elif abs(features.cross_venue_pressure) >= 0.35:
                specialist = "cross_venue_leadlag"

            regime = "micro_trend" if momentum * pressure > 0.10 else "micro_reversal" if momentum * pressure < -0.10 else "micro_balanced"

            legacy_rare_event_score = _unit(
                0.45 * dynamics
                + 0.30 * strength
                + 0.25 * features.pressure_persistence
            )

            rare_event_score = max(
                legacy_rare_event_score,
                (
                    kinematic_event_score
                    if kinematic_event_detected
                    else 0.0
                ),
            )

            qualified = True
            reason = "qualified"
            if not temporal_ready:
                qualified, reason = (
                    False,
                    "micro_temporal_history_warming",
                )
            elif direction == "flat":
                qualified, reason = False, "flat_micro_pressure"
            elif features.spread_bps > self.maximum_spread_bps:
                qualified, reason = False, "spread_too_wide"
            elif features.bid_depth_usd + features.ask_depth_usd < self.minimum_depth_usd:
                qualified, reason = False, "insufficient_depth"
            elif rare_event_score < 0.58:
                qualified, reason = (
                    False,
                    "micro_not_rare_enough",
                )
            elif confidence < self.minimum_confidence:
                qualified, reason = (
                    False,
                    "micro_confidence_below_threshold",
                )
            elif path_budget <= cost + self.minimum_edge_buffer_bps:
                qualified, reason = (
                    False,
                    "micro_predicted_magnitude_below_cost",
                )
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
                self.rejection_reasons[reason] = (
                    self.rejection_reasons.get(reason, 0) + 1
                )
            rows.append(row)
        return rows

    def health(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "horizons_seconds": list(self.HORIZONS),
            "assessments": self.assessments,
            "qualified": self.qualified,
            "rejected": self.rejected,
            "rejection_reasons": dict(
                sorted(self.rejection_reasons.items())
            ),
            "minimum_modeled_round_trip_cost_bps": self.minimum_modeled_round_trip_cost_bps,
            "short_path_distribution": True,
            "order_book_pressure": True,
            "public_trade_pressure": True,
            "cross_venue_reference": True,
            "dedicated_microstream_ready": True,
            "kinematic_rare_event_model": True,
            "midpoint_velocity_model": True,
            "midpoint_acceleration_model": True,
            "five_second_range_model": True,
            "depth_depletion_model": True,
            "kinematic_event_score_threshold": 0.60,
            "prospective_kinematic_specialist_prefix": (
                "kinematic_"
            ),
            "temporal_history_symbols": len(self._history),
            "temporal_history_max_samples_per_symbol": 64,
            "temporal_history_max_age_seconds": 90.0,
            "predictive_profit_claim": False,
            "automatic_promotion": False,
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }


class MicroAgentFoundry:
    VERSION = "1.48.0"

    def __init__(self, maximum_candidates_per_symbol: int = 2) -> None:
        self.maximum_candidates_per_symbol = max(1, int(maximum_candidates_per_symbol))
        self.proposals = 0

    def propose(
        self,
        assessments: Iterable[MicroPathAssessment],
        *,
        evidence_rankings: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        rankings = evidence_rankings or {}
        ranked_rows: list[
            tuple[MicroPathAssessment, Mapping[str, Any]]
        ] = []

        for row in assessments:
            if row.direction not in {"long", "short"}:
                continue
            if row.reason in {
                "insufficient_depth",
                "spread_too_wide",
                "flat_micro_pressure",
            }:
                continue

            key = (
                f"{row.specialist}|"
                f"{row.horizon_seconds}|"
                f"{row.regime}"
            )
            evidence = rankings.get(key) or {}

            if evidence.get("evidence_qualified") is not True:
                continue

            conservative_net = float(
                evidence.get(
                    "conservative_net_after_cost_bps"
                ) or 0.0
            )
            if conservative_net <= 0:
                continue

            ranked_rows.append((row, evidence))

        rows = ranked_rows
        rows.sort(
            key=lambda item: (
                float(
                    item[1].get(
                        "conservative_net_after_cost_bps"
                    ) or 0.0
                ),
                int(item[1].get("samples") or 0),
                -item[0].horizon_seconds,
            ),
            reverse=True,
        )
        output = []
        for row, evidence in rows[
            : self.maximum_candidates_per_symbol
        ]:
            self.proposals += 1

            conservative_net = float(
                evidence.get(
                    "conservative_net_after_cost_bps"
                ) or 0.0
            )
            evidence_accuracy = float(
                evidence.get("directional_accuracy") or 0.0
            )

            output.append(
                {
                    "candidate_kind": "microstructure_shadow_specialist",
                    "specialist": row.specialist,
                    "symbol": row.symbol,
                    "timeframe": f"micro-{row.horizon_seconds}s-{row.specialist}",
                    "horizon_seconds": row.horizon_seconds,
                    "side": row.direction,
                    "confidence": max(
                        0.01,
                        min(1.0, evidence_accuracy),
                    ),
                    "current_signal_confidence": row.confidence,
                    "expected_edge_bps": (
                        row.modeled_round_trip_cost_bps
                        + conservative_net
                    ),
                    "conservative_net_edge_bps": conservative_net,
                    "modeled_round_trip_cost_bps": row.modeled_round_trip_cost_bps,
                    "evidence_samples": int(
                        evidence.get("samples") or 0
                    ),
                    "evidence_average_net_bps": float(
                        evidence.get(
                            "average_net_after_cost_bps"
                        ) or 0.0
                    ),
                    "evidence_qualified": True,
                    "qualification_basis": (
                        "prospective_conservative_net_expectancy"
                    ),
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
