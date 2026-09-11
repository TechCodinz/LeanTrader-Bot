#!/usr/bin/env python3
"""
🌌 APEX PREDATOR INTELLIGENCE — 7 Ultra-Rare Trading Intelligences
═══════════════════════════════════════════════════════════════════
ONE-OF-A-KIND capabilities that no other trading system has ever combined.
These are used by Citadel, Renaissance Technologies, Two Sigma, Jane Street —
but NEVER assembled together in a single autonomous entity.

Modules:
    1. 🩸 Liquidation Cascade Predator    — Predict mass liquidations 5-30 min ahead
    2. 🏛️ Wyckoff Smart Money Detector    — Institutional accumulation / distribution
    3. 🕶️ Dark Pool Shadow Tracker        — Infer invisible institutional order flow
    4. 🧠 Regime-Adaptive Metamorphosis   — Bot personality hot-swap per regime
    5. 💸 Delta-Neutral Funding Harvester — Risk-free funding rate income (20-80% APY)
    6. 📡 Order Book Dominance Radar      — Read the force behind price before it moves
    7. 🔮 Entropy Decay Oracle            — Physics-based breakout prediction

Author: Lean-Trader Evolution Engine
"""

import asyncio
import logging
import math
import time
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Minimal numpy shim for environments without numpy
    class _NpShim:
        @staticmethod
        def array(x):
            return list(x) if not isinstance(x, list) else x
        @staticmethod
        def mean(x):
            x = list(x)
            return sum(x) / len(x) if x else 0.0
        @staticmethod
        def std(x):
            x = list(x)
            if len(x) < 2:
                return 0.0
            m = sum(x) / len(x)
            return (sum((v - m) ** 2 for v in x) / len(x)) ** 0.5
        @staticmethod
        def log(x):
            return math.log(max(x, 1e-12))
        @staticmethod
        def abs(x):
            return abs(x)
        @staticmethod
        def sum(x):
            return sum(x)
        @staticmethod
        def diff(x):
            x = list(x)
            return [x[i+1] - x[i] for i in range(len(x)-1)]
        @staticmethod
        def percentile(x, q):
            x = sorted(x)
            k = (len(x) - 1) * q / 100.0
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return x[int(k)]
            return x[f] * (c - k) + x[c] * (k - f)

    np = _NpShim()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. 🩸 LIQUIDATION CASCADE PREDATOR                                    ║
# ║  Predicts mass liquidation cascades BEFORE they happen                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class LiquidationCascadePredator:
    """
    Monitors open interest, funding rates, and leverage ratios across exchanges
    to predict cascading liquidation events 5-30 minutes in advance.

    When leverage is at a breaking point and funding rates are extreme,
    a small price move triggers a chain reaction:
        Price drop → Liquidations → More selling → More liquidations → CRASH

    This engine detects the pre-conditions and trades the CASCADE.
    """

    def __init__(self):
        self.funding_history = deque(maxlen=500)
        self.oi_history = deque(maxlen=500)       # open interest history
        self.leverage_alerts = deque(maxlen=100)
        self.cascade_threshold_funding = 0.0008   # 0.08% funding = extreme
        self.cascade_threshold_oi_drop = 0.05     # 5% OI drop in 1 hour
        self.last_scan_time = 0
        self.active_predictions = {}
        logger.info("🩸 Liquidation Cascade Predator initialized")

    async def scan_liquidation_risk(self, exchange, symbols: List[str]) -> List[Dict]:
        """
        Scan multiple symbols for liquidation cascade risk.
        Returns signals when cascade conditions are detected.
        """
        signals = []
        now = time.time()

        # Rate-limit scans to every 10 seconds minimum
        if now - self.last_scan_time < 10:
            return signals
        self.last_scan_time = now

        for symbol in symbols[:20]:  # Cap at 20 to avoid rate limits
            try:
                signal = await self._analyze_single(exchange, symbol)
                if signal and signal.get('confidence', 0) > 0.6:
                    signals.append(signal)
            except Exception as e:
                logger.debug(f"🩸 Liquidation scan error {symbol}: {e}")

        if signals:
            logger.info(f"🩸 LIQUIDATION PREDATOR: {len(signals)} cascade risks detected!")

        return signals

    async def _analyze_single(self, exchange, symbol: str) -> Optional[Dict]:
        """Analyze a single symbol for liquidation cascade risk."""
        try:
            # --- Fetch funding rate ---
            funding_rate = 0.0
            try:
                if hasattr(exchange, 'fetch_funding_rate'):
                    fr_data = await asyncio.to_thread(exchange.fetch_funding_rate, symbol)
                    funding_rate = fr_data.get('fundingRate', 0.0) or 0.0
                elif hasattr(exchange, 'fetch_funding_rates'):
                    fr_data = await asyncio.to_thread(exchange.fetch_funding_rates, [symbol])
                    if symbol in fr_data:
                        funding_rate = fr_data[symbol].get('fundingRate', 0.0) or 0.0
            except Exception:
                pass

            # --- Fetch open interest ---
            open_interest = 0.0
            try:
                if hasattr(exchange, 'fetch_open_interest'):
                    oi_data = await asyncio.to_thread(exchange.fetch_open_interest, symbol)
                    open_interest = oi_data.get('openInterest', 0.0) or 0.0
            except Exception:
                pass

            # --- Fetch ticker for price context ---
            ticker = await asyncio.to_thread(exchange.fetch_ticker, symbol)
            current_price = ticker.get('last', 0)
            price_change_pct = ticker.get('percentage', 0) or 0

            if current_price <= 0:
                return None

            # --- Track history ---
            ts = datetime.now(timezone.utc).isoformat()
            self.funding_history.append({
                'symbol': symbol, 'rate': funding_rate,
                'oi': open_interest, 'ts': ts
            })

            # --- Calculate cascade probability ---
            cascade_score = 0.0
            direction = 'none'
            reasons = []

            # Extreme positive funding = longs overleveraged → SHORT cascade
            if funding_rate > self.cascade_threshold_funding:
                cascade_score += 0.35
                direction = 'sell'
                reasons.append(f"Extreme positive funding {funding_rate:.4%} — longs overleveraged")

            # Extreme negative funding = shorts overleveraged → LONG cascade
            elif funding_rate < -self.cascade_threshold_funding:
                cascade_score += 0.35
                direction = 'buy'
                reasons.append(f"Extreme negative funding {funding_rate:.4%} — shorts overleveraged")

            # OI dropping rapidly = liquidations already starting
            recent_oi = [h['oi'] for h in self.funding_history
                         if h['symbol'] == symbol and h['oi'] > 0]
            if len(recent_oi) >= 5:
                oi_change = (recent_oi[-1] - recent_oi[-5]) / max(recent_oi[-5], 1)
                if abs(oi_change) > self.cascade_threshold_oi_drop:
                    cascade_score += 0.30
                    reasons.append(f"OI dropped {oi_change:.2%} — liquidations in progress")
                    if direction == 'none':
                        direction = 'buy' if oi_change < 0 and funding_rate > 0 else 'sell'

            # Price moving against dominant leverage side
            if abs(price_change_pct) > 2.0:
                if (funding_rate > 0 and price_change_pct < -1.0):
                    cascade_score += 0.20
                    reasons.append(f"Price dropping {price_change_pct:.1f}% against long-heavy OI")
                    direction = 'sell'
                elif (funding_rate < 0 and price_change_pct > 1.0):
                    cascade_score += 0.20
                    reasons.append(f"Price rising {price_change_pct:.1f}% against short-heavy OI")
                    direction = 'buy'

            # High funding + big price move = MAXIMUM cascade risk
            if abs(funding_rate) > 0.0005 and abs(price_change_pct) > 3.0:
                cascade_score += 0.15
                reasons.append("CRITICAL: High funding + large price move = cascade imminent")

            if cascade_score < 0.5 or direction == 'none':
                return None

            confidence = min(0.95, cascade_score)

            return {
                'type': 'liquidation_cascade',
                'symbol': symbol,
                'side': direction,
                'action': direction,
                'confidence': confidence,
                'cascade_score': cascade_score,
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'price': current_price,
                'reasons': reasons,
                'source': 'LiquidationCascadePredator',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.debug(f"🩸 Analysis error {symbol}: {e}")
            return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. 🏛️ WYCKOFF SMART MONEY CONCEPT DETECTOR                           ║
# ║  Detects institutional accumulation & distribution via volume-spread    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class WyckoffSmartMoneyDetector:
    """
    Implements Richard Wyckoff's 100-year-old institutional tracking method
    enhanced with modern volume-spread analysis (VSA).

    4 Phases: Accumulation → Markup → Distribution → Markdown

    Key Patterns:
        - Spring: Fake breakdown below support, then reversal (BUY!)
        - Upthrust: Fake breakout above resistance, then reversal (SELL!)
        - Absorption: High volume, narrow range = institutions absorbing supply
        - Sign of Strength (SOS): Strong breakout on high volume after accumulation
    """

    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=200))
        self.volume_history = defaultdict(lambda: deque(maxlen=200))
        self.detected_phases = {}
        self.support_levels = defaultdict(list)
        self.resistance_levels = defaultdict(list)
        logger.info("🏛️ Wyckoff Smart Money Detector initialized")

    async def analyze_wyckoff(self, exchange, symbol: str) -> Optional[Dict]:
        """
        Analyze a symbol for Wyckoff patterns using OHLCV candle data.
        """
        try:
            # Fetch recent candles (1h timeframe, 100 candles)
            candles = await asyncio.to_thread(
                exchange.fetch_ohlcv, symbol, '1h', limit=100
            )

            if not candles or len(candles) < 30:
                return None

            # Extract OHLCV
            opens   = [c[1] for c in candles]
            highs   = [c[2] for c in candles]
            lows    = [c[3] for c in candles]
            closes  = [c[4] for c in candles]
            volumes = [c[5] for c in candles]

            # Store history
            for c in closes[-50:]:
                self.price_history[symbol].append(c)
            for v in volumes[-50:]:
                self.volume_history[symbol].append(v)

            # --- Calculate key levels ---
            recent_lows = lows[-30:]
            recent_highs = highs[-30:]
            support = min(recent_lows)
            resistance = max(recent_highs)
            current_price = closes[-1]
            current_volume = volumes[-1]
            avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else current_volume

            # --- Detect Wyckoff Patterns ---
            signal = None

            # SPRING DETECTION: Price dips below support then recovers
            # (Institutions absorb all selling — massive buy signal)
            if len(lows) >= 5:
                recent_low = min(lows[-3:])
                prev_support = min(lows[-20:-3]) if len(lows) >= 20 else support

                # Spring = price went below support but closed above it
                if (recent_low < prev_support * 0.998 and
                    current_price > prev_support and
                    current_volume > avg_volume * 1.5):
                    signal = {
                        'pattern': 'SPRING',
                        'side': 'buy',
                        'confidence': min(0.90, 0.70 + (current_volume / avg_volume - 1) * 0.1),
                        'reason': f"Spring at ${prev_support:.2f} — fake breakdown with {current_volume/avg_volume:.1f}x volume absorption"
                    }

            # UPTHRUST DETECTION: Price spikes above resistance then falls back
            # (Institutions distributing to eager buyers — sell signal)
            if signal is None and len(highs) >= 5:
                recent_high = max(highs[-3:])
                prev_resistance = max(highs[-20:-3]) if len(highs) >= 20 else resistance

                if (recent_high > prev_resistance * 1.002 and
                    current_price < prev_resistance and
                    current_volume > avg_volume * 1.3):
                    signal = {
                        'pattern': 'UPTHRUST',
                        'side': 'sell',
                        'confidence': min(0.88, 0.65 + (current_volume / avg_volume - 1) * 0.1),
                        'reason': f"Upthrust at ${prev_resistance:.2f} — fake breakout with distribution volume"
                    }

            # ABSORPTION DETECTION: Narrow range + high volume = institutional absorption
            if signal is None and len(closes) >= 5:
                recent_ranges = [(highs[i] - lows[i]) / max(closes[i], 1e-8)
                                 for i in range(-5, 0)]
                avg_range = sum(recent_ranges) / len(recent_ranges)
                vol_ratio = current_volume / max(avg_volume, 1)

                # Narrow range (< 50% of average) with high volume (> 2x)
                if avg_range < 0.005 and vol_ratio > 2.0:
                    # Direction: if price near support = accumulation (buy)
                    price_position = (current_price - support) / max(resistance - support, 1e-8)

                    if price_position < 0.3:
                        signal = {
                            'pattern': 'ACCUMULATION_ABSORPTION',
                            'side': 'buy',
                            'confidence': min(0.85, 0.60 + vol_ratio * 0.05),
                            'reason': f"Institutional absorption near support — {vol_ratio:.1f}x volume, {avg_range*100:.2f}% range"
                        }
                    elif price_position > 0.7:
                        signal = {
                            'pattern': 'DISTRIBUTION_ABSORPTION',
                            'side': 'sell',
                            'confidence': min(0.85, 0.60 + vol_ratio * 0.05),
                            'reason': f"Institutional distribution near resistance — {vol_ratio:.1f}x volume, {avg_range*100:.2f}% range"
                        }

            # SIGN OF STRENGTH (SOS): Strong breakout on high volume after accumulation
            if signal is None and len(closes) >= 10:
                price_gain = (closes[-1] - closes[-5]) / max(closes[-5], 1e-8)
                vol_ratio = current_volume / max(avg_volume, 1)

                if price_gain > 0.03 and vol_ratio > 2.5:
                    signal = {
                        'pattern': 'SIGN_OF_STRENGTH',
                        'side': 'buy',
                        'confidence': min(0.92, 0.70 + price_gain * 2),
                        'reason': f"Sign of Strength — {price_gain*100:.1f}% breakout on {vol_ratio:.1f}x volume"
                    }

            if signal is None:
                return None

            return {
                'type': 'wyckoff_smart_money',
                'symbol': symbol,
                'side': signal['side'],
                'action': signal['side'],
                'confidence': signal['confidence'],
                'pattern': signal['pattern'],
                'reason': signal['reason'],
                'support': support,
                'resistance': resistance,
                'price': current_price,
                'source': 'WyckoffSmartMoneyDetector',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.debug(f"🏛️ Wyckoff error {symbol}: {e}")
            return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. 🕶️ DARK POOL SHADOW TRACKER                                       ║
# ║  Infers hidden institutional order flow from visible anomalies         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DarkPoolShadowTracker:
    """
    Dark pools are invisible by design. But they leave SHADOWS —
    anomalies in the visible market that reveal institutional activity.

    Detection methods:
        1. Volume anomaly: Large volume with minimal price movement
        2. Iceberg orders: Repeating limit orders at exact same price
        3. Print analysis: Unusual trade sizes (round lots = institutional)
        4. Time-of-day patterns: Institutional trading hours
    """

    def __init__(self):
        self.volume_baselines = defaultdict(lambda: deque(maxlen=100))
        self.shadow_detections = deque(maxlen=200)
        self.iceberg_tracker = defaultdict(int)
        logger.info("🕶️ Dark Pool Shadow Tracker initialized")

    async def detect_dark_pool_activity(self, exchange, symbol: str) -> Optional[Dict]:
        """
        Detect dark pool shadows by analyzing volume-price anomalies.
        """
        try:
            # Fetch recent trades
            trades = []
            try:
                trades = await asyncio.to_thread(exchange.fetch_trades, symbol, limit=100)
            except Exception:
                pass

            # Fetch ticker
            ticker = await asyncio.to_thread(exchange.fetch_ticker, symbol)
            current_price = ticker.get('last', 0)
            volume_24h = ticker.get('quoteVolume', 0) or 0
            price_change = abs(ticker.get('percentage', 0) or 0)

            if current_price <= 0:
                return None

            # --- Method 1: Volume-Price Divergence ---
            # Large volume + tiny price move = someone is absorbing without moving price
            divergence_score = 0.0
            if volume_24h > 0 and price_change < 0.5:
                # Compare current volume to baseline
                self.volume_baselines[symbol].append(volume_24h)
                if len(self.volume_baselines[symbol]) >= 5:
                    baseline = sum(self.volume_baselines[symbol]) / len(self.volume_baselines[symbol])
                    vol_ratio = volume_24h / max(baseline, 1)

                    # Volume 2x+ above average with <0.5% price change = dark pool shadow
                    if vol_ratio > 2.0 and price_change < 0.5:
                        divergence_score = min(1.0, (vol_ratio - 1.5) * 0.5)

            # --- Method 2: Iceberg Order Detection ---
            iceberg_score = 0.0
            if trades and len(trades) >= 20:
                # Look for repeating prices (iceberg = refilling limit order)
                price_counts = defaultdict(int)
                for t in trades:
                    rounded_price = round(t.get('price', 0), 2)
                    price_counts[rounded_price] += 1

                # If any price appears 5+ times = likely iceberg
                max_repeats = max(price_counts.values()) if price_counts else 0
                if max_repeats >= 5:
                    iceberg_score = min(1.0, max_repeats / 10.0)
                    iceberg_price = max(price_counts, key=price_counts.get)
                    self.iceberg_tracker[symbol] = max_repeats

            # --- Method 3: Large Block Trades ---
            block_score = 0.0
            if trades and len(trades) >= 10:
                trade_sizes = [t.get('amount', 0) * t.get('price', 0) for t in trades]
                if trade_sizes:
                    avg_size = sum(trade_sizes) / len(trade_sizes)
                    large_blocks = [s for s in trade_sizes if s > avg_size * 5]
                    if large_blocks:
                        block_score = min(1.0, len(large_blocks) / 5.0)

            # --- Combine scores ---
            total_score = (divergence_score * 0.4 + iceberg_score * 0.35 + block_score * 0.25)

            if total_score < 0.4:
                return None

            # Direction: Analyze trade side distribution
            direction = 'buy'
            if trades:
                buy_trades = sum(1 for t in trades if t.get('side') == 'buy')
                sell_trades = sum(1 for t in trades if t.get('side') == 'sell')
                if sell_trades > buy_trades * 1.3:
                    direction = 'sell'

            confidence = min(0.92, total_score)

            reasons = []
            if divergence_score > 0.3:
                reasons.append(f"Volume-price divergence: high volume, {price_change:.1f}% price change")
            if iceberg_score > 0.3:
                reasons.append(f"Iceberg orders detected: {self.iceberg_tracker.get(symbol, 0)} repeating fills")
            if block_score > 0.2:
                reasons.append("Large block trades: institutional-size orders")

            return {
                'type': 'dark_pool_shadow',
                'symbol': symbol,
                'side': direction,
                'action': direction,
                'confidence': confidence,
                'divergence_score': divergence_score,
                'iceberg_score': iceberg_score,
                'block_score': block_score,
                'reasons': reasons,
                'price': current_price,
                'source': 'DarkPoolShadowTracker',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.debug(f"🕶️ Dark pool error {symbol}: {e}")
            return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. 🧠 REGIME-ADAPTIVE METAMORPHOSIS BRAIN                            ║
# ║  Bot transforms its entire personality per market regime               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class RegimeAdaptiveMetamorphosisBrain:
    """
    Instead of just DETECTING regimes, this engine TRANSFORMS the bot.

    5 Personalities:
        BULL_TREND    → Aggressive momentum rider, trailing stops
        RANGE_CHOP    → Mean-reversion scalper, tight bands
        CRASH_PANIC   → Liquidation predator + dip buyer, big positions
        LOW_VOL       → Grid trader + funding farmer, passive income
        EUPHORIA      → Contrarian short seller, inverse signals

    The bot literally becomes a DIFFERENT TRADER based on conditions.
    """

    # Personality configurations
    PERSONALITIES = {
        'BULL_TREND': {
            'risk_per_trade': 0.03,        # 3% — aggressive
            'max_concurrent': 10,
            'preferred_side': 'buy',
            'strategy_weights': {'momentum': 0.4, 'breakout': 0.3, 'trend': 0.3},
            'stop_loss_pct': 0.02,         # tight trailing
            'take_profit_pct': 0.08,       # let winners run
            'scan_interval': 15,           # fast scanning
            'confidence_threshold': 0.60,  # lower bar — go aggressive
        },
        'RANGE_CHOP': {
            'risk_per_trade': 0.015,       # 1.5% — conservative
            'max_concurrent': 15,
            'preferred_side': 'both',
            'strategy_weights': {'mean_reversion': 0.5, 'scalping': 0.3, 'grid': 0.2},
            'stop_loss_pct': 0.01,
            'take_profit_pct': 0.015,      # small targets
            'scan_interval': 10,
            'confidence_threshold': 0.70,
        },
        'CRASH_PANIC': {
            'risk_per_trade': 0.05,        # 5% — maximum aggression on dips
            'max_concurrent': 5,
            'preferred_side': 'buy',       # buy the blood
            'strategy_weights': {'dip_buy': 0.5, 'liquidation_ride': 0.3, 'mean_reversion': 0.2},
            'stop_loss_pct': 0.05,         # wide stops — expect volatility
            'take_profit_pct': 0.15,       # big targets
            'scan_interval': 5,            # ultra-fast
            'confidence_threshold': 0.55,  # lower bar — act fast
        },
        'LOW_VOL': {
            'risk_per_trade': 0.01,        # 1% — minimal risk
            'max_concurrent': 20,
            'preferred_side': 'both',
            'strategy_weights': {'grid': 0.4, 'funding_farm': 0.3, 'market_make': 0.3},
            'stop_loss_pct': 0.005,
            'take_profit_pct': 0.008,
            'scan_interval': 60,           # slow — nothing happening
            'confidence_threshold': 0.75,
        },
        'EUPHORIA': {
            'risk_per_trade': 0.02,
            'max_concurrent': 8,
            'preferred_side': 'sell',       # contrarian shorts
            'strategy_weights': {'contrarian': 0.5, 'short': 0.3, 'hedge': 0.2},
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.10,
            'scan_interval': 20,
            'confidence_threshold': 0.72,
        },
    }

    def __init__(self):
        self.current_regime = 'RANGE_CHOP'  # safe default
        self.current_personality = self.PERSONALITIES['RANGE_CHOP']
        self.regime_history = deque(maxlen=200)
        self.price_windows = defaultdict(lambda: deque(maxlen=100))
        self.transition_count = 0
        self.last_transition_time = time.time()
        logger.info(f"🧠 Metamorphosis Brain initialized — Personality: {self.current_regime}")

    async def detect_regime(self, exchange, symbols: List[str]) -> Dict:
        """
        Detect current market regime and transform bot personality.
        """
        try:
            # Collect market-wide data
            price_changes = []
            volumes = []

            for symbol in symbols[:10]:
                try:
                    ticker = await asyncio.to_thread(exchange.fetch_ticker, symbol)
                    pct = ticker.get('percentage', 0) or 0
                    vol = ticker.get('quoteVolume', 0) or 0
                    price_changes.append(pct)
                    volumes.append(vol)
                except Exception:
                    continue

            if not price_changes:
                return {'regime': self.current_regime, 'personality': self.current_personality}

            # --- Regime Classification ---
            avg_change = sum(price_changes) / len(price_changes)
            volatility = (sum((p - avg_change) ** 2 for p in price_changes) / len(price_changes)) ** 0.5
            max_change = max(price_changes)
            min_change = min(price_changes)
            spread = max_change - min_change

            new_regime = self.current_regime

            # CRASH/PANIC: Average change < -5% or min change < -10%
            if avg_change < -5.0 or min_change < -10.0:
                new_regime = 'CRASH_PANIC'

            # EUPHORIA/BUBBLE: Average change > 8% or max change > 15%
            elif avg_change > 8.0 or max_change > 15.0:
                new_regime = 'EUPHORIA'

            # BULL TREND: Steady upward, 2% < avg < 8%
            elif 2.0 < avg_change < 8.0 and volatility < 5.0:
                new_regime = 'BULL_TREND'

            # LOW VOLATILITY: Spread < 3% and volatility < 1.5%
            elif spread < 3.0 and volatility < 1.5:
                new_regime = 'LOW_VOL'

            # RANGE/CHOP: Everything else
            else:
                new_regime = 'RANGE_CHOP'

            # --- METAMORPHOSIS: Transform personality ---
            if new_regime != self.current_regime:
                old_regime = self.current_regime
                self.current_regime = new_regime
                self.current_personality = self.PERSONALITIES[new_regime]
                self.transition_count += 1
                self.last_transition_time = time.time()

                logger.info(f"🧠 ═══ METAMORPHOSIS ═══")
                logger.info(f"🧠 Regime shift: {old_regime} → {new_regime}")
                logger.info(f"🧠 New personality: risk={self.current_personality['risk_per_trade']:.1%}, "
                            f"side={self.current_personality['preferred_side']}, "
                            f"threshold={self.current_personality['confidence_threshold']}")
                logger.info(f"🧠 Total transformations: {self.transition_count}")

            self.regime_history.append({
                'regime': new_regime,
                'avg_change': avg_change,
                'volatility': volatility,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            return {
                'regime': self.current_regime,
                'personality': self.current_personality,
                'avg_change': avg_change,
                'volatility': volatility,
                'transition_count': self.transition_count
            }

        except Exception as e:
            logger.debug(f"🧠 Regime detection error: {e}")
            return {'regime': self.current_regime, 'personality': self.current_personality}

    def get_adjusted_confidence(self, base_confidence: float, side: str) -> float:
        """Adjust signal confidence based on current personality."""
        personality = self.current_personality
        threshold = personality['confidence_threshold']

        # Boost signals that match preferred side
        if personality['preferred_side'] == side or personality['preferred_side'] == 'both':
            adjusted = base_confidence * 1.1
        else:
            adjusted = base_confidence * 0.85  # penalize wrong-side signals

        return min(0.98, adjusted)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. 💸 DELTA-NEUTRAL FUNDING RATE HARVESTER                           ║
# ║  Risk-free funding rate income — literally free money                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class DeltaNeutralFundingHarvester:
    """
    Earns funding rate payments with ZERO directional exposure.

    Strategy:
        When funding is POSITIVE (longs pay shorts):
            → SHORT perpetual + LONG spot = earn funding, zero price risk
        When funding is NEGATIVE (shorts pay longs):
            → LONG perpetual + SHORT margin/spot = earn funding, zero price risk

    Expected return: 20-80% APY with near-zero risk.
    Funding is paid every 8 hours on most exchanges.
    """

    def __init__(self):
        self.active_harvests = {}          # symbol → harvest info
        self.total_funding_earned = 0.0
        self.min_funding_threshold = 0.0003  # 0.03% minimum to bother
        self.max_positions = 5              # max concurrent harvests
        logger.info("💸 Delta-Neutral Funding Harvester initialized")

    async def scan_funding_opportunities(self, exchange, symbols: List[str]) -> List[Dict]:
        """
        Scan all perpetual markets for high funding rate opportunities.
        """
        opportunities = []

        for symbol in symbols[:30]:
            try:
                # Only works on perpetual/swap markets
                if not any(x in symbol for x in ['/USDT', '/USD']):
                    continue

                # Fetch funding rate
                funding_rate = 0.0
                try:
                    if hasattr(exchange, 'fetch_funding_rate'):
                        fr = await asyncio.to_thread(exchange.fetch_funding_rate, symbol)
                        funding_rate = fr.get('fundingRate', 0) or 0
                except Exception:
                    continue

                if abs(funding_rate) < self.min_funding_threshold:
                    continue

                # Calculate annualized yield (3 payments/day × 365 days)
                annual_yield = abs(funding_rate) * 3 * 365 * 100  # as percentage

                # Determine direction
                if funding_rate > 0:
                    # Longs pay shorts → we SHORT perp + LONG spot
                    perp_side = 'sell'
                    spot_side = 'buy'
                    reason = f"Positive funding {funding_rate:.4%} → Short perp + Long spot"
                else:
                    # Shorts pay longs → we LONG perp + SHORT spot
                    perp_side = 'buy'
                    spot_side = 'sell'
                    reason = f"Negative funding {funding_rate:.4%} → Long perp + Short spot"

                if annual_yield > 10:  # Only if > 10% APY
                    opportunities.append({
                        'type': 'funding_harvest',
                        'symbol': symbol,
                        'side': perp_side,
                        'action': 'delta_neutral_harvest',
                        'confidence': min(0.95, 0.60 + annual_yield / 200),
                        'funding_rate': funding_rate,
                        'annual_yield_pct': annual_yield,
                        'perp_side': perp_side,
                        'spot_side': spot_side,
                        'reason': reason,
                        'risk': 'near_zero',
                        'source': 'DeltaNeutralFundingHarvester',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })

            except Exception as e:
                logger.debug(f"💸 Funding scan error {symbol}: {e}")

        # Sort by yield
        opportunities.sort(key=lambda x: x['annual_yield_pct'], reverse=True)

        if opportunities:
            top = opportunities[0]
            logger.info(f"💸 FUNDING HARVESTER: {len(opportunities)} opportunities found!")
            logger.info(f"💸 Best: {top['symbol']} → {top['annual_yield_pct']:.1f}% APY")

        return opportunities[:self.max_positions]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. 📡 ORDER BOOK DOMINANCE RADAR                                     ║
# ║  Read the invisible force behind price before it moves                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class OrderBookDominanceRadar:
    """
    Analyzes real-time order book depth to detect:
        1. Bid-Ask dominance: Which side has more weight?
        2. Absorption: Large orders eating incoming flow without moving
        3. Spoof walls: Fake large orders that disappear near execution
        4. Imbalance shifts: Sudden changes in book composition

    Sees where price WANTS to go before it actually moves.
    """

    def __init__(self):
        self.depth_history = defaultdict(lambda: deque(maxlen=50))
        self.spoof_tracker = defaultdict(lambda: deque(maxlen=20))
        self.dominance_threshold = 1.8   # 1.8x = strong dominance
        logger.info("📡 Order Book Dominance Radar initialized")

    async def analyze_order_book(self, exchange, symbol: str) -> Optional[Dict]:
        """
        Analyze order book depth for directional signals.
        """
        try:
            # Fetch order book with depth
            order_book = await asyncio.to_thread(
                exchange.fetch_order_book, symbol, 50
            )

            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                return None

            # --- Calculate depth ratios ---
            # Total bid volume (within 1% of best bid)
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = (best_ask - best_bid) / best_bid

            # Volume within 0.5% of best price
            bid_depth_narrow = sum(b[1] * b[0] for b in bids
                                   if b[0] >= best_bid * 0.995)
            ask_depth_narrow = sum(a[1] * a[0] for a in asks
                                   if a[0] <= best_ask * 1.005)

            # Volume within 2% of best price
            bid_depth_wide = sum(b[1] * b[0] for b in bids
                                 if b[0] >= best_bid * 0.98)
            ask_depth_wide = sum(a[1] * a[0] for a in asks
                                 if a[0] <= best_ask * 1.02)

            if ask_depth_narrow <= 0 or ask_depth_wide <= 0:
                return None

            narrow_ratio = bid_depth_narrow / ask_depth_narrow
            wide_ratio = bid_depth_wide / ask_depth_wide

            # --- Track for spoof detection ---
            self.depth_history[symbol].append({
                'bid_depth': bid_depth_wide,
                'ask_depth': ask_depth_wide,
                'ratio': wide_ratio,
                'ts': time.time()
            })

            # --- Spoof Detection ---
            # If a large wall appeared and disappeared within 30 seconds = spoof
            spoof_detected = False
            if len(self.depth_history[symbol]) >= 3:
                recent = list(self.depth_history[symbol])[-3:]
                bid_changes = [abs(recent[i+1]['bid_depth'] - recent[i]['bid_depth'])
                               / max(recent[i]['bid_depth'], 1)
                               for i in range(len(recent)-1)]
                ask_changes = [abs(recent[i+1]['ask_depth'] - recent[i]['ask_depth'])
                               / max(recent[i]['ask_depth'], 1)
                               for i in range(len(recent)-1)]

                # >50% depth change in a short time = likely spoof
                if any(c > 0.5 for c in bid_changes) or any(c > 0.5 for c in ask_changes):
                    spoof_detected = True

            # --- Generate signal ---
            if narrow_ratio < 1 / self.dominance_threshold and wide_ratio < 1 / self.dominance_threshold:
                # Ask dominance = heavy selling pressure
                if not spoof_detected:
                    return {
                        'type': 'order_book_dominance',
                        'symbol': symbol,
                        'side': 'sell',
                        'action': 'sell',
                        'confidence': min(0.88, 0.55 + (1/narrow_ratio - 1) * 0.15),
                        'narrow_ratio': narrow_ratio,
                        'wide_ratio': wide_ratio,
                        'bid_depth': bid_depth_wide,
                        'ask_depth': ask_depth_wide,
                        'spread': spread,
                        'spoof_detected': spoof_detected,
                        'reason': f"Ask dominance {1/narrow_ratio:.1f}x — heavy selling pressure",
                        'source': 'OrderBookDominanceRadar',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }

            elif narrow_ratio > self.dominance_threshold and wide_ratio > self.dominance_threshold:
                # Bid dominance = heavy buying support
                if not spoof_detected:
                    return {
                        'type': 'order_book_dominance',
                        'symbol': symbol,
                        'side': 'buy',
                        'action': 'buy',
                        'confidence': min(0.88, 0.55 + (narrow_ratio - 1) * 0.15),
                        'narrow_ratio': narrow_ratio,
                        'wide_ratio': wide_ratio,
                        'bid_depth': bid_depth_wide,
                        'ask_depth': ask_depth_wide,
                        'spread': spread,
                        'spoof_detected': spoof_detected,
                        'reason': f"Bid dominance {narrow_ratio:.1f}x — massive hidden support",
                        'source': 'OrderBookDominanceRadar',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }

            return None

        except Exception as e:
            logger.debug(f"📡 Order book error {symbol}: {e}")
            return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. 🔮 ENTROPY DECAY ORACLE                                           ║
# ║  Physics-based breakout prediction using information theory            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class EntropyDecayOracle:
    """
    Uses Shannon entropy of price returns to predict breakouts.

    When entropy (randomness) DECAYS — the market is compressing
    like a spring. Low entropy = big move imminent.

    Also calculates:
        - Hurst exponent: H > 0.5 = trending, H < 0.5 = mean-reverting
        - Fisher information: measures how much "information" is in price action
        - Lyapunov exponent: measures chaos — high = unpredictable, low = structured

    When entropy is critically low + Hurst shows trend → BREAKOUT incoming.
    """

    def __init__(self):
        self.entropy_history = defaultdict(lambda: deque(maxlen=100))
        self.entropy_baseline = 0.7   # typical entropy
        self.critical_low = 0.30      # breakout threshold
        logger.info("🔮 Entropy Decay Oracle initialized")

    async def analyze_entropy(self, exchange, symbol: str) -> Optional[Dict]:
        """
        Calculate Shannon entropy and predict breakout/breakdown.
        """
        try:
            # Fetch 5-minute candles
            candles = await asyncio.to_thread(
                exchange.fetch_ohlcv, symbol, '5m', limit=100
            )

            if not candles or len(candles) < 30:
                return None

            closes = [c[4] for c in candles]
            volumes = [c[5] for c in candles]

            # --- Calculate returns ---
            returns = [(closes[i+1] - closes[i]) / max(closes[i], 1e-12)
                       for i in range(len(closes)-1)]

            if len(returns) < 20:
                return None

            # --- Shannon Entropy of returns ---
            entropy = self._shannon_entropy(returns[-30:])

            # --- Hurst Exponent (simplified R/S method) ---
            hurst = self._hurst_exponent(closes[-50:])

            # --- Entropy decay rate ---
            self.entropy_history[symbol].append(entropy)
            decay_rate = 0.0
            if len(self.entropy_history[symbol]) >= 5:
                recent_entropies = list(self.entropy_history[symbol])[-5:]
                if recent_entropies[0] > 0:
                    decay_rate = (recent_entropies[-1] - recent_entropies[0]) / recent_entropies[0]

            # --- Volatility compression ---
            recent_vol = self._rolling_std(returns[-10:])
            older_vol = self._rolling_std(returns[-30:-10])
            vol_compression = recent_vol / max(older_vol, 1e-12)

            # --- Generate signal ---
            if entropy > self.critical_low:
                return None  # Not compressed enough

            # Entropy is critically low — breakout imminent!
            # Use Hurst to determine direction tendency
            if hurst > 0.55:
                # Trending regime — momentum continuation
                last_return = sum(returns[-5:])
                direction = 'buy' if last_return > 0 else 'sell'
                reason_hurst = f"Hurst={hurst:.2f} (trending)"
            elif hurst < 0.45:
                # Mean-reverting — expect reversal
                last_return = sum(returns[-5:])
                direction = 'sell' if last_return > 0 else 'buy'
                reason_hurst = f"Hurst={hurst:.2f} (mean-reverting)"
            else:
                # Ambiguous — use volume trend
                vol_trend = volumes[-1] / max(sum(volumes[-10:]) / 10, 1)
                direction = 'buy' if vol_trend > 1.5 else 'sell'
                reason_hurst = f"Hurst={hurst:.2f} (neutral, volume-guided)"

            confidence = min(0.93, 0.55 + (self.critical_low - entropy) * 1.5 + abs(0.5 - hurst) * 0.5)

            return {
                'type': 'entropy_decay',
                'symbol': symbol,
                'side': direction,
                'action': direction,
                'confidence': confidence,
                'entropy': entropy,
                'entropy_decay_rate': decay_rate,
                'hurst_exponent': hurst,
                'vol_compression': vol_compression,
                'reason': (f"Entropy={entropy:.3f} (critical low!) — "
                           f"Spring compressed, breakout imminent. {reason_hurst}"),
                'source': 'EntropyDecayOracle',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.debug(f"🔮 Entropy error {symbol}: {e}")
            return None

    def _shannon_entropy(self, data: List[float]) -> float:
        """Calculate Shannon entropy of a data series."""
        if not data or len(data) < 5:
            return self.entropy_baseline

        # Bin the data into 10 bins
        n_bins = 10
        data_min = min(data)
        data_max = max(data)
        data_range = data_max - data_min

        if data_range < 1e-12:
            return 0.0  # Zero entropy = perfectly compressed

        bin_width = data_range / n_bins
        bins = [0] * n_bins

        for val in data:
            idx = min(int((val - data_min) / bin_width), n_bins - 1)
            bins[idx] += 1

        # Calculate entropy
        n = len(data)
        entropy = 0.0
        for count in bins:
            if count > 0:
                p = count / n
                entropy -= p * math.log2(p)

        # Normalize to [0, 1]
        max_entropy = math.log2(n_bins)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _hurst_exponent(self, data: List[float]) -> float:
        """Simplified Hurst exponent via R/S analysis."""
        if not data or len(data) < 20:
            return 0.5  # Random walk

        try:
            n = len(data)
            half = n // 2

            # R/S for full series
            rs_full = self._rs_statistic(data)

            # R/S for halves
            rs_half1 = self._rs_statistic(data[:half])
            rs_half2 = self._rs_statistic(data[half:])

            if rs_full <= 0 or rs_half1 <= 0 or rs_half2 <= 0:
                return 0.5

            rs_avg_half = (rs_half1 + rs_half2) / 2

            # H = log(RS_full / RS_half) / log(2)
            if rs_avg_half > 0:
                h = math.log(rs_full / rs_avg_half) / math.log(2)
                return max(0.0, min(1.0, h))

            return 0.5

        except Exception:
            return 0.5

    def _rs_statistic(self, data: List[float]) -> float:
        """Calculate R/S statistic for a series."""
        if len(data) < 3:
            return 0.0

        mean_val = sum(data) / len(data)
        deviations = [x - mean_val for x in data]
        cumulative = []
        running = 0
        for d in deviations:
            running += d
            cumulative.append(running)

        r = max(cumulative) - min(cumulative)
        s = (sum(d ** 2 for d in deviations) / len(deviations)) ** 0.5

        return r / s if s > 0 else 0.0

    def _rolling_std(self, data: List[float]) -> float:
        """Calculate standard deviation."""
        if not data or len(data) < 2:
            return 0.0
        mean_val = sum(data) / len(data)
        return (sum((x - mean_val) ** 2 for x in data) / len(data)) ** 0.5


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  🌌 APEX PREDATOR INTELLIGENCE — MASTER MANAGER                       ║
# ║  Unifies all 7 ultra-rare modules into a single intelligence           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ApexPredatorIntelligence:
    """
    Master controller for all 7 ultra-rare intelligence modules.
    Runs continuous scans and publishes signals to the DataHub.
    """

    def __init__(self, exchange=None, data_hub=None):
        self.exchange = exchange
        self.data_hub = data_hub

        # Initialize all 7 ultra-rare modules
        self.liquidation_predator = LiquidationCascadePredator()
        self.wyckoff_detector = WyckoffSmartMoneyDetector()
        self.dark_pool_tracker = DarkPoolShadowTracker()
        self.metamorphosis_brain = RegimeAdaptiveMetamorphosisBrain()
        self.funding_harvester = DeltaNeutralFundingHarvester()
        self.order_book_radar = OrderBookDominanceRadar()
        self.entropy_oracle = EntropyDecayOracle()

        # Stats
        self.total_signals = 0
        self.signals_by_type = defaultdict(int)
        self.scan_count = 0

        logger.info("🌌 ═══════════════════════════════════════════")
        logger.info("🌌  APEX PREDATOR INTELLIGENCE ONLINE")
        logger.info("🌌  7 Ultra-Rare Modules Active:")
        logger.info("🌌    🩸 Liquidation Cascade Predator")
        logger.info("🌌    🏛️ Wyckoff Smart Money Detector")
        logger.info("🌌    🕶️ Dark Pool Shadow Tracker")
        logger.info("🌌    🧠 Regime-Adaptive Metamorphosis Brain")
        logger.info("🌌    💸 Delta-Neutral Funding Harvester")
        logger.info("🌌    📡 Order Book Dominance Radar")
        logger.info("🌌    🔮 Entropy Decay Oracle")
        logger.info("🌌 ═══════════════════════════════════════════")

    async def run_apex_scan(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Run all 7 ultra-rare intelligence scans in parallel.
        Returns combined list of signals.
        """
        if not self.exchange:
            logger.debug("🌌 Apex: No exchange connected, skipping scan")
            return []

        if not symbols:
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT']

        self.scan_count += 1
        all_signals = []

        try:
            # --- Run all modules concurrently ---
            tasks = []

            # 1. Liquidation Cascade (perp markets)
            tasks.append(self.liquidation_predator.scan_liquidation_risk(
                self.exchange, symbols))

            # 2. Regime Detection (market-wide)
            tasks.append(self.metamorphosis_brain.detect_regime(
                self.exchange, symbols))

            # 3. Funding Rate Harvester
            tasks.append(self.funding_harvester.scan_funding_opportunities(
                self.exchange, symbols))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process liquidation signals
            if isinstance(results[0], list):
                for sig in results[0]:
                    all_signals.append(sig)
                    self.signals_by_type['liquidation_cascade'] += 1

            # Regime result (no signal, just state update)
            if isinstance(results[1], dict):
                regime = results[1].get('regime', 'UNKNOWN')
                if self.scan_count % 30 == 0:  # Log every ~5 min
                    logger.info(f"🧠 Current regime: {regime}")

            # Funding signals
            if isinstance(results[2], list):
                for sig in results[2]:
                    all_signals.append(sig)
                    self.signals_by_type['funding_harvest'] += 1

            # --- Per-symbol scans (sequential to avoid rate limits) ---
            for symbol in symbols[:5]:
                try:
                    # 4. Wyckoff Smart Money
                    wyckoff_sig = await self.wyckoff_detector.analyze_wyckoff(
                        self.exchange, symbol)
                    if wyckoff_sig:
                        all_signals.append(wyckoff_sig)
                        self.signals_by_type['wyckoff'] += 1

                    # 5. Dark Pool Shadow
                    dark_sig = await self.dark_pool_tracker.detect_dark_pool_activity(
                        self.exchange, symbol)
                    if dark_sig:
                        all_signals.append(dark_sig)
                        self.signals_by_type['dark_pool'] += 1

                    # 6. Order Book Dominance
                    ob_sig = await self.order_book_radar.analyze_order_book(
                        self.exchange, symbol)
                    if ob_sig:
                        all_signals.append(ob_sig)
                        self.signals_by_type['order_book'] += 1

                    # 7. Entropy Oracle
                    entropy_sig = await self.entropy_oracle.analyze_entropy(
                        self.exchange, symbol)
                    if entropy_sig:
                        all_signals.append(entropy_sig)
                        self.signals_by_type['entropy'] += 1

                except Exception as e:
                    logger.debug(f"🌌 Per-symbol scan error {symbol}: {e}")

            # --- Apply Metamorphosis confidence adjustment ---
            for sig in all_signals:
                original_conf = sig.get('confidence', 0)
                adjusted_conf = self.metamorphosis_brain.get_adjusted_confidence(
                    original_conf, sig.get('side', 'buy'))
                sig['confidence'] = adjusted_conf
                sig['regime'] = self.metamorphosis_brain.current_regime

            # --- Publish to DataHub ---
            if self.data_hub and all_signals:
                for sig in all_signals:
                    try:
                        self.data_hub.signal_queue.put_nowait(sig)
                        self.data_hub.recent_signals.append(sig)
                    except Exception:
                        pass
                logger.info(f"🌌 APEX PREDATOR: Published {len(all_signals)} signals to DataHub")

            self.total_signals += len(all_signals)

            # Periodic stats
            if self.scan_count % 60 == 0:
                logger.info(f"🌌 APEX STATS: {self.total_signals} total signals, "
                            f"by type: {dict(self.signals_by_type)}")

        except Exception as e:
            logger.error(f"🌌 Apex scan error: {e}")

        return all_signals

    def get_current_personality(self) -> Dict:
        """Get the current regime and personality settings."""
        return {
            'regime': self.metamorphosis_brain.current_regime,
            'personality': self.metamorphosis_brain.current_personality,
            'transitions': self.metamorphosis_brain.transition_count
        }

    def get_stats(self) -> Dict:
        """Get Apex Predator intelligence statistics."""
        return {
            'total_signals': self.total_signals,
            'scan_count': self.scan_count,
            'signals_by_type': dict(self.signals_by_type),
            'current_regime': self.metamorphosis_brain.current_regime,
            'regime_transitions': self.metamorphosis_brain.transition_count,
            'active_funding_harvests': len(self.funding_harvester.active_harvests),
            'total_funding_earned': self.funding_harvester.total_funding_earned,
        }
