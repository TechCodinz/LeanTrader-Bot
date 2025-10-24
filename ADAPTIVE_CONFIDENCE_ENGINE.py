#!/usr/bin/env python3
"""
ADAPTIVE CONFIDENCE ENGINE
Dynamically adjusts confidence thresholds based on:
- Market regime (trending/sideways/volatile)
- Pair characteristics (volatility, volume)
- Historical performance
- News/sentiment
- Trading session
- Recent win rate

This replaces static 80% threshold with intelligent 65-95% adaptive range
"""
import logging
from datetime import datetime, time
from typing import Dict, Optional
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveConfidenceEngine:
    """
    Intelligently adjusts confidence thresholds for maximum profitability
    
    Key Features:
    - Lower threshold in strong trends (catch momentum)
    - Higher threshold in choppy markets (avoid false signals)
    - Per-pair adjustment based on historical performance
    - Time-based adjustment (best sessions get lower threshold)
    - News-aware (lower threshold during high-impact news)
    """
    
    def __init__(self):
        # Base thresholds
        self.base_min_confidence = 0.75  # Default: 75%
        self.absolute_min = 0.65  # Never go below 65%
        self.absolute_max = 0.95  # Never go above 95%
        
        # Track per-pair performance
        self.pair_performance = {}  # {pair: {'wins': int, 'losses': int}}
        self.recent_trades = deque(maxlen=100)  # Last 100 trades
        
        # Market regime detection
        self.current_regime = "neutral"  # trending/sideways/volatile/neutral
        
        logger.info("🧠 Adaptive Confidence Engine initialized")
        logger.info(f"   Base threshold: {self.base_min_confidence*100}%")
        logger.info(f"   Range: {self.absolute_min*100}%-{self.absolute_max*100}%")
    
    def get_adaptive_threshold(
        self,
        pair: str,
        market_regime: str = "neutral",
        volatility: float = 0.0,
        volume_24h: float = 0.0,
        news_impact: str = "none",
        confidence: float = 0.0
    ) -> Dict[str, any]:
        """
        Calculate adaptive confidence threshold for a trading decision
        
        Returns: {
            'threshold': float,  # Actual threshold to use
            'adjustments': dict, # Breakdown of adjustments
            'reason': str       # Human-readable explanation
        }
        """
        
        threshold = self.base_min_confidence
        adjustments = {}
        reasons = []
        
        # ============================================================
        # 1. MARKET REGIME ADJUSTMENT (-10% to +15%)
        # ============================================================
        if market_regime == "trending":
            # Strong trend = lower threshold (catch momentum)
            adjustment = -0.10  # -10%
            adjustments['market_regime'] = adjustment
            threshold += adjustment
            reasons.append(f"trending market (-10%)")
            
        elif market_regime == "volatile":
            # High volatility = slightly lower (opportunities)
            adjustment = -0.05  # -5%
            adjustments['market_regime'] = adjustment
            threshold += adjustment
            reasons.append(f"volatile market (-5%)")
            
        elif market_regime == "sideways":
            # Choppy = higher threshold (avoid false signals)
            adjustment = +0.10  # +10%
            adjustments['market_regime'] = adjustment
            threshold += adjustment
            reasons.append(f"sideways market (+10%)")
        
        # ============================================================
        # 2. PAIR VOLATILITY ADJUSTMENT (-5% to +5%)
        # ============================================================
        if volatility > 5.0:
            # High volatility = lower threshold (big moves = opportunity)
            adjustment = -0.05
            adjustments['volatility'] = adjustment
            threshold += adjustment
            reasons.append(f"high volatility (-5%)")
        elif volatility < 1.0:
            # Low volatility = higher threshold (wait for certainty)
            adjustment = +0.05
            adjustments['volatility'] = adjustment
            threshold += adjustment
            reasons.append(f"low volatility (+5%)")
        
        # ============================================================
        # 3. VOLUME ADJUSTMENT (-5% to 0%)
        # ============================================================
        if volume_24h > 10000000:  # > $10M volume
            # High volume = lower threshold (liquid = reliable)
            adjustment = -0.05
            adjustments['volume'] = adjustment
            threshold += adjustment
            reasons.append(f"high volume (-5%)")
        
        # ============================================================
        # 4. HISTORICAL PERFORMANCE ADJUSTMENT (-10% to +10%)
        # ============================================================
        if pair in self.pair_performance:
            perf = self.pair_performance[pair]
            total = perf['wins'] + perf['losses']
            
            if total >= 5:  # Need at least 5 trades
                win_rate = perf['wins'] / total
                
                if win_rate >= 0.70:
                    # High win rate = trust this pair, lower threshold
                    adjustment = -0.10
                    adjustments['history'] = adjustment
                    threshold += adjustment
                    reasons.append(f"70%+ win rate (-10%)")
                    
                elif win_rate <= 0.40:
                    # Low win rate = be cautious, higher threshold
                    adjustment = +0.10
                    adjustments['history'] = adjustment
                    threshold += adjustment
                    reasons.append(f"<40% win rate (+10%)")
        
        # ============================================================
        # 5. NEWS/SENTIMENT ADJUSTMENT (-10% to 0%)
        # ============================================================
        if news_impact == "high":
            # Major news = lower threshold (capture event-driven moves)
            adjustment = -0.10
            adjustments['news'] = adjustment
            threshold += adjustment
            reasons.append(f"high-impact news (-10%)")
        elif news_impact == "medium":
            adjustment = -0.05
            adjustments['news'] = adjustment
            threshold += adjustment
            reasons.append(f"medium news (-5%)")
        
        # ============================================================
        # 6. TRADING SESSION ADJUSTMENT (-5% to 0%)
        # ============================================================
        current_hour = datetime.utcnow().hour
        
        # Best sessions (high liquidity)
        if 8 <= current_hour <= 16:  # European + US overlap
            adjustment = -0.05
            adjustments['session'] = adjustment
            threshold += adjustment
            reasons.append(f"prime trading session (-5%)")
        
        # ============================================================
        # 7. RECENT WIN STREAK ADJUSTMENT (-5% to +5%)
        # ============================================================
        if len(self.recent_trades) >= 10:
            recent_wins = sum(1 for t in list(self.recent_trades)[-10:] if t['result'] == 'win')
            recent_win_rate = recent_wins / 10
            
            if recent_win_rate >= 0.70:
                # Hot streak = be aggressive
                adjustment = -0.05
                adjustments['streak'] = adjustment
                threshold += adjustment
                reasons.append(f"hot streak 70%+ (-5%)")
                
            elif recent_win_rate <= 0.30:
                # Cold streak = be conservative
                adjustment = +0.05
                adjustments['streak'] = adjustment
                threshold += adjustment
                reasons.append(f"cold streak <30% (+5%)")
        
        # ============================================================
        # 8. CONFIDENCE PROXIMITY ADJUSTMENT (0% to -5%)
        # ============================================================
        # If confidence is very high (>90%), can afford to lower threshold
        if confidence >= 0.90:
            adjustment = -0.05
            adjustments['high_confidence'] = adjustment
            threshold += adjustment
            reasons.append(f"very high confidence (-5%)")
        
        # ============================================================
        # ENFORCE LIMITS
        # ============================================================
        threshold = max(self.absolute_min, min(self.absolute_max, threshold))
        
        # Build reason string
        reason = "Adaptive: " + ", ".join(reasons) if reasons else "Base threshold"
        
        return {
            'threshold': threshold,
            'adjustments': adjustments,
            'reason': reason,
            'base': self.base_min_confidence
        }
    
    def record_trade_result(self, pair: str, result: str, confidence: float):
        """
        Record trade outcome for learning
        
        Args:
            pair: Trading pair
            result: 'win' or 'loss'
            confidence: Confidence level used
        """
        
        # Update pair performance
        if pair not in self.pair_performance:
            self.pair_performance[pair] = {'wins': 0, 'losses': 0}
        
        if result == 'win':
            self.pair_performance[pair]['wins'] += 1
        else:
            self.pair_performance[pair]['losses'] += 1
        
        # Add to recent trades
        self.recent_trades.append({
            'pair': pair,
            'result': result,
            'confidence': confidence,
            'timestamp': datetime.now()
        })
        
        # Log learning
        total = self.pair_performance[pair]['wins'] + self.pair_performance[pair]['losses']
        win_rate = self.pair_performance[pair]['wins'] / total
        
        logger.info(f"📊 {pair} recorded: {result} (confidence: {confidence:.1%})")
        logger.info(f"   Win rate: {win_rate:.1%} ({self.pair_performance[pair]['wins']}/{total})")
    
    def get_stats(self) -> Dict:
        """Get current engine statistics"""
        
        total_trades = len(self.recent_trades)
        recent_wins = sum(1 for t in self.recent_trades if t['result'] == 'win')
        overall_win_rate = recent_wins / total_trades if total_trades > 0 else 0
        
        # Best/worst pairs
        best_pairs = []
        worst_pairs = []
        
        for pair, perf in self.pair_performance.items():
            total = perf['wins'] + perf['losses']
            if total >= 5:
                win_rate = perf['wins'] / total
                if win_rate >= 0.70:
                    best_pairs.append((pair, win_rate, total))
                elif win_rate <= 0.40:
                    worst_pairs.append((pair, win_rate, total))
        
        best_pairs.sort(key=lambda x: x[1], reverse=True)
        worst_pairs.sort(key=lambda x: x[1])
        
        return {
            'total_trades': total_trades,
            'overall_win_rate': overall_win_rate,
            'best_pairs': best_pairs[:5],
            'worst_pairs': worst_pairs[:5],
            'tracked_pairs': len(self.pair_performance),
            'current_regime': self.current_regime
        }


# Global singleton
_adaptive_engine = None

def get_adaptive_confidence_engine():
    """Get or create adaptive confidence engine"""
    global _adaptive_engine
    if _adaptive_engine is None:
        _adaptive_engine = AdaptiveConfidenceEngine()
    return _adaptive_engine
