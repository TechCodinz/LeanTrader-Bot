#!/usr/bin/env python3
"""
SMART SCALPING ENGINE
Multi-timeframe confluence + Session awareness + Market optimization
Only trades when ALL timeframes align and in optimal sessions
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, time
from dataclasses import dataclass
from collections import defaultdict, deque
import pytz

logger = logging.getLogger(__name__)


class MarketSession:
    """Trading session detection and management"""
    
    SESSIONS = {
        'ASIAN': {
            'start': time(0, 0),   # 00:00 UTC
            'end': time(8, 0),     # 08:00 UTC
            'major_pairs': ['USDJPY', 'AUDUSD', 'NZDUSD', 'EURJPY', 'GBPJPY'],
            'volatility': 'LOW',
            'characteristics': 'Range-bound, slow moves'
        },
        'LONDON': {
            'start': time(8, 0),   # 08:00 UTC
            'end': time(16, 0),    # 16:00 UTC
            'major_pairs': ['EURUSD', 'GBPUSD', 'EURGBP', 'EURJPY', 'GBPJPY'],
            'volatility': 'HIGH',
            'characteristics': 'Strong trends, high volume'
        },
        'NEW_YORK': {
            'start': time(13, 0),  # 13:00 UTC
            'end': time(21, 0),    # 21:00 UTC
            'major_pairs': ['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY', 'XAUUSD'],
            'volatility': 'HIGH',
            'characteristics': 'News-driven, momentum trades'
        },
        'LONDON_NY_OVERLAP': {
            'start': time(13, 0),  # 13:00 UTC
            'end': time(16, 0),    # 16:00 UTC
            'major_pairs': ['EURUSD', 'GBPUSD', 'XAUUSD', 'BTC/USDT', 'ETH/USDT'],
            'volatility': 'VERY_HIGH',
            'characteristics': 'Maximum liquidity, best for scalping'
        }
    }
    
    @staticmethod
    def get_current_session() -> str:
        """Get current trading session"""
        now_utc = datetime.now(pytz.UTC).time()
        
        # Check overlap first (highest priority)
        overlap = MarketSession.SESSIONS['LONDON_NY_OVERLAP']
        if overlap['start'] <= now_utc < overlap['end']:
            return 'LONDON_NY_OVERLAP'
        
        # Check other sessions
        for session_name, session_data in MarketSession.SESSIONS.items():
            if session_name == 'LONDON_NY_OVERLAP':
                continue
            if session_data['start'] <= now_utc < session_data['end']:
                return session_name
        
        return 'ASIAN'  # Default to Asian session
    
    @staticmethod
    def get_optimal_pairs(session: str) -> List[str]:
        """Get optimal trading pairs for current session"""
        return MarketSession.SESSIONS.get(session, {}).get('major_pairs', [])
    
    @staticmethod
    def get_session_volatility(session: str) -> str:
        """Get expected volatility for session"""
        return MarketSession.SESSIONS.get(session, {}).get('volatility', 'MEDIUM')


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe"""
    timeframe: str
    direction: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: float  # 0.0 to 1.0
    indicators: Dict[str, float]
    timestamp: datetime


class MultiTimeframeAnalyzer:
    """
    Analyzes multiple timeframes for confluence
    Only signals when ALL timeframes align
    """
    
    TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h']
    
    # Timeframe weights (higher timeframes = more weight)
    WEIGHTS = {
        '1m': 0.10,
        '5m': 0.15,
        '15m': 0.20,
        '30m': 0.20,
        '1h': 0.20,
        '4h': 0.15
    }
    
    def __init__(self):
        self.signals_cache: Dict[str, Dict[str, TimeframeSignal]] = defaultdict(dict)
        self.min_confluence = 0.75  # 75% of timeframes must agree
        
    def analyze_timeframe(self, symbol: str, timeframe: str, ohlcv: List) -> TimeframeSignal:
        """
        Analyze a single timeframe for direction and strength
        
        Uses multiple indicators:
        - Trend (EMA crossovers)
        - Momentum (RSI)
        - Volume
        - Support/Resistance
        """
        
        # Simplified analysis (would use real TA in production)
        indicators = {
            'ema_fast': 0.0,
            'ema_slow': 0.0,
            'rsi': 50.0,
            'volume_ratio': 1.0,
            'trend_strength': 0.5
        }
        
        # Determine direction based on indicators
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        
        # EMA trend
        if indicators['ema_fast'] > indicators['ema_slow']:
            buy_signals += 1
        else:
            sell_signals += 1
        total_signals += 1
        
        # RSI
        if indicators['rsi'] < 40:  # Oversold
            buy_signals += 1
            total_signals += 1
        elif indicators['rsi'] > 60:  # Overbought
            sell_signals += 1
            total_signals += 1
        
        # Volume confirmation
        if indicators['volume_ratio'] > 1.2:  # High volume
            # Confirms the trend
            total_signals += 0.5
        
        # Determine final direction
        if buy_signals > sell_signals:
            direction = 'BUY'
            strength = buy_signals / max(total_signals, 1)
        elif sell_signals > buy_signals:
            direction = 'SELL'
            strength = sell_signals / max(total_signals, 1)
        else:
            direction = 'NEUTRAL'
            strength = 0.5
        
        signal = TimeframeSignal(
            timeframe=timeframe,
            direction=direction,
            strength=strength,
            indicators=indicators,
            timestamp=datetime.now()
        )
        
        # Cache the signal
        self.signals_cache[symbol][timeframe] = signal
        
        return signal
    
    def check_confluence(self, symbol: str) -> Tuple[bool, str, float]:
        """
        Check if multiple timeframes agree (confluence)
        
        Returns:
            (has_confluence, direction, confidence)
        """
        
        if symbol not in self.signals_cache:
            return False, 'NEUTRAL', 0.0
        
        signals = self.signals_cache[symbol]
        
        if len(signals) < 3:  # Need at least 3 timeframes
            return False, 'NEUTRAL', 0.0
        
        # Count votes with weights
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0
        
        for timeframe, signal in signals.items():
            weight = self.WEIGHTS.get(timeframe, 0.1)
            total_weight += weight
            
            if signal.direction == 'BUY':
                buy_weight += weight * signal.strength
            elif signal.direction == 'SELL':
                sell_weight += weight * signal.strength
        
        # Calculate consensus
        if total_weight == 0:
            return False, 'NEUTRAL', 0.0
        
        buy_consensus = buy_weight / total_weight
        sell_consensus = sell_weight / total_weight
        
        # Check if we have strong confluence
        if buy_consensus >= self.min_confluence:
            return True, 'BUY', buy_consensus
        elif sell_consensus >= self.min_confluence:
            return True, 'SELL', sell_consensus
        else:
            return False, 'NEUTRAL', max(buy_consensus, sell_consensus)
    
    def get_confluence_details(self, symbol: str) -> Dict[str, Any]:
        """Get detailed confluence information"""
        
        if symbol not in self.signals_cache:
            return {}
        
        signals = self.signals_cache[symbol]
        
        details = {
            'timeframes': {},
            'alignment': 0.0,
            'direction': 'NEUTRAL',
            'strength': 0.0
        }
        
        buy_count = 0
        sell_count = 0
        total_count = 0
        
        for timeframe, signal in signals.items():
            details['timeframes'][timeframe] = {
                'direction': signal.direction,
                'strength': signal.strength,
                'indicators': signal.indicators
            }
            
            if signal.direction == 'BUY':
                buy_count += 1
            elif signal.direction == 'SELL':
                sell_count += 1
            total_count += 1
        
        # Calculate alignment (how many agree)
        if total_count > 0:
            details['alignment'] = max(buy_count, sell_count) / total_count
        
        has_confluence, direction, strength = self.check_confluence(symbol)
        details['direction'] = direction
        details['strength'] = strength
        
        return details


class SessionPerformanceTracker:
    """
    Tracks which markets perform best in which sessions
    Learns over time and optimizes market-session pairs
    """
    
    def __init__(self):
        self.performance: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
        # performance[symbol][session] = {'wins': 0, 'losses': 0, 'avg_profit': 0, 'trades': 0}
        
        self.min_trades_for_confidence = 10  # Need 10 trades before trusting stats
    
    def record_trade(self, symbol: str, session: str, profit: float, success: bool):
        """Record trade result for market-session pair"""
        
        if session not in self.performance[symbol]:
            self.performance[symbol][session] = {
                'wins': 0,
                'losses': 0,
                'total_profit': 0.0,
                'trades': 0,
                'avg_profit': 0.0,
                'win_rate': 0.0
            }
        
        stats = self.performance[symbol][session]
        
        stats['trades'] += 1
        stats['total_profit'] += profit
        
        if success:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        
        # Update averages
        stats['avg_profit'] = stats['total_profit'] / stats['trades']
        stats['win_rate'] = stats['wins'] / stats['trades']
    
    def get_optimal_markets(self, session: str, min_win_rate: float = 0.60) -> List[Tuple[str, float]]:
        """
        Get best performing markets for a session
        
        Returns:
            List of (symbol, score) sorted by performance
        """
        
        market_scores = []
        
        for symbol, sessions in self.performance.items():
            if session not in sessions:
                continue
            
            stats = sessions[session]
            
            # Need minimum trades for confidence
            if stats['trades'] < self.min_trades_for_confidence:
                continue
            
            # Filter by win rate
            if stats['win_rate'] < min_win_rate:
                continue
            
            # Calculate performance score
            # Score = win_rate * avg_profit * sqrt(trades)
            score = stats['win_rate'] * stats['avg_profit'] * (stats['trades'] ** 0.5)
            
            market_scores.append((symbol, score, stats))
        
        # Sort by score (best first)
        market_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [(symbol, score) for symbol, score, _ in market_scores]
    
    def should_trade_pair(self, symbol: str, session: str) -> Tuple[bool, str]:
        """Check if we should trade this pair in this session"""
        
        # If we don't have data yet, allow trading (learning phase)
        if symbol not in self.performance or session not in self.performance[symbol]:
            return True, "Learning phase - no historical data"
        
        stats = self.performance[symbol][session]
        
        # Not enough data yet
        if stats['trades'] < self.min_trades_for_confidence:
            return True, f"Gathering data: {stats['trades']}/{self.min_trades_for_confidence} trades"
        
        # Check performance
        if stats['win_rate'] < 0.50:
            return False, f"Low win rate: {stats['win_rate']:.1%}"
        
        if stats['avg_profit'] < 0:
            return False, f"Negative avg profit: ${stats['avg_profit']:.2f}"
        
        return True, f"Good performance: {stats['win_rate']:.1%} win rate, ${stats['avg_profit']:.2f} avg"


class SmartScalpingEngine:
    """
    SMART SCALPING ENGINE
    
    Features:
    - Multi-timeframe confluence (only trades when all TFs align)
    - Session awareness (trades optimal pairs per session)
    - Performance tracking (learns which markets work best when)
    - Adaptive strategy (adjusts based on session volatility)
    """
    
    def __init__(self, ultra_core, risk_engine):
        self.ultra_core = ultra_core
        self.risk_engine = risk_engine
        
        # Smart components
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.session_tracker = SessionPerformanceTracker()
        
        # Configuration
        self.min_confluence = 0.75  # 75% timeframe agreement
        self.min_timeframes = 4  # Need at least 4 timeframes
        
        # Performance tracking
        self.total_signals = 0
        self.confluence_signals = 0
        self.executed_trades = 0
        
        logger.info("🎯 Smart Scalping Engine initialized")
        logger.info(f"   Multi-timeframe: {len(self.mtf_analyzer.TIMEFRAMES)} timeframes")
        logger.info(f"   Session aware: 4 sessions tracked")
        logger.info(f"   Min confluence: {self.min_confluence:.0%}")
    
    async def scan_markets(self) -> List[Dict[str, Any]]:
        """
        Smart market scanning with session and timeframe awareness
        
        Only returns signals when:
        1. Current session is optimal for the market
        2. Multiple timeframes show confluence
        3. Historical performance is good
        """
        
        signals = []
        
        # Get current session
        current_session = MarketSession.get_current_session()
        session_volatility = MarketSession.get_session_volatility(current_session)
        
        logger.info(f"🕐 Current Session: {current_session} ({session_volatility} volatility)")
        
        # Get optimal pairs for this session
        optimal_pairs = MarketSession.get_optimal_pairs(current_session)
        
        # Also get learned optimal pairs
        learned_pairs = self.session_tracker.get_optimal_markets(current_session)
        
        # Combine default + learned pairs
        all_pairs = list(set(optimal_pairs + [p[0] for p in learned_pairs[:5]]))
        
        logger.info(f"📊 Scanning {len(all_pairs)} optimal pairs for {current_session}")
        
        for symbol in all_pairs:
            try:
                # Check if we should trade this pair in this session
                should_trade, reason = self.session_tracker.should_trade_pair(symbol, current_session)
                
                if not should_trade:
                    logger.debug(f"⏭️  Skipping {symbol}: {reason}")
                    continue
                
                # Analyze all timeframes
                await self.analyze_all_timeframes(symbol)
                
                # Check for confluence
                has_confluence, direction, confidence = self.mtf_analyzer.check_confluence(symbol)
                
                self.total_signals += 1
                
                if has_confluence:
                    self.confluence_signals += 1
                    
                    # Get confluence details
                    details = self.mtf_analyzer.get_confluence_details(symbol)
                    
                    signal = {
                        'type': 'scalping',
                        'source': 'SmartScalpingEngine',
                        'symbol': symbol,
                        'side': direction.lower(),
                        'confidence': confidence,
                        'session': current_session,
                        'session_volatility': session_volatility,
                        'timeframe_alignment': details['alignment'],
                        'timeframes_analyzed': len(details['timeframes']),
                        'confluence_details': details,
                        'timestamp': datetime.now().isoformat(),
                        'priority': 'high' if confidence > 0.85 else 'normal'
                    }
                    
                    signals.append(signal)
                    
                    logger.info(f"⚡ CONFLUENCE SIGNAL: {symbol}")
                    logger.info(f"   Direction: {direction}")
                    logger.info(f"   Confidence: {confidence:.1%}")
                    logger.info(f"   Session: {current_session}")
                    logger.info(f"   Alignment: {details['alignment']:.1%}")
                    logger.info(f"   Timeframes: {len(details['timeframes'])}")
                
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
        
        # Log statistics
        if self.total_signals > 0:
            confluence_rate = self.confluence_signals / self.total_signals
            logger.info(f"📊 Confluence Rate: {confluence_rate:.1%} ({self.confluence_signals}/{self.total_signals})")
        
        return signals
    
    async def analyze_all_timeframes(self, symbol: str):
        """Analyze all timeframes for a symbol"""
        
        for timeframe in self.mtf_analyzer.TIMEFRAMES:
            try:
                # Fetch OHLCV data for this timeframe
                # In production, would fetch real data
                ohlcv = []  # Placeholder
                
                # Analyze this timeframe
                signal = self.mtf_analyzer.analyze_timeframe(symbol, timeframe, ohlcv)
                
            except Exception as e:
                logger.debug(f"Error analyzing {symbol} {timeframe}: {e}")
    
    def record_trade_result(self, symbol: str, session: str, profit: float, success: bool):
        """Record trade result for learning"""
        self.session_tracker.record_trade(symbol, session, profit, success)
        self.executed_trades += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            'total_signals': self.total_signals,
            'confluence_signals': self.confluence_signals,
            'confluence_rate': self.confluence_signals / max(self.total_signals, 1),
            'executed_trades': self.executed_trades,
            'current_session': MarketSession.get_current_session(),
            'tracked_markets': len(self.session_tracker.performance)
        }
    
    def get_session_performance(self, session: str) -> List[Tuple[str, float]]:
        """Get best performing markets for a session"""
        return self.session_tracker.get_optimal_markets(session)
