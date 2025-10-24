
"""
ADVANCED TRADING ACTIONS
Professional trading strategies beyond simple buy/sell
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedTradingActions:
    """
    Determines optimal trading action based on:
    - Market conditions
    - Timeframe
    - Confidence level
    - Volatility
    - Position context
    """
    
    # Trading action types
    ACTIONS = {
        'LONG': 'Open leveraged long position (futures/perpetual)',
        'SHORT': 'Open leveraged short position (futures/perpetual)',
        'SCALP': 'Quick scalp trade (1-5 min)',
        'SWING': 'Swing trade (hours to days)',
        'DCA': 'Dollar cost average (accumulate)',
        'GRID': 'Grid trading (buy low, sell high repeatedly)',
        'ARBITRAGE': 'Cross-exchange arbitrage',
        'HEDGE': 'Hedge existing position',
        'SPOT_BUY': 'Simple spot buy',
        'SPOT_SELL': 'Simple spot sell',
        'TAKE_PROFIT': 'Take profit on existing position',
        'STOP_LOSS': 'Stop loss on existing position',
        'TRAILING_STOP': 'Trailing stop on existing position',
        'CLOSE_LONG': 'Close long position',
        'CLOSE_SHORT': 'Close short position',
    }
    
    def __init__(self):
        logger.info("📊 Advanced Trading Actions initialized")
        logger.info(f"   Available actions: {len(self.ACTIONS)}")
    
    def determine_action(
        self,
        symbol: str,
        direction: str,  # 'buy' or 'sell'
        confidence: float,
        timeframe: str = '1h',
        volatility: float = 0.02,
        market_regime: str = 'neutral',
        has_position: bool = False
    ) -> Dict[str, any]:
        """
        Determine optimal trading action based on multiple factors
        
        Returns: {
            'action': action type,
            'reason': why this action,
            'leverage': suggested leverage (if applicable),
            'duration': expected hold time
        }
        """
        
        # If we have an existing position, consider exit actions
        if has_position:
            if confidence < 0.70:
                return {
                    'action': 'STOP_LOSS',
                    'reason': 'Low confidence, protect position',
                    'leverage': 1,
                    'duration': 'immediate'
                }
            elif confidence > 0.85:
                return {
                    'action': 'TRAILING_STOP',
                    'reason': 'High confidence, let profits run',
                    'leverage': 1,
                    'duration': 'until trend break'
                }
        
        # HIGH CONFIDENCE (85%+) = Aggressive strategies
        if confidence >= 0.85:
            if timeframe in ['1m', '5m', '15m']:
                # Short timeframe + high confidence = SCALP
                return {
                    'action': 'SCALP',
                    'reason': 'High confidence + short timeframe',
                    'leverage': 3 if direction == 'buy' else 3,
                    'duration': '1-5 minutes'
                }
            
            elif volatility > 0.03:
                # High volatility + high confidence = LEVERAGE
                if direction == 'buy':
                    return {
                        'action': 'LONG',
                        'reason': 'High confidence + high volatility uptrend',
                        'leverage': 7,
                        'duration': '15min-2h'
                    }
                else:
                    return {
                        'action': 'SHORT',
                        'reason': 'High confidence + high volatility downtrend',
                        'leverage': 7,
                        'duration': '15min-2h'
                    }
            
            elif timeframe in ['4h', '1d']:
                # Long timeframe + high confidence = SWING
                return {
                    'action': 'SWING',
                    'reason': 'High confidence + long timeframe trend',
                    'leverage': 2,
                    'duration': '1-5 days'
                }
        
        # MEDIUM-HIGH CONFIDENCE (75-85%) = Moderate strategies
        elif confidence >= 0.75:
            if market_regime == 'sideways' or volatility < 0.01:
                # Sideways market = GRID TRADING
                return {
                    'action': 'GRID',
                    'reason': 'Sideways market, profit from oscillation',
                    'leverage': 1,
                    'duration': 'until breakout'
                }
            
            elif direction == 'buy' and volatility < 0.02:
                # Stable uptrend = DCA
                return {
                    'action': 'DCA',
                    'reason': 'Stable uptrend, accumulate gradually',
                    'leverage': 1,
                    'duration': 'hours to days'
                }
            
            else:
                # Standard leveraged position
                if direction == 'buy':
                    return {
                        'action': 'LONG',
                        'reason': 'Medium-high confidence uptrend',
                        'leverage': 3,
                        'duration': '1-4 hours'
                    }
                else:
                    return {
                        'action': 'SHORT',
                        'reason': 'Medium-high confidence downtrend',
                        'leverage': 3,
                        'duration': '1-4 hours'
                    }
        
        # MEDIUM CONFIDENCE (65-75%) = Conservative strategies
        else:
            if volatility > 0.05:
                # High volatility + medium confidence = HEDGE
                return {
                    'action': 'HEDGE',
                    'reason': 'High volatility, protect portfolio',
                    'leverage': 1,
                    'duration': 'until volatility drops'
                }
            
            else:
                # Standard spot trade
                if direction == 'buy':
                    return {
                        'action': 'SPOT_BUY',
                        'reason': 'Medium confidence, safe spot trade',
                        'leverage': 1,
                        'duration': 'flexible'
                    }
                else:
                    return {
                        'action': 'SPOT_SELL',
                        'reason': 'Medium confidence, safe spot sell',
                        'leverage': 1,
                        'duration': 'flexible'
                    }
    
    def check_arbitrage_opportunity(
        self,
        symbol: str,
        price_diff_pct: float
    ) -> Optional[Dict]:
        """
        Check if arbitrage opportunity exists
        """
        if price_diff_pct > 0.5:  # 0.5% price difference
            return {
                'action': 'ARBITRAGE',
                'reason': f'Price difference: {price_diff_pct:.2f}%',
                'leverage': 1,
                'duration': 'immediate'
            }
        return None


# Global instance
_advanced_actions = None

def get_advanced_actions():
    """Get singleton instance"""
    global _advanced_actions
    if _advanced_actions is None:
        _advanced_actions = AdvancedTradingActions()
    return _advanced_actions
