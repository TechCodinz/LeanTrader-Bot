"""
⏰ SESSION-AWARE TRADING ENGINE
Trade smarter by knowing when markets are most active

Features:
- London session (7:00-15:00 UTC): EUR, GBP pairs strongest
- New York session (12:00-20:00 UTC): USD pairs strongest  
- Asia session (23:00-7:00 UTC): JPY, AUD pairs strongest
- London-NY overlap (12:00-15:00 UTC): HIGHEST VOLUME - best time to trade
- Crypto: Peak during NY hours (13:00-22:00 UTC)

Adjusts position sizing and confidence based on session
"""

import logging
from typing import Dict, Tuple, List
from datetime import datetime, timezone, time as dt_time

logger = logging.getLogger(__name__)


class SessionAwareTrading:
    """
    Session-aware trading adjustments
    Increases confidence and position size during optimal sessions
    """
    
    def __init__(self):
        self.enabled = True
        
        # Session times (UTC)
        self.sessions = {
            'ASIA': (dt_time(23, 0), dt_time(7, 0)),
            'LONDON': (dt_time(7, 0), dt_time(15, 0)),
            'NEW_YORK': (dt_time(12, 0), dt_time(20, 0)),
            'CRYPTO_PEAK': (dt_time(13, 0), dt_time(22, 0))
        }
        
        logger.info("⏰ Session-Aware Trading initialized")
    
    def get_current_session(self) -> str:
        """Get current trading session"""
        now = datetime.now(timezone.utc).time()
        
        # Check for overlaps first (highest priority)
        if self._in_session('LONDON', now) and self._in_session('NEW_YORK', now):
            return 'LONDON-NY_OVERLAP'  # BEST TIME TO TRADE!
        
        # Individual sessions
        if self._in_session('ASIA', now):
            return 'ASIA'
        if self._in_session('LONDON', now):
            return 'LONDON'
        if self._in_session('NEW_YORK', now):
            return 'NEW_YORK'
        
        return 'OFF_HOURS'
    
    def _in_session(self, session: str, current_time: dt_time) -> bool:
        """Check if current time is in session"""
        start, end = self.sessions[session]
        
        # Handle sessions that cross midnight
        if start > end:
            return current_time >= start or current_time <= end
        else:
            return start <= current_time <= end
    
    def get_session_multiplier(self, symbol: str) -> Tuple[str, float]:
        """
        Get session-based confidence multiplier for a symbol
        
        Returns: (session_name, multiplier)
        """
        session = self.get_current_session()
        symbol_upper = symbol.upper()
        
        # Default multiplier
        multiplier = 1.0
        
        # LONDON-NY OVERLAP - Best time for ALL pairs
        if session == 'LONDON-NY_OVERLAP':
            multiplier = 1.25  # 25% boost during highest volume
            logger.info(f"⏰ LONDON-NY OVERLAP active - {symbol} gets 25% confidence boost!")
        
        # London session - Best for EUR, GBP
        elif session == 'LONDON':
            if 'EUR' in symbol_upper or 'GBP' in symbol_upper:
                multiplier = 1.15
            else:
                multiplier = 1.05
        
        # NY session - Best for USD pairs, crypto
        elif session == 'NEW_YORK':
            if 'USD' in symbol_upper or '/USDT' in symbol_upper:
                multiplier = 1.10
            else:
                multiplier = 1.05
        
        # Asia session - Best for JPY, AUD, Asian crypto
        elif session == 'ASIA':
            if 'JPY' in symbol_upper or 'AUD' in symbol_upper or 'CNY' in symbol_upper:
                multiplier = 1.15
            else:
                multiplier = 0.95  # Reduce for non-Asian pairs
        
        # Off hours - Reduce risk
        elif session == 'OFF_HOURS':
            multiplier = 0.80  # 20% reduction in low volume
        
        # Crypto-specific adjustment
        if '/USDT' in symbol_upper or '/USDC' in symbol_upper:
            crypto_session = self.get_current_session()
            if self._in_session('CRYPTO_PEAK', datetime.now(timezone.utc).time()):
                multiplier *= 1.10  # Extra 10% during crypto peak hours
        
        return (session, multiplier)
    
    def adjust_signal_for_session(self, signal: Dict) -> Dict:
        """
        Adjust signal confidence and position size based on session
        
        Input signal, returns adjusted signal
        """
        symbol = signal.get('symbol', '')
        
        if not symbol or not self.enabled:
            return signal
        
        session, multiplier = self.get_session_multiplier(symbol)
        
        # Adjust confidence
        original_confidence = signal.get('confidence', 0.5)
        adjusted_confidence = min(0.98, original_confidence * multiplier)
        
        # Add session info to signal
        signal['session'] = session
        signal['session_multiplier'] = multiplier
        signal['original_confidence'] = original_confidence
        signal['confidence'] = adjusted_confidence
        
        # Add to reasoning
        if multiplier > 1.0:
            session_note = f"\n⏰ Session Boost: {session} session (+{(multiplier-1)*100:.0f}% confidence)"
            signal['reasoning'] = signal.get('reasoning', '') + session_note
        elif multiplier < 1.0:
            session_note = f"\n⏰ Off-hours: {session} ({multiplier*100:.0f}% confidence)"
            signal['reasoning'] = signal.get('reasoning', '') + session_note
        
        if multiplier != 1.0:
            logger.info(
                f"⏰ {symbol}: {session} session → "
                f"{original_confidence*100:.0f}% → {adjusted_confidence*100:.0f}% "
                f"({multiplier:.2f}x)"
            )
        
        return signal
    
    def should_trade_now(self, symbol: str) -> bool:
        """Quick check if symbol should be traded in current session"""
        session, multiplier = self.get_session_multiplier(symbol)
        
        # Don't trade if multiplier is too low
        return multiplier >= 0.70
    
    def get_optimal_pairs_for_session(self) -> List[str]:
        """Get list of optimal pairs for current session"""
        session = self.get_current_session()
        
        if session == 'LONDON-NY_OVERLAP':
            # All pairs good during overlap
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
        
        elif session == 'LONDON':
            # Euro/GBP focus
            return ['BTC/USDT', 'ETH/USDT', 'EUR/USD', 'GBP/USD']
        
        elif session == 'NEW_YORK':
            # Crypto + USD pairs
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'PEPE/USDT']
        
        elif session == 'ASIA':
            # Asia-focused
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        else:  # OFF_HOURS
            # Only major liquid pairs
            return ['BTC/USDT', 'ETH/USDT']
