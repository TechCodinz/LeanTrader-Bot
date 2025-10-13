#!/usr/bin/env python3
"""
UTILITY INTEGRATION LAYER
Integrates all utility files: skillbook, sizer, guardrails, indicators, etc.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import all utility modules
try:
    import skillbook
    SKILLBOOK_AVAILABLE = True
except:
    SKILLBOOK_AVAILABLE = False

try:
    import sizer
    SIZER_AVAILABLE = True
except:
    SIZER_AVAILABLE = False

try:
    from guardrails import TradeGuard, GuardConfig
    GUARDRAILS_AVAILABLE = True
except:
    GUARDRAILS_AVAILABLE = False

try:
    import indicators
    INDICATORS_AVAILABLE = True
except:
    INDICATORS_AVAILABLE = False

logger = logging.getLogger(__name__)


class UtilityIntegrationLayer:
    """
    Central integration layer for all utility functions
    Provides unified access to: sizing, guardrails, indicators, skillbook
    """
    
    def __init__(self):
        # Initialize components
        self.skillbook_enabled = SKILLBOOK_AVAILABLE
        self.sizer_enabled = SIZER_AVAILABLE
        self.guardrails_enabled = GUARDRAILS_AVAILABLE
        self.indicators_enabled = INDICATORS_AVAILABLE
        
        # Setup guardrails
        if self.guardrails_enabled:
            self.trade_guard = TradeGuard(GuardConfig(
                cooldown_bars=3,
                max_loss_streak=3,
                daily_profit_lock_bps=50,
                spread_bps_threshold=8,
                max_trades_per_day=40
            ))
        else:
            self.trade_guard = None
        
        logger.info("⚙️  Utility Integration Layer initialized")
        logger.info(f"   Skillbook: {'✅' if self.skillbook_enabled else '❌'}")
        logger.info(f"   Sizer: {'✅' if self.sizer_enabled else '❌'}")
        logger.info(f"   Guardrails: {'✅' if self.guardrails_enabled else '❌'}")
        logger.info(f"   Indicators: {'✅' if self.indicators_enabled else '❌'}")
    
    def enhance_signal_with_sizing(self, signal: Dict[str, Any], equity: float = 1000.0) -> Dict[str, Any]:
        """Add position sizing to signal"""
        
        if not self.sizer_enabled:
            return signal
        
        try:
            # Use sizer to calculate position size
            enhanced = sizer.suggest_size(signal, equity)
            
            logger.debug(f"📏 Sized: {signal.get('symbol')} qty={enhanced.get('qty', 0):.6f} "
                        f"notional=${enhanced.get('notional_usd', 0):.2f}")
            
            return enhanced
            
        except Exception as e:
            logger.debug(f"Sizing error: {e}")
            return signal
    
    def check_guardrails(self, symbol: str, spread_bps: float = 5.0) -> bool:
        """Check if trade passes guardrails"""
        
        if not self.guardrails_enabled or not self.trade_guard:
            return True  # Allow if guardrails not available
        
        can_trade = self.trade_guard.can_enter_now(spread_bps)
        
        if not can_trade:
            logger.debug(f"🛑 Guardrail blocked: {symbol} (cooldown or limits)")
        
        return can_trade
    
    def record_trade_result(self, pnl: float):
        """Record trade result for guardrails"""
        
        if self.guardrails_enabled and self.trade_guard:
            self.trade_guard.record_exit(pnl)
    
    def update_skillbook(self, market: str, symbol: str, tf: str, atr_pct: float, bbw: float):
        """Update skillbook with market stats"""
        
        if not self.skillbook_enabled:
            return
        
        try:
            skillbook.update_vol_stats(market, symbol, tf, atr_pct, bbw)
            logger.debug(f"📚 Skillbook updated: {symbol} {tf}")
        except Exception as e:
            logger.debug(f"Skillbook update error: {e}")
    
    def get_personalized_thresholds(self, symbol: str, base_atr: float = 0.01, base_bbw: float = 0.02):
        """Get personalized thresholds for symbol"""
        
        if not self.skillbook_enabled:
            return base_atr, base_bbw
        
        try:
            return skillbook.personalized_thresholds(symbol, base_atr, base_bbw)
        except Exception as e:
            logger.debug(f"Threshold error: {e}")
            return base_atr, base_bbw
    
    def calculate_indicators(self, df) -> Optional[Dict[str, Any]]:
        """Calculate technical indicators"""
        
        if not self.indicators_enabled or df is None or len(df) < 20:
            return None
        
        try:
            result = {}
            
            # EMA
            result['ema_12'] = indicators.ema(df['close'], 12).iloc[-1]
            result['ema_26'] = indicators.ema(df['close'], 26).iloc[-1]
            
            # ATR
            result['atr'] = indicators.atr(df, 14).iloc[-1]
            result['atr_pct'] = result['atr'] / df['close'].iloc[-1]
            
            # RSI
            result['rsi'] = indicators.rsi(df['close'], 14).iloc[-1]
            
            # MACD
            macd_line, signal_line, hist = indicators.macd(df['close'])
            result['macd'] = macd_line.iloc[-1]
            result['macd_signal'] = signal_line.iloc[-1]
            result['macd_hist'] = hist.iloc[-1]
            
            # Supertrend
            result['supertrend_bullish'] = indicators.supertrend(df).iloc[-1]
            
            return result
            
        except Exception as e:
            logger.debug(f"Indicator error: {e}")
            return None
    
    def on_new_bar(self):
        """Called on each new bar"""
        
        if self.guardrails_enabled and self.trade_guard:
            self.trade_guard.on_new_bar()
    
    def reset_daily(self):
        """Reset daily stats"""
        
        if self.guardrails_enabled and self.trade_guard:
            self.trade_guard.reset_daily()
            logger.info("🔄 Daily guardrails reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get utility stats"""
        
        stats = {
            'skillbook_enabled': self.skillbook_enabled,
            'sizer_enabled': self.sizer_enabled,
            'guardrails_enabled': self.guardrails_enabled,
            'indicators_enabled': self.indicators_enabled
        }
        
        if self.guardrails_enabled and self.trade_guard:
            stats['trades_today'] = self.trade_guard.trades_today
            stats['loss_streak'] = self.trade_guard.loss_streak
            stats['cooldown'] = self.trade_guard.cooldown
            stats['paused'] = self.trade_guard.require_pause()
        
        return stats
