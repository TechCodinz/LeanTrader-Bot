"""
💰 STEADY PROFIT EXTRACTOR
Goal: Grow wallet steadily with consistent small profits

Strategy:
- Target: $0.50 - $5 per trade (based on balance)
- Frequency: 10-30 trades per day
- Win Rate Target: 65%+ (small wins, tight stops)
- Focus: Fast-moving pairs with clear patterns
- Risk: 1-2% per trade (conservative)

Daily Target: $10-40 profit from $42 balance
Weekly Target: $70-280 profit
Monthly Target: $300-1200 profit (7-28x returns!)
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class SteadyProfitExtractor:
    """
    Extracts steady profits through high-frequency, low-risk trading
    """
    
    def __init__(self, data_hub, execution_engine):
        self.data_hub = data_hub
        self.execution = execution_engine
        
        # Profit targets (scales with balance)
        self.min_profit_per_trade = 0.50  # $0.50 minimum
        self.target_profit_pct = 0.008  # 0.8% per trade (conservative)
        self.max_trades_per_day = 30
        
        # Risk management
        self.max_risk_pct = 0.015  # 1.5% risk per trade
        self.stop_loss_pct = 0.008  # 0.8% stop (tight!)
        self.take_profit_pct = 0.012  # 1.2% profit (1.5:1 R/R)
        
        # Trade tracking
        self.daily_trades = 0
        self.daily_profit = 0.0
        self.last_reset = datetime.now()
        self.trade_history = deque(maxlen=100)
        
        # Fast-moving pairs (updated dynamically)
        self.fast_pairs = [
            'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
            'PEPE/USDT', 'DOGE/USDT', 'SHIB/USDT',  # Volatile meme coins
            'WIF/USDT', 'BONK/USDT',  # High volatility
        ]
        
        logger.info("💰 Steady Profit Extractor initialized")
        logger.info(f"   Target: {self.target_profit_pct*100:.1f}% per trade")
        logger.info(f"   Stop Loss: {self.stop_loss_pct*100:.1f}%")
        logger.info(f"   Take Profit: {self.take_profit_pct*100:.1f}%")
        logger.info(f"   Max trades/day: {self.max_trades_per_day}")
    
    async def analyze_signal_for_extraction(self, signal: Dict) -> Optional[Dict]:
        """
        Analyze if signal is good for profit extraction
        
        Criteria:
        1. Fast-moving pair (high volatility)
        2. Clear direction (confidence 70%+)
        3. Not overtraded today
        4. Good risk/reward setup
        """
        
        # Reset daily counter at midnight
        if datetime.now().date() > self.last_reset.date():
            self.daily_trades = 0
            self.daily_profit = 0.0
            self.last_reset = datetime.now()
            logger.info(f"📊 Daily reset: {self.daily_trades} trades, ${self.daily_profit:.2f} profit")
        
        # Check daily limit
        if self.daily_trades >= self.max_trades_per_day:
            logger.debug(f"⚠️ Daily trade limit reached ({self.max_trades_per_day})")
            return None
        
        symbol = signal.get('symbol', 'UNKNOWN')
        confidence = signal.get('confidence', 0)
        side = signal.get('side', 'BUY')
        
        # Must be fast-moving pair
        if symbol not in self.fast_pairs:
            return None
        
        # Must have decent confidence
        if confidence < 0.70:
            return None
        
        # Calculate position size for profit extraction
        balance = 42.0  # Will get from exchange in production
        position_size = balance * 0.30  # Use 30% of balance per trade
        
        if position_size < 5:  # Minimum $5 position
            position_size = min(balance * 0.50, 10)  # Use up to 50% if balance is low
        
        # Calculate targets
        price = signal.get('price', 0)
        if price == 0:
            return None  # Need price
        
        if side.upper() == 'BUY':
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
        else:  # SELL
            stop_loss = price * (1 + self.stop_loss_pct)
            take_profit = price * (1 - self.take_profit_pct)
        
        # Expected profit
        expected_profit = position_size * self.take_profit_pct
        
        if expected_profit < self.min_profit_per_trade:
            logger.debug(f"⚠️ Expected profit too small: ${expected_profit:.2f}")
            return None
        
        # Create profit extraction trade
        extraction_trade = {
            'symbol': symbol,
            'side': side,
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size_usd': position_size,
            'expected_profit': expected_profit,
            'risk_amount': position_size * self.stop_loss_pct,
            'confidence': confidence,
            'strategy': 'SteadyProfitExtraction',
            'timestamp': datetime.now()
        }
        
        logger.info(
            f"💰 PROFIT EXTRACTION: {symbol} {side} "
            f"(${position_size:.0f} position → ${expected_profit:.2f} target)"
        )
        
        return extraction_trade
    
    def record_trade_result(self, trade: Dict, profit: float):
        """Record result of profit extraction trade"""
        self.daily_trades += 1
        self.daily_profit += profit
        
        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': trade['symbol'],
            'side': trade['side'],
            'profit': profit,
            'position_size': trade.get('position_size_usd', 0)
        })
        
        # Log progress
        if profit > 0:
            logger.info(
                f"✅ Profit extracted: ${profit:.2f} "
                f"(Daily: {self.daily_trades}/{self.max_trades_per_day} trades, ${self.daily_profit:.2f} total)"
            )
        else:
            logger.info(
                f"❌ Loss: ${profit:.2f} "
                f"(Daily: {self.daily_trades}/{self.max_trades_per_day} trades, ${self.daily_profit:.2f} total)"
            )
        
        # Daily summary
        if self.daily_trades % 5 == 0:
            win_rate = self._calculate_win_rate()
            logger.info(
                f"📊 DAILY PROGRESS: {self.daily_trades} trades, "
                f"${self.daily_profit:.2f} profit, "
                f"{win_rate*100:.0f}% win rate"
            )
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from recent trades"""
        if not self.trade_history:
            return 0.0
        
        wins = sum(1 for t in self.trade_history if t['profit'] > 0)
        return wins / len(self.trade_history)
    
    def get_stats(self) -> Dict:
        """Get profit extraction statistics"""
        win_rate = self._calculate_win_rate()
        
        return {
            'daily_trades': self.daily_trades,
            'daily_profit': self.daily_profit,
            'max_daily_trades': self.max_trades_per_day,
            'win_rate': win_rate,
            'total_tracked_trades': len(self.trade_history),
            'avg_profit_per_trade': self.daily_profit / max(self.daily_trades, 1)
        }
    
    async def run_profit_extraction(self):
        """
        Continuously monitor for profit extraction opportunities
        """
        logger.info("💰 Starting profit extraction engine...")
        
        while True:
            try:
                # Get recent signals
                if hasattr(self.data_hub, 'signal_queue'):
                    try:
                        signal = await asyncio.wait_for(
                            self.data_hub.signal_queue.get(),
                            timeout=1.0
                        )
                        
                        # Analyze for extraction
                        extraction_trade = await self.analyze_signal_for_extraction(signal)
                        
                        if extraction_trade:
                            # Execute immediately (auto-execution!)
                            logger.info(f"⚡ AUTO-EXECUTING profit extraction trade...")
                            # In production, would call execution engine here
                            
                    except asyncio.TimeoutError:
                        pass
                
                await asyncio.sleep(0.5)  # Fast loop for quick reactions
                
            except Exception as e:
                logger.error(f"Profit extraction error: {e}")
                await asyncio.sleep(5)


class FastScalper:
    """
    Ultra-fast scalper for quick 0.3-0.8% profits
    Targets: 20-50 trades/day, $0.30-$1 per trade
    """
    
    def __init__(self):
        self.min_move = 0.003  # 0.3% minimum
        self.target_move = 0.006  # 0.6% target
        self.hold_time = 60  # 1 minute max hold time
        
        logger.info("⚡ Fast Scalper initialized")
        logger.info(f"   Target: {self.target_move*100:.1f}% moves")
        logger.info(f"   Hold time: {self.hold_time}s max")
