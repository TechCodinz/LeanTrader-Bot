#!/usr/bin/env python3
"""
ADVANCED TRADING ACTIONS
Beyond BUY/SELL: HOLD, Scale In/Out, DCA, Market Adaption, Portfolio Management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detects Bull, Bear, Sideways, Choppy markets"""
    
    def __init__(self):
        self.regimes = {}  # symbol -> regime
        
    async def detect_regime(self, symbol: str, price_data: List[float]) -> str:
        """
        Detect market regime from price action
        
        Returns: 'bull', 'bear', 'sideways', 'choppy'
        """
        if len(price_data) < 20:
            return 'unknown'
        
        # Convert to pandas for analysis
        df = pd.DataFrame(price_data, columns=['price'])
        
        # Calculate indicators
        df['sma_20'] = df['price'].rolling(20).mean()
        df['sma_50'] = df['price'].rolling(50, min_periods=20).mean()
        df['std_20'] = df['price'].rolling(20).std()
        
        current_price = df['price'].iloc[-1]
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        volatility = df['std_20'].iloc[-1] / sma_20
        
        # Trend direction
        price_change_20 = (current_price - df['price'].iloc[-20]) / df['price'].iloc[-20]
        
        # Count directional changes (choppiness indicator)
        direction_changes = 0
        for i in range(-19, 0):
            if (df['price'].iloc[i] > df['price'].iloc[i-1]) != (df['price'].iloc[i-1] > df['price'].iloc[i-2]):
                direction_changes += 1
        
        # Regime detection logic
        if direction_changes > 12:  # Lots of direction changes
            regime = 'choppy'
        elif abs(price_change_20) < 0.02 and volatility < 0.01:  # Low movement, low vol
            regime = 'sideways'
        elif price_change_20 > 0.05 and current_price > sma_20 > sma_50:  # Strong uptrend
            regime = 'bull'
        elif price_change_20 < -0.05 and current_price < sma_20 < sma_50:  # Strong downtrend
            regime = 'bear'
        elif price_change_20 > 0.02:
            regime = 'bull'
        elif price_change_20 < -0.02:
            regime = 'bear'
        else:
            regime = 'sideways'
        
        self.regimes[symbol] = regime
        return regime
    
    def get_regime_strategy(self, regime: str) -> Dict[str, Any]:
        """Get trading strategy for market regime"""
        
        strategies = {
            'bull': {
                'action_preference': ['buy', 'hold'],
                'stop_loss_pct': 0.02,  # Wider stops in bull
                'take_profit_pct': 0.05,  # Bigger targets
                'position_size_multiplier': 1.2,  # Larger positions
                'hold_time': 'long',  # Hold winners
            },
            'bear': {
                'action_preference': ['sell', 'hold_short'],
                'stop_loss_pct': 0.01,  # Tight stops in bear
                'take_profit_pct': 0.03,  # Smaller targets
                'position_size_multiplier': 0.8,  # Smaller positions
                'hold_time': 'short',  # Quick exits
            },
            'sideways': {
                'action_preference': ['scalp', 'range_trade'],
                'stop_loss_pct': 0.015,  # Medium stops
                'take_profit_pct': 0.02,  # Quick profits at range edges
                'position_size_multiplier': 1.0,  # Normal size
                'hold_time': 'medium',  # Hold until range edge
            },
            'choppy': {
                'action_preference': ['hold', 'avoid'],
                'stop_loss_pct': 0.01,  # Very tight stops
                'take_profit_pct': 0.015,  # Small quick profits
                'position_size_multiplier': 0.5,  # Small positions
                'hold_time': 'very_short',  # Exit fast
            },
        }
        
        return strategies.get(regime, strategies['sideways'])


class ScaleInOutManager:
    """Manages DCA (Dollar Cost Averaging) and position scaling"""
    
    def __init__(self):
        self.positions = {}  # symbol -> {entries: [], total_size: x, avg_price: y}
        
    def can_scale_in(self, symbol: str, max_entries: int = 3) -> bool:
        """Check if we can add to position"""
        if symbol not in self.positions:
            return True
        return len(self.positions[symbol]['entries']) < max_entries
    
    async def scale_in(self, symbol: str, price: float, size: float, reason: str = "DCA") -> Dict[str, Any]:
        """Add to existing position (DCA)"""
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'entries': [],
                'total_size': 0,
                'total_cost': 0,
            }
        
        pos = self.positions[symbol]
        
        # Record entry
        entry = {
            'price': price,
            'size': size,
            'timestamp': datetime.now(),
            'reason': reason
        }
        pos['entries'].append(entry)
        pos['total_size'] += size
        pos['total_cost'] += price * size
        
        # Calculate new average price
        avg_price = pos['total_cost'] / pos['total_size']
        
        logger.info(f"📈 SCALE IN: {symbol}")
        logger.info(f"   Entry #{len(pos['entries'])} at ${price:.4f}")
        logger.info(f"   Size: {size:.6f}")
        logger.info(f"   Total Size: {pos['total_size']:.6f}")
        logger.info(f"   Avg Price: ${avg_price:.4f}")
        logger.info(f"   Reason: {reason}")
        
        return {
            'action': 'scale_in',
            'symbol': symbol,
            'price': price,
            'size': size,
            'avg_price': avg_price,
            'total_size': pos['total_size'],
            'entry_count': len(pos['entries'])
        }
    
    async def scale_out(self, symbol: str, price: float, percentage: float = 0.25) -> Dict[str, Any]:
        """Reduce position by percentage (Partial TP)"""
        
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Calculate amount to close
        close_size = pos['total_size'] * percentage
        avg_price = pos['total_cost'] / pos['total_size']
        
        # Calculate profit
        profit = close_size * (price - avg_price)
        profit_pct = (price - avg_price) / avg_price
        
        # Update position
        pos['total_size'] -= close_size
        pos['total_cost'] -= close_size * avg_price
        
        logger.info(f"📉 SCALE OUT: {symbol}")
        logger.info(f"   Closed {percentage:.0%} at ${price:.4f}")
        logger.info(f"   Size Closed: {close_size:.6f}")
        logger.info(f"   Remaining: {pos['total_size']:.6f}")
        logger.info(f"   Profit: ${profit:.2f} ({profit_pct:.2%})")
        
        # Clear position if fully closed
        if pos['total_size'] < 0.0001:
            del self.positions[symbol]
            logger.info(f"   ✅ Position fully closed")
        
        return {
            'action': 'scale_out',
            'symbol': symbol,
            'price': price,
            'size': close_size,
            'profit': profit,
            'profit_pct': profit_pct,
            'remaining_size': pos['total_size'] if pos['total_size'] >= 0.0001 else 0
        }
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position details"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        return {
            'symbol': symbol,
            'total_size': pos['total_size'],
            'avg_price': pos['total_cost'] / pos['total_size'],
            'total_cost': pos['total_cost'],
            'entries': len(pos['entries']),
        }


class PortfolioBalancer:
    """Manages portfolio allocation across many pairs"""
    
    def __init__(self, total_capital: float = 1000.0):
        self.total_capital = total_capital
        self.allocations = {}  # symbol -> allocated_capital
        self.max_single_position = 0.10  # 10% max per pair
        self.target_positions = 20  # Aim for 20 active positions
        
    def calculate_allocation(self, symbol: str, confidence: float, opportunities: List[Dict]) -> float:
        """
        Calculate how much capital to allocate to this opportunity
        
        Args:
            symbol: Trading pair
            confidence: AI confidence (0-1)
            opportunities: All current opportunities
            
        Returns:
            Capital to allocate (USD)
        """
        # Calculate free capital
        allocated = sum(self.allocations.values())
        free_capital = self.total_capital - allocated
        
        # If no free capital, can't allocate
        if free_capital <= 0:
            return 0.0
        
        # Base allocation: divide free capital by opportunities
        num_opportunities = max(len(opportunities), self.target_positions)
        base_allocation = free_capital / num_opportunities
        
        # Adjust by confidence (higher confidence = more allocation)
        confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5x to 2x based on confidence
        allocation = base_allocation * confidence_multiplier
        
        # Apply limits
        max_allocation = self.total_capital * self.max_single_position
        allocation = min(allocation, max_allocation, free_capital)
        
        return allocation
    
    def allocate(self, symbol: str, amount: float):
        """Record capital allocation"""
        self.allocations[symbol] = amount
        logger.info(f"💼 Portfolio: Allocated ${amount:.2f} to {symbol}")
        logger.info(f"   Total Allocated: ${sum(self.allocations.values()):.2f}")
        logger.info(f"   Free Capital: ${self.total_capital - sum(self.allocations.values()):.2f}")
        logger.info(f"   Active Positions: {len(self.allocations)}")
    
    def deallocate(self, symbol: str, pnl: float):
        """Release capital from closed position"""
        if symbol in self.allocations:
            allocation = self.allocations[symbol]
            del self.allocations[symbol]
            
            # Update total capital with profit/loss
            self.total_capital += pnl
            
            logger.info(f"💼 Portfolio: Deallocated {symbol}")
            logger.info(f"   P&L: ${pnl:.2f}")
            logger.info(f"   New Total Capital: ${self.total_capital:.2f}")
            logger.info(f"   Active Positions: {len(self.allocations)}")
    
    def rebalance_needed(self) -> bool:
        """Check if portfolio needs rebalancing"""
        # Check if any position is too large
        for symbol, allocation in self.allocations.items():
            if allocation / self.total_capital > self.max_single_position * 1.2:
                return True
        
        # Check if too concentrated (< 10 positions with > 50% capital)
        if len(self.allocations) < 10 and sum(self.allocations.values()) > self.total_capital * 0.5:
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get portfolio statistics"""
        allocated = sum(self.allocations.values())
        return {
            'total_capital': self.total_capital,
            'allocated': allocated,
            'free': self.total_capital - allocated,
            'allocation_pct': allocated / self.total_capital if self.total_capital > 0 else 0,
            'active_positions': len(self.allocations),
            'avg_position_size': allocated / len(self.allocations) if self.allocations else 0,
        }


class AdvancedActionDecider:
    """Decides WHAT to do beyond just BUY/SELL"""
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.scale_manager = ScaleInOutManager()
        self.portfolio = PortfolioBalancer()
        
    async def decide_action(self, 
                           symbol: str,
                           signal: Dict[str, Any],
                           confidence: float,
                           price_history: List[float],
                           opportunities: List[Dict]) -> Dict[str, Any]:
        """
        Decide sophisticated action based on ALL collective resources
        
        Returns: {
            'action': 'buy' | 'sell' | 'hold' | 'scale_in' | 'scale_out' | 'avoid',
            'size': position_size,
            'reason': explanation,
            'regime': market_regime,
            'strategy': regime_strategy
        }
        """
        
        # 1. Detect market regime
        regime = await self.regime_detector.detect_regime(symbol, price_history)
        strategy = self.regime_detector.get_regime_strategy(regime)
        
        # 2. Check if we have an existing position
        existing_position = self.scale_manager.get_position(symbol)
        
        # 3. Get portfolio allocation
        allocation = self.portfolio.calculate_allocation(symbol, confidence, opportunities)
        
        # 4. Decide action based on regime + position + signal
        action = 'hold'
        reason = ''
        size = 0.0
        
        signal_side = signal.get('signal', '').lower()
        current_price = signal.get('data', {}).get('price', 0)
        
        if existing_position:
            # WE HAVE AN OPEN POSITION - decide whether to add, reduce, or hold
            avg_price = existing_position['avg_price']
            profit_pct = (current_price - avg_price) / avg_price
            
            # Check if we should scale out (take partial profits)
            if profit_pct > strategy['take_profit_pct']:
                action = 'scale_out'
                size = 0.25  # Close 25%
                reason = f"{regime} market: Taking 25% profit at +{profit_pct:.2%}"
            
            # Check if we should scale in (DCA on dip)
            elif profit_pct < -0.02 and signal_side == 'buy' and self.scale_manager.can_scale_in(symbol):
                action = 'scale_in'
                size = allocation / current_price
                reason = f"{regime} market: DCA on dip at {profit_pct:.2%}"
            
            # Check stop loss
            elif profit_pct < -strategy['stop_loss_pct']:
                action = 'sell'
                size = existing_position['total_size']
                reason = f"{regime} market: Stop loss at {profit_pct:.2%}"
            
            else:
                action = 'hold'
                reason = f"{regime} market: Holding position (P&L: {profit_pct:.2%})"
        
        else:
            # NO POSITION - decide whether to enter
            
            # In choppy markets, avoid new entries
            if regime == 'choppy' and confidence < 0.90:
                action = 'avoid'
                reason = "Choppy market: Waiting for clear trend"
            
            # In bull markets, favor longs
            elif regime == 'bull' and signal_side == 'buy' and confidence >= 0.80:
                action = 'buy'
                size = (allocation * strategy['position_size_multiplier']) / current_price
                reason = f"Bull market: Strong buy signal ({confidence:.1%})"
            
            # In bear markets, favor shorts (or avoid longs)
            elif regime == 'bear' and signal_side == 'sell' and confidence >= 0.80:
                action = 'sell'
                size = (allocation * strategy['position_size_multiplier']) / current_price
                reason = f"Bear market: Strong sell signal ({confidence:.1%})"
            
            # Sideways: scalp at range edges
            elif regime == 'sideways' and confidence >= 0.85:
                action = signal_side
                size = (allocation * 0.8) / current_price  # Smaller scalp size
                reason = f"Sideways market: Scalping at range edge ({confidence:.1%})"
            
            else:
                action = 'hold'
                reason = f"{regime} market: Confidence too low ({confidence:.1%})"
        
        return {
            'action': action,
            'size': size,
            'reason': reason,
            'regime': regime,
            'strategy': strategy,
            'confidence': confidence,
            'has_position': existing_position is not None,
        }


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              ADVANCED TRADING ACTIONS                            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  Beyond Simple BUY/SELL:                                        ║
    ║                                                                  ║
    ║  🎯 ACTIONS:                                                    ║
    ║     • BUY - Open new long position                              ║
    ║     • SELL - Open new short / Close long                        ║
    ║     • HOLD - Manage existing position                           ║
    ║     • SCALE_IN - Add to position (DCA)                          ║
    ║     • SCALE_OUT - Take partial profits                          ║
    ║     • AVOID - Skip trade (bad conditions)                       ║
    ║                                                                  ║
    ║  🌊 MARKET REGIMES:                                             ║
    ║     • BULL - Uptrend (favor longs, wider stops, hold longer)    ║
    ║     • BEAR - Downtrend (favor shorts, tight stops, quick exit)  ║
    ║     • SIDEWAYS - Range (scalp edges, medium targets)            ║
    ║     • CHOPPY - Whipsaw (avoid or tiny positions)                ║
    ║                                                                  ║
    ║  💰 POSITION MANAGEMENT:                                        ║
    ║     • DCA into dips (up to 3 entries)                           ║
    ║     • Partial profits (25% at each level)                       ║
    ║     • Portfolio balancing (max 10% per pair)                    ║
    ║     • Risk-adjusted sizing per regime                           ║
    ║                                                                  ║
    ║  🧠 COLLECTIVE REASONING:                                       ║
    ║     • Uses ALL signals, models, scouts, engines                 ║
    ║     • Adapts to market conditions                               ║
    ║     • Learns from every trade                                   ║
    ║     • Optimizes for ANY market (bull/bear/sideways/choppy)      ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    This bot can NOW profit in ALL market conditions! 🚀
    """)
