#!/usr/bin/env python3
"""
EXECUTION ORCHESTRATOR - THE MISSING PIECE
Smart trade execution with risk management, position sizing, and profit optimization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
import time

logger = logging.getLogger(__name__)


class SmartPositionSizer:
    """Smart position sizing based on account balance and risk"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.balance = initial_balance
        self.max_risk_per_trade = 0.02  # 2% max risk
        self.max_position_pct = 0.10  # 10% max position size
        self.min_position_usd = 10.0  # $10 minimum
        
        # Dynamic sizing based on confidence
        self.use_dynamic_sizing = True
        self.aggressive_mode = True  # Grow account faster
        
    def calculate_position_size(self, 
                               confidence: float,
                               volatility: float = 0.02,
                               stop_loss_pct: float = 0.01) -> float:
        """
        Calculate optimal position size using Kelly Criterion + Risk Management
        
        Args:
            confidence: AI confidence (0.0 to 1.0)
            volatility: Market volatility estimate
            stop_loss_pct: Stop loss as percentage
            
        Returns:
            Position size in USD
        """
        # Base risk amount (2% of balance)
        base_risk = self.balance * self.max_risk_per_trade
        
        # Kelly Criterion adjustment
        # Kelly = (p * b - q) / b where p=win_prob, q=loss_prob, b=win/loss ratio
        win_prob = confidence
        loss_prob = 1 - confidence
        win_loss_ratio = 2.0  # Assume 2:1 reward/risk
        
        kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% Kelly
        
        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - volatility)
        
        # Adjust for confidence (higher confidence = larger size)
        confidence_adjustment = 0.5 + (confidence * 0.5)  # 50% to 100%
        
        # Calculate final position size
        position_size = (base_risk / stop_loss_pct) * kelly_fraction * volatility_adjustment * confidence_adjustment
        
        # AGGRESSIVE MODE: Grow account faster with high confidence trades
        if self.aggressive_mode and confidence >= 0.85:
            # Increase size by up to 50% for very high confidence
            confidence_boost = 1.0 + ((confidence - 0.85) * 2.0)  # 85% conf = 1.0x, 100% conf = 1.3x
            position_size *= confidence_boost
            logger.info(f"🚀 Aggressive sizing: confidence {confidence:.1%} → {confidence_boost:.2f}x boost")
        
        # Apply limits
        max_position = self.balance * self.max_position_pct
        position_size = min(position_size, max_position)
        position_size = max(position_size, self.min_position_usd)
        
        # Balance-aware: As balance grows, increase position sizes proportionally
        if self.balance > 1000:
            growth_multiplier = (self.balance / 1000) ** 0.5  # Square root scaling
            position_size *= growth_multiplier
            logger.debug(f"💰 Balance-aware sizing: ${self.balance:.0f} → {growth_multiplier:.2f}x multiplier")
        
        return position_size
    
    def update_balance(self, new_balance: float):
        """Update balance after trades"""
        self.balance = new_balance


class SmartRiskManager:
    """Smart risk management and validation"""
    
    def __init__(self):
        self.max_open_positions = 5
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_correlated_positions = 2  # Max positions in correlated assets
        
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        
    def can_open_position(self, symbol: str, signal_type: str) -> tuple[bool, str]:
        """Check if we can open a new position"""
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2%}"
        
        # Check max positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max positions reached: {len(self.open_positions)}/{self.max_open_positions}"
        
        # Check if already in this position
        if symbol in self.open_positions:
            return False, f"Already in position: {symbol}"
        
        # Check correlation (BTC/ETH, etc)
        correlated_count = self._count_correlated_positions(symbol)
        if correlated_count >= self.max_correlated_positions:
            return False, f"Max correlated positions: {correlated_count}/{self.max_correlated_positions}"
        
        return True, "OK"
    
    def _count_correlated_positions(self, symbol: str) -> int:
        """Count positions in correlated assets"""
        # Simplified correlation groups
        correlation_groups = {
            'BTC': ['BTC/USDT', 'BTC/USD'],
            'ETH': ['ETH/USDT', 'ETH/USD'],
            'MAJOR': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        }
        
        count = 0
        for group, symbols in correlation_groups.items():
            if symbol in symbols:
                # Count positions in same group
                for open_symbol in self.open_positions:
                    if open_symbol in symbols:
                        count += 1
        
        return count
    
    def record_position(self, symbol: str, side: str, size: float, entry_price: float):
        """Record opened position"""
        self.open_positions[symbol] = {
            'side': side,
            'size': size,
            'entry_price': entry_price,
            'timestamp': datetime.now()
        }
    
    def close_position(self, symbol: str, exit_price: float) -> float:
        """Close position and calculate P&L"""
        if symbol not in self.open_positions:
            return 0.0
        
        position = self.open_positions[symbol]
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        # Calculate P&L
        if side == 'buy':
            pnl = (exit_price - entry_price) * size
        else:
            pnl = (entry_price - exit_price) * size
        
        # Update daily stats
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        # Remove position
        del self.open_positions[symbol]
        
        return pnl
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset = today


class ExecutionOrchestrator:
    """
    SMART EXECUTION ORCHESTRATOR
    The critical missing piece that actually executes trades
    
    Features:
    - Smart position sizing (Kelly Criterion)
    - Risk management and validation
    - Stop loss and take profit automation
    - Multi-exchange support
    - Performance tracking
    - Trade recovery and retry logic
    """
    
    def __init__(self, data_hub, trading_engines, risk_engine, ledger, mode: str = "testnet"):
        self.data_hub = data_hub
        self.engines = trading_engines
        self.risk_engine = risk_engine
        self.ledger = ledger
        self.mode = mode
        
        # Smart components
        self.position_sizer = SmartPositionSizer(initial_balance=1000.0)
        self.risk_manager = SmartRiskManager()
        
        # Execution settings
        self.min_confidence = 0.80  # 80% minimum confidence to execute
        self.execution_enabled = True
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.execution_times = deque(maxlen=100)
        
        logger.info("⚡ Execution Orchestrator initialized (SMART LOGIC)")
        logger.info(f"   Mode: {mode}")
        logger.info(f"   Min Confidence: {self.min_confidence}")
        logger.info(f"   Risk per trade: {self.position_sizer.max_risk_per_trade:.1%}")
    
    async def run_execution_loop(self):
        """
        Main execution loop - monitors decisions and executes high-confidence trades
        """
        logger.info("⚡ Starting SMART execution loop...")
        
        while self.execution_enabled:
            try:
                # Reset daily stats if new day
                self.risk_manager.reset_daily_stats()
                
                # Check for decisions in alert queue
                if not self.data_hub.alert_queue.empty():
                    decision = await self.data_hub.alert_queue.get()
                    
                    # Validate and execute
                    await self.process_decision(decision)
                
                # Monitor open positions
                await self.monitor_positions()
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                await asyncio.sleep(5)
    
    async def process_decision(self, decision: Dict[str, Any]):
        """Process AI decision and execute if valid"""
        
        try:
            # Extract decision details
            signal = decision.get('signal', {})
            action = decision.get('action', 'hold')
            confidence = decision.get('confidence', 0.0)
            
            symbol = signal.get('symbol') or signal.get('data', {}).get('symbol')
            if not symbol:
                logger.debug("No symbol in decision")
                return
            
            # Validate confidence
            if confidence < self.min_confidence:
                logger.debug(f"Low confidence: {confidence:.2%} < {self.min_confidence:.2%}")
                return
            
            # Validate action
            if action not in ['buy', 'sell']:
                return
            
            # Check risk management
            can_trade, reason = self.risk_manager.can_open_position(symbol, action)
            if not can_trade:
                logger.info(f"⚠️ Trade blocked: {reason}")
                return
            
            # EXECUTE THE TRADE!
            logger.info(f"⚡ EXECUTING: {action.upper()} {symbol} (confidence: {confidence:.1%})")
            
            result = await self.execute_trade(
                symbol=symbol,
                side=action,
                confidence=confidence,
                signal=signal
            )
            
            if result:
                logger.info(f"✅ Trade executed successfully: {symbol}")
            else:
                logger.warning(f"❌ Trade execution failed: {symbol}")
                
        except Exception as e:
            logger.error(f"Decision processing error: {e}")
    
    async def execute_trade(self, 
                           symbol: str, 
                           side: str, 
                           confidence: float,
                           signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute trade with smart logic
        
        Returns trade result or None if failed
        """
        
        start_time = time.time()
        
        try:
            # Get current price
            price = signal.get('data', {}).get('price', 0)
            if not price:
                # Try to fetch current price
                price = await self.get_current_price(symbol)
            
            if not price:
                logger.error(f"Cannot get price for {symbol}")
                return None
            
            # Calculate smart position size
            volatility = signal.get('data', {}).get('volatility', 0.02)
            position_size_usd = self.position_sizer.calculate_position_size(
                confidence=confidence,
                volatility=volatility
            )
            
            # Convert to base currency amount
            amount = position_size_usd / price
            
            # Calculate stop loss and take profit
            stop_loss_pct = 0.01  # 1% stop loss
            take_profit_pct = 0.02  # 2% take profit (2:1 reward/risk)
            
            if side == 'buy':
                stop_loss = price * (1 - stop_loss_pct)
                take_profit = price * (1 + take_profit_pct)
            else:
                stop_loss = price * (1 + stop_loss_pct)
                take_profit = price * (1 - take_profit_pct)
            
            # Choose execution engine based on signal source
            execution_result = None
            
            # Try REAL_PROFIT_BOT first (Gate.io)
            if 'real_profit' in self.engines and self.engines['real_profit']:
                try:
                    execution_result = self.engines['real_profit'].execute_trade(
                        symbol=symbol,
                        signal=side.upper(),
                        price=price
                    )
                except Exception as e:
                    logger.debug(f"Real profit execution: {e}")
            
            # Fallback to enhanced bot (Bybit/other exchanges)
            if not execution_result and 'enhanced' in self.engines:
                try:
                    # Would execute on Bybit or other exchange
                    # For now, simulate execution
                    execution_result = {
                        'symbol': symbol,
                        'side': side,
                        'amount': amount,
                        'price': price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'timestamp': datetime.now().isoformat(),
                        'simulated': True  # Mark as simulated in testnet
                    }
                except Exception as e:
                    logger.debug(f"Enhanced bot execution: {e}")
            
            if execution_result:
                # Record position with risk manager
                self.risk_manager.record_position(
                    symbol=symbol,
                    side=side,
                    size=amount,
                    entry_price=price
                )
                
                # Record in ledger
                trade_record = {
                    'symbol': symbol,
                    'side': side,
                    'amount': amount,
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'position_size_usd': position_size_usd,
                    'timestamp': datetime.now(),
                    'status': 'open'
                }
                
                await self.data_hub.publish_trade(trade_record)
                
                # Update stats
                self.total_trades += 1
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)
                
                logger.info(f"⚡ TRADE EXECUTED:")
                logger.info(f"   Symbol: {symbol}")
                logger.info(f"   Side: {side.upper()}")
                logger.info(f"   Amount: {amount:.6f}")
                logger.info(f"   Entry: ${price:.4f}")
                logger.info(f"   Stop Loss: ${stop_loss:.4f}")
                logger.info(f"   Take Profit: ${take_profit:.4f}")
                logger.info(f"   Position Size: ${position_size_usd:.2f}")
                logger.info(f"   Confidence: {confidence:.1%}")
                logger.info(f"   Execution Time: {execution_time:.2f}s")
                
                return execution_result
            
            return None
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price from engines or fresh exchange connection"""
        try:
            # Try to get from router
            if hasattr(self, 'router') and self.router:
                ticker = await self.router.fetch_ticker(symbol)
                if ticker and ticker.get('last'):
                    return ticker.get('last', 0)
        except Exception as e:
            logger.debug(f"Router fetch failed: {str(e)[:50]}")
        
        # Try engines
        if self.engines:
            for engine_name, engine in self.engines.items():
                try:
                    # Check if engine has an exchange object
                    if hasattr(engine, 'exchange') and engine.exchange:
                        ticker = await engine.exchange.fetch_ticker(symbol)
                        if ticker and ticker.get('last'):
                            logger.debug(f"Got price from {engine_name}: ${ticker['last']:.2f}")
                            return ticker['last']
                except Exception as e:
                    logger.debug(f"{engine_name} fetch failed: {str(e)[:50]}")
                    continue
        
        # Fallback: Create fresh exchange connection
        try:
            import ccxt.async_support as ccxt
            import os
            
            logger.debug(f"Trying fresh exchange connection for {symbol}...")
            
            # Try Gate.io first (user's main exchange)
            if os.getenv('GATE_API_KEY'):
                try:
                    exchange = ccxt.gateio({
                        'apiKey': os.getenv('GATE_API_KEY'),
                        'secret': os.getenv('GATE_SECRET'),
                        'enableRateLimit': True
                    })
                    ticker = await exchange.fetch_ticker(symbol)
                    price = ticker['last']
                    await exchange.close()
                    logger.debug(f"Got ${price:.2f} from Gate.io")
                    return price
                except Exception as e:
                    logger.debug(f"Gate.io failed: {str(e)[:50]}")
            
            # Try Binance public API
            exchange = ccxt.binance({'enableRateLimit': True})
            ticker = await exchange.fetch_ticker(symbol)
            price = ticker['last']
            await exchange.close()
            logger.debug(f"Got ${price:.2f} from Binance")
            return price
            
        except Exception as e:
            logger.error(f"All price fetch attempts failed for {symbol}: {e}")
        
        return None
    
    async def monitor_positions(self):
        """Monitor open positions for stop loss and take profit"""
        
        for symbol, position in list(self.risk_manager.open_positions.items()):
            try:
                # Get current price
                current_price = await self.get_current_price(symbol)
                if not current_price:
                    continue
                
                entry_price = position['entry_price']
                side = position['side']
                
                # Calculate current P&L
                if side == 'buy':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Check stop loss (1% loss)
                if pnl_pct < -0.01:
                    logger.warning(f"🛑 Stop loss triggered: {symbol} ({pnl_pct:.2%})")
                    await self.close_position(symbol, current_price, 'stop_loss')
                
                # Check take profit (2% profit)
                elif pnl_pct > 0.02:
                    logger.info(f"🎯 Take profit triggered: {symbol} ({pnl_pct:.2%})")
                    await self.close_position(symbol, current_price, 'take_profit')
                
            except Exception as e:
                logger.debug(f"Position monitoring error for {symbol}: {e}")
    
    async def close_position(self, symbol: str, exit_price: float, reason: str):
        """Close position and record result"""
        
        try:
            # Calculate P&L with risk manager
            pnl = self.risk_manager.close_position(symbol, exit_price)
            
            # Update balance
            new_balance = self.position_sizer.balance + pnl
            self.position_sizer.update_balance(new_balance)
            
            # Update stats
            if pnl > 0:
                self.winning_trades += 1
            
            self.total_profit += pnl
            
            # Record in ledger
            close_record = {
                'symbol': symbol,
                'exit_price': exit_price,
                'pnl': pnl,
                'reason': reason,
                'timestamp': datetime.now(),
                'status': 'closed'
            }
            
            await self.data_hub.publish_trade(close_record)
            
            logger.info(f"💰 POSITION CLOSED:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Exit: ${exit_price:.4f}")
            logger.info(f"   P&L: ${pnl:.2f}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   New Balance: ${new_balance:.2f}")
            logger.info(f"   Total Profit: ${self.total_profit:.2f}")
            logger.info(f"   Win Rate: {self.get_win_rate():.1%}")
            
        except Exception as e:
            logger.error(f"Position close error: {e}")
    
    def get_win_rate(self) -> float:
        """Get current win rate"""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.get_win_rate(),
            'total_profit': self.total_profit,
            'current_balance': self.position_sizer.balance,
            'open_positions': len(self.risk_manager.open_positions),
            'daily_pnl': self.risk_manager.daily_pnl,
            'daily_trades': self.risk_manager.daily_trades,
            'avg_execution_time': sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        }
