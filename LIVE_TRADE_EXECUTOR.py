#!/usr/bin/env python3
"""
LIVE TRADE EXECUTOR - REAL TRADES ON BYBIT
Simple, safe, profitable execution
"""
import os
import ccxt
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LiveTradeExecutor:
    """Execute real trades on Bybit with proper risk management"""
    
    def __init__(self):
        # Check if we're in testnet mode
        trading_mode = os.getenv('TRADING_MODE', 'live').lower()
        use_testnet = trading_mode == 'testnet' or os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'
        
        if use_testnet:
            # Use TESTNET (fake money for training!)
            api_key = os.getenv('BYBIT_TESTNET_API_KEY')
            api_secret = os.getenv('BYBIT_TESTNET_API_SECRET')
            logger.info("🧪 TESTNET MODE - Using fake money for training!")
        else:
            # Use REAL trading
            api_key = os.getenv('BYBIT_API_KEY')
            api_secret = os.getenv('BYBIT_API_SECRET')
            logger.info("💰 LIVE MODE - Using REAL money!")
        
        if not api_key or not api_secret:
            raise ValueError("Bybit API keys must be set in .env")
        
        exchange_config = {
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Spot trading (safer than futures)
            }
        }
        
        # Add testnet flag if needed
        if use_testnet:
            exchange_config['options']['testnet'] = True
        
        self.exchange = ccxt.bybit(exchange_config)
        self.testnet_mode = use_testnet
        
        # Risk management
        self.max_position_size_usd = float(os.getenv('MAX_POSITION_SIZE', '50'))  # $50 max per trade
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '20'))
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', '0.80'))  # 80% minimum
        
        self.daily_trades = 0
        self.daily_profit = 0.0
        self.total_profit = 0.0
        
        mode_label = "TESTNET (Fake Money)" if self.testnet_mode else "LIVE (Real Money)"
        logger.info(f"✅ Live Executor initialized - {mode_label}")
        logger.info(f"   Exchange: Bybit {'Testnet' if self.testnet_mode else 'Mainnet'}")
        logger.info(f"   Max position: ${self.max_position_size_usd}")
        logger.info(f"   Max daily trades: {self.max_daily_trades}")
        logger.info(f"   Min confidence: {self.min_confidence:.0%}")
    
    async def execute_signal(self, symbol: str, side: str, confidence: float, 
                            entry: float, tp: float, sl: float):
        """
        Execute a trade based on signal
        
        Returns: (success: bool, trade_id: str, message: str)
        """
        
        # Safety checks
        if confidence < self.min_confidence:
            return False, None, f"Confidence {confidence:.1%} below minimum {self.min_confidence:.0%}"
        
        if self.daily_trades >= self.max_daily_trades:
            return False, None, f"Daily trade limit reached ({self.max_daily_trades})"
        
        try:
            # Get current balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            if usdt_balance < self.max_position_size_usd:
                return False, None, f"Insufficient balance: ${usdt_balance:.2f}"
            
            # Calculate position size
            position_size_usd = min(self.max_position_size_usd, usdt_balance * 0.1)  # Max 10% of balance
            amount = position_size_usd / entry
            
            # Execute market order
            order_side = 'buy' if side.upper() == 'BUY' else 'sell'
            
            logger.info(f"🔄 Executing {order_side.upper()} {symbol}: {amount:.4f} @ ${entry:.4f}")
            
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=order_side,
                amount=amount
            )
            
            self.daily_trades += 1
            
            # Place TP and SL orders (limit and stop-loss)
            if order['status'] == 'closed':
                trade_id = order['id']
                
                # Set take profit (limit sell/buy)
                tp_side = 'sell' if side.upper() == 'BUY' else 'buy'
                self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=tp_side,
                    amount=amount,
                    price=tp
                )
                
                # Set stop loss
                self.exchange.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side=tp_side,
                    amount=amount,
                    params={'stopPrice': sl}
                )
                
                msg = f"✅ Trade executed: {order_side.upper()} {amount:.4f} {symbol} @ ${entry:.4f}"
                logger.info(msg)
                
                return True, trade_id, msg
            else:
                return False, None, f"Order not filled: {order['status']}"
                
        except ccxt.InsufficientFunds as e:
            return False, None, f"Insufficient funds: {e}"
        except ccxt.NetworkError as e:
            return False, None, f"Network error: {e}"
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
            return False, None, f"Error: {e}"
    
    def get_stats(self):
        """Get trading statistics"""
        return {
            'daily_trades': self.daily_trades,
            'daily_profit': self.daily_profit,
            'total_profit': self.total_profit,
            'max_daily_trades': self.max_daily_trades,
            'trades_remaining': self.max_daily_trades - self.daily_trades
        }


# Global instance
_executor = None

def get_executor():
    """Get or create singleton executor"""
    global _executor
    if _executor is None:
        _executor = LiveTradeExecutor()
    return _executor
