"""
Divine Paper Trading System
Safe testing environment for the Divine Intelligence Trading Bot
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import asyncio
import yaml


class DivinePaperTrader:
    """
    Paper trading simulator with realistic market conditions.
    Tests the bot safely before going live.
    """
    
    def __init__(self, config_path: str = "divine_config.yml"):
        """Initialize paper trader with configuration."""
        self.config = self._load_config(config_path)
        
        # Paper trading account
        self.paper_balance = self.config['paper_trading']['starting_balance']
        self.initial_balance = self.paper_balance
        self.open_positions = {}
        self.trade_history = []
        self.order_id_counter = 1000
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
        
        # Market simulation
        self.market_prices = {}
        self.price_history = {}
        self.spread_bps = 5  # 0.05% spread
        self.slippage_bps = 3  # 0.03% slippage
        
        # Paper trading requirements
        self.requirements_met = False
        self.start_time = datetime.now()
        
        print("📝 Paper Trading System Initialized")
        print(f"   Starting Balance: ${self.paper_balance:.2f}")
        print(f"   Realistic Slippage: {self.slippage_bps/100:.3f}%")
        print(f"   Realistic Spread: {self.spread_bps/100:.3f}%")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default paper trading configuration."""
        return {
            'paper_trading': {
                'starting_balance': 1000.0,
                'realistic_slippage': True,
                'realistic_fees': True,
                'min_paper_trades': 100,
                'min_win_rate': 55,
                'min_profit_factor': 1.5
            }
        }
    
    def update_market_price(self, symbol: str, price: float, volume: float = None):
        """Update market price for symbol."""
        
        self.market_prices[symbol] = {
            'bid': price * (1 - self.spread_bps / 10000),
            'ask': price * (1 + self.spread_bps / 10000),
            'last': price,
            'volume': volume or np.random.uniform(1000000, 10000000),
            'timestamp': datetime.now()
        }
        
        # Store price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': price,
            'timestamp': datetime.now()
        })
        
        # Keep only last 1000 prices
        if len(self.price_history[symbol]) > 1000:
            self.price_history[symbol] = self.price_history[symbol][-1000:]
    
    def place_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place a paper order with realistic execution.
        """
        
        symbol = signal.get('symbol')
        side = signal.get('side')
        size = signal.get('position_size', 0)
        
        if not symbol or not side or size <= 0:
            return {
                'success': False,
                'reason': 'Invalid order parameters',
                'order_id': None
            }
        
        # Check balance
        if size > self.paper_balance * 0.95:  # Leave 5% buffer
            return {
                'success': False,
                'reason': 'Insufficient balance',
                'order_id': None
            }
        
        # Get execution price with slippage
        market_price = self.market_prices.get(symbol, {})
        if not market_price:
            return {
                'success': False,
                'reason': 'No market price available',
                'order_id': None
            }
        
        # Calculate execution price
        if side.upper() == 'BUY':
            base_price = market_price['ask']
            slippage_mult = 1 + (self.slippage_bps / 10000)
        else:
            base_price = market_price['bid']
            slippage_mult = 1 - (self.slippage_bps / 10000)
        
        execution_price = base_price * slippage_mult
        
        # Add random slippage for realism
        if self.config['paper_trading']['realistic_slippage']:
            random_slippage = np.random.normal(0, 0.0001)  # 0.01% std dev
            execution_price *= (1 + random_slippage)
        
        # Calculate fees
        fee = 0
        if self.config['paper_trading']['realistic_fees']:
            fee = size * 0.001  # 0.1% fee
        
        # Create order
        order_id = f"PAPER_{self.order_id_counter}"
        self.order_id_counter += 1
        
        position = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'size': size,
            'entry_price': execution_price,
            'current_price': execution_price,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'open_time': datetime.now(),
            'status': 'open',
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'fees_paid': fee,
            'divine_signal': signal.get('divine_intelligence', {})
        }
        
        # Store position
        self.open_positions[order_id] = position
        
        # Deduct from balance
        self.paper_balance -= (size + fee)
        
        # Update metrics
        self.metrics['total_trades'] += 1
        
        print(f"📝 Paper Order Placed:")
        print(f"   Order ID: {order_id}")
        print(f"   Symbol: {symbol}")
        print(f"   Side: {side}")
        print(f"   Size: ${size:.2f}")
        print(f"   Entry: ${execution_price:.4f}")
        print(f"   Fee: ${fee:.2f}")
        
        return {
            'success': True,
            'order_id': order_id,
            'execution_price': execution_price,
            'fee': fee,
            'position': position
        }
    
    def update_positions(self):
        """Update all open positions with current prices."""
        
        for order_id, position in list(self.open_positions.items()):
            symbol = position['symbol']
            
            # Get current price
            market_price = self.market_prices.get(symbol, {})
            if not market_price:
                continue
            
            current_price = market_price['last']
            position['current_price'] = current_price
            
            # Calculate unrealized PnL
            if position['side'].upper() == 'BUY':
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['size']
            else:
                position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['size']
            
            # Check stop loss
            if position.get('stop_loss'):
                if (position['side'].upper() == 'BUY' and current_price <= position['stop_loss']) or \
                   (position['side'].upper() == 'SELL' and current_price >= position['stop_loss']):
                    self.close_position(order_id, reason='stop_loss')
                    continue
            
            # Check take profit
            if position.get('take_profit'):
                if isinstance(position['take_profit'], list):
                    # Multiple take profit levels
                    for tp in position['take_profit']:
                        if (position['side'].upper() == 'BUY' and current_price >= tp) or \
                           (position['side'].upper() == 'SELL' and current_price <= tp):
                            self.close_position(order_id, reason='take_profit')
                            break
                else:
                    # Single take profit
                    if (position['side'].upper() == 'BUY' and current_price >= position['take_profit']) or \
                       (position['side'].upper() == 'SELL' and current_price <= position['take_profit']):
                        self.close_position(order_id, reason='take_profit')
    
    def close_position(self, order_id: str, reason: str = 'manual') -> Dict[str, Any]:
        """Close a paper position."""
        
        if order_id not in self.open_positions:
            return {
                'success': False,
                'reason': 'Position not found'
            }
        
        position = self.open_positions[order_id]
        
        # Get exit price with slippage
        symbol = position['symbol']
        market_price = self.market_prices.get(symbol, {})
        
        if not market_price:
            exit_price = position['current_price']
        else:
            if position['side'].upper() == 'BUY':
                # Selling - use bid with negative slippage
                exit_price = market_price['bid'] * (1 - self.slippage_bps / 10000)
            else:
                # Buying back - use ask with positive slippage
                exit_price = market_price['ask'] * (1 + self.slippage_bps / 10000)
        
        # Calculate final PnL
        if position['side'].upper() == 'BUY':
            gross_pnl = (exit_price - position['entry_price']) * position['size']
        else:
            gross_pnl = (position['entry_price'] - exit_price) * position['size']
        
        # Deduct fees
        exit_fee = position['size'] * 0.001 if self.config['paper_trading']['realistic_fees'] else 0
        net_pnl = gross_pnl - position['fees_paid'] - exit_fee
        
        # Update position
        position['exit_price'] = exit_price
        position['close_time'] = datetime.now()
        position['close_reason'] = reason
        position['realized_pnl'] = net_pnl
        position['status'] = 'closed'
        position['duration'] = (position['close_time'] - position['open_time']).total_seconds()
        
        # Update balance
        self.paper_balance += position['size'] + net_pnl
        
        # Move to history
        self.trade_history.append(position)
        del self.open_positions[order_id]
        
        # Update metrics
        self._update_metrics(position)
        
        print(f"📝 Paper Position Closed:")
        print(f"   Order ID: {order_id}")
        print(f"   Reason: {reason}")
        print(f"   Exit Price: ${exit_price:.4f}")
        print(f"   PnL: ${net_pnl:.2f}")
        print(f"   New Balance: ${self.paper_balance:.2f}")
        
        return {
            'success': True,
            'pnl': net_pnl,
            'exit_price': exit_price,
            'reason': reason
        }
    
    def _update_metrics(self, closed_position: Dict):
        """Update performance metrics after trade."""
        
        pnl = closed_position['realized_pnl']
        
        # Update win/loss counts
        if pnl > 0:
            self.metrics['winning_trades'] += 1
        else:
            self.metrics['losing_trades'] += 1
        
        # Update PnL metrics
        self.metrics['total_pnl'] += pnl
        
        if pnl > self.metrics['best_trade']:
            self.metrics['best_trade'] = pnl
        
        if pnl < self.metrics['worst_trade']:
            self.metrics['worst_trade'] = pnl
        
        # Calculate win rate
        total = self.metrics['winning_trades'] + self.metrics['losing_trades']
        if total > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / total
        
        # Calculate average win/loss
        if self.metrics['winning_trades'] > 0:
            wins = [t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] > 0]
            self.metrics['avg_win'] = np.mean(wins)
        
        if self.metrics['losing_trades'] > 0:
            losses = [t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] < 0]
            self.metrics['avg_loss'] = np.mean(losses)
        
        # Calculate profit factor
        gross_profit = sum(t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] > 0)
        gross_loss = abs(sum(t['realized_pnl'] for t in self.trade_history if t['realized_pnl'] < 0))
        
        if gross_loss > 0:
            self.metrics['profit_factor'] = gross_profit / gross_loss
        
        # Calculate Sharpe ratio (simplified)
        if len(self.trade_history) > 1:
            returns = [t['realized_pnl'] / self.initial_balance for t in self.trade_history]
            if np.std(returns) > 0:
                self.metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Calculate max drawdown
        peak = self.initial_balance
        for trade in self.trade_history:
            balance = self.initial_balance + sum(t['realized_pnl'] for t in self.trade_history[:self.trade_history.index(trade)+1])
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            if drawdown > self.metrics['max_drawdown']:
                self.metrics['max_drawdown'] = drawdown
    
    def check_requirements(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if paper trading requirements are met for live trading."""
        
        requirements = self.config['paper_trading']
        
        checks = {
            'min_trades': self.metrics['total_trades'] >= requirements['min_paper_trades'],
            'min_win_rate': self.metrics['win_rate'] * 100 >= requirements['min_win_rate'],
            'min_profit_factor': self.metrics['profit_factor'] >= requirements['min_profit_factor'],
            'profitable': self.metrics['total_pnl'] > 0,
            'time_tested': (datetime.now() - self.start_time).days >= 7  # At least 1 week
        }
        
        all_passed = all(checks.values())
        
        return all_passed, {
            'requirements_met': all_passed,
            'checks': checks,
            'metrics': self.metrics,
            'recommendation': 'Ready for live trading!' if all_passed else 'Continue paper trading'
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        # Calculate additional metrics
        total_return = ((self.paper_balance - self.initial_balance) / self.initial_balance) * 100
        
        days_trading = max(1, (datetime.now() - self.start_time).days)
        daily_return = total_return / days_trading
        
        # Check requirements
        requirements_met, req_details = self.check_requirements()
        
        report = {
            'account': {
                'initial_balance': self.initial_balance,
                'current_balance': self.paper_balance,
                'total_pnl': self.metrics['total_pnl'],
                'total_return_pct': total_return,
                'daily_return_pct': daily_return
            },
            'trades': {
                'total': self.metrics['total_trades'],
                'winners': self.metrics['winning_trades'],
                'losers': self.metrics['losing_trades'],
                'open': len(self.open_positions),
                'win_rate': f"{self.metrics['win_rate']*100:.1f}%"
            },
            'performance': {
                'profit_factor': round(self.metrics['profit_factor'], 2),
                'sharpe_ratio': round(self.metrics['sharpe_ratio'], 2),
                'max_drawdown': f"{self.metrics['max_drawdown']:.1f}%",
                'best_trade': f"${self.metrics['best_trade']:.2f}",
                'worst_trade': f"${self.metrics['worst_trade']:.2f}",
                'avg_win': f"${self.metrics['avg_win']:.2f}",
                'avg_loss': f"${self.metrics['avg_loss']:.2f}"
            },
            'requirements': req_details,
            'trading_days': days_trading,
            'status': 'READY FOR LIVE' if requirements_met else 'CONTINUE TESTING'
        }
        
        return report
    
    def save_results(self, filepath: str = "paper_trading_results.json"):
        """Save paper trading results to file."""
        
        results = {
            'report': self.get_performance_report(),
            'trade_history': [
                {
                    'order_id': t['order_id'],
                    'symbol': t['symbol'],
                    'side': t['side'],
                    'entry': t['entry_price'],
                    'exit': t.get('exit_price'),
                    'pnl': t.get('realized_pnl'),
                    'duration': t.get('duration'),
                    'reason': t.get('close_reason')
                }
                for t in self.trade_history
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Results saved to {filepath}")
    
    def reset(self):
        """Reset paper trading account."""
        
        self.paper_balance = self.initial_balance
        self.open_positions = {}
        self.trade_history = []
        self.metrics = {k: 0 for k in self.metrics}
        self.start_time = datetime.now()
        
        print("🔄 Paper trading account reset")


async def run_paper_trading_test():
    """Run a complete paper trading test."""
    
    print("\n" + "="*60)
    print("🚀 STARTING DIVINE PAPER TRADING TEST")
    print("="*60 + "\n")
    
    # Initialize paper trader
    paper_trader = DivinePaperTrader()
    
    # Simulate some market data
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    print("📈 Simulating market conditions...\n")
    
    # Run 50 simulated trades
    for i in range(50):
        # Update market prices
        for symbol in symbols:
            price = np.random.uniform(40000, 60000) if symbol == 'BTC/USDT' else \
                   np.random.uniform(2000, 4000) if symbol == 'ETH/USDT' else \
                   np.random.uniform(50, 150)
            
            paper_trader.update_market_price(symbol, price)
        
        # Generate a random signal
        if np.random.random() > 0.3:  # 70% chance of signal
            signal = {
                'symbol': np.random.choice(symbols),
                'side': np.random.choice(['BUY', 'SELL']),
                'position_size': np.random.uniform(50, 200),
                'confidence': np.random.uniform(0.6, 0.95),
                'stop_loss': None,  # Will be calculated
                'take_profit': None,  # Will be calculated
                'divine_intelligence': {
                    'consciousness_level': np.random.uniform(400, 700),
                    'manifestation_probability': np.random.uniform(0.4, 0.9)
                }
            }
            
            # Place order
            result = paper_trader.place_order(signal)
            
            if result['success']:
                # Set stop loss and take profit
                position = result['position']
                entry = position['entry_price']
                
                if position['side'] == 'BUY':
                    position['stop_loss'] = entry * 0.98  # 2% stop
                    position['take_profit'] = entry * 1.04  # 4% target
                else:
                    position['stop_loss'] = entry * 1.02
                    position['take_profit'] = entry * 0.96
        
        # Update positions
        paper_trader.update_positions()
        
        # Small delay for realism
        await asyncio.sleep(0.1)
        
        # Close some positions randomly
        if len(paper_trader.open_positions) > 0 and np.random.random() > 0.7:
            order_id = list(paper_trader.open_positions.keys())[0]
            paper_trader.close_position(order_id, reason='signal_exit')
    
    # Close all remaining positions
    for order_id in list(paper_trader.open_positions.keys()):
        paper_trader.close_position(order_id, reason='end_of_test')
    
    # Generate final report
    print("\n" + "="*60)
    print("📊 PAPER TRADING RESULTS")
    print("="*60 + "\n")
    
    report = paper_trader.get_performance_report()
    
    print(json.dumps(report, indent=2))
    
    # Save results
    paper_trader.save_results()
    
    print("\n" + "="*60)
    
    if report['requirements']['requirements_met']:
        print("✅ CONGRATULATIONS! Bot is ready for live trading!")
    else:
        print("⚠️ Continue paper trading to meet requirements")
    
    print("="*60)
    
    return paper_trader


if __name__ == "__main__":
    # Run paper trading test
    asyncio.run(run_paper_trading_test())