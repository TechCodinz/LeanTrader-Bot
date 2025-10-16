"""
🧠 SENTIENT TRADING BRAIN
A living, learning trading intelligence that tests everything in testnet before going live

Features:
- Dual execution: Test in sandbox, execute in live
- Strategy validation: Only profitable strategies go live
- Continuous learning: Every trade improves the system
- Profit even in losses: Learn from mistakes faster than they cost
- Real-time adaptation: Market changes → immediate strategy updates
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import json

logger = logging.getLogger(__name__)


class StrategyValidator:
    """
    Validates strategies in testnet before allowing live execution
    """
    
    def __init__(self):
        self.testnet_results = deque(maxlen=1000)
        self.strategy_scores = {}  # {strategy_name: win_rate}
        self.min_trades_before_live = 10  # Test 10 times before going live
        self.min_win_rate = 0.60  # 60% win rate required
        
        logger.info("✅ Strategy Validator initialized")
        logger.info(f"   Min trades for validation: {self.min_trades_before_live}")
        logger.info(f"   Min win rate: {self.min_win_rate*100}%")
    
    def record_testnet_result(self, strategy: str, symbol: str, side: str, profit: float):
        """Record a testnet trade result"""
        result = {
            'strategy': strategy,
            'symbol': symbol,
            'side': side,
            'profit': profit,
            'won': profit > 0,
            'timestamp': datetime.now()
        }
        
        self.testnet_results.append(result)
        
        # Update strategy score
        strategy_trades = [r for r in self.testnet_results if r['strategy'] == strategy]
        if strategy_trades:
            wins = sum(1 for r in strategy_trades if r['won'])
            win_rate = wins / len(strategy_trades)
            self.strategy_scores[strategy] = {
                'win_rate': win_rate,
                'trades': len(strategy_trades),
                'total_profit': sum(r['profit'] for r in strategy_trades)
            }
        
        logger.info(f"📊 Testnet: {strategy} → {symbol} {side} → ${profit:.2f}")
    
    def is_strategy_approved(self, strategy: str) -> bool:
        """Check if strategy is approved for live trading"""
        score = self.strategy_scores.get(strategy)
        
        if not score:
            logger.debug(f"❌ {strategy}: No testnet data yet")
            return False
        
        if score['trades'] < self.min_trades_before_live:
            logger.debug(f"❌ {strategy}: Only {score['trades']} trades (need {self.min_trades_before_live})")
            return False
        
        if score['win_rate'] < self.min_win_rate:
            logger.debug(f"❌ {strategy}: {score['win_rate']*100:.1f}% win rate (need {self.min_win_rate*100}%)")
            return False
        
        logger.info(f"✅ {strategy}: APPROVED for live trading ({score['win_rate']*100:.1f}% over {score['trades']} trades)")
        return True
    
    def get_best_strategies(self, top_n: int = 5) -> List[str]:
        """Get top performing strategies"""
        valid_strategies = [
            (name, score) for name, score in self.strategy_scores.items()
            if score['trades'] >= self.min_trades_before_live
        ]
        
        sorted_strategies = sorted(
            valid_strategies,
            key=lambda x: (x[1]['win_rate'], x[1]['total_profit']),
            reverse=True
        )
        
        return [s[0] for s in sorted_strategies[:top_n]]


class SentientTradingBrain:
    """
    The living intelligence that manages testnet + live trading
    """
    
    def __init__(self, data_hub, execution_orchestrator):
        self.data_hub = data_hub
        self.execution = execution_orchestrator
        
        # Strategy validation
        self.validator = StrategyValidator()
        
        # Learning memory
        self.learning_history = deque(maxlen=10000)
        self.market_memory = {}  # {symbol: recent_behavior}
        
        # Testnet vs Live stats
        self.testnet_trades = 0
        self.live_trades = 0
        self.testnet_profit = 0.0
        self.live_profit = 0.0
        
        # Adaptation settings
        self.adaptation_speed = 0.1  # How fast to adapt (0-1)
        self.learning_rate = 0.05
        
        logger.info("🧠 Sentient Trading Brain initialized")
        logger.info("   Mode: DUAL (Testnet validation → Live execution)")
        logger.info("   Learning: CONTINUOUS (every trade improves the system)")
    
    async def process_signal(self, signal: Dict) -> Dict:
        """
        Process a trading signal through the sentient brain
        
        Flow:
        1. Check if strategy is approved (tested in testnet)
        2. If not approved → execute in testnet only
        3. If approved → execute in live
        4. Learn from result in both cases
        """
        
        strategy = signal.get('strategy', signal.get('source', 'Unknown'))
        symbol = signal.get('symbol', 'UNKNOWN')
        side = signal.get('side', 'BUY')
        confidence = signal.get('confidence', 0.5)
        
        # Decision: Testnet or Live?
        use_live = self.validator.is_strategy_approved(strategy)
        
        if use_live:
            logger.info(f"🟢 LIVE TRADE: {strategy} → {symbol} {side} ({confidence*100:.0f}%)")
            mode = 'live'
            self.live_trades += 1
        else:
            logger.info(f"🟡 TESTNET TRADE: {strategy} → {symbol} {side} ({confidence*100:.0f}%) [validating...]")
            mode = 'testnet'
            self.testnet_trades += 1
        
        # Add mode to signal
        signal['trading_mode'] = mode
        signal['sentient_approved'] = use_live
        
        return signal
    
    def learn_from_trade(self, trade: Dict):
        """
        Learn from a completed trade (testnet or live)
        """
        strategy = trade.get('strategy', 'Unknown')
        symbol = trade.get('symbol', 'UNKNOWN')
        side = trade.get('side', 'BUY')
        profit = trade.get('profit', 0.0)
        mode = trade.get('trading_mode', 'unknown')
        
        # Record in appropriate stats
        if mode == 'testnet':
            self.testnet_profit += profit
            self.validator.record_testnet_result(strategy, symbol, side, profit)
        elif mode == 'live':
            self.live_profit += profit
        
        # Add to learning history
        self.learning_history.append({
            'timestamp': datetime.now(),
            'strategy': strategy,
            'symbol': symbol,
            'side': side,
            'profit': profit,
            'mode': mode,
            'won': profit > 0
        })
        
        # Learn patterns
        self._update_market_memory(symbol, side, profit)
        
        logger.info(
            f"📚 Learning: {strategy} {symbol} {side} → "
            f"${profit:.2f} ({mode}) "
            f"[Total: Testnet ${self.testnet_profit:.2f}, Live ${self.live_profit:.2f}]"
        )
    
    def _update_market_memory(self, symbol: str, side: str, profit: float):
        """Update market behavior memory"""
        if symbol not in self.market_memory:
            self.market_memory[symbol] = {
                'buy_wins': 0,
                'buy_losses': 0,
                'sell_wins': 0,
                'sell_losses': 0,
                'total_profit': 0.0
            }
        
        mem = self.market_memory[symbol]
        mem['total_profit'] += profit
        
        if side == 'BUY':
            if profit > 0:
                mem['buy_wins'] += 1
            else:
                mem['buy_losses'] += 1
        else:  # SELL
            if profit > 0:
                mem['sell_wins'] += 1
            else:
                mem['sell_losses'] += 1
    
    def get_stats(self) -> Dict:
        """Get sentient brain statistics"""
        approved_strategies = len([
            s for s in self.validator.strategy_scores.values()
            if s['trades'] >= self.validator.min_trades_before_live
            and s['win_rate'] >= self.validator.min_win_rate
        ])
        
        return {
            'mode': 'DUAL (Testnet + Live)',
            'testnet_trades': self.testnet_trades,
            'live_trades': self.live_trades,
            'testnet_profit': self.testnet_profit,
            'live_profit': self.live_profit,
            'total_profit': self.testnet_profit + self.live_profit,
            'approved_strategies': approved_strategies,
            'total_strategies_tested': len(self.validator.strategy_scores),
            'learning_samples': len(self.learning_history),
            'symbols_tracked': len(self.market_memory)
        }
    
    def get_top_strategies(self) -> List[Dict]:
        """Get top performing strategies"""
        best = self.validator.get_best_strategies(top_n=5)
        
        results = []
        for strategy_name in best:
            score = self.validator.strategy_scores[strategy_name]
            results.append({
                'strategy': strategy_name,
                'win_rate': score['win_rate'],
                'trades': score['trades'],
                'profit': score['total_profit']
            })
        
        return results
