#!/usr/bin/env python3
"""
🤖 AUTO LIVE TRIGGER SYSTEM
Automatically switches between testnet learning and live trading based on performance

Features:
- Monitors testnet performance in real-time
- Auto-approves profitable strategies (60%+ win rate, 10+ trades)
- Auto-starts live bot when strategies proven
- Auto-pauses live bot if performance drops
- No manual intervention needed
"""

import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List
import json
import os

logger = logging.getLogger(__name__)


class StrategyPerformance:
    """Track performance of individual strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.trades = []
        self.total_profit = 0.0
        self.wins = 0
        self.losses = 0
    
    def add_trade(self, profit: float, timestamp: datetime):
        """Record a trade result"""
        self.trades.append({
            'profit': profit,
            'timestamp': timestamp
        })
        self.total_profit += profit
        if profit > 0:
            self.wins += 1
        else:
            self.losses += 1
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades
    
    @property
    def is_approved(self) -> bool:
        """Strategy approved for live trading?"""
        return self.total_trades >= 10 and self.win_rate >= 0.60
    
    def get_status(self) -> Dict:
        return {
            'name': self.name,
            'trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit': self.total_profit,
            'approved': self.is_approved
        }


class AutoLiveTrigger:
    """
    Automatically manages testnet → live transition
    """
    
    def __init__(self):
        self.strategies = {
            'scalping': StrategyPerformance('Scalping'),
            'arbitrage': StrategyPerformance('Arbitrage'),
            'moon': StrategyPerformance('Moon Spotter'),
            'hedge_fund': StrategyPerformance('Hedge Fund'),
            'evolution': StrategyPerformance('Evolution Models')
        }
        
        self.live_bot_running = False
        self.testnet_bot_running = True
        
        self.min_trades_for_approval = 10
        self.min_win_rate = 0.60  # 60%
        self.pause_threshold = 0.55  # Pause live if drops below 55%
        
        logger.info("🤖 Auto Live Trigger System initialized")
        logger.info(f"   Approval: {self.min_trades_for_approval}+ trades, {self.min_win_rate:.0%}+ win rate")
        logger.info(f"   Pause threshold: {self.pause_threshold:.0%}")
    
    def record_testnet_trade(self, strategy: str, profit: float):
        """Record a testnet trade result"""
        if strategy in self.strategies:
            self.strategies[strategy].add_trade(profit, datetime.now())
            
            # Check if strategy just got approved
            if self.strategies[strategy].is_approved:
                logger.info(f"✅ {strategy} APPROVED for live trading!")
                logger.info(f"   Trades: {self.strategies[strategy].total_trades}")
                logger.info(f"   Win rate: {self.strategies[strategy].win_rate:.1%}")
    
    def get_approved_strategies(self) -> List[str]:
        """Get list of approved strategies"""
        return [
            name for name, perf in self.strategies.items()
            if perf.is_approved
        ]
    
    def should_start_live_bot(self) -> bool:
        """Should we start the live bot?"""
        approved = self.get_approved_strategies()
        
        # Need at least 2 approved strategies before going live
        if len(approved) >= 2:
            logger.info(f"🚀 READY FOR LIVE: {len(approved)} strategies approved")
            return True
        
        return False
    
    def should_pause_live_bot(self) -> bool:
        """Should we pause the live bot?"""
        if not self.live_bot_running:
            return False
        
        # Check recent performance of live strategies
        for name, perf in self.strategies.items():
            if perf.is_approved:
                # Check last 20 trades
                recent_trades = perf.trades[-20:]
                if len(recent_trades) >= 10:
                    recent_wins = sum(1 for t in recent_trades if t['profit'] > 0)
                    recent_win_rate = recent_wins / len(recent_trades)
                    
                    if recent_win_rate < self.pause_threshold:
                        logger.warning(f"⚠️  {name} performance dropped to {recent_win_rate:.1%}")
                        logger.warning(f"   Pausing live bot for re-training")
                        return True
        
        return False
    
    async def start_live_bot(self):
        """Start the live trading bot"""
        try:
            logger.info("🚀 AUTO-STARTING LIVE BOT...")
            subprocess.run(['sudo', 'systemctl', 'start', 'trading-bot-live'], check=True)
            self.live_bot_running = True
            logger.info("✅ Live bot started successfully")
        except Exception as e:
            logger.error(f"Failed to start live bot: {e}")
    
    async def pause_live_bot(self):
        """Pause the live trading bot"""
        try:
            logger.warning("⏸️  AUTO-PAUSING LIVE BOT...")
            subprocess.run(['sudo', 'systemctl', 'stop', 'trading-bot-live'], check=True)
            self.live_bot_running = False
            logger.info("✅ Live bot paused - returning to testnet learning")
        except Exception as e:
            logger.error(f"Failed to pause live bot: {e}")
    
    async def run_auto_trigger_loop(self):
        """Main loop - monitors and triggers live/pause"""
        logger.info("🤖 Starting Auto Live Trigger loop...")
        
        while True:
            try:
                # Check if we should start live
                if not self.live_bot_running and self.should_start_live_bot():
                    await self.start_live_bot()
                
                # Check if we should pause live
                elif self.live_bot_running and self.should_pause_live_bot():
                    await self.pause_live_bot()
                
                # Log status every 5 minutes
                await self.log_status()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Auto trigger loop error: {e}")
                await asyncio.sleep(300)
    
    async def log_status(self):
        """Log current status"""
        approved = self.get_approved_strategies()
        testing = [name for name, perf in self.strategies.items() if not perf.is_approved]
        
        logger.info("🤖 AUTO TRIGGER STATUS:")
        logger.info(f"   Live bot: {'🟢 RUNNING' if self.live_bot_running else '⚪ PAUSED'}")
        logger.info(f"   Approved strategies: {len(approved)}")
        if approved:
            for name in approved:
                perf = self.strategies[name]
                logger.info(f"      ✅ {name}: {perf.total_trades} trades, {perf.win_rate:.1%} win rate")
        logger.info(f"   Testing: {len(testing)}")
        if testing:
            for name in testing:
                perf = self.strategies[name]
                logger.info(f"      🧪 {name}: {perf.total_trades}/{self.min_trades_for_approval} trades, {perf.win_rate:.1%} win rate")


# Global instance
auto_trigger = AutoLiveTrigger()


def record_trade(strategy: str, profit: float):
    """Record a trade (called by trading engines)"""
    auto_trigger.record_testnet_trade(strategy, profit)


async def start_auto_trigger():
    """Start the auto trigger system"""
    await auto_trigger.run_auto_trigger_loop()


if __name__ == "__main__":
    asyncio.run(start_auto_trigger())
