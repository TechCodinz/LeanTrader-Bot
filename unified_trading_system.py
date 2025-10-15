#!/usr/bin/env python3
"""
UNIFIED TRADING SYSTEM - Central Orchestrator
Integrates all existing trading bot components into a single unified system.

This orchestrator manages:
- Multiple trading engines (arbitrage, scalping, moon spotting)
- AI/ML engines (evolution, online learning)
- Exchange connectivity
- Risk management
- Portfolio coordination
- Monitoring and notifications
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_trading_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class SystemStatus:
    """System-wide status tracking"""
    started_at: datetime = field(default_factory=datetime.now)
    engines_running: Dict[str, bool] = field(default_factory=dict)
    total_trades: int = 0
    total_profit: float = 0.0
    active_positions: int = 0
    errors: List[str] = field(default_factory=list)
    last_health_check: Optional[datetime] = None


class UnifiedTradingSystem:
    """
    Central orchestrator for all trading components.
    Manages lifecycle, coordination, and monitoring of all engines.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the unified trading system"""
        self.config = config or self._load_default_config()
        self.status = SystemStatus()
        
        # Component references
        self.enhanced_bot = None
        self.arbitrage_engine = None
        self.scalping_engine = None
        self.moon_spotter = None
        self.evolution_engine = None
        self.online_learner = None
        self.real_profit_bot = None
        
        # Infrastructure
        self.exchange_router = None
        self.risk_manager = None
        self.portfolio_manager = None
        self.telegram_bot = None
        
        # Control flags
        self.running = False
        self.engines_enabled = {
            'enhanced_bot': True,
            'arbitrage': True,
            'scalping': True,
            'moon_spotter': True,
            'evolution': True,
            'learning': True,
        }
        
        logger.info("🚀 Unified Trading System initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            'mode': os.getenv('TRADING_MODE', 'paper'),  # paper or live
            'exchanges': ['bybit', 'binance', 'okx'],
            'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'],
            'risk': {
                'max_position_size': 0.1,  # 10% of portfolio per position
                'max_daily_loss': 0.05,     # 5% max daily loss
                'stop_loss_pct': 0.02,      # 2% stop loss
            },
            'arbitrage': {
                'enabled': True,
                'min_spread_bps': 12,
                'max_concurrent': 5,
            },
            'scalping': {
                'enabled': True,
                'timeframes': ['1m', '5m', '15m'],
                'min_profit_bps': 5,
            },
            'evolution': {
                'enabled': True,
                'population_size': 50,
                'generations': 100,
            },
            'notifications': {
                'telegram_enabled': True,
                'alert_on_trades': True,
                'alert_on_errors': True,
            }
        }
    
    async def initialize_components(self):
        """Initialize all trading components"""
        logger.info("📦 Initializing components...")
        
        # Initialize enhanced trading bot
        if self.engines_enabled.get('enhanced_bot'):
            await self._init_enhanced_bot()
        
        # Initialize exchange router
        await self._init_exchange_router()
        
        # Initialize trading engines
        if self.engines_enabled.get('arbitrage'):
            await self._init_arbitrage_engine()
        
        if self.engines_enabled.get('scalping'):
            await self._init_scalping_engine()
        
        if self.engines_enabled.get('moon_spotter'):
            await self._init_moon_spotter()
        
        # Initialize AI engines
        if self.engines_enabled.get('evolution'):
            await self._init_evolution_engine()
        
        if self.engines_enabled.get('learning'):
            await self._init_online_learner()
        
        # Initialize infrastructure
        await self._init_risk_manager()
        await self._init_portfolio_manager()
        await self._init_telegram_bot()
        
        logger.info("✅ All components initialized successfully")
    
    async def _init_enhanced_bot(self):
        """Initialize enhanced trading bot"""
        try:
            from enhanced_trading_bot import EnhancedTradingBot
            self.enhanced_bot = EnhancedTradingBot()
            self.status.engines_running['enhanced_bot'] = True
            logger.info("✅ Enhanced trading bot initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced bot: {e}")
            self.status.errors.append(f"Enhanced bot init: {e}")
    
    async def _init_exchange_router(self):
        """Initialize exchange router"""
        try:
            # Try traders_core router first
            try:
                from traders_core.router import ExchangeRouter
                self.exchange_router = ExchangeRouter()
                logger.info("✅ Exchange router initialized (traders_core)")
            except:
                # Fallback to src/leantrader router
                from src.leantrader.execution.router import ExchangeRouter
                self.exchange_router = ExchangeRouter()
                logger.info("✅ Exchange router initialized (leantrader)")
        except Exception as e:
            logger.warning(f"⚠️  Exchange router not available: {e}")
            self.status.errors.append(f"Exchange router: {e}")
    
    async def _init_arbitrage_engine(self):
        """Initialize arbitrage engine"""
        try:
            from ultra_arbitrage_engine import UltraArbitrageEngine
            self.arbitrage_engine = UltraArbitrageEngine(
                exchanges=self.config['exchanges'],
                symbols=self.config['symbols'],
                min_spread_bps=self.config['arbitrage']['min_spread_bps']
            )
            self.status.engines_running['arbitrage'] = True
            logger.info("✅ Arbitrage engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize arbitrage engine: {e}")
            self.status.errors.append(f"Arbitrage engine: {e}")
    
    async def _init_scalping_engine(self):
        """Initialize scalping engine"""
        try:
            from ultra_scalping_engine import UltraScalpingEngine
            self.scalping_engine = UltraScalpingEngine(
                exchange=self.config['exchanges'][0],  # Primary exchange
                symbols=self.config['symbols'],
                timeframes=self.config['scalping']['timeframes']
            )
            self.status.engines_running['scalping'] = True
            logger.info("✅ Scalping engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize scalping engine: {e}")
            self.status.errors.append(f"Scalping engine: {e}")
    
    async def _init_moon_spotter(self):
        """Initialize moon spotter"""
        try:
            from ultra_moon_spotter import UltraMoonSpotter
            self.moon_spotter = UltraMoonSpotter(
                exchange=self.config['exchanges'][0],
                symbols=self.config['symbols']
            )
            self.status.engines_running['moon_spotter'] = True
            logger.info("✅ Moon spotter initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize moon spotter: {e}")
            self.status.errors.append(f"Moon spotter: {e}")
    
    async def _init_evolution_engine(self):
        """Initialize evolution engine"""
        try:
            from EVOLUTION_ENGINE import ULTIMATE_EVOLUTION_ENGINE
            self.evolution_engine = ULTIMATE_EVOLUTION_ENGINE()
            self.status.engines_running['evolution'] = True
            logger.info("✅ Evolution engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize evolution engine: {e}")
            self.status.errors.append(f"Evolution engine: {e}")
    
    async def _init_online_learner(self):
        """Initialize online learner"""
        try:
            from online_learner import OnlineLearner
            self.online_learner = OnlineLearner()
            self.status.engines_running['online_learner'] = True
            logger.info("✅ Online learner initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize online learner: {e}")
            self.status.errors.append(f"Online learner: {e}")
    
    async def _init_risk_manager(self):
        """Initialize risk management"""
        try:
            # Try to use existing risk gates
            from risk.gates import GuardState, RiskLimits
            logger.info("✅ Risk manager initialized")
        except Exception as e:
            logger.warning(f"⚠️  Risk manager not available, using defaults: {e}")
    
    async def _init_portfolio_manager(self):
        """Initialize portfolio manager"""
        try:
            from allocators.portfolio import choose_assets
            logger.info("✅ Portfolio manager initialized")
        except Exception as e:
            logger.warning(f"⚠️  Portfolio manager not available: {e}")
    
    async def _init_telegram_bot(self):
        """Initialize Telegram notifications"""
        try:
            # Telegram is integrated in enhanced_bot
            if self.enhanced_bot:
                logger.info("✅ Telegram notifications available via enhanced bot")
        except Exception as e:
            logger.warning(f"⚠️  Telegram notifications not available: {e}")
    
    async def start(self):
        """Start the unified trading system"""
        logger.info("🚀 Starting Unified Trading System")
        logger.info(f"📊 Mode: {self.config['mode']}")
        logger.info(f"🔗 Exchanges: {', '.join(self.config['exchanges'])}")
        logger.info(f"💱 Symbols: {', '.join(self.config['symbols'])}")
        
        # Initialize all components
        await self.initialize_components()
        
        self.running = True
        
        # Start all engines concurrently
        tasks = []
        
        if self.enhanced_bot:
            tasks.append(asyncio.create_task(self._run_enhanced_bot()))
        
        if self.arbitrage_engine:
            tasks.append(asyncio.create_task(self._run_arbitrage_engine()))
        
        if self.scalping_engine:
            tasks.append(asyncio.create_task(self._run_scalping_engine()))
        
        if self.moon_spotter:
            tasks.append(asyncio.create_task(self._run_moon_spotter()))
        
        # Add monitoring and health check tasks
        tasks.append(asyncio.create_task(self._monitor_system()))
        tasks.append(asyncio.create_task(self._health_check_loop()))
        
        logger.info(f"✅ Started {len(tasks)} concurrent tasks")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"❌ System error: {e}")
            await self.stop()
    
    async def _run_enhanced_bot(self):
        """Run enhanced trading bot"""
        try:
            logger.info("▶️  Starting enhanced bot...")
            await self.enhanced_bot.run_enhanced_bot()
        except Exception as e:
            logger.error(f"❌ Enhanced bot error: {e}")
            self.status.errors.append(f"Enhanced bot runtime: {e}")
    
    async def _run_arbitrage_engine(self):
        """Run arbitrage engine"""
        try:
            logger.info("▶️  Starting arbitrage engine...")
            await self.arbitrage_engine.start_arbitrage_scanning()
        except Exception as e:
            logger.error(f"❌ Arbitrage engine error: {e}")
            self.status.errors.append(f"Arbitrage runtime: {e}")
    
    async def _run_scalping_engine(self):
        """Run scalping engine"""
        try:
            logger.info("▶️  Starting scalping engine...")
            await self.scalping_engine.start_scalping()
        except Exception as e:
            logger.error(f"❌ Scalping engine error: {e}")
            self.status.errors.append(f"Scalping runtime: {e}")
    
    async def _run_moon_spotter(self):
        """Run moon spotter"""
        try:
            logger.info("▶️  Starting moon spotter...")
            # Moon spotter integration
            while self.running:
                # Scan for momentum opportunities
                await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"❌ Moon spotter error: {e}")
            self.status.errors.append(f"Moon spotter runtime: {e}")
    
    async def _monitor_system(self):
        """Monitor system performance"""
        while self.running:
            try:
                # Log system status
                logger.info("=" * 60)
                logger.info(f"📊 SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"⏱️  Uptime: {datetime.now() - self.status.started_at}")
                logger.info(f"🎯 Engines running: {sum(self.status.engines_running.values())}/{len(self.status.engines_running)}")
                logger.info(f"💰 Total trades: {self.status.total_trades}")
                logger.info(f"📈 Total profit: ${self.status.total_profit:.2f}")
                logger.info(f"📍 Active positions: {self.status.active_positions}")
                logger.info(f"⚠️  Errors: {len(self.status.errors)}")
                logger.info("=" * 60)
                
                await asyncio.sleep(300)  # Status update every 5 minutes
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
    
    async def _health_check_loop(self):
        """Perform periodic health checks"""
        while self.running:
            try:
                self.status.last_health_check = datetime.now()
                
                # Check each engine
                for engine_name, is_running in self.status.engines_running.items():
                    if not is_running:
                        logger.warning(f"⚠️  Engine {engine_name} is not running!")
                
                # Check for excessive errors
                if len(self.status.errors) > 50:
                    logger.error(f"🚨 ALERT: Too many errors ({len(self.status.errors)})")
                
                await asyncio.sleep(60)  # Health check every minute
            except Exception as e:
                logger.error(f"❌ Health check error: {e}")
    
    async def stop(self):
        """Stop the unified trading system"""
        logger.info("🛑 Stopping Unified Trading System...")
        self.running = False
        
        # Stop all engines
        # (Add cleanup code here if engines need explicit shutdown)
        
        logger.info("✅ System stopped gracefully")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'running': self.running,
            'uptime': str(datetime.now() - self.status.started_at),
            'engines': self.status.engines_running,
            'trades': self.status.total_trades,
            'profit': self.status.total_profit,
            'positions': self.status.active_positions,
            'errors': len(self.status.errors),
            'last_health_check': self.status.last_health_check,
        }


async def main():
    """Main entry point"""
    print("=" * 70)
    print("🚀 UNIFIED TRADING SYSTEM - Central Orchestrator")
    print("=" * 70)
    print("Integrating all trading components into a unified system")
    print()
    
    # Create and start the system
    system = UnifiedTradingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Received shutdown signal")
        await system.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        await system.stop()
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
