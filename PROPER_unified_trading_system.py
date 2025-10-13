#!/usr/bin/env python3
"""
PROPER UNIFIED TRADING SYSTEM - Real Integration
Uses actual dependencies and wires engines together correctly
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('proper_unified_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class ProperUnifiedTradingSystem:
    """
    Proper unified trading system that actually uses the real dependencies
    and coordinates engines correctly.
    """
    
    def __init__(self):
        logger.info("🚀 Initializing PROPER Unified Trading System")
        
        # Core dependencies that ALL engines need
        self.ultra_core = None
        self.risk_engine = None
        self.pattern_memory = None
        self.brain = None
        
        # Trading engines
        self.arbitrage_engine = None
        self.scalping_engine = None
        self.moon_spotter = None
        self.real_profit_bot = None
        self.enhanced_bot = None
        
        # Status tracking
        self.running = False
        self.active_engines = []
        
    def initialize_core_dependencies(self):
        """Initialize core dependencies that engines share"""
        logger.info("📦 Initializing core dependencies...")
        
        try:
            from ultra_core import UltraCore
            self.ultra_core = UltraCore()
            logger.info("✅ UltraCore initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize UltraCore: {e}")
            return False
        
        try:
            from risk_engine import RiskEngine
            self.risk_engine = RiskEngine()
            logger.info("✅ RiskEngine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RiskEngine: {e}")
            return False
        
        try:
            from pattern_memory import PatternMemory
            self.pattern_memory = PatternMemory()
            logger.info("✅ PatternMemory initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PatternMemory: {e}")
            return False
        
        try:
            from brain import Brain
            self.brain = Brain()
            logger.info("✅ Brain initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Brain: {e}")
            return False
        
        return True
    
    def initialize_arbitrage_engine(self):
        """Initialize arbitrage engine with proper dependencies"""
        try:
            from ultra_arbitrage_engine import UltraArbitrageEngine
            
            # Pass REAL dependencies
            self.arbitrage_engine = UltraArbitrageEngine(
                ultra_core=self.ultra_core,
                risk_engine=self.risk_engine
            )
            
            self.active_engines.append('arbitrage')
            logger.info("✅ Arbitrage Engine initialized with dependencies")
            return True
        except Exception as e:
            logger.error(f"❌ Arbitrage Engine failed: {e}")
            return False
    
    def initialize_scalping_engine(self):
        """Initialize scalping engine with proper dependencies"""
        try:
            from ultra_scalping_engine import UltraScalpingEngine
            
            # Pass REAL dependencies
            self.scalping_engine = UltraScalpingEngine(
                ultra_core=self.ultra_core,
                risk_engine=self.risk_engine
            )
            
            self.active_engines.append('scalping')
            logger.info("✅ Scalping Engine initialized with dependencies")
            return True
        except Exception as e:
            logger.error(f"❌ Scalping Engine failed: {e}")
            return False
    
    def initialize_moon_spotter(self):
        """Initialize moon spotter (standalone, no dependencies needed)"""
        try:
            from ultra_moon_spotter import MicroMoonSpotter
            
            self.moon_spotter = MicroMoonSpotter()
            
            self.active_engines.append('moon_spotter')
            logger.info("✅ Moon Spotter initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Moon Spotter failed: {e}")
            return False
    
    def initialize_real_profit_bot(self):
        """Initialize REAL_PROFIT_BOT (Gate.io, standalone)"""
        try:
            from REAL_PROFIT_BOT import REAL_PROFIT_BOT
            
            self.real_profit_bot = REAL_PROFIT_BOT()
            
            self.active_engines.append('real_profit')
            logger.info("✅ REAL_PROFIT_BOT initialized (Gate.io LIVE)")
            return True
        except Exception as e:
            logger.error(f"❌ REAL_PROFIT_BOT failed: {e}")
            return False
    
    def initialize_enhanced_bot(self):
        """Initialize enhanced trading bot (Bybit testnet)"""
        try:
            from enhanced_trading_bot import EnhancedTradingBot
            
            self.enhanced_bot = EnhancedTradingBot()
            
            self.active_engines.append('enhanced')
            logger.info("✅ Enhanced Bot initialized (Bybit testnet)")
            return True
        except Exception as e:
            logger.error(f"❌ Enhanced Bot failed: {e}")
            return False
    
    async def start(self):
        """Start the unified trading system"""
        logger.info("=" * 70)
        logger.info("🚀 STARTING PROPER UNIFIED TRADING SYSTEM")
        logger.info("=" * 70)
        
        # Step 1: Initialize core dependencies
        if not self.initialize_core_dependencies():
            logger.error("❌ Core dependencies failed to initialize - aborting")
            return
        
        logger.info("\n📊 Initializing trading engines...")
        
        # Step 2: Initialize engines (each can fail independently)
        self.initialize_arbitrage_engine()
        self.initialize_scalping_engine()
        self.initialize_moon_spotter()
        self.initialize_real_profit_bot()
        self.initialize_enhanced_bot()
        
        logger.info(f"\n✅ {len(self.active_engines)} engines initialized: {', '.join(self.active_engines)}")
        
        self.running = True
        
        # Step 3: Start engines concurrently
        tasks = []
        
        if self.arbitrage_engine:
            tasks.append(asyncio.create_task(self._run_arbitrage()))
        
        if self.scalping_engine:
            tasks.append(asyncio.create_task(self._run_scalping()))
        
        if self.moon_spotter:
            tasks.append(asyncio.create_task(self._run_moon_spotter()))
        
        if self.real_profit_bot:
            tasks.append(asyncio.create_task(self._run_real_profit_bot()))
        
        if self.enhanced_bot:
            tasks.append(asyncio.create_task(self._run_enhanced_bot()))
        
        # Monitoring task
        tasks.append(asyncio.create_task(self._monitor()))
        
        logger.info(f"\n▶️  Running {len(tasks)} concurrent tasks...")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("\n⚠️  Shutdown signal received")
            self.running = False
        except Exception as e:
            logger.error(f"\n❌ System error: {e}")
            self.running = False
    
    async def _run_arbitrage(self):
        """Run arbitrage engine"""
        try:
            logger.info("▶️  Starting Arbitrage Engine...")
            await self.arbitrage_engine.start_arbitrage_scanning()
        except Exception as e:
            logger.error(f"❌ Arbitrage engine error: {e}")
    
    async def _run_scalping(self):
        """Run scalping engine"""
        try:
            logger.info("▶️  Starting Scalping Engine...")
            await self.scalping_engine.start_scalping()
        except Exception as e:
            logger.error(f"❌ Scalping engine error: {e}")
    
    async def _run_moon_spotter(self):
        """Run moon spotter"""
        try:
            logger.info("▶️  Starting Moon Spotter...")
            while self.running:
                gems = await self.moon_spotter.scan_for_new_gems()
                if gems:
                    logger.info(f"🌙 Found {len(gems)} potential micro-moons")
                await asyncio.sleep(300)  # Scan every 5 minutes
        except Exception as e:
            logger.error(f"❌ Moon spotter error: {e}")
    
    async def _run_real_profit_bot(self):
        """Run REAL_PROFIT_BOT"""
        try:
            logger.info("▶️  Starting REAL_PROFIT_BOT (Gate.io)...")
            # Note: REAL_PROFIT_BOT.run() is not async, need to wrap it
            await asyncio.to_thread(self.real_profit_bot.run)
        except Exception as e:
            logger.error(f"❌ REAL_PROFIT_BOT error: {e}")
    
    async def _run_enhanced_bot(self):
        """Run enhanced trading bot"""
        try:
            logger.info("▶️  Starting Enhanced Bot (Bybit)...")
            await self.enhanced_bot.run_enhanced_bot()
        except Exception as e:
            logger.error(f"❌ Enhanced bot error: {e}")
    
    async def _monitor(self):
        """Monitor system status"""
        while self.running:
            try:
                logger.info("\n" + "=" * 70)
                logger.info(f"📊 SYSTEM STATUS - {len(self.active_engines)} engines active")
                logger.info("=" * 70)
                
                # Get performance from each engine
                if self.arbitrage_engine:
                    try:
                        perf = self.arbitrage_engine.get_performance_summary()
                        logger.info(f"💰 Arbitrage: {perf.get('total_profit', 0):.2f} profit")
                    except:
                        pass
                
                if self.scalping_engine:
                    try:
                        perf = self.scalping_engine.get_performance_summary()
                        logger.info(f"⚡ Scalping: {perf.get('total_profit', 0):.2f} profit")
                    except:
                        pass
                
                logger.info("=" * 70)
                
                await asyncio.sleep(300)  # Status every 5 minutes
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(60)


async def main():
    """Main entry point"""
    print("=" * 70)
    print("🚀 PROPER UNIFIED TRADING SYSTEM")
    print("=" * 70)
    print("This version ACTUALLY uses real dependencies and proper integration")
    print()
    
    system = ProperUnifiedTradingSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
