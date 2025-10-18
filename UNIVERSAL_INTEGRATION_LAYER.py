#!/usr/bin/env python3
"""
UNIVERSAL INTEGRATION LAYER
Wires ALL components together - provides universal access to all systems
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


class UniversalIntegrationLayer:
    """
    Universal Integration Layer - Provides access to ALL systems
    
    This layer wires together:
    - Core infrastructure (26 systems in MASTER_ORCHESTRATOR)
    - All standalone bots
    - All utility functions
    - All trading engines
    - All AI/ML systems
    
    Everything is accessible from one place.
    """
    
    def __init__(self):
        self.systems = {}
        self.bots = {}
        self.engines = {}
        self.utils = {}
        self.ai_systems = {}
        
        logger.info("🔌 Universal Integration Layer initializing...")
    
    def load_core_systems(self):
        """Load all 26 core systems from MASTER_ORCHESTRATOR"""
        logger.info("Loading core systems...")
        
        try:
            from router import ExchangeRouter
            self.systems['router'] = ExchangeRouter
            logger.info("  ✅ ExchangeRouter")
        except Exception as e:
            logger.warning(f"  ⚠️  ExchangeRouter: {e}")
        
        try:
            from risk_engine import RiskEngine
            self.systems['risk_engine'] = RiskEngine
            logger.info("  ✅ RiskEngine")
        except Exception as e:
            logger.warning(f"  ⚠️  RiskEngine: {e}")
        
        try:
            from brain import Brain
            self.systems['brain'] = Brain
            logger.info("  ✅ Brain")
        except Exception as e:
            logger.warning(f"  ⚠️  Brain: {e}")
        
        try:
            from pattern_memory import PatternMemory
            self.systems['pattern_memory'] = PatternMemory
            logger.info("  ✅ PatternMemory")
        except Exception as e:
            logger.warning(f"  ⚠️  PatternMemory: {e}")
        
        try:
            from ledger import Ledger
            self.systems['ledger'] = Ledger
            logger.info("  ✅ Ledger")
        except Exception as e:
            logger.warning(f"  ⚠️  Ledger: {e}")
        
        try:
            from ultra_core import UltraCore
            self.systems['ultra_core'] = UltraCore
            logger.info("  ✅ UltraCore")
        except Exception as e:
            logger.warning(f"  ⚠️  UltraCore: {e}")
        
        try:
            from awareness import SituationalAwareness
            self.systems['awareness'] = SituationalAwareness
            logger.info("  ✅ SituationalAwareness")
        except Exception as e:
            logger.warning(f"  ⚠️  SituationalAwareness: {e}")
        
        try:
            from hivemind import HiveCoordinator
            self.systems['hivemind'] = HiveCoordinator
            logger.info("  ✅ HiveCoordinator")
        except Exception as e:
            logger.warning(f"  ⚠️  HiveCoordinator: {e}")
        
        try:
            from gloaware import GlobalAwareness
            self.systems['gloaware'] = GlobalAwareness
            logger.info("  ✅ GlobalAwareness")
        except Exception as e:
            logger.warning(f"  ⚠️  GlobalAwareness: {e}")
        
        logger.info(f"✅ Loaded {len(self.systems)}/9 core infrastructure systems")
    
    def load_trading_engines(self):
        """Load all trading engines"""
        logger.info("Loading trading engines...")
        
        try:
            from ultra_arbitrage_engine import UltraArbitrageEngine
            self.engines['arbitrage'] = UltraArbitrageEngine
            logger.info("  ✅ UltraArbitrageEngine")
        except Exception as e:
            logger.warning(f"  ⚠️  UltraArbitrageEngine: {e}")
        
        try:
            from ultra_scalping_engine import UltraScalpingEngine
            self.engines['scalping'] = UltraScalpingEngine
            logger.info("  ✅ UltraScalpingEngine")
        except Exception as e:
            logger.warning(f"  ⚠️  UltraScalpingEngine: {e}")
        
        try:
            from ultra_moon_spotter import UltraMoonSpotter
            self.engines['moon_spotter'] = UltraMoonSpotter
            logger.info("  ✅ UltraMoonSpotter")
        except Exception as e:
            logger.warning(f"  ⚠️  UltraMoonSpotter: {e}")
        
        try:
            from REAL_PROFIT_BOT import RealProfitBot
            self.engines['real_profit'] = RealProfitBot
            logger.info("  ✅ RealProfitBot")
        except Exception as e:
            logger.warning(f"  ⚠️  RealProfitBot: {e}")
        
        try:
            from enhanced_trading_bot import EnhancedTradingBot
            self.engines['enhanced'] = EnhancedTradingBot
            logger.info("  ✅ EnhancedTradingBot")
        except Exception as e:
            logger.warning(f"  ⚠️  EnhancedTradingBot: {e}")
        
        logger.info(f"✅ Loaded {len(self.engines)}/5 trading engines")
    
    def load_ai_systems(self):
        """Load all AI/ML systems"""
        logger.info("Loading AI/ML systems...")
        
        try:
            from EVOLUTION_ENGINE import ULTIMATE_EVOLUTION_ENGINE
            self.ai_systems['evolution'] = ULTIMATE_EVOLUTION_ENGINE
            logger.info("  ✅ EvolutionEngine")
        except Exception as e:
            logger.warning(f"  ⚠️  EvolutionEngine: {e}")
        
        try:
            from working_450_models_bot import working_450_models_bot
            self.ai_systems['models_450'] = working_450_models_bot
            logger.info("  ✅ 450+ Models Bot")
        except Exception as e:
            logger.warning(f"  ⚠️  450+ Models: {e}")
        
        try:
            from ultra_swarm_consciousness import UltraSwarmConsciousness
            self.ai_systems['swarm'] = UltraSwarmConsciousness
            logger.info("  ✅ UltraSwarmConsciousness")
        except Exception as e:
            logger.warning(f"  ⚠️  SwarmConsciousness: {e}")
        
        try:
            from divine_intelligence_core import DivineIntelligence
            self.ai_systems['divine'] = DivineIntelligence
            logger.info("  ✅ DivineIntelligence")
        except Exception as e:
            logger.warning(f"  ⚠️  DivineIntelligence: {e}")
        
        try:
            from ml_strategy_engine import MLStrategyEngine
            self.ai_systems['ml_strategy'] = MLStrategyEngine
            logger.info("  ✅ MLStrategyEngine")
        except Exception as e:
            logger.warning(f"  ⚠️  MLStrategyEngine: {e}")
        
        try:
            from online_learner import OnlineLearner
            self.ai_systems['online_learner'] = OnlineLearner
            logger.info("  ✅ OnlineLearner")
        except Exception as e:
            logger.warning(f"  ⚠️  OnlineLearner: {e}")
        
        logger.info(f"✅ Loaded {len(self.ai_systems)}/6 AI/ML systems")
    
    def load_all(self):
        """Load all available systems"""
        logger.info("=" * 80)
        logger.info("🔌 LOADING ALL SYSTEMS")
        logger.info("=" * 80)
        
        self.load_core_systems()
        self.load_trading_engines()
        self.load_ai_systems()
        
        total_loaded = len(self.systems) + len(self.engines) + len(self.ai_systems)
        
        logger.info("=" * 80)
        logger.info(f"✅ LOADED {total_loaded} SYSTEMS TOTAL")
        logger.info("=" * 80)
        
        return total_loaded
    
    def get_system(self, name: str):
        """Get any system by name"""
        if name in self.systems:
            return self.systems[name]
        if name in self.engines:
            return self.engines[name]
        if name in self.ai_systems:
            return self.ai_systems[name]
        return None
    
    def list_all(self) -> Dict[str, List[str]]:
        """List all available systems"""
        return {
            'core_systems': list(self.systems.keys()),
            'trading_engines': list(self.engines.keys()),
            'ai_systems': list(self.ai_systems.keys()),
        }


# Global instance
_integration_layer = None

def get_integration_layer() -> UniversalIntegrationLayer:
    """Get the global integration layer instance"""
    global _integration_layer
    if _integration_layer is None:
        _integration_layer = UniversalIntegrationLayer()
        _integration_layer.load_all()
    return _integration_layer


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         UNIVERSAL INTEGRATION LAYER - TEST MODE              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    layer = get_integration_layer()
    
    print("\n📊 AVAILABLE SYSTEMS:")
    all_systems = layer.list_all()
    
    print(f"\n🎯 Core Systems ({len(all_systems['core_systems'])}):")
    for name in all_systems['core_systems']:
        print(f"  • {name}")
    
    print(f"\n⚡ Trading Engines ({len(all_systems['trading_engines'])}):")
    for name in all_systems['trading_engines']:
        print(f"  • {name}")
    
    print(f"\n🤖 AI/ML Systems ({len(all_systems['ai_systems'])}):")
    for name in all_systems['ai_systems']:
        print(f"  • {name}")
    
    total = sum(len(v) for v in all_systems.values())
    print(f"\n✅ TOTAL: {total} systems accessible through Universal Integration Layer")
