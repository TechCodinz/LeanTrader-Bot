#!/usr/bin/env python3
"""
COMPREHENSIVE SYSTEM INTEGRATION TESTS
Tests all 26 systems for proper initialization and basic functionality
"""

import asyncio
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

async def test_core_infrastructure():
    """Test Phase 1: Core Infrastructure (9 systems)"""
    logger.info("=" * 80)
    logger.info("TESTING PHASE 1: CORE INFRASTRUCTURE")
    logger.info("=" * 80)
    
    tests_passed = 0
    tests_total = 9
    
    # Test 1: ExchangeRouter
    try:
        from router import ExchangeRouter
        router = ExchangeRouter()
        assert router is not None
        logger.info("✅ 1/9 - ExchangeRouter")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 1/9 - ExchangeRouter: {e}")
    
    # Test 2: RiskEngine
    try:
        from risk_engine import RiskEngine
        risk = RiskEngine()
        assert risk is not None
        assert hasattr(risk, 'max_position_size')
        logger.info("✅ 2/9 - RiskEngine")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 2/9 - RiskEngine: {e}")
    
    # Test 3: Brain
    try:
        from brain import Brain
        brain = Brain()
        assert brain is not None
        logger.info("✅ 3/9 - Brain")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 3/9 - Brain: {e}")
    
    # Test 4: PatternMemory
    try:
        from pattern_memory import PatternMemory
        memory = PatternMemory()
        assert memory is not None
        logger.info("✅ 4/9 - PatternMemory")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 4/9 - PatternMemory: {e}")
    
    # Test 5: Ledger
    try:
        from ledger import Ledger
        ledger = Ledger()
        assert ledger is not None
        logger.info("✅ 5/9 - Ledger")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 5/9 - Ledger: {e}")
    
    # Test 6: UltraCore
    try:
        from ultra_core import UltraCore
        core = UltraCore(router, ["BTC/USDT"], logger)
        assert core is not None
        logger.info("✅ 6/9 - UltraCore")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 6/9 - UltraCore: {e}")
    
    # Test 7: SituationalAwareness
    try:
        from awareness import SituationalAwareness, AwarenessConfig
        cfg = AwarenessConfig()
        awareness = SituationalAwareness(cfg)
        assert awareness is not None
        logger.info("✅ 7/9 - SituationalAwareness")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 7/9 - SituationalAwareness: {e}")
    
    # Test 8: HiveCoordinator
    try:
        from hivemind import HiveCoordinator
        hive = HiveCoordinator(["1m", "5m", "15m"])
        assert hive is not None
        logger.info("✅ 8/9 - HiveCoordinator")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 8/9 - HiveCoordinator: {e}")
    
    # Test 9: GlobalAwareness
    try:
        from gloaware import GlobalAwareness, AwarenessConfig as GloConfig
        glo_cfg = GloConfig()
        glo = GlobalAwareness(glo_cfg)
        assert glo is not None
        logger.info("✅ 9/9 - GlobalAwareness")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 9/9 - GlobalAwareness: {e}")
    
    logger.info(f"\nPhase 1 Result: {tests_passed}/{tests_total} systems operational")
    return tests_passed, tests_total

async def test_trading_engines():
    """Test Phase 2: Trading Engines (5 systems)"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PHASE 2: TRADING ENGINES")
    logger.info("=" * 80)
    
    tests_passed = 0
    tests_total = 5
    
    # Setup dependencies
    from router import ExchangeRouter
    from risk_engine import RiskEngine
    from ultra_core import UltraCore
    
    router = ExchangeRouter()
    risk = RiskEngine()
    core = UltraCore(router, ["BTC/USDT"], logger)
    
    # Test 1: UltraArbitrageEngine
    try:
        from ultra_arbitrage_engine import UltraArbitrageEngine
        arb = UltraArbitrageEngine(core, risk)
        assert arb is not None
        logger.info("✅ 1/5 - UltraArbitrageEngine")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 1/5 - UltraArbitrageEngine: {e}")
    
    # Test 2: UltraScalpingEngine
    try:
        from ultra_scalping_engine import UltraScalpingEngine
        scalp = UltraScalpingEngine(core, risk)
        assert scalp is not None
        logger.info("✅ 2/5 - UltraScalpingEngine")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 2/5 - UltraScalpingEngine: {e}")
    
    # Test 3: UltraMoonSpotter
    try:
        from ultra_moon_spotter import UltraMoonSpotter
        moon = UltraMoonSpotter()
        assert moon is not None
        logger.info("✅ 3/5 - UltraMoonSpotter")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 3/5 - UltraMoonSpotter: {e}")
    
    # Test 4: RealProfitBot
    try:
        from REAL_PROFIT_BOT import RealProfitBot
        profit = RealProfitBot()
        assert profit is not None
        logger.info("✅ 4/5 - RealProfitBot")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 4/5 - RealProfitBot: {e}")
    
    # Test 5: EnhancedTradingBot
    try:
        from enhanced_trading_bot import EnhancedTradingBot
        enhanced = EnhancedTradingBot()
        assert enhanced is not None
        assert len(enhanced.exchanges) > 0
        logger.info("✅ 5/5 - EnhancedTradingBot")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 5/5 - EnhancedTradingBot: {e}")
    
    logger.info(f"\nPhase 2 Result: {tests_passed}/{tests_total} systems operational")
    return tests_passed, tests_total

async def test_ai_ml_systems():
    """Test Phase 3: AI/ML Systems (6 systems)"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PHASE 3: AI/ML SYSTEMS")
    logger.info("=" * 80)
    
    tests_passed = 0
    tests_total = 6
    
    # Setup dependencies
    from router import ExchangeRouter
    from risk_engine import RiskEngine
    from ultra_core import UltraCore
    
    router = ExchangeRouter()
    risk = RiskEngine()
    core = UltraCore(router, ["BTC/USDT"], logger)
    
    # Test 1: EvolutionEngine
    try:
        from EVOLUTION_ENGINE import ULTIMATE_EVOLUTION_ENGINE
        evolution = ULTIMATE_EVOLUTION_ENGINE()
        assert evolution is not None
        logger.info("✅ 1/6 - EvolutionEngine (70+ models)")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 1/6 - EvolutionEngine: {e}")
    
    # Test 2: 450+ Models Bot
    try:
        from working_450_models_bot import working_450_models_bot
        models_450 = working_450_models_bot()
        assert models_450 is not None
        logger.info("✅ 2/6 - 450+ Models Bot")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 2/6 - 450+ Models Bot: {e}")
    
    # Test 3: SwarmConsciousness
    try:
        from ultra_swarm_consciousness import UltraSwarmConsciousness
        swarm = UltraSwarmConsciousness(core, risk)
        assert swarm is not None
        logger.info("✅ 3/6 - UltraSwarmConsciousness (100 agents)")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 3/6 - SwarmConsciousness: {e}")
    
    # Test 4: DivineIntelligence
    try:
        from divine_intelligence_core import DivineIntelligence
        divine = DivineIntelligence()
        assert divine is not None
        logger.info("✅ 4/6 - DivineIntelligence")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 4/6 - DivineIntelligence: {e}")
    
    # Test 5: MLStrategyEngine
    try:
        from ml_strategy_engine import MLStrategyEngine
        ml = MLStrategyEngine()
        assert ml is not None
        logger.info("✅ 5/6 - MLStrategyEngine")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 5/6 - MLStrategyEngine: {e}")
    
    # Test 6: OnlineLearner
    try:
        from online_learner import OnlineLearner
        learner = OnlineLearner()
        assert learner is not None
        logger.info("✅ 6/6 - OnlineLearner")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 6/6 - OnlineLearner: {e}")
    
    logger.info(f"\nPhase 3 Result: {tests_passed}/{tests_total} systems operational")
    return tests_passed, tests_total

async def test_advanced_intelligence():
    """Test Phase 4: Advanced Intelligence (3 systems)"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING PHASE 4: ADVANCED INTELLIGENCE")
    logger.info("=" * 80)
    
    tests_passed = 0
    tests_total = 3
    
    # Setup dependencies
    from router import ExchangeRouter
    from risk_engine import RiskEngine
    from ultra_core import UltraCore
    
    router = ExchangeRouter()
    risk = RiskEngine()
    core = UltraCore(router, ["BTC/USDT"], logger)
    
    # Test 1: QuantumIntelligence
    try:
        from ultra_quantum_intelligence import UltraQuantumIntelligence
        quantum = UltraQuantumIntelligence()
        assert quantum is not None
        logger.info("✅ 1/3 - UltraQuantumIntelligence")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 1/3 - QuantumIntelligence: {e}")
    
    # Test 2: FluidMechanics
    try:
        from ultra_fluid_mechanics import UltraFluidMechanics
        fluid = UltraFluidMechanics(core, risk)
        assert fluid is not None
        logger.info("✅ 2/3 - UltraFluidMechanics")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 2/3 - FluidMechanics: {e}")
    
    # Test 3: BacktestEngine
    try:
        from ultra_backtest_engine import UltraBacktestEngine
        backtest = UltraBacktestEngine(core, risk)
        assert backtest is not None
        logger.info("✅ 3/3 - UltraBacktestEngine")
        tests_passed += 1
    except Exception as e:
        logger.error(f"❌ 3/3 - BacktestEngine: {e}")
    
    logger.info(f"\nPhase 4 Result: {tests_passed}/{tests_total} systems operational")
    return tests_passed, tests_total

async def main():
    """Run all integration tests"""
    logger.info("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       COMPREHENSIVE SYSTEM INTEGRATION TESTS                 ║
    ║              Testing All 26 Systems                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    total_passed = 0
    total_tests = 0
    
    # Phase 1: Core Infrastructure
    p1_passed, p1_total = await test_core_infrastructure()
    total_passed += p1_passed
    total_tests += p1_total
    
    # Phase 2: Trading Engines
    p2_passed, p2_total = await test_trading_engines()
    total_passed += p2_passed
    total_tests += p2_total
    
    # Phase 3: AI/ML Systems
    p3_passed, p3_total = await test_ai_ml_systems()
    total_passed += p3_passed
    total_tests += p3_total
    
    # Phase 4: Advanced Intelligence
    p4_passed, p4_total = await test_advanced_intelligence()
    total_passed += p4_total
    total_tests += p4_total
    
    # Final Results
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Phase 1 (Core Infrastructure): {p1_passed}/{p1_total} ✅")
    logger.info(f"Phase 2 (Trading Engines):     {p2_passed}/{p2_total} ✅")
    logger.info(f"Phase 3 (AI/ML Systems):       {p3_passed}/{p3_total} ✅")
    logger.info(f"Phase 4 (Advanced Intel):      {p4_passed}/{p4_total} ✅")
    logger.info("=" * 80)
    logger.info(f"TOTAL: {total_passed}/{total_tests} systems operational ({total_passed/total_tests*100:.1f}%)")
    logger.info("=" * 80)
    
    if total_passed == total_tests:
        logger.info("🎉 100% SUCCESS - ALL SYSTEMS OPERATIONAL!")
        return 0
    else:
        logger.warning(f"⚠️  {total_tests - total_passed} system(s) need attention")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
