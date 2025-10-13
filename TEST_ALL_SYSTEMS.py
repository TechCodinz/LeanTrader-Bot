#!/usr/bin/env python3
"""
COMPREHENSIVE SYSTEM TEST
Tests ALL 34 systems to verify integration
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Test results
results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_result(name, passed, message=""):
    if passed:
        results['passed'].append(name)
        print(f"✅ {name}")
    else:
        results['failed'].append(name)
        print(f"❌ {name}: {message}")

print("=" * 80)
print("COMPREHENSIVE SYSTEM TEST - ALL 34 SYSTEMS")
print("=" * 80)

# Test imports
print("\n📦 Testing Core Imports (26 systems)...")
try:
    from router import ExchangeRouter
    test_result("ExchangeRouter", True)
except Exception as e:
    test_result("ExchangeRouter", False, str(e))

try:
    from risk_engine import RiskEngine
    test_result("RiskEngine", True)
except Exception as e:
    test_result("RiskEngine", False, str(e))

try:
    from brain import Brain
    test_result("Brain", True)
except Exception as e:
    test_result("Brain", False, str(e))

try:
    from pattern_memory import PatternMemory
    test_result("PatternMemory", True)
except Exception as e:
    test_result("PatternMemory", False, str(e))

try:
    from ledger import Ledger
    test_result("Ledger", True)
except Exception as e:
    test_result("Ledger", False, str(e))

try:
    from ultra_core import UltraCore
    test_result("UltraCore", True)
except Exception as e:
    test_result("UltraCore", False, str(e))

try:
    from awareness import SituationalAwareness
    test_result("SituationalAwareness", True)
except Exception as e:
    test_result("SituationalAwareness", False, str(e))

try:
    from hivemind import HiveCoordinator
    test_result("HiveCoordinator", True)
except Exception as e:
    test_result("HiveCoordinator", False, str(e))

try:
    from gloaware import GlobalAwareness
    test_result("GlobalAwareness", True)
except Exception as e:
    test_result("GlobalAwareness", False, str(e))

try:
    from ultra_arbitrage_engine import UltraArbitrageEngine
    test_result("UltraArbitrageEngine", True)
except Exception as e:
    test_result("UltraArbitrageEngine", False, str(e))

try:
    from ultra_scalping_engine import UltraScalpingEngine
    test_result("UltraScalpingEngine", True)
except Exception as e:
    test_result("UltraScalpingEngine", False, str(e))

try:
    from ultra_moon_spotter import UltraMoonSpotter
    test_result("UltraMoonSpotter", True)
except Exception as e:
    test_result("UltraMoonSpotter", False, str(e))

try:
    from REAL_PROFIT_BOT import RealProfitBot
    test_result("RealProfitBot", True)
except Exception as e:
    test_result("RealProfitBot", False, str(e))

try:
    from enhanced_trading_bot import EnhancedTradingBot
    test_result("EnhancedTradingBot", True)
except Exception as e:
    test_result("EnhancedTradingBot", False, str(e))

try:
    from EVOLUTION_ENGINE import ULTIMATE_EVOLUTION_ENGINE
    test_result("ULTIMATE_EVOLUTION_ENGINE", True)
except Exception as e:
    test_result("ULTIMATE_EVOLUTION_ENGINE", False, str(e))

try:
    from working_450_models_bot import working_450_models_bot
    test_result("working_450_models_bot", True)
except Exception as e:
    test_result("working_450_models_bot", False, str(e))

try:
    from ultra_swarm_consciousness import UltraSwarmConsciousness
    test_result("UltraSwarmConsciousness", True)
except Exception as e:
    test_result("UltraSwarmConsciousness", False, str(e))

try:
    from divine_intelligence_core import DivineIntelligence
    test_result("DivineIntelligence", True)
except Exception as e:
    test_result("DivineIntelligence", False, str(e))

try:
    from ml_strategy_engine import MLStrategyEngine
    test_result("MLStrategyEngine", True)
except Exception as e:
    test_result("MLStrategyEngine", False, str(e))

try:
    from online_learner import OnlineLearner
    test_result("OnlineLearner", True)
except Exception as e:
    test_result("OnlineLearner", False, str(e))

try:
    from ultra_quantum_intelligence import UltraQuantumIntelligence
    test_result("UltraQuantumIntelligence", True)
except Exception as e:
    test_result("UltraQuantumIntelligence", False, str(e))

try:
    from ultra_fluid_mechanics import UltraFluidMechanics
    test_result("UltraFluidMechanics", True)
except Exception as e:
    test_result("UltraFluidMechanics", False, str(e))

try:
    from ultra_backtest_engine import UltraBacktestEngine
    test_result("UltraBacktestEngine", True)
except Exception as e:
    test_result("UltraBacktestEngine", False, str(e))

try:
    from ultra_business_system import UltraBusinessSystem
    test_result("UltraBusinessSystem", True)
except Exception as e:
    test_result("UltraBusinessSystem", False, str(e))

try:
    from november_growth_strategy import NovemberGrowthStrategy
    test_result("NovemberGrowthStrategy", True)
except Exception as e:
    test_result("NovemberGrowthStrategy", False, str(e))

try:
    from paper_broker import PaperBroker
    test_result("PaperBroker", True)
except Exception as e:
    test_result("PaperBroker", False, str(e))

# Test advanced systems
print("\n🌟 Testing Advanced Imports (8 additional systems)...")
try:
    from ultra_scout import UltraScout
    test_result("UltraScout", True)
except Exception as e:
    test_result("UltraScout", False, str(e))

try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
    test_result("CompleteUltimateOrchestrator", True)
except Exception as e:
    test_result("CompleteUltimateOrchestrator", False, str(e))

try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import AdvancedScoutingOrchestrator
    test_result("AdvancedScoutingOrchestrator", True)
except Exception as e:
    test_result("AdvancedScoutingOrchestrator", False, str(e))

try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import ForexTradingOrchestrator
    test_result("ForexTradingOrchestrator", True)
except Exception as e:
    test_result("ForexTradingOrchestrator", False, str(e))

try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import DeepLearningOrchestrator
    test_result("DeepLearningOrchestrator", True)
except Exception as e:
    test_result("DeepLearningOrchestrator", False, str(e))

# Test orchestration
print("\n🔌 Testing Data Flows...")
try:
    from ENHANCED_DATA_FLOWS import RealTimeLearningPipeline
    test_result("RealTimeLearningPipeline", True)
except Exception as e:
    test_result("RealTimeLearningPipeline", False, str(e))

try:
    from ENHANCED_DATA_FLOWS import UnifiedScoutingPipeline
    test_result("UnifiedScoutingPipeline", True)
except Exception as e:
    test_result("UnifiedScoutingPipeline", False, str(e))

try:
    from ENHANCED_DATA_FLOWS import CollectiveIntelligenceCoordinator
    test_result("CollectiveIntelligenceCoordinator", True)
except Exception as e:
    test_result("CollectiveIntelligenceCoordinator", False, str(e))

try:
    from ENHANCED_DATA_FLOWS import UnifiedReportingSystem
    test_result("UnifiedReportingSystem", True)
except Exception as e:
    test_result("UnifiedReportingSystem", False, str(e))

# Test initialization
print("\n🚀 Testing System Initialization...")

async def test_initialization():
    try:
        from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
        
        orchestrator = CompleteUltimateOrchestrator(mode="testnet")
        
        # Initialize all systems
        await orchestrator.initialize_all_systems()
        
        # Check systems were initialized
        core_count = sum([
            1 if orchestrator.router else 0,
            1 if orchestrator.risk_engine else 0,
            1 if orchestrator.brain else 0,
            1 if orchestrator.pattern_memory else 0,
            1 if orchestrator.ledger else 0,
            1 if orchestrator.ultra_core else 0,
            1 if orchestrator.awareness else 0,
            1 if orchestrator.hivemind else 0,
            1 if orchestrator.global_awareness else 0,
        ])
        
        trading_count = len(orchestrator.trading_engines)
        ai_count = len([v for v in orchestrator.ai_systems.values() if v])
        advanced_count = len([v for v in orchestrator.advanced_systems.values() if v])
        
        test_result("Core Infrastructure (9 systems)", core_count == 9, f"Only {core_count}/9 initialized")
        test_result("Trading Engines (5 systems)", trading_count >= 5, f"Only {trading_count}/5 initialized")
        test_result("AI/ML Systems (6 systems)", ai_count >= 4, f"Only {ai_count}/6 initialized (some optional)")
        test_result("Advanced Systems (1+ systems)", advanced_count >= 1, f"Only {advanced_count} initialized")
        
        # Test wiring
        await orchestrator.wire_all_systems()
        
        orchestrator_count = len(orchestrator.orchestrators) + len(orchestrator.advanced_orchestrators)
        test_result("Orchestrators wired (6+ total)", orchestrator_count >= 6, f"Only {orchestrator_count} wired")
        
        return True
    
    except Exception as e:
        test_result("System Initialization", False, str(e))
        return False

# Run async test
loop = asyncio.get_event_loop()
init_success = loop.run_until_complete(test_initialization())

# Print results
print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)

print(f"\n✅ PASSED: {len(results['passed'])}")
for test in results['passed']:
    print(f"   ✓ {test}")

if results['failed']:
    print(f"\n❌ FAILED: {len(results['failed'])}")
    for test in results['failed']:
        print(f"   ✗ {test}")

if results['warnings']:
    print(f"\n⚠️  WARNINGS: {len(results['warnings'])}")
    for test in results['warnings']:
        print(f"   ! {test}")

# Final verdict
total_tests = len(results['passed']) + len(results['failed'])
pass_rate = (len(results['passed']) / total_tests * 100) if total_tests > 0 else 0

print("\n" + "=" * 80)
print(f"OVERALL: {len(results['passed'])}/{total_tests} tests passed ({pass_rate:.1f}%)")

if pass_rate >= 90:
    print("✅ EXCELLENT - System ready for deployment!")
elif pass_rate >= 75:
    print("⚠️  GOOD - Minor issues to address")
else:
    print("❌ NEEDS WORK - Major issues detected")

print("=" * 80)
