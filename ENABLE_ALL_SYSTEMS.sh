#!/bin/bash
# Enable ALL bot components and make them work
echo "🚀 ENABLING ALL BOT COMPONENTS..."
echo ""

cd ~/trading_bot

# Backup
cp COMPLETE_ULTIMATE_ORCHESTRATOR.py COMPLETE_ULTIMATE_ORCHESTRATOR.py.backup.enable_all.$(date +%Y%m%d_%H%M%S)
echo "✅ Backup created"
echo ""

# Replace the availability flags section with proper imports and checks
python3 << 'PYEOF'
print("📝 Setting up proper system imports and availability checks...")

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the flags section
flags_start = content.find('# ALL SYSTEM AVAILABILITY FLAGS')
if flags_start == -1:
    print("❌ Flags section not found")
    exit(1)

flags_end = content.find('\nfrom ULTIMATE_ORCHESTRATOR', flags_start)

# Replace with proper try/except imports
proper_imports = '''# ============================================================================
# ULTRA SYSTEM IMPORTS WITH AVAILABILITY CHECKS
# ============================================================================

# Try to import each system and set availability flag
print("🔧 Loading Ultra Systems...")

# Ultra Arbitrage Engine
try:
    from ultra_arbitrage_engine import UltraArbitrageEngine
    ULTRA_ARBITRAGE_AVAILABLE = True
    print("   ✅ Ultra Arbitrage Engine available")
except ImportError as e:
    ULTRA_ARBITRAGE_AVAILABLE = False
    print(f"   ⚠️  Ultra Arbitrage Engine not available: {e}")

# Ultra Scalping Engine
try:
    from ultra_scalping_engine import UltraScalpingEngine
    ULTRA_SCALPING_AVAILABLE = True
    print("   ✅ Ultra Scalping Engine available")
except ImportError:
    ULTRA_SCALPING_AVAILABLE = False
    print("   ⚠️  Ultra Scalping Engine not available")

# Ultra Moon System
try:
    from ultra_moon_system import UltraMoonSystem
    ULTRA_MOON_AVAILABLE = True
    ULTRA_MOON_SPOTTER_AVAILABLE = True
    print("   ✅ Ultra Moon System available")
except ImportError:
    ULTRA_MOON_AVAILABLE = False
    ULTRA_MOON_SPOTTER_AVAILABLE = False
    print("   ⚠️  Ultra Moon System not available")

# Ultra Quantum Intelligence
try:
    from ultra_quantum_intelligence import UltraQuantumIntelligence
    ULTRA_QUANTUM_AVAILABLE = True
    print("   ✅ Ultra Quantum Intelligence available")
except ImportError:
    ULTRA_QUANTUM_AVAILABLE = False
    print("   ⚠️  Ultra Quantum Intelligence not available")

# Ultra Swarm Consciousness
try:
    from ultra_swarm_consciousness import UltraSwarmConsciousness
    ULTRA_SWARM_AVAILABLE = True
    print("   ✅ Ultra Swarm Consciousness available")
except ImportError:
    ULTRA_SWARM_AVAILABLE = False
    print("   ⚠️  Ultra Swarm Consciousness not available")

# Ultra Fluid Mechanics
try:
    from ultra_fluid_mechanics import UltraFluidMechanics
    ULTRA_FLUID_AVAILABLE = True
    print("   ✅ Ultra Fluid Mechanics available")
except ImportError:
    ULTRA_FLUID_AVAILABLE = False
    print("   ⚠️  Ultra Fluid Mechanics not available")

# Ultra ML Pipeline
try:
    from ultra_ml_pipeline import UltraMLPipeline
    ULTRA_ML_AVAILABLE = True
    print("   ✅ Ultra ML Pipeline available")
except ImportError:
    ULTRA_ML_AVAILABLE = False
    print("   ⚠️  Ultra ML Pipeline not available")

# Trade Planner
try:
    from trade_planner import TradePlanner
    TRADE_PLANNER_AVAILABLE = True
    print("   ✅ Trade Planner available")
except ImportError:
    TRADE_PLANNER_AVAILABLE = False
    print("   ⚠️  Trade Planner not available")

# Working 450 Models Bot
try:
    from working_450_models_bot import working_450_models_bot
    MODELS_450_AVAILABLE = True
    WORKING_450_MODELS_AVAILABLE = True
    ULTIMATE_BOT_AVAILABLE = True
    print("   ✅ 450+ Models Bot available")
except ImportError:
    MODELS_450_AVAILABLE = False
    WORKING_450_MODELS_AVAILABLE = False
    ULTIMATE_BOT_AVAILABLE = False
    print("   ⚠️  450+ Models Bot not available")

# Trader Core
try:
    from trader_core import TraderCore
    TRADER_CORE_AVAILABLE = True
    print("   ✅ Trader Core available")
except ImportError:
    TRADER_CORE_AVAILABLE = False
    print("   ⚠️  Trader Core not available")

# Unified Trading System
try:
    from unified_trading_system import UnifiedTradingSystem
    UNIFIED_TRADING_AVAILABLE = True
    print("   ✅ Unified Trading System available")
except ImportError:
    UNIFIED_TRADING_AVAILABLE = False
    print("   ⚠️  Unified Trading System not available")

# Divine Intelligence (already loaded above)
DIVINE_INTELLIGENCE_AVAILABLE = DIVINE_FEATURES_AVAILABLE

print(f"\\n✅ System scan complete!")
print(f"   Available systems: {sum([ULTRA_ARBITRAGE_AVAILABLE, ULTRA_SCALPING_AVAILABLE, ULTRA_MOON_AVAILABLE, ULTRA_QUANTUM_AVAILABLE, ULTRA_SWARM_AVAILABLE, ULTRA_FLUID_AVAILABLE, ULTRA_ML_AVAILABLE, TRADE_PLANNER_AVAILABLE, MODELS_450_AVAILABLE, TRADER_CORE_AVAILABLE, UNIFIED_TRADING_AVAILABLE])}/11")

'''

# Replace the section
content = content[:flags_start] + proper_imports + content[flags_end:]

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Updated system imports with auto-detection")
PYEOF

echo ""
echo "🧪 Testing import..."
python3 << 'PYEOF'
try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
    print("✅ Import successful - systems will auto-detect!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Ready to restart with ALL available systems!"
    echo ""
    echo "🚀 Restarting bot..."
else
    echo "❌ Fix failed"
    exit 1
fi
