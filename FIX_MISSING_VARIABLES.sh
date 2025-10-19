#!/bin/bash
echo "🔧 Adding missing variable definitions..."
echo ""

cd ~/trading_bot

# Backup
cp COMPLETE_ULTIMATE_ORCHESTRATOR.py COMPLETE_ULTIMATE_ORCHESTRATOR.py.backup.vars.$(date +%Y%m%d_%H%M%S)
echo "✅ Backup created"

# Add missing variable definitions at the top of the file after imports
python3 << 'PYEOF'
print("📝 Adding missing system availability flags...")

with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if variables are already defined
if 'ULTRA_ARBITRAGE_AVAILABLE' in content:
    print("✅ Variables already defined")
else:
    # Find where to insert (after DIVINE_FEATURES_AVAILABLE section)
    # Look for the warning section and add before imports
    
    insertion_point = content.find('# Import complete unified orchestrator')
    
    if insertion_point == -1:
        insertion_point = content.find('from ULTIMATE_ORCHESTRATOR import')
    
    if insertion_point > 0:
        # Define all the missing availability flags
        new_vars = '''
# ============================================================================
# SYSTEM AVAILABILITY FLAGS
# ============================================================================
# Import availability checks for ultra systems
try:
    from ultra_arbitrage_engine import UltraArbitrageEngine
    ULTRA_ARBITRAGE_AVAILABLE = True
except:
    ULTRA_ARBITRAGE_AVAILABLE = False

try:
    from ultra_scalping_engine import UltraScalpingEngine
    ULTRA_SCALPING_AVAILABLE = True
except:
    ULTRA_SCALPING_AVAILABLE = False

try:
    from ultra_moon_system import UltraMoonSystem
    ULTRA_MOON_AVAILABLE = True
except:
    ULTRA_MOON_AVAILABLE = False

try:
    from ultra_swarm_consciousness import UltraSwarmConsciousness
    ULTRA_SWARM_AVAILABLE = True
except:
    ULTRA_SWARM_AVAILABLE = False

try:
    from ultra_fluid_mechanics import UltraFluidMechanics
    ULTRA_FLUID_AVAILABLE = True
except:
    ULTRA_FLUID_AVAILABLE = False

try:
    from ultra_ml_pipeline import UltraMLPipeline
    ULTRA_ML_AVAILABLE = True
except:
    ULTRA_ML_AVAILABLE = False

try:
    from trade_planner import TradePlanner
    TRADE_PLANNER_AVAILABLE = True
except:
    TRADE_PLANNER_AVAILABLE = False

try:
    from working_450_models_bot import working_450_models_bot
    MODELS_450_AVAILABLE = True
except:
    MODELS_450_AVAILABLE = False

try:
    from trader_core import TraderCore
    TRADER_CORE_AVAILABLE = True
except:
    TRADER_CORE_AVAILABLE = False

try:
    from unified_trading_system import UnifiedTradingSystem
    UNIFIED_TRADING_AVAILABLE = True
except:
    UNIFIED_TRADING_AVAILABLE = False

'''
        content = content[:insertion_point] + new_vars + '\n' + content[insertion_point:]
        
        with open('COMPLETE_ULTIMATE_ORCHESTRATOR.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Added system availability flags")
    else:
        print("❌ Could not find insertion point")
        exit(1)

PYEOF

echo ""
echo "🧪 Testing import..."
python3 << 'PYEOF'
try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
    print("✅ Import successful!")
except NameError as e:
    print(f"❌ Still missing: {e}")
except Exception as e:
    print(f"⚠️  Other error: {type(e).__name__}: {e}")
PYEOF

echo ""
echo "✅ Fix complete! Try starting the bot again."
