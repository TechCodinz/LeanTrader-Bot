#!/usr/bin/env python3
"""
MAP FILE USAGE - Show exactly which files are used vs unused
"""

import ast
from pathlib import Path
import re

# Files directly used by MASTER_ORCHESTRATOR_FIXED.py
MASTER_ORCHESTRATOR_IMPORTS = [
    # Core infrastructure
    'router.py', 'risk_engine.py', 'brain.py', 'pattern_memory.py', 'ledger.py',
    'ultra_core.py', 'awareness.py', 'hivemind.py', 'gloaware.py',
    
    # Trading engines
    'ultra_arbitrage_engine.py', 'ultra_scalping_engine.py', 'ultra_moon_spotter.py',
    'REAL_PROFIT_BOT.py', 'enhanced_trading_bot.py',
    
    # AI/ML
    'EVOLUTION_ENGINE.py', 'working_450_models_bot.py', 'ultra_swarm_consciousness.py',
    'divine_intelligence_core.py', 'ml_strategy_engine.py', 'online_learner.py',
    
    # Advanced
    'ultra_quantum_intelligence.py', 'ultra_fluid_mechanics.py', 'ultra_backtest_engine.py',
    
    # Business
    'ultra_business_system.py', 'november_growth_strategy.py',
    
    # Utilities
    'paper_broker.py'
]

# Files indirectly used (imported by the above)
INDIRECT_IMPORTS = {
    'ultra_core.py': ['ultra_scout.py'],
    'REAL_PROFIT_BOT.py': ['bybit_adapter.py'],
    'enhanced_trading_bot.py': [],
    # Add more as discovered
}

def analyze_imports(file_path):
    """Analyze what a Python file imports"""
    try:
        content = file_path.read_text()
        imports = []
        
        # Find all import statements
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('from ') and ' import ' in line:
                module = line.split('from ')[1].split(' import')[0].strip()
                if not module.startswith('.'):  # Skip relative imports for now
                    imports.append(module + '.py')
            elif line.startswith('import ') and not line.startswith('import sys'):
                module = line.split('import ')[1].split(' as')[0].split(',')[0].strip()
                if '.' not in module:  # Skip package imports
                    imports.append(module + '.py')
        
        return imports
    except:
        return []

def main():
    workspace = Path('/workspace')
    all_py_files = sorted([f.name for f in workspace.glob('*.py')])
    
    print("=" * 100)
    print("FILE USAGE MAP - COMPLETE BREAKDOWN")
    print("=" * 100)
    
    # Category 1: Core systems (used by MASTER_ORCHESTRATOR)
    core_systems = set(MASTER_ORCHESTRATOR_IMPORTS)
    
    # Category 2: Dependencies (imported by core systems)
    dependencies = set()
    for core_file in core_systems:
        file_path = workspace / core_file
        if file_path.exists():
            deps = analyze_imports(file_path)
            dependencies.update([d for d in deps if d in all_py_files])
    
    # Category 3: Utility/Test files
    utility_patterns = ['test_', 'debug_', 'smoke_', 'diag', 'verify_', 'check', 'deploy']
    utility_files = set([f for f in all_py_files if any(p in f.lower() for p in utility_patterns)])
    
    # Category 4: Alternative/old versions
    alternative_files = set([f for f in all_py_files if any(x in f.lower() for x in ['_old', 'backup', 'copy', 'alternative'])])
    
    # Category 5: Standalone bots (not used by MASTER_ORCHESTRATOR)
    bot_files = set([f for f in all_py_files if 'bot' in f.lower() or 'BOT' in f]) - core_systems
    
    # Category 6: Everything else
    other_files = set(all_py_files) - core_systems - dependencies - utility_files - alternative_files - bot_files
    
    print(f"\n📊 TOTAL FILES: {len(all_py_files)} Python files\n")
    
    print("=" * 100)
    print("🎯 CATEGORY 1: CORE SYSTEMS (Used by MASTER_ORCHESTRATOR)")
    print("=" * 100)
    print(f"Count: {len(core_systems)} files\n")
    for f in sorted(core_systems):
        print(f"  ✅ {f}")
    
    print(f"\n{'=' * 100}")
    print("🔗 CATEGORY 2: DEPENDENCIES (Imported by core systems)")
    print("=" * 100)
    print(f"Count: {len(dependencies)} files\n")
    for f in sorted(dependencies):
        print(f"  🔗 {f}")
    
    print(f"\n{'=' * 100}")
    print("🔧 CATEGORY 3: UTILITY/TEST FILES")
    print("=" * 100)
    print(f"Count: {len(utility_files)} files\n")
    for f in sorted(utility_files):
        print(f"  🔧 {f}")
    
    print(f"\n{'=' * 100}")
    print("🤖 CATEGORY 4: STANDALONE BOTS (Not used by MASTER_ORCHESTRATOR)")
    print("=" * 100)
    print(f"Count: {len(bot_files)} files\n")
    for f in sorted(bot_files):
        print(f"  🤖 {f}")
    
    print(f"\n{'=' * 100}")
    print("📁 CATEGORY 5: OTHER/ALTERNATIVE FILES")
    print("=" * 100)
    print(f"Count: {len(other_files)} files\n")
    for f in sorted(other_files):
        print(f"  📁 {f}")
    
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print("=" * 100)
    print(f"Core Systems (MASTER_ORCHESTRATOR):  {len(core_systems):3d} files  (Critical)")
    print(f"Dependencies (imported):             {len(dependencies):3d} files  (Important)")
    print(f"Utility/Test:                        {len(utility_files):3d} files  (Optional)")
    print(f"Standalone Bots:                     {len(bot_files):3d} files  (Optional)")
    print(f"Other/Alternative:                   {len(other_files):3d} files  (Optional)")
    print(f"{'-' * 100}")
    print(f"TOTAL:                               {len(all_py_files):3d} files")
    
    critical_count = len(core_systems) + len(dependencies)
    optional_count = len(utility_files) + len(bot_files) + len(other_files)
    
    print(f"\n🎯 CRITICAL for MASTER_ORCHESTRATOR: {critical_count} files")
    print(f"📦 OPTIONAL (nice to have):          {optional_count} files")
    
    return core_systems, dependencies, utility_files, bot_files, other_files

if __name__ == "__main__":
    core, deps, util, bots, other = main()
    
    print("\n" + "=" * 100)
    print("CONCLUSION")
    print("=" * 100)
    print(f"✅ {len(core)} core systems + {len(deps)} dependencies = {len(core)+len(deps)} files MUST work")
    print(f"⚠️  {len(util)+len(bots)+len(other)} other files are optional")
    print("=" * 100)
