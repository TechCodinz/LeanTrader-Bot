#!/usr/bin/env python3
"""
ULTRA DEEP SCAN - Find EVERYTHING not integrated
Including routers, managers, engines, systems, etc.
"""

import os
from pathlib import Path

workspace = Path('/workspace')

# Load orchestrator
orch_file = workspace / 'COMPLETE_ULTIMATE_ORCHESTRATOR.py'
with open(orch_file, 'r') as f:
    orch_content = f.read()

# Key system files to check
key_systems = {
    'router.py': 'Main routing system',
    'market_router.py': 'Market routing',
    'dex_router.py': 'DEX routing',
    'exchange_manager.py': 'Exchange management',
    'risk_engine.py': 'Risk management engine',
    'risk_guard.py': 'Risk guard system',
    'ml_strategy_engine.py': 'ML strategy engine',
    'analyzer.py': 'Market analyzer',
    'OMNISCIENT_EXECUTION_ENGINE.py': 'Omniscient execution',
    'OMNISCIENT_TRADING_MODE.py': 'Omniscient trading mode',
    'PREMIUM_VIP_TELEGRAM_SYSTEM.py': 'Premium VIP Telegram',
    'nobel_complete_system.py': 'Nobel complete system',
    'nobel_hedge_fund_system.py': 'Nobel hedge fund',
    'nobel_risk_management.py': 'Nobel risk management',
    'nobel_simple_system.py': 'Nobel simple system',
    'unified_trading_system.py': 'Unified trading system',
    'november_growth_strategy.py': 'November growth strategy',
    'INTEGRATE_EVERYTHING_CLEAN.py': 'Integration system',
}

print("=" * 80)
print("ULTRA DEEP SCAN - ALL SYSTEMS")
print("=" * 80)
print()

missing = []
integrated = []

for filename, description in sorted(key_systems.items()):
    filepath = workspace / filename
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        module_name = filename.replace('.py', '')
        
        # Check if imported
        is_imported = (
            f"from {module_name} import" in orch_content or 
            f"import {module_name}" in orch_content
        )
        
        if is_imported:
            integrated.append((filename, description, size_kb))
            print(f"✅ {filename:<45} ({size_kb:>6.1f} KB) INTEGRATED")
        else:
            missing.append((filename, description, size_kb))
            print(f"❌ {filename:<45} ({size_kb:>6.1f} KB) NOT INTEGRATED!")
            print(f"   → {description}")
    else:
        print(f"⚪ {filename:<45} NOT FOUND")

print()
print("=" * 80)
print(f"INTEGRATED: {len(integrated)}/{len(key_systems)}")
print(f"MISSING:    {len(missing)}/{len(key_systems)}")
print("=" * 80)

if missing:
    print()
    print("CRITICAL MISSING SYSTEMS:")
    print("-" * 80)
    total_kb = 0
    for filename, description, size_kb in missing:
        print(f"  {filename:<40} {size_kb:>6.1f} KB - {description}")
        total_kb += size_kb
    print("-" * 80)
    print(f"  TOTAL MISSING CODE: {total_kb:.1f} KB")
    print()

