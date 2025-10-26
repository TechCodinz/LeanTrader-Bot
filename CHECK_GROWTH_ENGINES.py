#!/usr/bin/env python3
"""Check for growth and trading engines"""

import os
from pathlib import Path

workspace = Path('/workspace')
orch_file = workspace / 'COMPLETE_ULTIMATE_ORCHESTRATOR.py'
with open(orch_file, 'r') as f:
    orch_content = f.read()

# Growth and trading engines to check
engines = {
    'MICRO_TRADING_BOT.py': 'Micro wallet grower ($1 to infinite)',
    'ULTRA_GOLDMINE_FEATURES.py': 'Ultra goldmine (18 rare strategies)',
    'DYNAMIC_MARKET_SCANNER.py': 'Dynamic market scanner (all patterns)',
    'continuous_ultra_bot.py': 'Continuous ultra bot',
    'ultra_continuous_trading.py': 'Ultra continuous trading (already checked)',
    'ultra_multi_platform_scanner.py': 'Multi-platform scanner (already checked)',
    'ultra_forex_master.py': 'Ultra Forex Master (already checked)',
}

print("=" * 80)
print("GROWTH & CONTINUOUS TRADING ENGINES CHECK")
print("=" * 80)
print()

missing = []
integrated = []
found_files = []

for filename, description in sorted(engines.items()):
    filepath = workspace / filename
    if filepath.exists():
        found_files.append(filename)
        size_kb = filepath.stat().st_size / 1024
        module_name = filename.replace('.py', '')
        
        # Check multiple patterns
        is_imported = any([
            f"from {module_name} import" in orch_content,
            f"import {module_name}" in orch_content,
        ])
        
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
print(f"INTEGRATED: {len(integrated)}/{len(found_files)}")
print(f"MISSING:    {len(missing)}/{len(found_files)}")
print("=" * 80)

if missing:
    print()
    print("MISSING GROWTH & TRADING ENGINES:")
    for filename, description, size_kb in missing:
        print(f"  {filename:<45} {size_kb:>6.1f} KB")
        print(f"    → {description}")
    print()

# Additional checks for specific features
print()
print("SPECIFIC FEATURE CHECKS:")
print("-" * 80)

features = {
    'Micro wallet growth': 'MICRO_TRADING_BOT' in orch_content or 'MICRO_GATE_BOT' in orch_content,
    'Gold trading': 'gold' in orch_content.lower() or 'xau' in orch_content.lower(),
    'Continuous trading': 'ultra_continuous_trading' in orch_content,
    'Dynamic scanner': 'DYNAMIC_MARKET_SCANNER' in orch_content or 'DynamicMarketScanner' in orch_content,
    'Ultra goldmine': 'ULTRA_GOLDMINE_FEATURES' in orch_content or 'UltraGoldmineManager' in orch_content,
    'Forex/TradFi': 'ultra_forex_master' in orch_content,
    'Multi-platform scan': 'ultra_multi_platform_scanner' in orch_content,
}

for feature, present in features.items():
    print(f"  {feature:<30} {'✅' if present else '❌'}")

print()
print("=" * 80)

