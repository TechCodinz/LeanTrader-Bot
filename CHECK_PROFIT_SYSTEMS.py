#!/usr/bin/env python3
"""
Check for steady profit engines and advanced trading features
"""

import os
from pathlib import Path

workspace = Path('/workspace')
orch_file = workspace / 'COMPLETE_ULTIMATE_ORCHESTRATOR.py'
with open(orch_file, 'r') as f:
    orch_content = f.read()

# Profit and trading systems to check
profit_systems = {
    'SESSION_AWARE_TRADING.py': 'Session-aware trading',
    'REAL_PROFIT_BOT.py': 'Real profit bot',
    'SMART_SCALPING_ENGINE.py': 'Smart scalping',
    'FINAL_PROFIT_OPTIMIZATION.py': 'Profit optimization',
    'critical_features_addon.py': 'Critical features (sizing, margin)',
    'guardrails.py': 'Guardrails & limits',
    'allocators/sizing.py': 'Position sizing',
    'allocators/portfolio.py': 'Portfolio allocation',
    'execution_adv.py': 'Advanced execution',
    'futures_signals.py': 'Futures signals',
}

print("=" * 80)
print("PROFIT ENGINE & ADVANCED TRADING FEATURES CHECK")
print("=" * 80)
print()

missing = []
integrated = []

for filename, description in sorted(profit_systems.items()):
    filepath = workspace / filename
    if filepath.exists():
        size_kb = filepath.stat().st_size / 1024
        module_name = filename.replace('.py', '').replace('/', '.')
        
        # Check multiple patterns
        is_imported = any([
            f"from {filename.replace('.py', '')} import" in orch_content,
            f"import {filename.replace('.py', '')}" in orch_content,
            filename.replace('.py', '').replace('/', '.') in orch_content,
        ])
        
        if is_imported:
            integrated.append((filename, description, size_kb))
            print(f"✅ {filename:<40} ({size_kb:>6.1f} KB) INTEGRATED")
        else:
            missing.append((filename, description, size_kb))
            print(f"❌ {filename:<40} ({size_kb:>6.1f} KB) NOT INTEGRATED!")
            print(f"   → {description}")
    else:
        print(f"⚪ {filename:<40} NOT FOUND")

print()
print("=" * 80)
print(f"INTEGRATED: {len(integrated)}/{len([f for f in profit_systems.keys() if (workspace/f).exists()])}")
print(f"MISSING:    {len(missing)}/{len([f for f in profit_systems.keys() if (workspace/f).exists()])}")
print("=" * 80)

if missing:
    print()
    print("MISSING PROFIT SYSTEMS:")
    for filename, description, size_kb in missing:
        print(f"  {filename:<40} {size_kb:>6.1f} KB - {description}")
    print()

