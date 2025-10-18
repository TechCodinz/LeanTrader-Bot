#!/usr/bin/env python3
"""Check if user-mentioned files are integrated"""

import sys
from pathlib import Path

# Files mentioned by user
mentioned_files = [
    'brain.py',
    'ultra_launcher_advanced.py', 
    'ultra_ml_pipeline.py',
    'ultra_multi_platform_scanner.py',
    'ultra_backtest_engine.py',
    'skillbook.py',
    'sizer.py',
    'research.py',
    'research_optuna.py',
    'pattern_memory.py',
    'research_optuna_walk.py',
    'risk.py',
    'risk_engine.py',
    'regime.py',
    'guardrails.py',
    'import_check.py',
    'indicators.py',
    'ledger.py',
    'risk_guard.py',
]

print("=" * 80)
print("CHECKING USER-MENTIONED FILES")
print("=" * 80)

# Check which exist
existing = []
missing = []

for f in mentioned_files:
    path = Path(f)
    # Also check in subdirectories
    matches = list(Path('.').rglob(f))
    
    if path.exists() or matches:
        existing.append(f)
        if matches:
            print(f"✅ {f} (found at {matches[0]})")
        else:
            print(f"✅ {f}")
    else:
        missing.append(f)
        print(f"❌ {f} - NOT FOUND")

print("\n" + "=" * 80)
print(f"EXISTING: {len(existing)}/{len(mentioned_files)}")
print(f"MISSING: {len(missing)}/{len(mentioned_files)}")
print("=" * 80)

if existing:
    print("\n📁 EXISTING FILES:")
    for f in existing:
        print(f"  • {f}")

if missing:
    print("\n❌ MISSING FILES:")
    for f in missing:
        print(f"  • {f}")

# Now check if existing ones are imported in orchestrator
print("\n" + "=" * 80)
print("CHECKING INTEGRATION IN ORCHESTRATOR")
print("=" * 80)

orchestrator_file = 'COMPLETE_UNIFIED_ORCHESTRATOR.py'

if Path(orchestrator_file).exists():
    with open(orchestrator_file, 'r') as f:
        content = f.read()
    
    integrated = []
    not_integrated = []
    
    for file in existing:
        module = file.replace('.py', '')
        
        # Check if imported
        if f'from {module} import' in content or f'import {module}' in content:
            integrated.append(file)
            print(f"✅ {file} - INTEGRATED")
        else:
            not_integrated.append(file)
            print(f"⚠️  {file} - NOT INTEGRATED in orchestrator")
    
    print("\n" + "=" * 80)
    print(f"INTEGRATED: {len(integrated)}/{len(existing)}")
    print(f"NOT INTEGRATED: {len(not_integrated)}/{len(existing)}")
    print("=" * 80)
    
    if not_integrated:
        print("\n⚠️  FILES THAT NEED INTEGRATION:")
        for f in not_integrated:
            print(f"  • {f}")

