#!/usr/bin/env python3
"""Check all additional files mentioned by user"""

from pathlib import Path

# Files mentioned
files = [
    'critical_features_addon.py', 'cross_examiner.py', 'crypto_cctx.py',
    'crypto_diag.py', 'crypto_trader.py', 'dashboard_app.py',
    'dashboard_reporter.py', 'data_sources.py', 'debug_imports.py',
    'debug_imports_run.py', 'brain_loop.py', 'build_history.py',
    'build_universe.py', 'bybit_adapter.py', 'bybit_bot.py',
    'bybit_smoke.py', 'calendar_ingest.py', 'awareness.py',
    'backtest_fx.py', 'bandit_report.py', 'calendar_utils.py',
    'charting.py', 'cmd_reader.py', 'acct_portfolio.py',
    'alpha_engines.py', 'analyzer.py', 'BALANCE_CHECKER.py'
]

# Directories mentioned
directories = [
    'allocators', 'analytics', 'brokers', 'cli', 'core', 'data',
    'execution', 'features', 'integrations', 'research', 'risk',
    'scanners', 'signals', 'strategies', 'tools', 'utils'
]

print("=" * 80)
print("CHECKING ADDITIONAL FILES")
print("=" * 80)

existing_files = []
missing_files = []

for f in files:
    matches = list(Path('.').rglob(f))
    if matches:
        existing_files.append((f, matches[0]))
        print(f"✅ {f} (at {matches[0]})")
    else:
        missing_files.append(f)
        print(f"❌ {f} - NOT FOUND")

print("\n" + "=" * 80)
print(f"FILES: {len(existing_files)}/{len(files)} found")
print("=" * 80)

print("\n" + "=" * 80)
print("CHECKING DIRECTORIES")
print("=" * 80)

existing_dirs = []
for d in directories:
    if Path(d).is_dir():
        # Count .py files
        py_files = list(Path(d).rglob('*.py'))
        existing_dirs.append((d, len(py_files)))
        print(f"✅ {d}/ ({len(py_files)} Python files)")
    else:
        print(f"❌ {d}/ - NOT FOUND")

print("\n" + "=" * 80)
print(f"DIRECTORIES: {len(existing_dirs)}/{len(directories)} found")
print("=" * 80)

# Check if already integrated
print("\n" + "=" * 80)
print("CHECKING INTEGRATION STATUS")
print("=" * 80)

orchestrator = Path('COMPLETE_ULTIMATE_ORCHESTRATOR.py')
if orchestrator.exists():
    with open(orchestrator, 'r') as f:
        content = f.read()
    
    critical_files = [
        'awareness.py',
        'data_sources.py',
        'bybit_adapter.py',
        'alpha_engines.py',
        'analyzer.py'
    ]
    
    for f in critical_files:
        if f in [x[0] for x in existing_files]:
            module = f.replace('.py', '')
            if f'import {module}' in content or f'from {module}' in content:
                print(f"✅ {f} - INTEGRATED")
            else:
                print(f"⚠️  {f} - NOT INTEGRATED")

