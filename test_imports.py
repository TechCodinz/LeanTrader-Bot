#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

# Add workspace to path
sys.path.insert(0, '/workspace')
os.environ['PYTHONPATH'] = '/workspace'

print("Testing imports...")
print("="*50)

# Test risk imports
try:
    from risk.calendar_gates import is_high_impact_window
    print("✅ risk.calendar_gates imported")
except Exception as e:
    print(f"❌ risk.calendar_gates: {e}")

try:
    from risk.guards import GuardState, RiskLimits
    print("✅ risk.guards imported")
except Exception as e:
    print(f"❌ risk.guards: {e}")

# Test w3guard imports
try:
    from w3guard.guards import estimate_price_impact
    print("✅ w3guard.guards imported")
except Exception as e:
    print(f"❌ w3guard.guards: {e}")

# Test research imports
try:
    from research.evolution.ga_trader import run_ga
    print("✅ research.evolution.ga_trader imported")
except Exception as e:
    print(f"❌ research.evolution.ga_trader: {e}")

# Test main modules
modules = [
    "router",
    "trader_core",
    "risk_guard",
    "mt5_adapter",
    "mt5_signals",
    "bybit_adapter",
    "paper_broker",
    "tools.web_crawler"
]

print("\nTesting main modules...")
print("-"*50)

for module in modules:
    try:
        exec(f"import {module}")
        print(f"✅ {module} imported")
    except Exception as e:
        print(f"❌ {module}: {e}")

print("\n" + "="*50)
print("Import test complete!")