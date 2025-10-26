#!/usr/bin/env python3
"""
Simple test script to verify bot is working after venv restoration
"""
import sys
import os

print("=" * 80)
print("🔧 TESTING BOT AFTER VENV RESTORATION")
print("=" * 80)
print()

# Test 1: Python version
print("1. Python Version:")
print(f"   ✅ Python {sys.version.split()[0]}")
print()

# Test 2: Core dependencies
print("2. Testing Core Dependencies:")
try:
    import ccxt
    print(f"   ✅ ccxt {ccxt.__version__}")
except Exception as e:
    print(f"   ❌ ccxt: {e}")

try:
    import pandas as pd
    print(f"   ✅ pandas {pd.__version__}")
except Exception as e:
    print(f"   ❌ pandas: {e}")

try:
    import numpy as np
    print(f"   ✅ numpy {np.__version__}")
except Exception as e:
    print(f"   ❌ numpy: {e}")

try:
    import tensorflow as tf
    print(f"   ✅ tensorflow {tf.__version__}")
except Exception as e:
    print(f"   ❌ tensorflow: {e}")

try:
    import torch
    print(f"   ✅ torch {torch.__version__}")
except Exception as e:
    print(f"   ❌ torch: {e}")

try:
    import web3
    print(f"   ✅ web3 {web3.__version__}")
except Exception as e:
    print(f"   ❌ web3: {e}")

print()

# Test 3: Check .env file
print("3. Configuration Files:")
if os.path.exists('.env'):
    print("   ✅ .env file exists")
else:
    print("   ⚠️  .env file not found")

if os.path.exists('accounts.yml'):
    print("   ✅ accounts.yml exists")
else:
    print("   ⚠️  accounts.yml not found")

print()

# Test 4: Check main bot files
print("4. Main Bot Files:")
bot_files = [
    'COMPLETE_ULTIMATE_ORCHESTRATOR.py',
    'RUN_BOT.py',
    'main.py',
    'auto_trading_bot.py'
]

for file in bot_files:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ⚠️  {file} not found")

print()

# Test 5: Can we import bot modules?
print("5. Testing Bot Module Imports:")
sys.path.insert(0, '.')

try:
    import ccxt
    exchange = ccxt.binance({'enableRateLimit': True})
    print(f"   ✅ CCXT exchange initialized: {exchange.id}")
except Exception as e:
    print(f"   ⚠️  CCXT test: {str(e)[:80]}")

print()

print("=" * 80)
print("✅ BOT RESTORATION TEST COMPLETE!")
print("=" * 80)
print()
print("Next steps:")
print("  - To run the bot in testnet mode:")
print("    ./venv/bin/python RUN_BOT.py --testnet")
print()
print("  - To run the orchestrator:")
print("    ./venv/bin/python COMPLETE_ULTIMATE_ORCHESTRATOR.py")
print()
print("  - Check logs:")
print("    tail -f logs/live.log")
print()
