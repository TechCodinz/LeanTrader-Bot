#!/usr/bin/env python3
"""
Check if we can do internal transfer via API
If yes, automate the transfer from Funding to Unified account
"""

import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

# Setup Bybit testnet
bybit = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_SECRET'),
    'enableRateLimit': True,
})
bybit.set_sandbox_mode(True)

print("🔍 Checking Bybit testnet account structure...")
print("")

# Check all account balances
try:
    print("💰 Funding Account Balance:")
    funding = bybit.fetch_balance({'type': 'funding'})
    print(f"   USDT: {funding.get('USDT', {}).get('free', 0)}")
except Exception as e:
    print(f"   Error: {e}")

print("")

try:
    print("💰 Unified Account Balance:")
    unified = bybit.fetch_balance({'type': 'unified'})
    print(f"   USDT: {unified.get('USDT', {}).get('free', 0)}")
except Exception as e:
    print(f"   Error: {e}")

print("")

try:
    print("💰 Spot Account Balance:")
    spot = bybit.fetch_balance({'type': 'spot'})
    print(f"   USDT: {spot.get('USDT', {}).get('free', 0)}")
except Exception as e:
    print(f"   Error: {e}")

print("")
print("═══════════════════════════════════════════════════════")
print("🔄 ATTEMPTING INTERNAL TRANSFER")
print("═══════════════════════════════════════════════════════")
print("")

# Try to transfer from funding to unified
try:
    # Bybit internal transfer API
    # https://bybit-exchange.github.io/docs/v5/asset/create-inter-transfer
    
    print("Attempting to transfer 17055 USDT from FUNDING to UNIFIED...")
    
    transfer_result = bybit.transfer(
        code='USDT',
        amount=17055,
        fromAccount='funding',
        toAccount='unified'
    )
    
    print("✅ TRANSFER SUCCESSFUL!")
    print(f"   Result: {transfer_result}")
    print("")
    print("🎉 Your $17,055 is now in Unified Account!")
    print("   The bot should start trading immediately!")
    
except Exception as e:
    print(f"❌ Transfer failed: {e}")
    print("")
    print("📋 MANUAL TRANSFER REQUIRED")
    print("")
    print("Please transfer funds manually:")
    print("1. Go to: https://testnet.bybit.com/user/assets/home")
    print("2. Click 'Transfer'")
    print("3. From: Funding Account → To: Unified Trading Account")
    print("4. Amount: 17055 USDT")
    print("5. Confirm")
    print("")
    print("See BYBIT_MANUAL_TRANSFER_GUIDE.md for detailed steps!")
