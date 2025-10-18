#!/usr/bin/env python3
"""
Diagnose WHY Bybit says "Insufficient balance" when we have $17k
"""

import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

print("═══════════════════════════════════════════════════════")
print("🔍 BYBIT TESTNET DIAGNOSTIC")
print("═══════════════════════════════════════════════════════")
print("")

# Setup Bybit testnet
bybit = ccxt.bybit({
    'apiKey': os.getenv('BYBIT_API_KEY'),
    'secret': os.getenv('BYBIT_SECRET'),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
    }
})
bybit.set_sandbox_mode(True)

print("1️⃣ Checking ALL account types...")
print("")

# Check all possible account types
for account_type in ['spot', 'unified', 'contract', 'funding']:
    try:
        balance = bybit.fetch_balance({'type': account_type})
        usdt = balance.get('USDT', {}).get('free', 0)
        if usdt > 0:
            print(f"   ✅ {account_type.upper()}: {usdt} USDT")
        else:
            print(f"   ⚪ {account_type.upper()}: 0 USDT")
    except Exception as e:
        print(f"   ❌ {account_type.upper()}: Error - {str(e)[:80]}")

print("")
print("2️⃣ Checking what 'defaultType' the exchange is using...")
print(f"   Current defaultType: {bybit.options.get('defaultType', 'not set')}")
print("")

print("3️⃣ Testing a SMALL order to see the exact error...")
print("")

# Try to place a tiny test order
try:
    print("   Attempting: BUY 0.0001 BTC (about $6.50)")
    
    # Get current price first
    ticker = bybit.fetch_ticker('BTC/USDT')
    price = ticker['last']
    print(f"   BTC price: ${price:.2f}")
    
    # Try market buy with explicit params
    test_order = bybit.create_market_buy_order(
        'BTC/USDT',
        0.0001,  # Tiny amount
        params={
            'accountType': 'UNIFIED',  # Try this
        }
    )
    
    print("   ✅ ORDER SUCCEEDED!")
    print(f"   Order ID: {test_order.get('id')}")
    print("")
    print("🎉 THE BOT SHOULD WORK NOW!")
    
except Exception as e:
    error_msg = str(e)
    print(f"   ❌ Order failed: {error_msg}")
    print("")
    
    if "170131" in error_msg or "Insufficient balance" in error_msg:
        print("💡 DIAGNOSIS:")
        print("")
        print("   The issue is: Bybit API is not recognizing the account parameter!")
        print("")
        print("   Possible fixes:")
        print("   1. Use 'category' instead of 'accountType'")
        print("   2. Set account at exchange level, not per-order")
        print("   3. Use different API method for unified account")
        print("")
        
        # Try alternative parameters
        print("4️⃣ Trying alternative parameters...")
        print("")
        
        for param_name in ['category', 'type', 'accountCategory']:
            try:
                print(f"   Testing with {param_name}='spot'...")
                test = bybit.create_market_buy_order(
                    'BTC/USDT',
                    0.0001,
                    params={param_name: 'spot'}
                )
                print(f"   ✅ SUCCESS with {param_name}='spot'!")
                print("")
                print(f"🔥 THE FIX: Use params={{'{param_name}': 'spot'}}")
                break
            except Exception as e2:
                print(f"   ❌ Failed: {str(e2)[:80]}")

print("")
print("═══════════════════════════════════════════════════════")
print("📊 SUMMARY")
print("═══════════════════════════════════════════════════════")
print("")
print("Run this diagnostic to find the exact fix needed!")
