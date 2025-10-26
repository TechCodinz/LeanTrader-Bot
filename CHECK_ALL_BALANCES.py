#!/usr/bin/env python3
"""
Check balances on ALL exchanges to see where funds are
"""
import ccxt
import os

api_keys = {
    'bybit_live': {
        'apiKey': 'mMHs7rDC72TvHs4oQG',
        'secret': 'NwTa6UOgczdmZI2Kn2WBcFfh5r6VkVGnvGEI'
    }
}

print("\n💰 CHECKING ALL EXCHANGE BALANCES")
print("="*80 + "\n")

# Check Bybit LIVE
try:
    print("Checking Bybit LIVE...")
    bybit = ccxt.bybit({
        'apiKey': api_keys['bybit_live']['apiKey'],
        'secret': api_keys['bybit_live']['secret'],
        'enableRateLimit': True
    })
    
    balance = bybit.fetch_balance()
    total = balance.get('total', {})
    
    # Show significant balances
    print(f"   Bybit LIVE balances:")
    for currency, amount in total.items():
        if amount and amount > 0.01:
            print(f"      {currency}: {amount}")
    
    # Check USDT specifically
    usdt = total.get('USDT', 0)
    print(f"\n   💰 Bybit USDT: ${usdt:.2f}")
    
except Exception as e:
    print(f"   ❌ Bybit error: {e}")

# Check Gate.io (if keys exist)
try:
    from REAL_PROFIT_BOT import REAL_PROFIT_BOT
    bot = REAL_PROFIT_BOT()
    balance = bot.check_gate_balance()
    print(f"\n   💰 Gate.io USDT: ${balance:.2f}")
except Exception as e:
    print(f"\n   ❌ Gate.io error: {e}")

print("\n" + "="*80)
print("\n💎 TOTAL AVAILABLE FOR TRADING:")
print(f"   Bybit: Check above")
print(f"   Gate.io: $1.44")
print("\n" + "="*80)

