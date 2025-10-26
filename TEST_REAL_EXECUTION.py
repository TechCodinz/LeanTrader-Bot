#!/usr/bin/env python3
"""
Test REAL_PROFIT_BOT execution directly to see what's failing
"""
import os
os.environ['TRADING_MODE'] = 'live'
os.environ['ENABLE_LIVE'] = 'true'
os.environ['ALLOW_LIVE'] = 'true'

from REAL_PROFIT_BOT import REAL_PROFIT_BOT

print("\n🔥 Testing REAL_PROFIT_BOT execution...")
print("="*80)

try:
    bot = REAL_PROFIT_BOT()
    print(f"✅ Bot created")
    print(f"   Exchange: {bot.gate.name}")
    print(f"   Pairs: {bot.crypto_pairs}")
    
    # Check balance
    print(f"\n💰 Checking Gate.io balance...")
    try:
        balance = bot.check_gate_balance()
        print(f"   ✅ Balance: ${balance:.2f}")
        
        if balance < 10:
            print(f"   ⚠️  LOW BALANCE! Need at least $10 to trade")
            print(f"   Current balance: ${balance:.2f}")
    except Exception as e:
        print(f"   ❌ Balance check failed: {e}")
        balance = 0
    
    # Try to analyze first pair
    if bot.crypto_pairs:
        symbol = bot.crypto_pairs[0]
        print(f"\n🔍 Analyzing {symbol}...")
        
        signal, confidence, price = bot.analyze_market(symbol)
        print(f"   Signal: {signal}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Price: ${price:.4f}")
        
        if signal in ['BUY', 'SELL'] and confidence >= 0.70:
            print(f"\n⚡ HIGH CONFIDENCE SIGNAL!")
            print(f"   Would execute: {signal} {symbol} @ ${price:.4f}")
            
            # Calculate required balance
            position_size = bot.position_sizes.get(symbol, 0.01)
            required = price * position_size * 1.2
            print(f"   Position size: {position_size}")
            print(f"   Required balance: ${required:.2f}")
            print(f"   Current balance: ${balance:.2f}")
            
            if balance >= required:
                print(f"   ✅ SUFFICIENT BALANCE - WOULD EXECUTE!")
                print(f"\n💰 Calling execute_trade()...")
                
                try:
                    result = bot.execute_trade(symbol, signal, price)
                    if result:
                        print(f"   ✅ ORDER PLACED!")
                        print(f"   Result: {result}")
                    else:
                        print(f"   ❌ execute_trade returned None")
                except Exception as e:
                    print(f"   ❌ execute_trade failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ❌ INSUFFICIENT BALANCE")
                print(f"      Need: ${required:.2f}")
                print(f"      Have: ${balance:.2f}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)

