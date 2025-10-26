#!/usr/bin/env python3
"""
Test if REAL_PROFIT_BOT actually executes trades
"""
import os
os.environ['TRADING_MODE'] = 'live'
os.environ['ENABLE_LIVE'] = 'true'
os.environ['ALLOW_LIVE'] = 'true'

from REAL_PROFIT_BOT import REAL_PROFIT_BOT

print("\n🔥 Testing REAL_PROFIT_BOT execution...")
print("="*80 + "\n")

try:
    bot = REAL_PROFIT_BOT()
    print(f"✅ Bot initialized")
    print(f"✅ Exchange: {bot.gate.name if hasattr(bot, 'gate') else 'Unknown'}")
    print(f"✅ Trading pairs: {len(bot.pairs)}")
    
    # Check balance
    try:
        balance = bot.check_gate_balance()
        print(f"✅ Balance: ${balance:.2f}")
    except Exception as e:
        print(f"⚠️  Balance check: {e}")
    
    # Try one analysis
    if bot.pairs:
        symbol = bot.pairs[0]
        print(f"\n🔍 Analyzing {symbol}...")
        
        signal, confidence, price = bot.analyze_market(symbol)
        print(f"   Signal: {signal}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Price: ${price:.4f}")
        
        if signal in ['BUY', 'SELL'] and confidence >= 0.70:
            print(f"\n⚡ Would execute: {signal} {symbol} @ ${price:.4f}")
            print(f"   (Not executing to avoid accidental trade)")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)

