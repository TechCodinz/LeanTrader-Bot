#!/usr/bin/env python3
"""
Activate MICRO_WALLET_GROWER to trade with $1.44 balance
Show ACTUAL trades, positions, profits
"""
import os
os.environ['TRADING_MODE'] = 'live'
os.environ['ENABLE_LIVE'] = 'true'
os.environ['ALLOW_LIVE'] = 'true'

from MICRO_TRADING_BOT import MICRO_GATE_BOT

print("\n" + "="*80)
print("💎 ACTIVATING MICRO WALLET GROWER")
print("="*80)
print("\n✅ This engine trades from $1 to INFINITE!")
print("✅ Perfect for your $1.44 Gate.io balance\n")

try:
    bot = MICRO_GATE_BOT()
    print(f"✅ Bot initialized")
    print(f"   Exchange: {bot.gate.name}")
    print(f"   Pairs: {bot.crypto_pairs}")
    
    # Check balance
    print(f"\n💰 Checking balance...")
    balance = bot.check_gate_balance()
    print(f"   ✅ Balance: ${balance:.2f}")
    
    if balance < 1:
        print(f"   ❌ Need at least $1 to start")
    else:
        print(f"   ✅ SUFFICIENT! Can trade with ${balance:.2f}\n")
        print(f"💎 MICRO positions (designed for small balances):")
        for symbol, size in list(bot.position_sizes.items())[:5]:
            price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 150 if 'SOL' in symbol else 50
            required = price * size
            print(f"      {symbol}: {size} (~${required:.2f})")
        
        # Try ONE trade cycle
        print(f"\n🚀 Running ONE trading cycle...")
        print(f"="*80 + "\n")
        
        for symbol in bot.crypto_pairs[:3]:  # Test first 3 pairs
            print(f"📊 Analyzing {symbol}...")
            
            try:
                action, confidence, price, sl, tp = bot.analyze_market(symbol)
                print(f"   Signal: {action}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Price: ${price:.4f}")
                
                if action in ['BUY', 'SELL'] and confidence >= 0.70:
                    print(f"\n   ⚡ HIGH CONFIDENCE! Attempting execution...")
                    
                    result = bot.execute_trade(symbol, action, price, sl, tp)
                    
                    if result:
                        print(f"   ✅ ORDER PLACED!")
                        print(f"      {result}")
                    else:
                        print(f"   ⚠️  Not executed (insufficient balance or error)")
                else:
                    print(f"   ⏸️  Low confidence ({confidence:.1%}) - skipping")
                    
                print()
                
            except Exception as e:
                print(f"   ❌ Error: {e}\n")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*80 + "\n")

