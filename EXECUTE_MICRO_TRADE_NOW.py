#!/usr/bin/env python3
"""
Execute ACTUAL trades with MICRO_WALLET_GROWER using correct parameters
"""
import os
os.environ['TRADING_MODE'] = 'live'
os.environ['ENABLE_LIVE'] = 'true'

from MICRO_TRADING_BOT import MICRO_GATE_BOT

print("\n" + "="*80)
print("💎 EXECUTING REAL TRADES WITH $1.44 BALANCE")
print("="*80 + "\n")

bot = MICRO_GATE_BOT()

# Check balance
balance = bot.check_gate_balance()
print(f"✅ Balance: ${balance:.2f}\n")

# Execute one full trading cycle
print("🚀 EXECUTING TRADING CYCLE...")
print("="*80 + "\n")

for i in range(10):  # Try 10 trading cycles
    print(f"Cycle {i+1}:")
    
    for symbol in bot.crypto_pairs:
        try:
            # Analyze
            action, confidence, price, sl, tp = bot.analyze_market(symbol)
            confidence_pct = confidence / 100  # Fix percentage
            
            print(f"   {symbol}: {action} @ ${price:.4f} ({confidence_pct:.1f}% conf)")
            
            if action in ['BUY', 'SELL'] and confidence_pct >= 70:
                print(f"      ⚡ HIGH CONFIDENCE! Executing...")
                
                # Execute with correct parameters (symbol, action, price)
                result = bot.execute_trade(symbol, action, price)
                
                if result:
                    print(f"      ✅ ORDER PLACED!")
                    print(f"         Order ID: {result.get('id', 'N/A')}")
                    print(f"         Amount: {result.get('amount', 0)}")
                    print(f"         Status: {result.get('status', 'unknown')}")
                else:
                    print(f"      ⚠️  No execution (insufficient balance?)")
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)[:100]}")
    
    print()
    
    # Check new balance
    new_balance = bot.check_gate_balance()
    profit = new_balance - balance
    print(f"   💰 Balance: ${new_balance:.2f} (P&L: ${profit:+.2f})\n")
    
    if profit > 0:
        print(f"   🎉 PROFIT MADE: ${profit:.2f}!")
        break
    
    import time
    time.sleep(5)  # Wait 5 seconds between cycles

print("="*80)

