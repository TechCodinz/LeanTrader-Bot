#!/usr/bin/env python3
"""
🚨 EMERGENCY: CLOSE ALL POSITIONS & RECOVER CAPITAL 🚨
"""
import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

print("🚨 EMERGENCY: Closing ALL Gate.io positions...")
print("="*60)

# Connect to Gate.io
gate = ccxt.gateio({
    'apiKey': os.getenv('GATEIO_API_KEY'),
    'secret': os.getenv('GATEIO_SECRET'),
    'options': {'defaultType': 'spot'}
})

try:
    # Get all balances
    balance = gate.fetch_balance()
    print(f"\n💰 Current USDT Balance: ${balance['USDT']['free']:.2f}")
    
    total_value = 0
    positions_closed = 0
    
    # Close ALL non-USDT positions
    for coin, amounts in balance['total'].items():
        if coin != 'USDT' and amounts > 0:
            try:
                symbol = f"{coin}/USDT"
                if symbol not in gate.markets:
                    print(f"⚠️  Skipping {symbol} (not in markets)")
                    continue
                
                # Get current price
                ticker = gate.fetch_ticker(symbol)
                current_price = ticker['last']
                position_value = amounts * current_price
                
                print(f"\n📊 {symbol}:")
                print(f"   Amount: {amounts:.8f} {coin}")
                print(f"   Price: ${current_price:.6f}")
                print(f"   Value: ${position_value:.2f}")
                
                # Close position COMPLETELY
                if position_value > 0.5:  # Only close if worth more than $0.50
                    print(f"   🔄 CLOSING FULL POSITION...")
                    
                    # Create market sell order for FULL amount
                    order = gate.create_market_sell_order(
                        symbol, 
                        amounts,  # FULL amount
                    )
                    
                    print(f"   ✅ CLOSED! Order ID: {order['id']}")
                    print(f"   💰 Recovered ~${position_value:.2f}")
                    
                    total_value += position_value
                    positions_closed += 1
                else:
                    print(f"   ⏭️  Too small to close (${position_value:.2f})")
                    
            except Exception as e:
                print(f"   ❌ Error closing {coin}: {e}")
    
    # Get final balance
    print("\n" + "="*60)
    final_balance = gate.fetch_balance()
    final_usdt = final_balance['USDT']['free']
    
    print(f"\n✅ EMERGENCY CLOSURE COMPLETE!")
    print(f"   Positions closed: {positions_closed}")
    print(f"   Value recovered: ${total_value:.2f}")
    print(f"   Final USDT balance: ${final_usdt:.2f}")
    
except Exception as e:
    print(f"\n❌ EMERGENCY ERROR: {e}")
    import traceback
    traceback.print_exc()
