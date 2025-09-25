#!/usr/bin/env python3
import ccxt

# Check actual Gate.io balance
gate_config = {
    'apiKey': 'a0508d8aadf3bcb76e16f4373e1f3a76',
    'secret': '451770a07dbede1b87bb92f5ce98e24029d2fe91e0053be2ec41771c953113f9',
    'sandbox': False,
    'enableRateLimit': True
}

gate = ccxt.gate(gate_config)

try:
    balance = gate.fetch_balance()
    usdt_balance = balance['USDT']['free']
    total_balance = balance['USDT']['total']
    
    print(f"💰 USDT Free Balance: {usdt_balance}")
    print(f"💰 USDT Total Balance: {total_balance}")
    
    # Calculate safe position sizes (use 80% of available balance)
    safe_balance = float(usdt_balance) * 0.8
    
    print(f"\n🎯 RECOMMENDED POSITION SIZES FOR ${safe_balance:.2f} SAFE BALANCE:")
    print(f"BTC: {safe_balance * 0.4 / 43000:.6f} BTC (~${safe_balance * 0.4:.2f})")
    print(f"ETH: {safe_balance * 0.3 / 2500:.6f} ETH (~${safe_balance * 0.3:.2f})")
    print(f"BNB: {safe_balance * 0.2 / 150:.4f} BNB (~${safe_balance * 0.2:.2f})")
    print(f"SOL: {safe_balance * 0.1 / 10:.4f} SOL (~${safe_balance * 0.1:.2f})")
    
except Exception as e:
    print(f"❌ Balance check error: {e}")