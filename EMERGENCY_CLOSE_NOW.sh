#!/bin/bash
# EMERGENCY: Close all Gate.io positions manually

echo "🚨 EMERGENCY POSITION CLOSER 🚨"
echo ""
echo "Go to Gate.io and MANUALLY close all positions:"
echo ""
echo "1. Login to https://www.gate.io/myaccount/mywallet"
echo "2. Go to Spot Wallet"
echo "3. For EACH coin (except USDT):"
echo "   - Click 'Trade'"
echo "   - Click 'Sell'"
echo "   - Enter 100% of balance"
echo "   - Click 'Market Order'"
echo "   - Confirm sale"
echo ""
echo "This will convert everything back to USDT!"
echo ""
echo "OR run this Python command:"
echo ""
cat << 'EOFPYTHON'
python3 << 'EOF'
import ccxt, os, sys
sys.path.insert(0, '/workspace')
from load_env import load_credentials
creds = load_credentials()
gate = ccxt.gateio({
    'apiKey': creds['GATEIO_API_KEY'],
    'secret': creds['GATEIO_SECRET_KEY'],
    'options': {'defaultType': 'spot'}
})
balance = gate.fetch_balance()
print(f"USDT: ${balance['USDT']['free']:.2f}")
for coin, amt in balance['total'].items():
    if coin != 'USDT' and amt > 0:
        try:
            symbol = f"{coin}/USDT"
            ticker = gate.fetch_ticker(symbol)
            value = amt * ticker['last']
            if value > 0.5:
                print(f"Closing {symbol}: {amt:.8f} (${value:.2f})")
                gate.create_market_sell_order(symbol, amt)
                print(f"  ✅ CLOSED!")
        except Exception as e:
            print(f"  ❌ {coin}: {e}")
final = gate.fetch_balance()
print(f"Final USDT: ${final['USDT']['free']:.2f}")
EOF
EOFPYTHON
