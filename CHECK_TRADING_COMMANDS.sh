#!/bin/bash
# CHECK IF BOT IS ACTUALLY TRADING
# Run these commands on VPS to verify trading activity

echo "================================================================================"
echo "🔍 CHECKING BOT TRADING ACTIVITY"
echo "================================================================================"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. CHECK BOT LOGS FOR TRADE ACTIVITY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Last 50 lines with trade-related activity:"
tail -100 /root/trading_bot/bot.log | grep -iE "trade|order|execute|buy|sell|position" | tail -20

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. CHECK FOR DATABASE FILES (Ledger)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ls -lh /root/trading_bot/*.db 2>/dev/null || echo "No database files yet"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. CHECK BOT STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

systemctl status trading-bot --no-pager | head -15

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. TEST EXCHANGE CONNECTION (Python)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 << 'PYEOF'
import ccxt
import os

# Test Bybit connection
try:
    bybit = ccxt.bybit({
        'apiKey': 'REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE',
        'secret': 'REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE',
        'options': {'defaultType': 'spot'}
    })
    bybit.set_sandbox_mode(True)  # Testnet

    balance = bybit.fetch_balance()
    print("✅ Bybit Testnet Connection: SUCCESS")
    print(f"   USDT Balance: {balance.get('USDT', {}).get('free', 0)}")

    # Check recent orders
    try:
        orders = bybit.fetch_orders(limit=10)
        print(f"   Recent orders: {len(orders)}")
        if orders:
            print("   ✅ BOT HAS PLACED ORDERS!")
            for order in orders[:3]:
                print(f"      - {order['symbol']}: {order['side']} {order['amount']}")
        else:
            print("   ⚠️  No orders yet")
    except Exception as e:
        print(f"   Orders check: {e}")

except Exception as e:
    print(f"❌ Bybit Error: {e}")

print()

# Test Gate.io connection
try:
    gate = ccxt.gateio({
        'apiKey': 'REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE',
        'secret': 'REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE'
    })

    balance = gate.fetch_balance()
    print("✅ Gate.io Testnet Connection: SUCCESS")
    print(f"   USDT Balance: {balance.get('USDT', {}).get('free', 0)}")

    # Check recent orders
    try:
        orders = gate.fetch_orders(limit=10)
        print(f"   Recent orders: {len(orders)}")
        if orders:
            print("   ✅ BOT HAS PLACED ORDERS!")
            for order in orders[:3]:
                print(f"      - {order['symbol']}: {order['side']} {order['amount']}")
        else:
            print("   ⚠️  No orders yet")
    except Exception as e:
        print(f"   Orders check: {e}")

except Exception as e:
    print(f"❌ Gate.io Error: {e}")

PYEOF

echo ""
echo "================================================================================"
echo "📊 SUMMARY"
echo "================================================================================"
echo ""

# Count trade mentions in last 100 lines
trade_count=$(tail -100 /root/trading_bot/bot.log | grep -ic "trade executed\|order placed")

if [ $trade_count -gt 0 ]; then
    echo "✅ Found $trade_count trade-related log entries"
    echo "   Bot is likely trading! Check exchange order history to confirm."
else
    echo "⚠️  No trade executions found in recent logs"
    echo "   Bot might not be generating signals yet (too early)"
    echo "   Or execution might not be working"
    echo "   Check again in 2-4 hours"
fi

echo ""
echo "================================================================================"
echo "🎯 NEXT STEPS:"
echo "================================================================================"
echo ""
echo "1. If you see orders above: ✅ Bot is trading!"
echo "2. If no orders: ⚠️  Too early OR not executing"
echo "3. Wait 4-6 hours and run this script again"
echo "4. Check exchange websites directly for confirmation"
echo ""
echo "================================================================================"
