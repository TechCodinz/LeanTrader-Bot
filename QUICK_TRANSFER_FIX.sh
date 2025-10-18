#!/bin/bash
# 🚨 QUICK FIX: Try API transfer first, guide user if it fails

set -e

cd /root/trading_bot
source venv/bin/activate

echo "═══════════════════════════════════════════════════════"
echo "🚨 BYBIT FUND TRANSFER - UNLOCKING \$17K"
echo "═══════════════════════════════════════════════════════"
echo ""

# Pull latest
git fetch origin main
git checkout origin/main -- check_bybit_internal_transfer.py BYBIT_MANUAL_TRANSFER_GUIDE.md

echo "🔄 Attempting automatic transfer via API..."
echo ""

python3 check_bybit_internal_transfer.py

echo ""
echo "═══════════════════════════════════════════════════════"
echo "📋 NEXT STEPS"
echo "═══════════════════════════════════════════════════════"
echo ""

if [ $? -eq 0 ]; then
    echo "✅ If transfer succeeded, restart bot:"
    echo "   sudo systemctl restart trading-bot-testnet"
    echo ""
    echo "   Then monitor trades:"
    echo "   sudo journalctl -u trading-bot-testnet -f | grep 'TRADE EXECUTED'"
else
    echo "⚠️  API transfer failed. Manual transfer required:"
    echo ""
    echo "1. Open: https://testnet.bybit.com/user/assets/home"
    echo "2. Click 'Transfer'"
    echo "3. From: Funding → To: Unified Trading"
    echo "4. Amount: 17055 USDT"
    echo "5. Confirm"
    echo ""
    echo "See BYBIT_MANUAL_TRANSFER_GUIDE.md for screenshots!"
fi
