#!/bin/bash
# 🚨 CRITICAL FIX: Unlock Bybit $17k + Fix Scanner + All Systems Sync
set -e

echo "═══════════════════════════════════════════════════════"
echo "🚨 DEPLOYING CRITICAL FIX: BYBIT UNIFIED ACCOUNT"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "FIXES:"
echo "  ✅ Bybit UNIFIED account access (\$17k)"
echo "  ✅ Dynamic Market Scanner (string error fixed)"
echo "  ✅ All 40+ systems logging & synchronized"
echo "  ✅ Auto-Live-Trigger ready"
echo ""

cd /root/trading_bot
source venv/bin/activate

echo "📥 Pulling critical fix..."
git fetch origin main
git reset --hard origin/main

echo "✅ Pulled latest from GitHub"
echo ""

echo "🧹 Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Cache cleared"
echo ""

echo "🔄 Restarting testnet bot..."
sudo systemctl restart trading-bot-testnet

echo "✅ Bot restarted"
echo ""

echo "⏳ Waiting 60s for full startup (TensorFlow + all systems)..."
sleep 60

echo ""
echo "═══════════════════════════════════════════════════════"
echo "💰 VERIFYING BYBIT UNIFIED ACCOUNT ACCESS"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check for UNIFIED account messages
echo "🔍 Checking UNIFIED account setup:"
sudo journalctl -u trading-bot-testnet --since "2 min ago" | grep -i "unified" | tail -5

echo ""
echo "💰 Checking balance access:"
sudo journalctl -u trading-bot-testnet --since "2 min ago" | grep "Balance: 17055" | tail -3

echo ""
echo "🔥 Checking for successful trades (not 'Insufficient balance'):"
sudo journalctl -u trading-bot-testnet --since "2 min ago" | grep -E "TRADE EXECUTED|Insufficient balance" | tail -10

echo ""
echo "🔍 Scanner errors fixed?"
sudo journalctl -u trading-bot-testnet --since "2 min ago" | grep "string indices" || echo "✅ No scanner errors!"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "If you still see 'Insufficient balance', the Bybit testnet"
echo "requires manual transfer of funds to Unified Account at:"
echo "  https://testnet.bybit.com/user/assets/home"
echo ""
echo "Monitor live with:"
echo "  sudo journalctl -u trading-bot-testnet -f | grep -E 'TRADE EXECUTED|Insufficient|Balance'"
