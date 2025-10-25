#!/bin/bash
###############################################################################
# 🧪 DEPLOY TESTNET MODE - Train bot with FAKE MONEY first!
###############################################################################

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         🧪 DEPLOYING BYBIT TESTNET MODE 🧪                      ║"
echo "║    Bot will trade with FAKE MONEY to prove profitability!       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

cd ~/trading_bot

# Step 1: Pull latest fixes
echo "📥 Pulling latest code..."
git stash
git pull origin cursor/discover-profitable-trading-pairs-5d1e
echo "✅ Code updated!"
echo ""

# Step 2: Check if testnet keys are set
echo "🔍 Checking .env configuration..."
if ! grep -q "BYBIT_TESTNET_API_KEY" .env; then
    echo "❌ Testnet keys not found in .env!"
    echo ""
    echo "⚠️  YOU NEED TO:"
    echo "   1. Get testnet API keys from: https://testnet.bybit.com"
    echo "   2. Edit .env: nano .env"
    echo "   3. Add your testnet keys"
    echo "   4. Run this script again"
    echo ""
    exit 1
fi

TESTNET_KEY=$(grep "BYBIT_TESTNET_API_KEY=" .env | cut -d'=' -f2)
if [ "$TESTNET_KEY" == "YOUR_TESTNET_KEY_HERE" ] || [ -z "$TESTNET_KEY" ]; then
    echo "❌ Please add your REAL Bybit testnet API keys to .env!"
    echo ""
    echo "📖 Read SETUP_BYBIT_TESTNET.md for instructions"
    echo ""
    exit 1
fi

echo "✅ Testnet keys found!"
echo ""

# Step 3: Verify trading mode
echo "🔧 Ensuring testnet mode is enabled..."
sed -i 's/TRADING_MODE=.*/TRADING_MODE=testnet/' .env
sed -i 's/BYBIT_TESTNET=.*/BYBIT_TESTNET=true/' .env
sed -i 's/ENABLE_LIVE=.*/ENABLE_LIVE=false/' .env
echo "✅ Testnet mode enabled!"
echo ""

# Step 4: Restart bot
echo "🔄 Restarting bot..."
pkill -9 -f RUN_BOT.py || true
sleep 2
./start_bot.sh
sleep 5
echo "✅ Bot restarted!"
echo ""

# Step 5: Check if testnet is active
echo "🔍 Checking testnet initialization..."
sleep 5
if grep -q "TESTNET MODE" bot.log; then
    echo "✅ TESTNET MODE ACTIVE - Bot is using fake money!"
else
    echo "⚠️  Could not confirm testnet mode in logs"
    echo "   Showing last 20 log lines:"
    tail -20 bot.log
    echo ""
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ DEPLOYMENT COMPLETE!                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 MONITOR YOUR BOT:"
echo "   tail -f bot.log | grep -E 'TESTNET|executed|profit'"
echo ""
echo "🌐 CHECK TESTNET TRADES:"
echo "   https://testnet.bybit.com/user/assets/wallet"
echo ""
echo "📈 WHEN YOU SEE PROFITS:"
echo "   The bot will prove profitability on testnet first!"
echo "   After consistent profits, we can switch to live trading."
echo ""
