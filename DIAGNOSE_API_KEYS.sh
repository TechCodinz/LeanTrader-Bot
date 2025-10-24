#!/bin/bash
#
# Diagnose API Key Issues
#

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  API KEY DIAGNOSTICS                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd ~/trading_bot || exit 1

echo "1️⃣  Checking .env file..."
if [ -f ".env" ]; then
    echo "   ✅ .env exists"
    echo ""
    echo "   API Keys found (not showing values):"
    grep -E "BYBIT|GATE|BINANCE" .env | cut -d'=' -f1 | while read key; do
        VALUE=$(grep "^${key}=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
        if [ -n "$VALUE" ] && [ "$VALUE" != "your_key_here" ]; then
            LEN=${#VALUE}
            echo "      ✅ $key (${LEN} characters)"
        else
            echo "      ❌ $key (empty or default)"
        fi
    done
else
    echo "   ❌ .env file not found!"
fi
echo ""

echo "2️⃣  Checking what mode bot is running in..."
if grep -q "mode='testnet'" RUN_BOT.py; then
    echo "   📊 Bot mode: TESTNET"
elif grep -q "mode='live'" RUN_BOT.py; then
    echo "   💰 Bot mode: LIVE"
else
    echo "   Looking at logs..."
    grep -E "Mode:|TESTNET|LIVE" bot.log | tail -3
fi
echo ""

echo "3️⃣  Checking which exchange is being used..."
grep -E "TRADING EXCHANGE|Using.*for.*trading" bot.log | tail -5
echo ""

echo "4️⃣  Checking API key errors in logs..."
grep -E "API key\|retCode.*10003" bot.log | tail -3
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DIAGNOSIS                                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if testnet keys vs mainnet
echo "The error 'retCode:10003' means:"
echo "   1. API key is for TESTNET but bot is using MAINNET Bybit"
echo "   2. OR API key permissions are wrong"
echo "   3. OR API key is expired/disabled"
echo ""

echo "SOLUTIONS:"
echo ""
echo "Option 1: Use Gate.io instead of Bybit (already configured)"
echo "   Your logs show 'Using Gate.io for real trading'"
echo "   Gate.io seems to be working!"
echo ""
echo "Option 2: Get correct Bybit API keys"
echo "   Go to Bybit → API Management"
echo "   Create NEW API key with:"
echo "   - Read permission"
echo "   - Trade permission" 
echo "   - For MAINNET (not testnet)"
echo ""
echo "Option 3: Disable Bybit, rely on Gate.io"
echo "   Edit LIVE_TRADE_EXECUTOR.py"
echo "   Comment out Bybit execution attempts"
echo ""
echo "═══════════════════════════════════════════════════════════════"
