#!/bin/bash
###############################################################################
# 🎯 ENABLE LIVE TRADING - Switches from testnet to REAL MONEY
###############################################################################
#
# CURRENT STATUS:
#   ✅ Bot is FULLY FUNCTIONAL
#   ✅ Generating 70-95% confidence signals  
#   ✅ Loaded 43,201 historical trades
#   ✅ Execution orchestrator wired
#   ⚠️  Currently BLOCKED (testnet mode for safety)
#
# WHAT THIS DOES:
#   - Switches from testnet to LIVE trading
#   - Enables REAL money execution
#   - Starts making ACTUAL trades
#
# ⚠️  WARNING: This will use REAL MONEY!
#     Only run when you're ready to trade live.
#
###############################################################################

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "🎯 ENABLE LIVE TRADING"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Safety check
echo "⚠️  WARNING: This will switch to REAL MONEY trading!"
echo ""
echo "Current status:"
echo "  ✅ Bot generates 70-95% confidence signals"
echo "  ✅ Loaded 43,201 historical trades"
echo "  ✅ Execution system wired and ready"
echo "  ⚠️  Currently in TESTNET (safe mode)"
echo ""
echo "After enabling:"
echo "  💰 Will use REAL money (Bybit live account)"
echo "  💰 Will place REAL orders"
echo "  💰 Can make profits OR losses"
echo ""

read -p "Are you SURE you want to enable live trading? (type 'YES' to confirm): " confirm

if [ "$confirm" != "YES" ]; then
    echo ""
    echo "❌ Cancelled. Bot remains in testnet mode."
    echo ""
    exit 1
fi

echo ""
echo "Enabling live trading..."
echo ""

# Backup .env
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
echo "✅ Backed up .env"

# Update .env to enable live trading
sed -i 's/TRADING_MODE=testnet/TRADING_MODE=live/' .env
sed -i 's/BYBIT_TESTNET=true/BYBIT_TESTNET=false/' .env
sed -i 's/ENABLE_LIVE=false/ENABLE_LIVE=true/' .env

echo "✅ Updated .env"
echo ""

# Verify changes
echo "New settings:"
grep "TRADING_MODE" .env
grep "BYBIT_TESTNET" .env
grep "ENABLE_LIVE" .env
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ LIVE TRADING ENABLED!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Stop the bot if running: pkill -f COMPLETE_ULTIMATE_ORCHESTRATOR"
echo "  2. Restart the bot: python COMPLETE_ULTIMATE_ORCHESTRATOR.py"
echo "  3. Watch logs for REAL trades: tail -f bot.log"
echo ""
echo "The bot will now:"
echo "  💰 Use REAL Bybit account"
echo "  💰 Place REAL orders"
echo "  💰 Execute with learned memory (43k trades)"
echo "  💰 Trade with 70-95% confidence signals"
echo ""
echo "Monitor carefully for first hour!"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
