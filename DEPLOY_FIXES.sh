#!/bin/bash

# ============================================================================
# DEPLOY ARBITRAGE + MARKET SCANNER + EXECUTION FIXES
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║    🔧 DEPLOYING FIXES FOR ARBITRAGE + SCANNER + EXECUTION          ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

echo ""
echo "🔧 FIXES INCLUDED:"
echo "   ✅ Arbitrage: Extract actual ccxt exchanges from engines"
echo "   ✅ Market Scanner: Create fallback connections (Gate.io, Binance)"
echo "   ✅ Trade Execution: Enhanced price fetching (router → engines → fallback)"
echo ""

cd /root/trading_bot || { echo "❌ /root/trading_bot not found!"; exit 1; }

echo "🔄 Step 1: Pull latest fixes..."
git pull origin cursor/check-and-update-trading-bot-service-0f23

echo ""
echo "🔄 Step 2: Restart live bot..."
sudo systemctl restart trading-bot-live

echo ""
echo "⏳ Waiting 15 seconds for bot to fully start..."
sleep 15

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "🔍 CHECKING WHAT'S WORKING:"
echo ""

echo "1️⃣  Arbitrage Status:"
sudo journalctl -u trading-bot-live --since "30 seconds ago" --no-pager | grep -i "arbitrage.*started\|valid exchanges\|arbitrage.*wired" | tail -5

echo ""
echo "2️⃣  Market Scanner Status:"
sudo journalctl -u trading-bot-live --since "30 seconds ago" --no-pager | grep -i "market.*scanner.*started\|scanning.*gateio\|scanning.*binance\|active pairs" | tail -5

echo ""
echo "3️⃣  Trade Execution Status:"
sudo journalctl -u trading-bot-live --since "1 minute ago" --no-pager | grep -i "executing:\|got.*from.*exchange\|trade.*executed" | tail -5

echo ""
echo "4️⃣  VIP Signals (should still work):"
sudo journalctl -u trading-bot-live --since "1 minute ago" --no-pager | grep -i "vip.*success" | tail -3

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📊 WHAT TO EXPECT:"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Arbitrage:"
echo "   - Should see 'ARBITRAGE SCANNER STARTED'"
echo "   - Should see 'Valid exchanges: [gateio, binance, ...]'"
echo "   - Should find arbitrage opportunities within minutes"
echo ""
echo "✅ Market Scanner:"
echo "   - Should see 'DYNAMIC MARKET SCANNER STARTED'"
echo "   - Should see 'Scanning gateio...' or 'Scanning binance...'"
echo "   - Should see 'Active pairs: X' (where X > 0)"
echo "   - Will find 20-50+ pairs on first scan"
echo ""
echo "✅ Trade Execution:"
echo "   - Should see 'Got \$X from [exchange]' when executing trades"
echo "   - No more 'Cannot get price' errors"
echo "   - Trades will actually execute!"
echo ""
echo "✅ VIP Signals:"
echo "   - Still working with TP1/TP2/TP3"
echo "   - No changes to signal quality"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "🚀 NOW MONITORING LIVE LOGS..."
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Watch for:"
echo "  • 'ARBITRAGE SCANNER STARTED'"
echo "  • 'DYNAMIC MARKET SCANNER STARTED'"
echo "  • '🔥 TRENDING: [coin]'"
echo "  • '💰 Arbitrage found'"
echo "  • '⚡ EXECUTING: [trade]'"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Show live logs
sudo journalctl -u trading-bot-live -f --since "1 minute ago"
