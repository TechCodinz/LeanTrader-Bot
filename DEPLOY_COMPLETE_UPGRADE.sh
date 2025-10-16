#!/bin/bash

# ============================================================================
# DEPLOY COMPLETE BOT UPGRADE
# All systems + News + Session-Aware + Hedge Fund Features
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║    🚀 DEPLOYING COMPLETE BOT UPGRADE - ALL SYSTEMS ACTIVATED        ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

echo ""
echo "🎯 WHAT'S BEING DEPLOYED:"
echo ""
echo "FIXES:"
echo "   ✅ Market scanner bug fixed (was getting dict, now gets ccxt exchanges)"
echo "   ✅ Arbitrage real execution logic (was simulated, now real)"
echo "   ✅ Lowered market scanner thresholds (\$1M volume, 1.5% change)"
echo ""
echo "NEW FEATURES:"
echo "   ✅ NEWS TRADING: CoinGecko trending, sentiment analysis"
echo "   ✅ SESSION-AWARE: 10-25% confidence boost during peak hours"
echo "   ✅ HEDGE FUND ARSENAL: Pairs trading, volatility, smart routing"
echo "   ✅ DAILY PROFIT TRACKING: See today's arbitrage profits"
echo ""

cd /root/trading_bot || { echo "❌ Directory not found!"; exit 1; }

echo "🔄 Pulling latest code..."
git pull origin cursor/check-and-update-trading-bot-service-0f23

echo ""
echo "🔄 Restarting bot..."
sudo systemctl restart trading-bot-live

echo ""
echo "⏳ Waiting 20 seconds for full startup..."
sleep 20

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "🔍 CHECKING SYSTEMS (30 second snapshot):"
echo ""

echo "1️⃣  Exchanges for arbitrage:"
sudo journalctl -u trading-bot-live --since "40 seconds ago" --no-pager | grep -i "added to arbitrage" | tail -10

echo ""
echo "2️⃣  Market Scanner:"
sudo journalctl -u trading-bot-live --since "40 seconds ago" --no-pager | grep -E "Market.*Scanner|Found.*USDT pairs|Active pairs|TRENDING" | tail -10

echo ""
echo "3️⃣  News Trading:"
sudo journalctl -u trading-bot-live --since "40 seconds ago" --no-pager | grep -i "news.*trading.*started\|news.*wired\|trending coins" | tail -5

echo ""
echo "4️⃣  Session-Aware:"
sudo journalctl -u trading-bot-live --since "40 seconds ago" --no-pager | grep -i "session.*aware.*wired\|current session" | tail -5

echo ""
echo "5️⃣  Hedge Fund:"
sudo journalctl -u trading-bot-live --since "40 seconds ago" --no-pager | grep -i "hedge.*fund.*wired\|hedge.*fund.*started" | tail -5

echo ""
echo "6️⃣  Recent VIP signals:"
sudo journalctl -u trading-bot-live --since "1 minute ago" --no-pager | grep "VIP.*SUCCESS" | tail -3

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📊 WHAT TO EXPECT:"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Within 1-2 minutes:"
echo "  ✅ '4-7 exchanges added to arbitrage'"
echo "  ✅ 'Found XXX USDT pairs on gateio'"
echo "  ✅ 'Active pairs: 30-60' (not 0!)"
echo "  ✅ 'NEWS TRADING ENGINE STARTED'"
echo "  ✅ 'SESSION-AWARE TRADING WIRED'"
echo "  ✅ 'HEDGE FUND ARSENAL STARTED'"
echo "  ✅ 'Current session: LONDON-NY' (or whatever session)"
echo ""
echo "Within 10 minutes:"
echo "  ✅ '🔥 TRENDING: PEPE/USDT +5.3%'"
echo "  ✅ '📰 News signal: DOGE trending'"
echo "  ✅ '🏦 Pairs trade: ETH/USDT BUY (z-score: -2.3)'"
echo "  ✅ '⏰ BTC/USDT: LONDON-NY session → 75% → 94% (1.25x)'"
echo "  ✅ '💰 Arbitrage found: BTC/USDT ...'"
echo "  ✅ '📊 TODAY'S ARBITRAGE STATS: Profit \$X.XX'"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "🚀 MONITORING LIVE LOGS..."
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "Watch for NEW features:"
echo "  📰 News signals (trending coins)"
echo "  ⏰ Session adjustments (confidence boosts)"
echo "  🏦 Hedge fund signals (pairs trading)"
echo "  💰 Arbitrage profit tracking"
echo "  🔍 Market scanner finding 50+ pairs"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Show live logs
sudo journalctl -u trading-bot-live -f --since "1 minute ago"
