#!/bin/bash

# ============================================================================
# DEPLOY MARKET EXPANSION + VIP ENHANCEMENTS + ARBITRAGE ACTIVATION
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║    🚀 DEPLOYING MARKET EXPANSION + VIP ENHANCEMENTS                 ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📦 What's being deployed:"
echo "   ✅ Market scan: 5 coins → 70+ pairs"
echo "   ✅ VIP signals: TP1/TP2/TP3 + cross-timeframe analysis"
echo "   ✅ FREE signals: TP1/TP2/TP3 + better formatting"
echo "   ✅ Dynamic scanner: Auto-discovers trending pairs"
echo "   ✅ Arbitrage: Now active and trading"
echo ""

cd /root/trading_bot || { echo "❌ /root/trading_bot not found!"; exit 1; }

echo "🔄 Step 1: Pull latest code..."
git pull origin cursor/check-and-update-trading-bot-service-0f23

echo ""
echo "🔄 Step 2: Restart live bot..."
sudo systemctl restart trading-bot-live

echo ""
echo "⏳ Waiting 10 seconds for bot to start..."
sleep 10

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════════"

echo ""
echo "📊 Check what's happening:"
echo ""
echo "1️⃣  Monitor live logs:"
echo "   sudo journalctl -u trading-bot-live -f"
echo ""
echo "2️⃣  Check market scanner (should see 70+ pairs):"
echo "   sudo journalctl -u trading-bot-live --since '1 minute ago' | grep -i 'market.*scanner\\|active pairs\\|expanded universe'"
echo ""
echo "3️⃣  Check VIP signals (should see TP1/TP2/TP3):"
echo "   sudo journalctl -u trading-bot-live --since '1 minute ago' | grep -E 'VIP.*TP1\\|VIP.*SUCCESS\\|send_signal_to_vip'"
echo ""
echo "4️⃣  Check arbitrage activity:"
echo "   sudo journalctl -u trading-bot-live --since '1 minute ago' | grep -i 'arbitrage\\|arb.*found'"
echo ""
echo "5️⃣  Check dynamic scanner:"
echo "   sudo journalctl -u trading-bot-live --since '2 minutes ago' | grep -i 'trending\\|volume.*leaders\\|discovered'"
echo ""

echo "🎯 WHAT TO EXPECT IN TELEGRAM:"
echo ""
echo "FREE Channel:"
echo "  • Signals with TP1, TP2, TP3"
echo "  • Better formatting"
echo "  • Risk/Reward ratios"
echo ""
echo "VIP Channel:"
echo "  • TP1/TP2/TP3 levels (30%, 40%, 30% allocation)"
echo "  • Cross-timeframe analysis (15m, 1h, 4h)"
echo "  • Strategy confirmations"
echo "  • Detailed AI reasoning"
echo "  • 'Analysis' button (charts coming)"
echo ""
echo "Signals:"
echo "  • 10-20x more opportunities (70+ pairs instead of 5)"
echo "  • Trending coins auto-discovered"
echo "  • Arbitrage opportunities"
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "🚀 NOW MONITORING..."
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Show live startup logs
sudo journalctl -u trading-bot-live -f --since "30 seconds ago"
