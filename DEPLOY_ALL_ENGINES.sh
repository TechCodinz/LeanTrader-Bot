#!/bin/bash
# ULTIMATE ENGINE INTEGRATION DEPLOYMENT
# Run this ONE command to integrate ALL engines!

cd ~/bot

echo "🚀 ULTIMATE ENGINE INTEGRATION - Deploying ALL engines..."
echo ""

# Pull latest from branch
echo "📦 Pulling latest code..."
git fetch origin vps-working-snapshot-20251028-0402
git checkout vps-working-snapshot-20251028-0402
git pull

# Run the comprehensive fix first (gets ultra_core.get_market_data)
echo ""
echo "🔧 Step 1: Running comprehensive engine fix..."
python3 COMPREHENSIVE_ENGINE_FIX.py

# Run the ultimate integration (adds all missing engines)
echo ""
echo "🔧 Step 2: Running ultimate engine integration..."
python3 ULTIMATE_ENGINE_INTEGRATION.py

# Restart bot
echo ""
echo "🔄 Step 3: Restarting bot with ALL engines..."
pkill -f COMPLETE_ULTIMATE_ORCHESTRATOR
sleep 5
nohup python COMPLETE_ULTIMATE_ORCHESTRATOR.py > bot.log 2>&1 &

echo ""
echo "⏳ Waiting for bot to initialize (40 seconds)..."
sleep 40

# Monitor ALL engines
echo ""
echo "👀 Monitoring ALL engine signals (2 minutes)..."
echo "   You should see signals from:"
echo "   - 🌙 Moon Spotter"
echo "   - ⚡ Ultra Scalping"
echo "   - 💎 Ultra Arbitrage"
echo "   - 💎 Ultra Rare Engines (10 engines!)"
echo "   - 👁️  Omniscient Execution"
echo "   - 🧠 Adaptive Confidence"
echo "   - 📊 Smart Scalping"
echo ""
timeout 120 tail -f bot.log | grep --line-buffered -E "🌙 MOON|ULTRA.*SCALP|💎 RARE ENGINE|👁️  OMNISCIENT|execute_trade|Balance:"

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
echo ""
echo "📊 Check status:"
echo "   ps aux | grep COMPLETE_ULTIMATE_ORCHESTRATOR"
echo ""
echo "📈 Monitor profit:"
echo "   ./watch_micro.sh"
