#!/bin/bash
# Deploy MICRO + PIG fixes to VPS

echo "🚀 DEPLOYING FIXES TO VPS"
echo "=========================="
echo ""
echo "Fixes:"
echo "  ✅ PIG/USDT close loop stopped"
echo "  ✅ MICRO visibility added"
echo "  ✅ MICRO fallback pairs increased to 15"
echo "  ✅ All errors now visible"
echo ""
echo "Branch: analyze-vps-current"
echo ""

# Pull fixes on VPS
ssh root@75.119.149.117 << 'ENDSSH'
cd ~/bot

echo "📥 Pulling fixes from GitHub..."
git fetch origin
git checkout analyze-vps-current || git checkout -b analyze-vps-current origin/analyze-vps-current
git pull origin analyze-vps-current

echo ""
echo "🔄 Restarting bot..."
pkill -9 -f COMPLETE_ULTIMATE_ORCHESTRATOR
sleep 3

echo "🚀 Starting bot with fixes..."
nohup python COMPLETE_ULTIMATE_ORCHESTRATOR.py > bot.log 2>&1 &

echo ""
echo "⏳ Waiting 10 seconds for bot to initialize..."
sleep 10

echo ""
echo "📊 Showing recent logs (watch for MICRO activity):"
echo "=================================================="
tail -30 bot.log

echo ""
echo "✅ DEPLOYED!"
echo ""
echo "Watch logs with:"
echo "  tail -f bot.log | grep -E 'MICRO|PIG'"
ENDSSH

echo ""
echo "✅ Deploy complete!"
echo ""
echo "To watch MICRO activity:"
echo "  ssh root@75.119.149.117 'tail -f ~/bot/bot.log | grep MICRO'"
echo ""
