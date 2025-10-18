#!/bin/bash
# 🚀 COMPLETE SYSTEM ACTIVATION - DEPLOY ALL FIXES
# This script deploys all fixes to VPS and activates all 40+ systems

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════"
echo "🚀 DEPLOYING ALL SYSTEM FIXES TO VPS"
echo "═══════════════════════════════════════════════════════"
echo ""

cd /root/trading_bot

# Ensure we're in venv
if [[ "$VIRTUAL_ENV" != "/root/trading_bot/venv" ]]; then
    echo "⚠️  Activating venv..."
    source venv/bin/activate
fi

echo "✅ In venv: $VIRTUAL_ENV"
echo ""

# Pull ALL fixes from GitHub
echo "📥 Pulling all fixes from GitHub..."
git fetch origin main
git reset --hard origin/main

echo "✅ All fixes pulled"
echo ""

# Clear all Python cache
echo "🧹 Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "✅ Cache cleared"
echo ""

# Restart testnet bot
echo "🔄 Restarting testnet bot..."
sudo systemctl restart trading-bot-testnet

echo "✅ Bot restarting..."
echo ""

# Wait for full initialization
echo "⏳ Waiting 60 seconds for full startup (TensorFlow loading)..."
sleep 60

echo ""
echo "═══════════════════════════════════════════════════════"
echo "📊 VERIFICATION - ALL SYSTEMS CHECK"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check bot status
echo "🔍 Bot Status:"
sudo systemctl status trading-bot-testnet | head -12
echo ""

# Check Bybit balance
echo "💰 Checking Bybit Testnet Balance:"
sudo journalctl -u trading-bot-testnet --since "2 min ago" | grep -E "Bybit.*Balance|TESTNET MODE|UNIFIED" | tail -5
echo ""

# Check all systems initialized
echo "📊 Systems Initialized:"
sudo journalctl -u trading-bot-testnet --since "2 min ago" | grep -E "WIRED|initialized" | wc -l
echo "systems detected"
echo ""

# Check active engines
echo "🔥 Active Engines (with logging):"
sudo journalctl -u trading-bot-testnet --since "2 min ago" | grep -E "Scalper|Moon|Arbitrage|News|Quantum|GOLDMINE|DIVINE|Brain|Swarm|Hedge" | tail -20
echo ""

# Check for errors
echo "❌ Any Errors?"
sudo journalctl -u trading-bot-testnet --since "2 min ago" | grep -iE "error|failed" | tail -10
echo ""

echo "═══════════════════════════════════════════════════════"
echo "✅ DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "📋 Next: Monitor bot for 5 minutes to verify all systems working"
echo ""
echo "Run this to monitor live:"
echo "  sudo journalctl -u trading-bot-testnet -f"
