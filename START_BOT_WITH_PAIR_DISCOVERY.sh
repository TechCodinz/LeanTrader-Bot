#!/bin/bash
#
# START BOT WITH DYNAMIC PAIR DISCOVERY
# This runs your complete trading system with automatic pair discovery
#

echo "════════════════════════════════════════════════════════════════"
echo "  STARTING COMPLETE ULTIMATE ORCHESTRATOR"
echo "  WITH DYNAMIC PAIR DISCOVERY (5000+ pairs!)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
fi

# Check if required dependencies are installed
echo "📦 Checking dependencies..."
python3 -c "import ccxt; print('✅ ccxt installed')" 2>/dev/null || {
    echo "⚠️  Installing ccxt..."
    pip3 install ccxt
}

# Verify DYNAMIC_PAIR_DISCOVERY imports correctly
echo "🔍 Verifying Dynamic Pair Discovery..."
python3 -c "from DYNAMIC_PAIR_DISCOVERY import get_discovery_engine; print('✅ Discovery engine ready')" || {
    echo "❌ Discovery engine failed to import!"
    exit 1
}

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🚀 LAUNCHING BOT..."
echo "════════════════════════════════════════════════════════════════"
echo ""

# Run the complete orchestrator (default: testnet mode)
# Use --mode=testnet (safe testing) or --mode=live (real money)
python3 COMPLETE_ULTIMATE_ORCHESTRATOR.py --mode=testnet

# If you want to run in background:
# nohup python3 COMPLETE_ULTIMATE_ORCHESTRATOR.py --mode=testnet > bot.log 2>&1 &
# echo "✅ Bot started in background! PID: $!"
# echo "📋 View logs: tail -f bot.log"
