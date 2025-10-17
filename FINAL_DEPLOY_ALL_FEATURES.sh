#!/bin/bash
# FINAL DEPLOYMENT - EVERYTHING INCLUDING PREMIUM VIP TELEGRAM

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║    🔮 FINAL DEPLOYMENT - ALL 55+ SYSTEMS + PREMIUM VIP PLATFORM      ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

cd /root/trading_bot

# ============================================================================
# STEP 1: Pull Latest Code
# ============================================================================
echo "📥 STEP 1/4: Pulling latest code..."
echo "══════════════════════════════════════════════════════════════════════"
git fetch origin
git checkout cursor/check-and-update-trading-bot-service-0f23
git pull origin cursor/check-and-update-trading-bot-service-0f23
echo "✅ Latest code with Premium VIP system pulled"
echo ""

# ============================================================================
# STEP 2: Ensure Venv Exists
# ============================================================================
echo "🔧 STEP 2/4: Checking virtual environment..."
echo "══════════════════════════════════════════════════════════════════════"

if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✅ Virtual environment ready"
echo ""

# ============================================================================
# STEP 3: Install ALL Dependencies (Including Telegram!)
# ============================================================================
echo "📦 STEP 3/4: Installing ALL dependencies..."
echo "══════════════════════════════════════════════════════════════════════"
echo "   This includes the Premium VIP Telegram system..."
echo ""

# Upgrade pip
pip install --upgrade pip -q

# Core packages
echo "   Installing core packages..."
pip install numpy pandas scipy scikit-learn -q

# Trading
echo "   Installing trading packages..."
pip install ccxt pandas-ta -q

# Blockchain
echo "   Installing blockchain packages..."
pip install web3 eth-account eth-utils -q

# Async
echo "   Installing async packages..."
pip install aiohttp websockets requests -q

# NLP
echo "   Installing NLP packages..."
pip install nltk textblob vaderSentiment -q

# Utilities
echo "   Installing utilities..."
pip install python-dotenv pyyaml toml -q

# CRITICAL: Telegram
echo "   Installing Telegram bot (CRITICAL!)..."
pip install python-telegram-bot -q

# Web scraping (for UltraScout news/social features)
echo "   Installing web scraping packages..."
pip install beautifulsoup4 lxml -q

echo ""
echo "✅ ALL dependencies installed (including Telegram!)"
echo ""

# Verify critical packages
echo "🔍 Verifying installations..."
python3 -c "import numpy, pandas, ccxt, telegram; print('   ✅ All critical packages OK!')" 2>&1 || echo "   ⚠️  Some packages may have issues"
echo ""

# ============================================================================
# STEP 4: Restart Bot
# ============================================================================
echo "🔄 STEP 4/4: Restarting bot with ALL features..."
echo "══════════════════════════════════════════════════════════════════════"
sudo systemctl restart trading-bot

echo "⏳ Waiting 25 seconds for startup..."
sleep 25

echo ""
echo "📊 SERVICE STATUS:"
echo "══════════════════════════════════════════════════════════════════════"
systemctl status trading-bot --no-pager | head -15

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║                  ✅ DEPLOYMENT COMPLETE! 🎉                           ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if running
if systemctl is-active --quiet trading-bot; then
    echo "✅ Bot is RUNNING!"
    echo ""
    
    # Check for features
    echo "🔍 Checking loaded features..."
    FEATURES=$(journalctl -u trading-bot --since "1 minute ago" --no-pager | grep -E "LOADED|ACTIVE" | tail -10)
    
    if [ -n "$FEATURES" ]; then
        echo "$FEATURES"
    else
        echo "   Features loading... check logs in 30 seconds"
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "📱 CHECK YOUR TELEGRAM!"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    echo "You should receive admin notification:"
    echo "  '🚀 BOT STARTED - 55+ systems active'"
    echo ""
    echo "If you DON'T receive notification:"
    echo "  1. Check logs: journalctl -u trading-bot -n 100"
    echo "  2. Verify Telegram token in .env"
    echo "  3. Check admin chat ID is correct"
    echo ""
    
else
    echo "❌ Bot is NOT running - checking error..."
    echo ""
    echo "Error logs:"
    journalctl -u trading-bot -n 50 --no-pager | tail -30
    echo ""
    echo "To debug:"
    echo "  journalctl -u trading-bot -n 100 --no-pager"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "🌟 PREMIUM VIP SYSTEM FEATURES:"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  📱 Admin gets notifications for EVERYTHING"
echo "  📢 Free channel gets basic signals"
echo "  🌟 VIP channel gets premium signals with ONE-CLICK TRADING"
echo "  💰 USDT payment system for subscriptions"
echo "  🔑 Users add their own exchange APIs"
echo "  ⚡ Interactive trading from Telegram"
echo ""
echo "  💵 Revenue: \$50/month per VIP user"
echo "  🎯 Target: 100+ VIP users = \$5,000/month"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "📖 Read FIX_CRASH_AND_ENABLE_VIP.md for full documentation"
echo ""
