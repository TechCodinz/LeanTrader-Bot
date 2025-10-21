#!/bin/bash
# COPY AND PASTE THIS ENTIRE SCRIPT INTO YOUR VPS TERMINAL
# It will create the fix script and run it automatically

cat > /root/trading_bot/fix_and_restart.sh << 'EOFSCRIPT'
#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           TRADING BOT FIX & RESTART                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Go to trading bot directory
cd /root/trading_bot || cd ~/trading_bot || { echo "❌ Cannot find trading_bot directory"; exit 1; }

echo "📁 Working directory: $(pwd)"
echo ""

# Stop any existing instances
echo "🛑 Stopping existing bot..."
pkill -9 -f RUN_BOT.py 2>/dev/null
screen -S trading_bot -X quit 2>/dev/null
sleep 2

# Clear cache
echo "🧹 Clearing cache..."
rm -rf __pycache__ */__pycache__ 2>/dev/null

# Set environment variables
echo "🔧 Setting environment variables..."
export TELEGRAM_BOT_TOKEN='8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg'
export VIP_CHANNEL_ID='-1002983007302'
export FREE_CHANNEL_ID='-1002930953007'
export BYBIT_API_KEY='mMHs7rDC72TvHs4oQG'
export BYBIT_API_SECRET='NwTa6UOgczdmZI2Kn2WBcFfh5r6VkVGnvGEI'
export MAX_POSITION_SIZE='50'
export MAX_DAILY_TRADES='20'
export MIN_CONFIDENCE='0.80'

# Check if RUN_BOT.py exists
if [ ! -f "RUN_BOT.py" ]; then
    echo "❌ RUN_BOT.py not found in $(pwd)"
    echo "   Files here:"
    ls -la *.py 2>/dev/null | head -10
    exit 1
fi

echo "✅ Found RUN_BOT.py"
echo ""

# Start bot
echo "🚀 Starting bot in screen session..."
screen -dmS trading_bot bash -c "cd ~/trading_bot && python3 -B RUN_BOT.py > bot.log 2>&1"

sleep 4

# Check if it started
BOT_PID=$(pgrep -f "RUN_BOT.py")
if [ -n "$BOT_PID" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ BOT STARTED SUCCESSFULLY!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "   Process ID: $BOT_PID"
    echo ""
    echo "📊 MONITORING COMMANDS:"
    echo ""
    echo "   Watch live signals:"
    echo "   $ tail -f bot.log | grep '✅'"
    echo ""
    echo "   View all logs:"
    echo "   $ tail -f bot.log"
    echo ""
    echo "   Check status:"
    echo "   $ ./status.sh"
    echo ""
    echo "   Signal count:"
    echo "   $ grep -c '✅ VIP' bot.log"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📺 Showing first 10 seconds of output..."
    echo ""
    timeout 10 tail -f bot.log 2>/dev/null || echo "   (Waiting for initial output...)"
    echo ""
    echo "✅ Bot is running! Check logs with: tail -f bot.log"
    echo ""
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ BOT FAILED TO START"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    if [ -f "bot.log" ]; then
        echo "📋 Last 30 lines of log:"
        echo ""
        tail -30 bot.log
    else
        echo "⚠️  No bot.log file found"
    fi
    echo ""
    exit 1
fi
EOFSCRIPT

# Make it executable
chmod +x /root/trading_bot/fix_and_restart.sh

# Run it
echo "✅ Script created! Running now..."
echo ""
/root/trading_bot/fix_and_restart.sh
