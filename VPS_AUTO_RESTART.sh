#!/bin/bash

# Quick auto-restart script (no prompts)

echo "🔄 VPS Trading Bot Auto-Restart"
echo "================================"
echo ""

cd ~/trading_bot || cd /workspace

echo "🛑 Stopping any existing bot..."
pkill -9 -f RUN_BOT.py 2>/dev/null
screen -S trading_bot -X quit 2>/dev/null
sleep 2

echo "🧹 Clearing cache..."
rm -rf __pycache__ */__pycache__ 2>/dev/null

echo "🚀 Starting bot..."

# Export environment variables
export TELEGRAM_BOT_TOKEN='8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg'
export VIP_CHANNEL_ID='-1002983007302'
export FREE_CHANNEL_ID='-1002930953007'
export BYBIT_API_KEY='mMHs7rDC72TvHs4oQG'
export BYBIT_API_SECRET='NwTa6UOgczdmZI2Kn2WBcFfh5r6VkVGnvGEI'
export MAX_POSITION_SIZE='50'
export MAX_DAILY_TRADES='20'
export MIN_CONFIDENCE='0.80'

# Start in screen session
screen -dmS trading_bot bash -c "cd ~/trading_bot && python3 -B RUN_BOT.py > bot.log 2>&1"

sleep 3

# Check if started
if pgrep -f "RUN_BOT.py" > /dev/null; then
    echo ""
    echo "✅ Bot started successfully! (PID: $(pgrep -f RUN_BOT.py))"
    echo ""
    echo "📝 Monitor with: tail -f bot.log"
    echo ""
    echo "Showing first 5 seconds of output..."
    timeout 5 tail -f bot.log 2>/dev/null || echo "(waiting for output...)"
    echo ""
else
    echo "❌ Failed to start bot"
    if [ -f "bot.log" ]; then
        echo ""
        echo "Last 20 lines of log:"
        tail -20 bot.log
    fi
    exit 1
fi
