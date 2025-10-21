#!/bin/bash

# VPS Trading Bot Check and Restart Script
# Run this on your VPS to check status and restart if needed

echo "╔════════════════════════════════════════════════════════════╗"
echo "║        VPS TRADING BOT CHECK & RESTART UTILITY             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Ensure we're in the right directory
cd ~/trading_bot || { echo "❌ Error: ~/trading_bot directory not found!"; exit 1; }

echo "📁 Current directory: $(pwd)"
echo ""

# Check if RUN_BOT.py exists
if [ ! -f "RUN_BOT.py" ]; then
    echo "❌ Error: RUN_BOT.py not found in $(pwd)"
    echo "   Please ensure you're in the correct trading_bot directory"
    exit 1
fi

echo "✅ RUN_BOT.py found"
echo ""

# Check current bot status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 CURRENT STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BOT_PID=$(pgrep -f "RUN_BOT.py")
if [ -n "$BOT_PID" ]; then
    echo "✅ Bot is currently RUNNING (PID: $BOT_PID)"
    echo ""
    echo "   Process details:"
    ps aux | grep $BOT_PID | grep -v grep
else
    echo "❌ Bot is NOT currently running"
fi

echo ""

# Check for bot.log
if [ -f "bot.log" ]; then
    echo "✅ Log file exists: bot.log"
    LOG_SIZE=$(du -h bot.log | cut -f1)
    LOG_LINES=$(wc -l < bot.log)
    echo "   Size: $LOG_SIZE ($LOG_LINES lines)"
    echo ""
    
    # Show recent activity
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 RECENT ACTIVITY (Last 10 entries)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -10 bot.log | sed 's/^/   /'
    echo ""
    
    # Show signal statistics
    VIP_COUNT=$(grep -c "✅ VIP #" bot.log 2>/dev/null || echo 0)
    FREE_COUNT=$(grep -c "✅ FREE #" bot.log 2>/dev/null || echo 0)
    UNIQUE_PAIRS=$(grep "Decision:" bot.log 2>/dev/null | grep -oE "[A-Z]{2,5}/[A-Z]{2,5}" | sort -u | wc -l)
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 SIGNAL STATISTICS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   VIP Signals: $VIP_COUNT"
    echo "   FREE Signals: $FREE_COUNT"
    echo "   Unique Pairs: $UNIQUE_PAIRS"
    echo ""
    
    # Show latest VIP signals
    if [ "$VIP_COUNT" -gt 0 ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "💎 LATEST VIP SIGNALS (Last 5)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        grep "✅ VIP #" bot.log | tail -5 | sed 's/^/   /'
        echo ""
    fi
    
    # Show discovered pairs
    TOTAL_DISCOVERED=$(grep "TOTAL DISCOVERED" bot.log 2>/dev/null | tail -1)
    if [ -n "$TOTAL_DISCOVERED" ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🔍 PAIR DISCOVERY"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "   $TOTAL_DISCOVERED"
        echo ""
    fi
else
    echo "⚠️  Log file not found: bot.log"
    echo "   (Will be created when bot starts)"
    echo ""
fi

# Ask for restart
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 RESTART OPTIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "Do you want to restart the bot? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🛑 Stopping existing bot instances..."
    pkill -9 -f RUN_BOT.py 2>/dev/null
    screen -S trading_bot -X quit 2>/dev/null
    sleep 2
    
    echo "🧹 Clearing Python cache..."
    rm -rf __pycache__ */__pycache__ 2>/dev/null
    
    echo "🚀 Starting bot in background..."
    
    # Export environment variables (from start_bot.sh)
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
    
    # Verify it started
    NEW_PID=$(pgrep -f "RUN_BOT.py")
    if [ -n "$NEW_PID" ]; then
        echo ""
        echo "✅ Bot successfully restarted! (PID: $NEW_PID)"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📝 MONITORING COMMANDS"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
        echo "   Attach to screen session:"
        echo "   $ screen -r trading_bot"
        echo ""
        echo "   Stop bot:"
        echo "   $ ./stop_bot.sh"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "🎯 Showing live output for 5 seconds..."
        echo ""
        sleep 1
        timeout 5 tail -f bot.log 2>/dev/null || echo "   (waiting for initial output...)"
        echo ""
        echo "✅ Bot is now running in background!"
        echo ""
    else
        echo ""
        echo "❌ Error: Bot failed to start!"
        echo ""
        echo "Checking for errors in log..."
        if [ -f "bot.log" ]; then
            echo ""
            tail -20 bot.log
        fi
        exit 1
    fi
else
    echo ""
    echo "ℹ️  Restart cancelled. Bot status unchanged."
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Script completed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
