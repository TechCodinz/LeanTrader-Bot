#!/bin/bash

echo "🚀 Starting Trading Bot..."

# Kill any existing processes
pkill -9 -f RUN_BOT.py
sleep 2

# Clear cache
rm -rf __pycache__ */__pycache__

# Export environment variables
export TELEGRAM_BOT_TOKEN='REDACTED_ROTATED_TELEGRAM_BOT_TOKEN__SET_VIA_MOUNTED_SECRET_FILE'
export VIP_CHANNEL_ID='-1002983007302'
export FREE_CHANNEL_ID='-1002930953007'
export BYBIT_API_KEY='REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE'
export BYBIT_API_SECRET='REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE'
export MAX_POSITION_SIZE='50'
export MAX_DAILY_TRADES='20'
export MIN_CONFIDENCE='0.80'

# Start in screen session
screen -dmS trading_bot bash -c "cd ~/trading_bot && python3 -B RUN_BOT.py > bot.log 2>&1"

echo "✅ Bot started in background!"
echo ""
echo "📝 Useful commands:"
echo "   View logs: tail -f bot.log"
echo "   Check status: ./status.sh"
echo "   Attach to screen: screen -r trading_bot"
echo "   Stop bot: ./stop_bot.sh"
