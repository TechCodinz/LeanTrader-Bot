#!/bin/bash
###############################################################################
# TRADING BOT - VPS DEPLOYMENT SCRIPT
# Run this on your VPS to deploy and start the bot
###############################################################################

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         TRADING BOT - VPS DEPLOYMENT                      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# 1. UPDATE SYSTEM
echo "1️⃣ Updating system..."
sudo apt-get update -y
sudo apt-get upgrade -y

# 2. INSTALL PYTHON 3.11+
echo "2️⃣ Installing Python..."
sudo apt-get install -y python3 python3-pip python3-venv

# 3. INSTALL REQUIRED PACKAGES
echo "3️⃣ Installing system packages..."
sudo apt-get install -y git curl wget screen htop

# 4. CREATE BOT DIRECTORY
echo "4️⃣ Setting up bot directory..."
mkdir -p ~/trading_bot
cd ~/trading_bot

# 5. INSTALL PYTHON DEPENDENCIES
echo "5️⃣ Installing Python dependencies..."
pip3 install --upgrade pip

# Core trading
pip3 install ccxt pandas numpy python-telegram-bot python-dotenv

# Technical analysis
pip3 install ta-lib scikit-learn tensorflow

# Visualization
pip3 install matplotlib mplfinance

# Additional
pip3 install aiohttp requests asyncio

echo "✅ Dependencies installed!"

# 6. CREATE .ENV FILE
echo "6️⃣ Creating .env file..."
cat > .env << 'EOF'
# Telegram
TELEGRAM_BOT_TOKEN=8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg
VIP_CHANNEL_ID=-1002983007302
FREE_CHANNEL_ID=-1002930953007

# Bybit
BYBIT_API_KEY=mMHs7rDC72TvHs4oQG
BYBIT_API_SECRET=NwTa6UOgczdmZI2Kn2WBcFfh5r6VkVGnvGEI

# Trading Limits (Safe defaults)
MAX_POSITION_SIZE=50
MAX_DAILY_TRADES=20
MIN_CONFIDENCE=0.80
EOF

echo "✅ .env configured!"

# 7. CREATE START SCRIPT
echo "7️⃣ Creating start script..."
cat > start_bot.sh << 'STARTEOF'
#!/bin/bash

# Kill any existing bot processes
pkill -9 -f RUN_BOT.py

# Clear Python cache
rm -rf __pycache__ */__pycache__

# Export environment variables
export TELEGRAM_BOT_TOKEN='8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg'
export VIP_CHANNEL_ID='-1002983007302'
export FREE_CHANNEL_ID='-1002930953007'
export BYBIT_API_KEY='mMHs7rDC72TvHs4oQG'
export BYBIT_API_SECRET='NwTa6UOgczdmZI2Kn2WBcFfh5r6VkVGnvGEI'
export MAX_POSITION_SIZE='50'
export MAX_DAILY_TRADES='20'
export MIN_CONFIDENCE='0.80'

# Start bot in screen session
screen -dmS trading_bot bash -c "cd ~/trading_bot && python3 -B RUN_BOT.py > bot.log 2>&1"

echo "✅ Bot started in screen session 'trading_bot'"
echo "   View logs: tail -f ~/trading_bot/bot.log"
echo "   Attach to screen: screen -r trading_bot"
STARTEOF

chmod +x start_bot.sh

# 8. CREATE STOP SCRIPT
echo "8️⃣ Creating stop script..."
cat > stop_bot.sh << 'STOPEOF'
#!/bin/bash
pkill -9 -f RUN_BOT.py
screen -S trading_bot -X quit
echo "✅ Bot stopped"
STOPEOF

chmod +x stop_bot.sh

# 9. CREATE STATUS SCRIPT
echo "9️⃣ Creating status script..."
cat > status.sh << 'STATUSEOF'
#!/bin/bash

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                    BOT STATUS                             ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if bot is running
if pgrep -f RUN_BOT.py > /dev/null; then
    echo "✅ Bot is RUNNING"
    echo "   PID: $(pgrep -f RUN_BOT.py)"
else
    echo "❌ Bot is NOT running"
fi

echo ""
echo "📱 Telegram Signals Sent:"
echo "   VIP: $(grep -c '✅ VIP #' bot.log 2>/dev/null || echo 0)"
echo "   FREE: $(grep -c '✅ FREE #' bot.log 2>/dev/null || echo 0)"

echo ""
echo "📊 Pairs Trading: $(grep 'Decision:' bot.log 2>/dev/null | grep -oE '[A-Z]{2,5}/[A-Z]{2,5}' | sort -u | wc -l)"

echo ""
echo "💰 Recent Signals:"
tail -10 bot.log 2>/dev/null | grep "Decision:" | tail -5

echo ""
echo "📈 Latest Activity:"
tail -5 bot.log 2>/dev/null
STATUSEOF

chmod +x status.sh

# 10. CREATE AUTO-RESTART CRON
echo "🔟 Setting up auto-restart..."
cat > restart_cron.sh << 'CRONEOF'
#!/bin/bash
# Add to crontab: */30 * * * * ~/trading_bot/restart_cron.sh

if ! pgrep -f RUN_BOT.py > /dev/null; then
    cd ~/trading_bot && ./start_bot.sh
    echo "$(date): Bot was down, restarted" >> restart.log
fi
CRONEOF

chmod +x restart_cron.sh

# Add to crontab
(crontab -l 2>/dev/null; echo "*/30 * * * * ~/trading_bot/restart_cron.sh") | crontab -

echo "✅ Auto-restart configured (checks every 30 minutes)"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                   DEPLOYMENT COMPLETE!                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "📝 NEXT STEPS:"
echo ""
echo "1. Copy your bot files to ~/trading_bot/"
echo "   scp -r /workspace/* user@your-vps:~/trading_bot/"
echo ""
echo "2. Start the bot:"
echo "   cd ~/trading_bot && ./start_bot.sh"
echo ""
echo "3. Check status:"
echo "   ./status.sh"
echo ""
echo "4. View live logs:"
echo "   tail -f bot.log"
echo ""
echo "5. Attach to screen session:"
echo "   screen -r trading_bot"
echo ""
echo "🎯 READY TO MAKE MONEY!"
