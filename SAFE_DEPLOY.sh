#!/bin/bash
# SAFE DEPLOYMENT SCRIPT
# You enter your token interactively (never written to file)

echo "================================================================================"
echo "🚀 SAFE DEPLOYMENT - Interactive Token Entry"
echo "================================================================================"
echo ""

# Cleanup first
echo "Step 1: Cleanup..."
sudo rm -rf /opt/Lean-Trader
docker rm 994920290b67 2>/dev/null
sudo pkill -9 -f python
cd ~ && sudo rm -rf venv env bot* trade* 2>/dev/null

echo "✅ Cleanup complete"
echo ""

# Get token securely
echo "Step 2: GitHub Authentication"
echo ""
echo "Get your GitHub token from: https://github.com/settings/tokens"
echo "  1. Click 'Generate new token (classic)'"
echo "  2. Select 'repo' permission"
echo "  3. Copy the ENTIRE token"
echo ""
read -sp "Paste your GitHub token here (hidden): " GH_TOKEN
echo ""
echo ""

if [ -z "$GH_TOKEN" ]; then
    echo "❌ No token provided. Exiting."
    exit 1
fi

echo "✅ Token received (not shown for security)"
echo ""

# Clone
echo "Step 3: Cloning repository..."
cd ~
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  https://${GH_TOKEN}@github.com/Techcodinz/Lean-Trader.git trading_bot

if [ $? -ne 0 ]; then
    echo "❌ Clone failed. Check your token and try again."
    exit 1
fi

echo "✅ Repository cloned"
echo ""

# Clear token from memory
unset GH_TOKEN

cd trading_bot

# Create .env
echo "Step 4: Creating .env file..."
cat > .env << 'ENVEOF'
TELEGRAM_BOT_TOKEN=8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg
TG_ADMIN_CHAT_ID=5329503447
TG_FREE_CHAT_ID=-1002930953007
TG_VIP_CHAT_ID=-1002983007302
BYBIT_API_KEY=N8BMgWdfisCtkvfZk8
BYBIT_SECRET=BIu7c65FQnDsd6kBmctU7gK9bBbzY15vi8oe
BYBIT_TESTNET=true
GATEIO_TESTNET_API_KEY=590f4e3cb2a8cfcaa66fe1a3a646e4b1
GATEIO_TESTNET_SECRET=e1e5614876dfd2aa9c59beabd035c2af08a186b5f818209640c66e98225ca37b
GATEIO_LIVE_API_KEY=bbdcedbd7f719a87c851356cf4dd3c20
GATEIO_LIVE_SECRET=068996eb5877b74abf3595aedbc4f0778fe64e7f37d88c01b5af41f62e4d9c26
GATEIO_MODE=testnet
NEWSAPI_KEY=11dc35597f524f868ef814ad0bc9d216
ETHERSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
BSCSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
POLYGONSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
ENVEOF

echo "✅ .env created"
echo ""

# Install
echo "Step 5: Installing dependencies..."
pip3 install -r requirements.txt

echo "✅ Dependencies installed"
echo ""

# Create systemd service
echo "Step 6: Creating systemd service (for permanent operation)..."
sudo tee /etc/systemd/system/trading-bot.service > /dev/null << 'SVCEOF'
[Unit]
Description=Advanced AI Trading Bot - 40 Systems
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/trading_bot
ExecStart=/usr/bin/python3 /root/trading_bot/RUN_BOT.py --testnet
Restart=always
RestartSec=10
StandardOutput=append:/root/trading_bot/bot.log
StandardError=append:/root/trading_bot/bot_error.log

[Install]
WantedBy=multi-user.target
SVCEOF

echo "✅ Service created"
echo ""

# Start service
echo "Step 7: Starting bot..."
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

sleep 3

echo ""
echo "================================================================================"
echo "✅ BOT DEPLOYED!"
echo "================================================================================"
echo ""
echo "Check status:"
echo "  systemctl status trading-bot"
echo ""
echo "View logs:"
echo "  tail -f /root/trading_bot/bot.log"
echo ""
echo "Management:"
echo "  sudo systemctl restart trading-bot  (restart)"
echo "  sudo systemctl stop trading-bot     (stop)"
echo "  sudo systemctl start trading-bot    (start)"
echo ""
echo "Bot is running 24/7 with auto-restart!"
echo "Check Telegram in 2-3 minutes for startup message!"
echo "================================================================================"
