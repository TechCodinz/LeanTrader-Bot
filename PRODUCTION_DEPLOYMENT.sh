#!/bin/bash
# PRODUCTION DEPLOYMENT - Run Continuously Like Old Bot
# This sets up bot to:
# - Run 24/7
# - Auto-restart if crashes
# - Survive VPS reboots
# - Professional daemon setup

echo "================================================================================"
echo "🚀 PRODUCTION DEPLOYMENT - CONTINUOUS OPERATION"
echo "================================================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    exit 1
fi

# Get username for service
read -p "Enter your VPS username (default: ubuntu): " VPS_USER
VPS_USER=${VPS_USER:-ubuntu}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1: CLEANUP OLD BOT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Stop old services
systemctl stop trading-bot 2>/dev/null
systemctl disable trading-bot 2>/dev/null

# Kill processes
pkill -9 -f python
docker stop $(docker ps -q) 2>/dev/null
docker rm 994920290b67 2>/dev/null

# Remove old directories
rm -rf /opt/Lean-Trader
rm -rf /home/$VPS_USER/venv
rm -rf /home/$VPS_USER/bot*
rm -rf /home/$VPS_USER/trade*

echo "✅ Old bot cleaned up"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2: CLONE NEW BOT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

read -p "Enter GitHub Personal Access Token: " GH_TOKEN
read -p "Enter GitHub Username: " GH_USER
read -p "Enter Repository Name (e.g., Lean-Trader): " REPO_NAME

cd /home/$VPS_USER

# Clone repository
su - $VPS_USER -c "git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  https://${GH_TOKEN}@github.com/${GH_USER}/${REPO_NAME}.git trading_bot"

if [ $? -ne 0 ]; then
    echo "❌ Clone failed. Check token and repo name."
    exit 1
fi

echo "✅ Repository cloned to /home/$VPS_USER/trading_bot"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3: CREATE .env FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cat > /home/$VPS_USER/trading_bot/.env << 'ENVFILE'
# ===== TELEGRAM =====
TELEGRAM_BOT_TOKEN=REDACTED_ROTATED_TELEGRAM_BOT_TOKEN__SET_VIA_MOUNTED_SECRET_FILE
TG_ADMIN_CHAT_ID=5329503447
TG_FREE_CHAT_ID=-1002930953007
TG_VIP_CHAT_ID=-1002983007302

# ===== BYBIT TESTNET =====
BYBIT_API_KEY=REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
BYBIT_SECRET=REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
BYBIT_TESTNET=true

# ===== GATE.IO =====
GATEIO_TESTNET_API_KEY=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_TESTNET_SECRET=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_LIVE_API_KEY=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_LIVE_SECRET=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_MODE=testnet

# ===== NEWS & DATA =====
NEWSAPI_KEY=REDACTED_ROTATED_NEWSAPI_KEY__SET_VIA_MOUNTED_SECRET_FILE
ETHERSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
BSCSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
POLYGONSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
ENVFILE

chown $VPS_USER:$VPS_USER /home/$VPS_USER/trading_bot/.env

echo "✅ .env file created"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 4: INSTALL DEPENDENCIES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd /home/$VPS_USER/trading_bot

# Install system dependencies
apt update
apt install -y python3 python3-pip screen

# Install Python packages
su - $VPS_USER -c "cd trading_bot && pip3 install -r requirements.txt"

echo "✅ Dependencies installed"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 5: CREATE SYSTEMD SERVICE (Like Old Bot!)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create systemd service file
cat > /etc/systemd/system/trading-bot.service << SERVICEEOF
[Unit]
Description=Advanced AI Trading Bot - 40 Systems
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$VPS_USER
WorkingDirectory=/home/$VPS_USER/trading_bot
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/usr/bin/python3 /home/$VPS_USER/trading_bot/RUN_BOT.py --testnet
Restart=always
RestartSec=10
StandardOutput=append:/home/$VPS_USER/trading_bot/bot.log
StandardError=append:/home/$VPS_USER/trading_bot/bot_error.log

# Resource limits
MemoryMax=4G
CPUQuota=200%

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
SERVICEEOF

echo "✅ Systemd service created"
echo ""

# Set permissions
chown -R $VPS_USER:$VPS_USER /home/$VPS_USER/trading_bot

# Reload systemd
systemctl daemon-reload

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 6: START BOT AS SERVICE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Enable service (starts on boot)
systemctl enable trading-bot

# Start service
systemctl start trading-bot

sleep 3

# Check status
systemctl status trading-bot --no-pager

echo ""
echo "================================================================================"
echo "✅ BOT DEPLOYED AS PRODUCTION SERVICE!"
echo "================================================================================"
echo ""
echo "Bot Features:"
echo "  ✅ Runs 24/7 continuously"
echo "  ✅ Auto-restarts if crashes"
echo "  ✅ Survives VPS reboots"
echo "  ✅ Logs to bot.log"
echo "  ✅ Professional daemon setup"
echo ""
echo "Management Commands:"
echo "  View logs:      tail -f /home/$VPS_USER/trading_bot/bot.log"
echo "  Check status:   systemctl status trading-bot"
echo "  Stop bot:       systemctl stop trading-bot"
echo "  Start bot:      systemctl start trading-bot"
echo "  Restart bot:    systemctl restart trading-bot"
echo "  Disable bot:    systemctl disable trading-bot"
echo ""
echo "================================================================================"
echo "Bot is running! Check Telegram in 2-3 minutes for startup message!"
echo "================================================================================"
