#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# 🚀 ULTRA TRADING BOT - ONE-COMMAND VPS DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════
# Deploy the complete trading bot to a fresh Ubuntu/Debian VPS
# Requirements: Ubuntu 20.04+ or Debian 11+ with root access
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "
╔══════════════════════════════════════════════════════════════════╗
║         🚀 ULTRA TRADING BOT - VPS DEPLOYMENT                    ║
║                                                                  ║
║  Deploying 28 Ultra AI Systems:                                 ║
║  • Quantum Computing Engine                                     ║
║  • ULTRASONIC PhD Strategies                                    ║
║  • GOLDMINE Features                                            ║
║  • DIVINE Intelligence                                          ║
║  • 16 Background Trading Loops                                  ║
║  • 8 Passive Utility Systems                                    ║
╚══════════════════════════════════════════════════════════════════╝
"

# ═══════════════════════════════════════════════════════════════════
# STEP 1: System Preparation
# ═══════════════════════════════════════════════════════════════════
echo "📦 Step 1/6: Installing system dependencies..."

apt-get update -qq
apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

echo "✅ System dependencies installed"

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Clone Repository
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "📥 Step 2/6: Cloning trading bot repository..."

cd /root
if [ -d "trading_bot" ]; then
    echo "⚠️  trading_bot directory exists. Backing up..."
    mv trading_bot trading_bot_backup_$(date +%s)
fi

git clone https://github.com/TechCodinz/Lean-Trader.git trading_bot
cd trading_bot

echo "✅ Repository cloned"

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Python Virtual Environment
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "🐍 Step 3/6: Setting up Python virtual environment..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "✅ Python environment ready"

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Environment Configuration
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "⚙️  Step 4/6: Setting up environment configuration..."

if [ ! -f ".env" ]; then
    cat > .env << 'ENVFILE'
# ═══════════════════════════════════════════════════════════════════
# ULTRA TRADING BOT - ENVIRONMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Gate.io API (REQUIRED)
GATEIO_MODE=live                    # 'live' or 'testnet'
GATEIO_LIVE_API_KEY=your_live_api_key_here
GATEIO_LIVE_SECRET=your_live_secret_here
GATEIO_TESTNET_API_KEY=your_testnet_api_key_here
GATEIO_TESTNET_SECRET=your_testnet_secret_here

# Telegram Notifications (OPTIONAL)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_ADMIN_ID=your_telegram_user_id_here

# Additional Exchanges (OPTIONAL)
BINANCE_API_KEY=
BINANCE_SECRET=
BYBIT_API_KEY=
BYBIT_SECRET=
MEXC_API_KEY=
MEXC_SECRET=

# Risk Management (OPTIONAL - has defaults)
MAX_POSITION_SIZE=100.0
MAX_DAILY_LOSS=0.10
EMERGENCY_STOP_ENABLED=true

ENVFILE
    echo "⚠️  .env file created. IMPORTANT: Edit /root/trading_bot/.env with your API keys!"
    echo ""
    echo "Minimum required:"
    echo "  - GATEIO_LIVE_API_KEY"
    echo "  - GATEIO_LIVE_SECRET"
else
    echo "✅ .env file already exists"
fi

echo "✅ Configuration ready"

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Systemd Service Setup
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "🔧 Step 5/6: Creating systemd services..."

# Live trading service
cat > /etc/systemd/system/trading-bot-live.service << 'LIVESERVICE'
[Unit]
Description=Trading Bot - LIVE (Real Trading)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/trading_bot
Environment="PATH=/root/trading_bot/venv/bin"
ExecStart=/root/trading_bot/venv/bin/python3 /root/trading_bot/RUN_BOT.py --live --auto-confirm
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
MemoryMax=2G

[Install]
WantedBy=multi-user.target
LIVESERVICE

# Testnet trading service
cat > /etc/systemd/system/trading-bot-testnet.service << 'TESTSERVICE'
[Unit]
Description=Trading Bot - TESTNET (Learning Mode)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/trading_bot
Environment="PATH=/root/trading_bot/venv/bin"
ExecStart=/root/trading_bot/venv/bin/python3 /root/trading_bot/RUN_BOT.py --testnet --auto-confirm
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
MemoryMax=2G

[Install]
WantedBy=multi-user.target
TESTSERVICE

systemctl daemon-reload

echo "✅ Systemd services created"

# ═══════════════════════════════════════════════════════════════════
# STEP 6: Final Setup
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "🎯 Step 6/6: Final configuration..."

# Git config for easy updates
git config --global pull.rebase false

echo "✅ Deployment complete!"

echo "
╔══════════════════════════════════════════════════════════════════╗
║                   ✅ DEPLOYMENT COMPLETE!                        ║
╚══════════════════════════════════════════════════════════════════╝

📝 NEXT STEPS:

1. Configure your API keys:
   nano /root/trading_bot/.env

2. Start the bot:
   sudo systemctl start trading-bot-live

3. Check status:
   sudo systemctl status trading-bot-live

4. View live logs:
   sudo journalctl -u trading-bot-live -f

═══════════════════════════════════════════════════════════════════

📚 FULL DOCUMENTATION:
   cat /root/trading_bot/BOT_OVERVIEW.md
   cat /root/trading_bot/MANAGEMENT_COMMANDS.md

═══════════════════════════════════════════════════════════════════
"
