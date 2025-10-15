#!/bin/bash
# UPDATE SYSTEMD SERVICE TO USE VIRTUAL ENVIRONMENT

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           UPDATING SYSTEMD SERVICE FOR VENV                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Get paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
SERVICE_FILE="/etc/systemd/system/trading-bot.service"

# Check if venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: bash setup_venv.sh first"
    exit 1
fi

echo "✅ Found venv Python: $VENV_PYTHON"
echo ""

# Backup current service file
if [ -f "$SERVICE_FILE" ]; then
    echo "💾 Backing up current service file..."
    cp "$SERVICE_FILE" "$SERVICE_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✅ Backup created"
fi

# Create new service file
echo ""
echo "📝 Creating new service file with venv..."

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Advanced AI Trading Bot - 55+ Systems (VENV)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
Environment="PATH=$SCRIPT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="VIRTUAL_ENV=$SCRIPT_DIR/venv"
ExecStart=$VENV_PYTHON $SCRIPT_DIR/RUN_BOT.py --testnet
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=2G
CPUQuota=150%

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Service file updated"
echo ""

# Reload systemd
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload
echo "✅ Daemon reloaded"
echo ""

# Enable service
echo "🔧 Enabling service..."
systemctl enable trading-bot
echo "✅ Service enabled"
echo ""

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ SYSTEMD SERVICE UPDATED!                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Service is now configured to use:"
echo "  Python: $VENV_PYTHON"
echo "  Virtual Env: $SCRIPT_DIR/venv"
echo ""
echo "To restart the bot with new configuration:"
echo "  sudo systemctl restart trading-bot"
echo ""
echo "To check status:"
echo "  sudo systemctl status trading-bot"
echo ""
echo "To view logs:"
echo "  journalctl -u trading-bot -f"
