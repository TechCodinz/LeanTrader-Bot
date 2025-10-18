#!/bin/bash
echo "🚀 ULTRA TRADING SYSTEM - VPS DEPLOYMENT"
echo "========================================"

# Configuration
REPO_URL="https://github.com/TechCodinz/Lean-Trader.git"
REPO_DIR="/opt/leantrader"

echo "📦 Cloning Ultra Trading System..."
rm -rf $REPO_DIR
git clone $REPO_URL $REPO_DIR
cd $REPO_DIR

echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ccxt pyyaml feedparser prometheus-client requests beautifulsoup4 numpy pandas

echo "📁 Creating required directories..."
mkdir -p /var/log/leantrader
mkdir -p /opt/leantrader/inbox_signals
mkdir -p /opt/leantrader/out/meta
mkdir -p /opt/leantrader/data

echo "⚙️ Setting up systemd services..."
cat > /etc/systemd/system/leantrader.service << 'EOF'
[Unit]
Description=LeanTrader Orchestrator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/leantrader
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/opt/leantrader
EnvironmentFile=-/opt/leantrader/.env
ExecStart=/opt/leantrader/venv/bin/python /opt/leantrader/runtime/unified_runner.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/leantrader/orchestrator.log
StandardError=append:/var/log/leantrader/orchestrator.err

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/leantrader-router.service << 'EOF'
[Unit]
Description=LeanTrader Auto Env Router
After=network.target

[Service]
User=root
WorkingDirectory=/opt/leantrader
ExecStart=/opt/leantrader/venv/bin/python /opt/leantrader/tools/auto_env_router.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "🔧 Setting permissions..."
chmod +x /opt/leantrader/tools/*.py
chmod +x /opt/leantrader/scripts/*.sh
chmod +x /opt/leantrader/start_ultra_system.sh
chown -R root:root /opt/leantrader
chown -R root:root /var/log/leantrader

echo "📊 Installing additional tools..."
curl -sL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /usr/local/bin/yq
chmod +x /usr/local/bin/yq
apt-get update && apt-get install -y tmux

echo "🔄 Enabling services..."
systemctl daemon-reload
systemctl enable leantrader
systemctl enable leantrader-router

echo "📝 Creating environment configuration..."
cat > /opt/leantrader/.env << 'EOF'
# Trading Mode
ENABLE_LIVE=false
ALLOW_LIVE=false
LIVE_CONFIRM=NO

# Exchange Configuration
EXCHANGE_ID=paper
PAPER_START_CASH=5000

# Risk Management
RISK_PER_TRADE=0.02
MAX_POSITIONS=5
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.10

# Meta-Brain Settings
META_BRAIN_ENABLED=true
ENSEMBLE_LEARNING=true
PERFORMANCE_TRACKING=true

# Copy Signals
COPY_SIGNALS_ENABLED=true
SIGNALS_INBOX_DIR=/opt/leantrader/inbox_signals

# Multi-Exchange Swarm
SWARM_ENABLED=true
PARALLEL_TRAINING=true
EXCHANGE_ISOLATION=true

# Logging
LOG_LEVEL=INFO
EOF

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "======================"
echo ""
echo "📋 Next steps:"
echo "1. Configure API keys: nano /opt/leantrader/.env"
echo "2. Start the system: /opt/leantrader/start_ultra_system.sh"
echo "3. Monitor: systemctl status leantrader"
echo "4. View logs: journalctl -u leantrader -f"
echo ""
echo "🚀 Ready to make profits! 💰"
