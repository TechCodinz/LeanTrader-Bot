#!/bin/bash

# Nobel Prize Hedge Fund System - VPS Deployment Package
# =====================================================

set -e

echo "🏆 NOBEL PRIZE HEDGE FUND SYSTEM - VPS DEPLOYMENT"
echo "=================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Update system
print_header "Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_success "System updated"

# Install essential packages
print_header "Installing essential packages..."
sudo apt install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential libssl-dev libffi-dev \
    git curl wget unzip htop tree jq \
    sqlite3 redis-server nginx \
    supervisor ufw fail2ban \
    cron logrotate \
    vim nano htop iotop nethogs \
    screen tmux

print_success "Essential packages installed"

# Create nobel user
print_header "Creating nobel user..."
sudo useradd -m -s /bin/bash nobel || true
sudo usermod -aG sudo nobel || true
print_success "Nobel user created"

# Create application directory
print_header "Setting up application directory..."
sudo mkdir -p /opt/nobel-hedge-fund
sudo chown nobel:nobel /opt/nobel-hedge-fund
sudo chmod 755 /opt/nobel-hedge-fund

# Copy application files
print_header "Copying application files..."
sudo cp -r /workspace/* /opt/nobel-hedge-fund/
sudo chown -R nobel:nobel /opt/nobel-hedge-fund
print_success "Application files copied"

# Create virtual environment
print_header "Creating Python virtual environment..."
cd /opt/nobel-hedge-fund
sudo -u nobel python3 -m venv venv
sudo -u nobel venv/bin/pip install --upgrade pip
print_success "Virtual environment created"

# Install Python dependencies
print_header "Installing Python dependencies..."
sudo -u nobel venv/bin/pip install \
    numpy pandas scikit-learn ccxt python-telegram-bot \
    asyncio websockets aiohttp requests yfinance \
    talib-binary schedule psutil \
    sqlite3 joblib pickle-mixin

print_success "Python dependencies installed"

# Create systemd service
print_header "Creating systemd service..."
sudo tee /etc/systemd/system/nobel-hedge-fund.service > /dev/null << 'EOF'
[Unit]
Description=Nobel Prize Hedge Fund Trading System
After=network.target redis.service
Wants=network.target

[Service]
Type=simple
User=nobel
Group=nobel
WorkingDirectory=/opt/nobel-hedge-fund
Environment=PATH=/opt/nobel-hedge-fund/venv/bin
ExecStart=/opt/nobel-hedge-fund/venv/bin/python3 /opt/nobel-hedge-fund/nobel_complete_system.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nobel-hedge-fund

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/nobel-hedge-fund

[Install]
WantedBy=multi-user.target
EOF

print_success "Systemd service created"

# Create configuration file
print_header "Creating configuration file..."
sudo -u nobel tee /opt/nobel-hedge-fund/configs/nobel_config.json > /dev/null << 'EOF'
{
    "system": {
        "name": "Nobel Hedge Fund System",
        "version": "2.0.0",
        "environment": "production",
        "log_level": "INFO"
    },
    "exchanges": {
        "bybit": {
            "api_key": "YOUR_BYBIT_API_KEY",
            "secret": "YOUR_BYBIT_SECRET",
            "sandbox": false,
            "testnet": false,
            "enabled": true
        },
        "binance": {
            "api_key": "YOUR_BINANCE_API_KEY",
            "secret": "YOUR_BINANCE_SECRET",
            "sandbox": false,
            "enabled": true
        },
        "okx": {
            "api_key": "YOUR_OKX_API_KEY",
            "secret": "YOUR_OKX_SECRET",
            "sandbox": false,
            "enabled": true
        }
    },
    "telegram": {
        "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
        "channels": {
            "admin": "YOUR_ADMIN_CHAT_ID",
            "free": "YOUR_FREE_CHANNEL_ID",
            "vip": "YOUR_VIP_CHANNEL_ID"
        },
        "enabled": true
    },
    "trading": {
        "max_positions": 20,
        "max_risk_per_trade": 0.02,
        "max_daily_risk": 0.10,
        "min_confidence": 0.75,
        "scalp_timeframes": ["1m", "5m", "15m"],
        "swing_timeframes": ["1h", "4h", "1d"],
        "scalp_profit_target": 0.005,
        "swing_profit_target": 0.02,
        "stop_loss_multiplier": 2.0
    },
    "ai": {
        "retrain_interval": 3600,
        "feature_window": 100,
        "prediction_horizon": 10
    },
    "risk": {
        "max_drawdown": 0.15,
        "var_confidence": 0.95,
        "correlation_threshold": 0.7,
        "volatility_threshold": 0.05
    }
}
EOF

print_success "Configuration file created"

# Create directories
print_header "Creating application directories..."
sudo -u nobel mkdir -p /opt/nobel-hedge-fund/{data,models,logs,backups}
sudo -u nobel mkdir -p /opt/nobel-hedge-fund/data/{market_data,signals,positions,performance}
sudo -u nobel mkdir -p /opt/nobel-hedge-fund/models/{ai,technical,sentiment}
sudo -u nobel mkdir -p /opt/nobel-hedge-fund/logs/{trading,errors,performance}
sudo -u nobel mkdir -p /opt/nobel-hedge-fund/backups/{daily,weekly,monthly}
print_success "Directories created"

# Create management scripts
print_header "Creating management scripts..."

# Start script
sudo -u nobel tee /opt/nobel-hedge-fund/start_nobel.sh > /dev/null << 'EOF'
#!/bin/bash

echo "🏆 Starting Nobel Prize Hedge Fund System..."

# Check if system is already running
if systemctl is-active --quiet nobel-hedge-fund; then
    echo "⚠️ System is already running"
    exit 1
fi

# Start the service
sudo systemctl start nobel-hedge-fund

# Wait for startup
sleep 5

# Check status
if systemctl is-active --quiet nobel-hedge-fund; then
    echo "✅ Nobel Hedge Fund System started successfully"
    echo "📊 System is now running and monitoring markets..."
    echo "📋 View logs: sudo journalctl -u nobel-hedge-fund -f"
    echo "🛑 Stop system: sudo systemctl stop nobel-hedge-fund"
else
    echo "❌ Failed to start Nobel Hedge Fund System"
    echo "📋 Check logs: sudo journalctl -u nobel-hedge-fund"
    exit 1
fi
EOF

# Stop script
sudo -u nobel tee /opt/nobel-hedge-fund/stop_nobel.sh > /dev/null << 'EOF'
#!/bin/bash

echo "🛑 Stopping Nobel Prize Hedge Fund System..."

# Stop the service
sudo systemctl stop nobel-hedge-fund

# Wait for shutdown
sleep 5

# Check status
if systemctl is-active --quiet nobel-hedge-fund; then
    echo "⚠️ System is still running, force stopping..."
    sudo systemctl kill nobel-hedge-fund
    sleep 2
fi

if ! systemctl is-active --quiet nobel-hedge-fund; then
    echo "✅ Nobel Hedge Fund System stopped successfully"
else
    echo "❌ Failed to stop Nobel Hedge Fund System"
    exit 1
fi
EOF

# Status script
sudo -u nobel tee /opt/nobel-hedge-fund/status_nobel.sh > /dev/null << 'EOF'
#!/bin/bash

echo "📊 Nobel Prize Hedge Fund System Status"
echo "======================================="

# System status
if systemctl is-active --quiet nobel-hedge-fund; then
    echo "✅ System Status: RUNNING"
    echo "🕐 Uptime: $(systemctl show nobel-hedge-fund --property=ActiveEnterTimestamp --value)"
else
    echo "❌ System Status: STOPPED"
fi

# Service status
echo ""
echo "🔧 Service Status:"
systemctl status nobel-hedge-fund --no-pager

# System resources
echo ""
echo "💻 System Resources:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"

# Recent logs
echo ""
echo "📋 Recent Logs:"
sudo journalctl -u nobel-hedge-fund --no-pager -n 10
EOF

# Make scripts executable
sudo chmod +x /opt/nobel-hedge-fund/start_nobel.sh
sudo chmod +x /opt/nobel-hedge-fund/stop_nobel.sh
sudo chmod +x /opt/nobel-hedge-fund/status_nobel.sh

print_success "Management scripts created"

# Create backup script
print_header "Creating backup script..."
sudo -u nobel tee /opt/nobel-hedge-fund/backup_nobel.sh > /dev/null << 'EOF'
#!/bin/bash

echo "💾 Backing up Nobel Hedge Fund System..."

BACKUP_DIR="/opt/nobel-hedge-fund/backups/daily"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="nobel_backup_${TIMESTAMP}.tar.gz"

# Create backup
cd /opt/nobel-hedge-fund
tar -czf "${BACKUP_DIR}/${BACKUP_FILE}" \
    data/ models/ logs/ configs/ \
    nobel_complete.db \
    --exclude="logs/*.log" \
    --exclude="data/market_data/*.csv"

echo "✅ Backup created: ${BACKUP_FILE}"

# Clean old backups (keep last 7 days)
find /opt/nobel-hedge-fund/backups/daily -name "nobel_backup_*.tar.gz" -mtime +7 -delete

echo "🧹 Old backups cleaned up"
EOF

sudo chmod +x /opt/nobel-hedge-fund/backup_nobel.sh
print_success "Backup script created"

# Setup cron jobs
print_header "Setting up cron jobs..."
sudo -u nobel crontab -l 2>/dev/null | grep -v "nobel" | sudo -u nobel crontab - || true
(sudo -u nobel crontab -l 2>/dev/null; echo "0 0 * * * /opt/nobel-hedge-fund/backup_nobel.sh") | sudo -u nobel crontab -
(sudo -u nobel crontab -l 2>/dev/null; echo "0 2 * * 0 /opt/nobel-hedge-fund/backup_nobel.sh && tar -czf /opt/nobel-hedge-fund/backups/weekly/nobel_weekly_$(date +\%Y\%m\%d).tar.gz /opt/nobel-hedge-fund/data /opt/nobel-hedge-fund/models") | sudo -u nobel crontab -
print_success "Cron jobs set up"

# Setup log rotation
print_header "Setting up log rotation..."
sudo tee /etc/logrotate.d/nobel-hedge-fund > /dev/null << 'EOF'
/opt/nobel-hedge-fund/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 nobel nobel
    postrotate
        systemctl reload nobel-hedge-fund > /dev/null 2>&1 || true
    endscript
}
EOF

print_success "Log rotation configured"

# Setup firewall
print_header "Configuring firewall..."
sudo ufw --force enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
print_success "Firewall configured"

# Setup fail2ban
print_header "Configuring fail2ban..."
sudo tee /etc/fail2ban/jail.d/nobel-hedge-fund.conf > /dev/null << 'EOF'
[nobel-hedge-fund]
enabled = true
port = 22
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
EOF

sudo systemctl enable fail2ban
sudo systemctl start fail2ban
print_success "Fail2ban configured"

# Setup Redis
print_header "Configuring Redis..."
sudo systemctl enable redis-server
sudo systemctl start redis-server
print_success "Redis configured"

# Reload systemd
print_header "Reloading systemd..."
sudo systemctl daemon-reload
print_success "Systemd reloaded"

# Enable service
print_header "Enabling Nobel service..."
sudo systemctl enable nobel-hedge-fund
print_success "Service enabled"

# Create monitoring script
print_header "Creating monitoring script..."
sudo -u nobel tee /opt/nobel-hedge-fund/monitor_nobel.sh > /dev/null << 'EOF'
#!/bin/bash

echo "🔍 Nobel Hedge Fund System Monitor"
echo "=================================="

# Check system status
if systemctl is-active --quiet nobel-hedge-fund; then
    echo "✅ System: RUNNING"
else
    echo "❌ System: STOPPED"
    echo "🔄 Attempting to restart..."
    sudo systemctl restart nobel-hedge-fund
    sleep 5
    if systemctl is-active --quiet nobel-hedge-fund; then
        echo "✅ System restarted successfully"
    else
        echo "❌ Failed to restart system"
    fi
fi

# Check system resources
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
DISK_USAGE=$(df -h / | awk 'NR==2{print $5}' | cut -d'%' -f1)

echo "💻 Resources: CPU ${CPU_USAGE}%, Memory ${MEMORY_USAGE}%, Disk ${DISK_USAGE}%"

# Alert if resources are high
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "⚠️ High CPU usage: ${CPU_USAGE}%"
fi

if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo "⚠️ High memory usage: ${MEMORY_USAGE}%"
fi

if [ "$DISK_USAGE" -gt 80 ]; then
    echo "⚠️ High disk usage: ${DISK_USAGE}%"
fi

# Check recent errors
ERROR_COUNT=$(sudo journalctl -u nobel-hedge-fund --since "1 hour ago" | grep -i error | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️ $ERROR_COUNT errors in the last hour"
fi

echo "✅ Monitor check completed"
EOF

sudo chmod +x /opt/nobel-hedge-fund/monitor_nobel.sh
print_success "Monitoring script created"

# Create setup completion script
print_header "Creating setup completion script..."
sudo -u nobel tee /opt/nobel-hedge-fund/setup_complete.sh > /dev/null << 'EOF'
#!/bin/bash

echo "🎉 Nobel Prize Hedge Fund System Setup Complete!"
echo "==============================================="
echo ""
echo "📋 Next Steps:"
echo "1. Edit configuration file:"
echo "   sudo nano /opt/nobel-hedge-fund/configs/nobel_config.json"
echo ""
echo "2. Add your API keys and Telegram tokens"
echo ""
echo "3. Start the system:"
echo "   /opt/nobel-hedge-fund/start_nobel.sh"
echo ""
echo "4. Check status:"
echo "   /opt/nobel-hedge-fund/status_nobel.sh"
echo ""
echo "5. View logs:"
echo "   sudo journalctl -u nobel-hedge-fund -f"
echo ""
echo "6. Stop system:"
echo "   /opt/nobel-hedge-fund/stop_nobel.sh"
echo ""
echo "📁 Application Directory: /opt/nobel-hedge-fund"
echo "📊 Logs: /opt/nobel-hedge-fund/logs/"
echo "💾 Backups: /opt/nobel-hedge-fund/backups/"
echo "⚙️ Config: /opt/nobel-hedge-fund/configs/"
echo ""
echo "🏆 Your Nobel Prize Hedge Fund System is ready!"
EOF

sudo chmod +x /opt/nobel-hedge-fund/setup_complete.sh
print_success "Setup completion script created"

# Final system check
print_header "Running final system check..."
cd /opt/nobel-hedge-fund
sudo -u nobel venv/bin/python3 -c "
import sys
import importlib

required_modules = ['numpy', 'pandas', 'ccxt', 'sklearn', 'telegram', 'asyncio', 'yfinance']

missing_modules = []
for module in required_modules:
    try:
        importlib.import_module(module)
        print(f'✅ {module}')
    except ImportError:
        missing_modules.append(module)
        print(f'❌ {module}')

if missing_modules:
    print(f'\\nMissing modules: {missing_modules}')
    sys.exit(1)
else:
    print('\\n✅ All required modules are installed')
"

if [ $? -eq 0 ]; then
    print_success "System check passed"
else
    print_error "System check failed"
    exit 1
fi

# Create final summary
print_header "DEPLOYMENT SUMMARY"
echo "===================="
echo "✅ System packages installed and updated"
echo "✅ Nobel user created with proper permissions"
echo "✅ Application files copied to /opt/nobel-hedge-fund"
echo "✅ Python virtual environment created"
echo "✅ All dependencies installed"
echo "✅ Systemd service configured"
echo "✅ Configuration file created"
echo "✅ Management scripts created"
echo "✅ Backup system configured"
echo "✅ Cron jobs scheduled"
echo "✅ Log rotation configured"
echo "✅ Firewall configured"
echo "✅ Fail2ban configured"
echo "✅ Redis configured"
echo "✅ Monitoring system set up"
echo ""
echo "🏆 NOBEL PRIZE HEDGE FUND SYSTEM IS READY FOR VPS DEPLOYMENT!"
echo ""
echo "📋 To complete setup:"
echo "1. Run: /opt/nobel-hedge-fund/setup_complete.sh"
echo "2. Edit configuration with your API keys"
echo "3. Start the system"
echo ""
echo "🚀 Your Nobel Prize Hedge Fund System is ready to dominate markets!"
print_success "VPS deployment package completed successfully!"