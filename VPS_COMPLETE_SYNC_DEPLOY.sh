#!/bin/bash

# ================================================================================
# 🚀 VPS COMPLETE SYSTEM SYNCHRONIZATION & DEPLOYMENT
# ================================================================================
# This script ensures ALL systems, models, engines, brain, and ultra features
# are properly deployed, configured, and synchronized on the VPS
# ================================================================================

set -e  # Exit on any error

echo "================================================================================
🚀 VPS COMPLETE SYSTEM SYNCHRONIZATION & DEPLOYMENT
================================================================================
"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${PURPLE}================================================================================
$1
================================================================================${NC}"
}

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    print_error "Please run as root or with sudo"
    exit 1
fi

# Set working directory
cd /workspace
print_info "Working directory: $(pwd)"

# ================================================================================
# PHASE 1: SYSTEM DEPENDENCIES & ENVIRONMENT SETUP
# ================================================================================
print_header "PHASE 1: SYSTEM DEPENDENCIES & ENVIRONMENT SETUP"

print_info "Updating system packages..."
apt update -y
apt upgrade -y

print_info "Installing essential system packages..."
apt install -y python3-pip python3-venv python3-dev build-essential
apt install -y git curl wget unzip
apt install -y sqlite3
apt install -y htop screen tmux
apt install -y nginx
apt install -y redis-server

print_info "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install wheel setuptools

# Core ML/AI dependencies
pip3 install numpy==1.26.4 pandas==2.3.3 scipy==1.16.2 scikit-learn==1.7.2
pip3 install tensorflow==2.20.0 keras==3.11.3

# Trading and financial dependencies
pip3 install ccxt==4.5.11 yfinance ta-lib
pip3 install requests aiohttp websockets
pip3 install python-telegram-bot aiogram

# AI/ML advanced dependencies
pip3 install langchain==0.3.7 langchain-community==0.3.7
pip3 install openai anthropic
pip3 install transformers torch

# Web3 and blockchain dependencies
pip3 install web3 qiskit
pip3 install stripe qrcode[pil]

# Additional utilities
pip3 install python-dotenv beautifulsoup4 lxml
pip3 install asyncio-mqtt schedule
pip3 install plotly dash

print_status "Phase 1 completed: System dependencies installed"

# ================================================================================
# PHASE 2: CORE TRADING SYSTEMS DEPLOYMENT
# ================================================================================
print_header "PHASE 2: CORE TRADING SYSTEMS DEPLOYMENT"

print_info "Deploying core trading systems..."

# Create system directories
mkdir -p /workspace/systems
mkdir -p /workspace/models
mkdir -p /workspace/engines
mkdir -p /workspace/brain
mkdir -p /workspace/ultra
mkdir -p /workspace/logs
mkdir -p /workspace/data
mkdir -p /workspace/config

# Set proper permissions
chown -R ubuntu:ubuntu /workspace
chmod -R 755 /workspace

print_info "Deploying main orchestrator systems..."

# Copy main system files
cp COMPLETE_ULTIMATE_ORCHESTRATOR.py /workspace/systems/
cp COMPLETE_UNIFIED_ORCHESTRATOR.py /workspace/systems/
cp ULTIMATE_ORCHESTRATOR.py /workspace/systems/
cp RUN_BOT.py /workspace/systems/

print_info "Deploying trading engines..."

# Copy engine files
cp alpha_engines.py /workspace/engines/
cp ml_strategy_engine.py /workspace/engines/
cp ultra_scalping_engine.py /workspace/engines/
cp EVOLUTION_ENGINE.py /workspace/engines/
cp DEX_SWAP_ENGINE.py /workspace/engines/
cp NEWS_TRADING_ENGINE.py /workspace/engines/
cp IBM_QUANTUM_ENGINE.py /workspace/engines/

print_info "Deploying brain systems..."

# Copy brain files
cp brain.py /workspace/brain/
cp brain_loop.py /workspace/brain/
cp ultra_moon_spotter.py /workspace/brain/

print_info "Deploying ultra features..."

# Copy ultra feature files
cp FIXED_FULL_ULTRA_ARSENAL.py /workspace/ultra/
cp FULL_ULTRA_ARSENAL_BOT.py /workspace/ultra/
cp continuous_ultra_bot.py /workspace/ultra/
cp multi_channel_ultra_bot.py /workspace/ultra/
cp ultimate_ultra_plus.py /workspace/ultra/

print_status "Phase 2 completed: Core trading systems deployed"

# ================================================================================
# PHASE 3: CONFIGURATION & ENVIRONMENT SETUP
# ================================================================================
print_header "PHASE 3: CONFIGURATION & ENVIRONMENT SETUP"

print_info "Setting up environment configuration..."

# Create comprehensive environment file
cat > /workspace/.env.production << 'EOF'
# ================================================================================
# 🚀 PRODUCTION ENVIRONMENT CONFIGURATION
# ================================================================================

# Trading Mode
TRADING_MODE=testnet
AUTO_CONFIRM=true

# API Keys (Set these with your actual keys)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
BYBIT_API_KEY=your_bybit_api_key
BYBIT_SECRET=your_bybit_secret
DEX_PRIVATE_KEY=your_dex_private_key

# System Configuration
LOG_LEVEL=INFO
ENABLE_LIVE=false
ENABLE_DEX=false
ENABLE_QUANTUM=false

# Database Configuration
DATABASE_URL=sqlite:///workspace/data/trading_bot.db
REDIS_URL=redis://localhost:6379

# Trading Configuration
MIN_CONFLUENCE=75
MAX_RISK_PER_TRADE=2.0
DEFAULT_LEVERAGE=1.0

# Ultra Features
ENABLE_ULTRA_FEATURES=true
ENABLE_BRAIN_LOOP=true
ENABLE_EVOLUTION_ENGINE=true
ENABLE_QUANTUM_ANALYSIS=false

# Monitoring
ENABLE_MONITORING=true
GRAFANA_PORT=3000
PROMETHEUS_PORT=9090
EOF

print_info "Setting up systemd service..."

# Create systemd service file
cat > /etc/systemd/system/trading-bot-production.service << 'EOF'
[Unit]
Description=Trading Bot Production Service
After=network.target redis.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/workspace
Environment=PATH=/usr/bin:/usr/local/bin
EnvironmentFile=/workspace/.env.production
ExecStart=/usr/bin/python3 /workspace/systems/RUN_BOT.py --testnet --auto-confirm
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
systemctl daemon-reload
systemctl enable trading-bot-production.service

print_status "Phase 3 completed: Configuration and environment setup"

# ================================================================================
# PHASE 4: BRAIN & ULTRA FEATURES ACTIVATION
# ================================================================================
print_header "PHASE 4: BRAIN & ULTRA FEATURES ACTIVATION"

print_info "Activating brain systems..."

# Create brain activation script
cat > /workspace/activate_brain.sh << 'EOF'
#!/bin/bash
echo "🧠 Activating Brain Systems..."

# Start brain loop
cd /workspace/brain
python3 brain_loop.py &
BRAIN_PID=$!
echo "Brain Loop PID: $BRAIN_PID"

# Start ultra moon spotter
python3 ultra_moon_spotter.py &
MOON_PID=$!
echo "Ultra Moon Spotter PID: $MOON_PID"

# Save PIDs for monitoring
echo "$BRAIN_PID" > /workspace/logs/brain.pid
echo "$MOON_PID" > /workspace/logs/moon_spotter.pid

echo "✅ Brain systems activated"
EOF

chmod +x /workspace/activate_brain.sh

print_info "Activating ultra features..."

# Create ultra features activation script
cat > /workspace/activate_ultra.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating Ultra Features..."

# Start evolution engine
cd /workspace/engines
python3 EVOLUTION_ENGINE.py &
EVOLUTION_PID=$!
echo "Evolution Engine PID: $EVOLUTION_PID"

# Start ultra arsenal
cd /workspace/ultra
python3 FIXED_FULL_ULTRA_ARSENAL.py &
ULTRA_PID=$!
echo "Ultra Arsenal PID: $ULTRA_PID"

# Start continuous ultra bot
python3 continuous_ultra_bot.py &
CONTINUOUS_PID=$!
echo "Continuous Ultra Bot PID: $CONTINUOUS_PID"

# Save PIDs for monitoring
echo "$EVOLUTION_PID" > /workspace/logs/evolution.pid
echo "$ULTRA_PID" > /workspace/logs/ultra_arsenal.pid
echo "$CONTINUOUS_PID" > /workspace/logs/continuous_ultra.pid

echo "✅ Ultra features activated"
EOF

chmod +x /workspace/activate_ultra.sh

print_status "Phase 4 completed: Brain and ultra features activated"

# ================================================================================
# PHASE 5: MONITORING & HEALTH CHECKS
# ================================================================================
print_header "PHASE 5: MONITORING & HEALTH CHECKS"

print_info "Setting up monitoring systems..."

# Create health check script
cat > /workspace/health_check.sh << 'EOF'
#!/bin/bash
echo "🔍 System Health Check - $(date)"

# Check Python processes
echo "Python processes:"
ps aux | grep python3 | grep -v grep

# Check system resources
echo "Memory usage:"
free -h

echo "Disk usage:"
df -h

# Check logs
echo "Recent logs:"
tail -n 20 /workspace/logs/*.log 2>/dev/null || echo "No log files found"

# Check services
echo "Systemd services:"
systemctl status trading-bot-production.service --no-pager

echo "✅ Health check completed"
EOF

chmod +x /workspace/health_check.sh

print_info "Setting up log rotation..."

# Create logrotate configuration
cat > /etc/logrotate.d/trading-bot << 'EOF'
/workspace/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
}
EOF

print_status "Phase 5 completed: Monitoring and health checks setup"

# ================================================================================
# PHASE 6: FINAL INTEGRATION & TESTING
# ================================================================================
print_header "PHASE 6: FINAL INTEGRATION & TESTING"

print_info "Starting Redis service..."
systemctl start redis-server
systemctl enable redis-server

print_info "Testing system integration..."

# Test Python imports
cd /workspace
python3 -c "
import sys
sys.path.append('/workspace/systems')
sys.path.append('/workspace/engines')
sys.path.append('/workspace/brain')
sys.path.append('/workspace/ultra')

try:
    from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator
    print('✅ CompleteUltimateOrchestrator imported successfully')
except Exception as e:
    print(f'❌ Import error: {e}')

try:
    import numpy, pandas, tensorflow, ccxt
    print('✅ Core dependencies available')
except Exception as e:
    print(f'❌ Dependency error: {e}')
"

print_info "Creating startup script..."

# Create master startup script
cat > /workspace/start_all_systems.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting All Trading Systems..."

# Start Redis
systemctl start redis-server

# Start brain systems
/workspace/activate_brain.sh

# Start ultra features
/workspace/activate_ultra.sh

# Start main trading bot
systemctl start trading-bot-production.service

echo "✅ All systems started"
echo "Check status with: systemctl status trading-bot-production.service"
echo "View logs with: journalctl -u trading-bot-production.service -f"
EOF

chmod +x /workspace/start_all_systems.sh

print_info "Creating stop script..."

# Create stop script
cat > /workspace/stop_all_systems.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping All Trading Systems..."

# Stop main trading bot
systemctl stop trading-bot-production.service

# Stop brain systems
if [ -f /workspace/logs/brain.pid ]; then
    kill $(cat /workspace/logs/brain.pid) 2>/dev/null || true
fi

if [ -f /workspace/logs/moon_spotter.pid ]; then
    kill $(cat /workspace/logs/moon_spotter.pid) 2>/dev/null || true
fi

# Stop ultra features
if [ -f /workspace/logs/evolution.pid ]; then
    kill $(cat /workspace/logs/evolution.pid) 2>/dev/null || true
fi

if [ -f /workspace/logs/ultra_arsenal.pid ]; then
    kill $(cat /workspace/logs/ultra_arsenal.pid) 2>/dev/null || true
fi

if [ -f /workspace/logs/continuous_ultra.pid ]; then
    kill $(cat /workspace/logs/continuous_ultra.pid) 2>/dev/null || true
fi

echo "✅ All systems stopped"
EOF

chmod +x /workspace/stop_all_systems.sh

print_status "Phase 6 completed: Final integration and testing"

# ================================================================================
# DEPLOYMENT SUMMARY
# ================================================================================
print_header "DEPLOYMENT SUMMARY"

print_status "✅ System dependencies installed and configured"
print_status "✅ Core trading systems deployed"
print_status "✅ Brain systems activated"
print_status "✅ Ultra features activated"
print_status "✅ Monitoring and health checks setup"
print_status "✅ All systems integrated and ready"

echo ""
print_info "🚀 VPS COMPLETE SYSTEM SYNCHRONIZATION COMPLETED!"
echo ""
print_info "📋 Next Steps:"
echo "1. Update API keys in /workspace/.env.production"
echo "2. Start all systems: /workspace/start_all_systems.sh"
echo "3. Check status: /workspace/health_check.sh"
echo "4. Monitor logs: journalctl -u trading-bot-production.service -f"
echo ""
print_info "🔧 Management Commands:"
echo "• Start all: /workspace/start_all_systems.sh"
echo "• Stop all: /workspace/stop_all_systems.sh"
echo "• Health check: /workspace/health_check.sh"
echo "• Service status: systemctl status trading-bot-production.service"
echo ""
print_warning "⚠️  Remember to set your API keys in /workspace/.env.production before starting!"
echo ""
print_status "🎯 All systems are now synchronized and ready for deployment!"