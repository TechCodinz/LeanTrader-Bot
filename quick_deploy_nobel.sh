#!/bin/bash

# Quick Nobel Prize Hedge Fund System Deployment
# =============================================

set -e

echo "🏆 NOBEL PRIZE HEDGE FUND SYSTEM - QUICK DEPLOYMENT"
echo "===================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

# Check Python version
print_header "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
print_status "Python $python_version detected"

# Install essential packages
print_header "Installing essential packages..."
pip3 install --user numpy pandas scikit-learn ccxt python-telegram-bot asyncio websockets aiohttp requests

# Create directories
print_header "Creating system directories..."
mkdir -p data models logs backups configs
mkdir -p data/market_data data/signals data/positions data/performance
mkdir -p models/ai models/technical models/sentiment
mkdir -p logs/trading logs/errors logs/performance
print_success "Directories created"

# Create configuration
print_header "Creating configuration..."
cat > configs/nobel_config.json << 'EOF'
{
    "system": {
        "name": "Nobel Hedge Fund System",
        "version": "1.0.0",
        "environment": "production",
        "log_level": "INFO"
    },
    "exchanges": {
        "bybit": {
            "api_key": "g1mhPqKrOBp9rnqb4G",
            "secret": "s9KCIelCqPwJOOWAXNoWqFHtiauRQr9PLeqG",
            "sandbox": true,
            "testnet": true,
            "enabled": true
        }
    },
    "telegram": {
        "bot_token": "8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg",
        "channels": {
            "admin": "5329503447",
            "free": "-1002930953007",
            "vip": "-1002983007302"
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
        "stop_loss_multiplier": 2.0,
        "trailing_stop_percent": 0.02
    },
    "ai": {
        "retrain_interval": 3600,
        "feature_window": 100,
        "prediction_horizon": 10,
        "ensemble_weights": [0.3, 0.3, 0.4]
    },
    "risk": {
        "max_drawdown": 0.15,
        "var_confidence": 0.95,
        "correlation_threshold": 0.7,
        "volatility_threshold": 0.05
    }
}
EOF
print_success "Configuration created"

# Create startup script
print_header "Creating startup script..."
cat > start_nobel_quick.sh << 'EOF'
#!/bin/bash

echo "🏆 Starting Nobel Prize Hedge Fund System..."

# Check if main file exists
if [ ! -f "nobel_hedge_fund_system.py" ]; then
    echo "❌ Nobel Hedge Fund System file not found!"
    exit 1
fi

# Start the system
echo "🚀 Starting Nobel Hedge Fund System..."
python3 nobel_hedge_fund_system.py &
NOBEL_PID=$!

# Save PID
echo $NOBEL_PID > nobel_system.pid

echo "✅ Nobel Hedge Fund System started (PID: $NOBEL_PID)"
echo "📊 System is now running and monitoring markets..."
echo "🛑 To stop: kill $NOBEL_PID"
EOF

chmod +x start_nobel_quick.sh
print_success "Startup script created"

# Create stop script
print_header "Creating stop script..."
cat > stop_nobel_quick.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping Nobel Prize Hedge Fund System..."

if [ -f "nobel_system.pid" ]; then
    NOBEL_PID=$(cat nobel_system.pid)
    if ps -p $NOBEL_PID > /dev/null; then
        echo "🛑 Stopping Nobel system (PID: $NOBEL_PID)..."
        kill $NOBEL_PID
        sleep 2
        
        if ps -p $NOBEL_PID > /dev/null; then
            echo "⚠️ Force killing Nobel system..."
            kill -9 $NOBEL_PID
        fi
        
        echo "✅ Nobel Hedge Fund System stopped"
    else
        echo "⚠️ Nobel system not running"
    fi
    rm -f nobel_system.pid
else
    echo "⚠️ No PID file found"
fi
EOF

chmod +x stop_nobel_quick.sh
print_success "Stop script created"

# Create status script
print_header "Creating status script..."
cat > status_nobel_quick.sh << 'EOF'
#!/bin/bash

echo "📊 Nobel Prize Hedge Fund System Status"
echo "======================================="

if [ -f "nobel_system.pid" ]; then
    NOBEL_PID=$(cat nobel_system.pid)
    if ps -p $NOBEL_PID > /dev/null; then
        echo "✅ Nobel Hedge Fund System: RUNNING (PID: $NOBEL_PID)"
    else
        echo "❌ Nobel Hedge Fund System: NOT RUNNING (stale PID file)"
    fi
else
    echo "❌ Nobel Hedge Fund System: NOT RUNNING (no PID file)"
fi

echo "💻 System Resources:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"

if [ -f "nobel_hedge_fund.log" ]; then
    echo ""
    echo "📋 Recent Log Activity:"
    tail -5 nobel_hedge_fund.log
fi
EOF

chmod +x status_nobel_quick.sh
print_success "Status script created"

# Test the system
print_header "Testing system components..."
python3 -c "
import sys
import importlib

required_modules = ['numpy', 'pandas', 'ccxt', 'sklearn', 'telegram', 'asyncio']

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
    print_success "System test passed"
else
    print_error "System test failed"
    exit 1
fi

# Final summary
print_header "DEPLOYMENT SUMMARY"
echo "===================="
echo "✅ Essential packages installed"
echo "✅ System directories created"
echo "✅ Configuration files created"
echo "✅ Management scripts created"
echo ""
echo "🏆 NOBEL PRIZE HEDGE FUND SYSTEM IS READY!"
echo ""
echo "To start the system:"
echo "  ./start_nobel_quick.sh"
echo ""
echo "To check status:"
echo "  ./status_nobel_quick.sh"
echo ""
echo "To stop the system:"
echo "  ./stop_nobel_quick.sh"
echo ""
echo "Configuration: configs/nobel_config.json"
echo "Logs: nobel_hedge_fund.log"
echo ""
print_success "Quick deployment completed successfully!"