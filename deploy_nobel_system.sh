#!/bin/bash

# Nobel Prize Hedge Fund Trading System Deployment Script
# ======================================================

set -e

echo "🏆 NOBEL PRIZE HEDGE FUND SYSTEM DEPLOYMENT"
echo "============================================="

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
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[HEADER]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check Python version
print_header "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_success "Python $python_version is compatible"
else
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
print_header "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_header "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_header "Upgrading pip..."
pip install --upgrade pip
print_success "Pip upgraded"

# Install system dependencies
print_header "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    libpq-dev \
    libhdf5-dev \
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    htop \
    tree \
    jq

print_success "System dependencies installed"

# Install TA-Lib
print_header "Installing TA-Lib..."
if ! python3 -c "import talib" 2>/dev/null; then
    cd /tmp
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    sudo ldconfig
    cd /workspace
    print_success "TA-Lib installed"
else
    print_warning "TA-Lib already installed"
fi

# Install Python dependencies
print_header "Installing Python dependencies..."
pip install -r nobel_requirements.txt
print_success "Python dependencies installed"

# Create necessary directories
print_header "Creating system directories..."
mkdir -p data models logs backups configs
mkdir -p data/market_data data/signals data/positions data/performance
mkdir -p models/ai models/technical models/sentiment
mkdir -p logs/trading logs/errors logs/performance
mkdir -p backups/daily backups/weekly backups/monthly
print_success "Directories created"

# Create configuration files
print_header "Creating configuration files..."

# Main config
cat > configs/nobel_config.json << EOF
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
        },
        "binance": {
            "api_key": "",
            "secret": "",
            "sandbox": true,
            "enabled": false
        },
        "okx": {
            "api_key": "",
            "secret": "",
            "sandbox": true,
            "enabled": false
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
        "ensemble_weights": [0.3, 0.3, 0.4],
        "models": {
            "price_predictor": "RandomForestRegressor",
            "volatility_predictor": "GradientBoostingRegressor",
            "sentiment_predictor": "MLPRegressor"
        }
    },
    "risk": {
        "max_drawdown": 0.15,
        "var_confidence": 0.95,
        "correlation_threshold": 0.7,
        "volatility_threshold": 0.05,
        "position_sizing": "kelly_criterion"
    },
    "monitoring": {
        "performance_update_interval": 30,
        "health_check_interval": 60,
        "alert_thresholds": {
            "max_drawdown": 0.10,
            "min_win_rate": 0.60,
            "max_daily_loss": 0.05
        }
    }
}
EOF

# Environment file
cat > .env << EOF
# Nobel Hedge Fund System Environment Variables
NOBEL_ENV=production
NOBEL_LOG_LEVEL=INFO
NOBEL_DEBUG=false
NOBEL_DATABASE_URL=sqlite:///nobel_hedge_fund.db
NOBEL_REDIS_URL=redis://localhost:6379
NOBEL_TELEGRAM_TOKEN=8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg
NOBEL_BYBIT_API_KEY=g1mhPqKrOBp9rnqb4G
NOBEL_BYBIT_SECRET=s9KCIelCqPwJOOWAXNoWqFHtiauRQr9PLeqG
NOBEL_BYBIT_TESTNET=true
EOF

print_success "Configuration files created"

# Create systemd service
print_header "Creating systemd service..."
sudo tee /etc/systemd/system/nobel-hedge-fund.service > /dev/null << EOF
[Unit]
Description=Nobel Prize Hedge Fund Trading System
After=network.target
Wants=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/workspace
Environment=PATH=/workspace/venv/bin
ExecStart=/workspace/venv/bin/python3 /workspace/nobel_hedge_fund_system.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=nobel-hedge-fund

[Install]
WantedBy=multi-user.target
EOF

print_success "Systemd service created"

# Create monitoring script
print_header "Creating monitoring script..."
cat > monitor_nobel_system.py << 'EOF'
#!/usr/bin/env python3
"""
Nobel Hedge Fund System Monitor
===============================
"""

import psutil
import time
import logging
import subprocess
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_system_health():
    """Check system health metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Check if Nobel system is running
        nobel_running = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'nobel_hedge_fund_system.py' in ' '.join(proc.info['cmdline']):
                    nobel_running = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Log health status
        logger.info(f"System Health - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%, Nobel Running: {nobel_running}")
        
        # Alert if issues
        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent}%")
        
        if memory_percent > 80:
            logger.warning(f"High memory usage: {memory_percent}%")
        
        if disk_percent > 90:
            logger.warning(f"High disk usage: {disk_percent}%")
        
        if not nobel_running:
            logger.error("Nobel Hedge Fund System is not running!")
            # Try to restart
            try:
                subprocess.run(['sudo', 'systemctl', 'restart', 'nobel-hedge-fund'], check=True)
                logger.info("Attempted to restart Nobel system")
            except subprocess.CalledProcessError:
                logger.error("Failed to restart Nobel system")
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent,
            'nobel_running': nobel_running,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return None

def main():
    """Main monitoring loop"""
    logger.info("Starting Nobel Hedge Fund System Monitor")
    
    while True:
        try:
            health = check_system_health()
            if health:
                # Save health data
                with open('logs/system_health.json', 'a') as f:
                    f.write(json.dumps(health) + '\n')
            
            time.sleep(60)  # Check every minute
            
        except KeyboardInterrupt:
            logger.info("Monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
EOF

chmod +x monitor_nobel_system.py
print_success "Monitoring script created"

# Create backup script
print_header "Creating backup script..."
cat > backup_nobel_system.py << 'EOF'
#!/usr/bin/env python3
"""
Nobel Hedge Fund System Backup
==============================
"""

import os
import shutil
import sqlite3
import json
import gzip
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_database():
    """Backup the database"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"backups/daily/nobel_hedge_fund_{timestamp}.db"
        
        # Create backup directory if it doesn't exist
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)
        
        # Copy database
        shutil.copy2('nobel_hedge_fund.db', backup_file)
        
        # Compress backup
        with open(backup_file, 'rb') as f_in:
            with gzip.open(f"{backup_file}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove uncompressed file
        os.remove(backup_file)
        
        logger.info(f"Database backed up to {backup_file}.gz")
        return f"{backup_file}.gz"
        
    except Exception as e:
        logger.error(f"Database backup error: {e}")
        return None

def backup_logs():
    """Backup log files"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = f"backups/daily/logs_{timestamp}"
        
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy log files
        log_files = ['nobel_hedge_fund.log', 'logs/live.log', 'logs/paper_run.log']
        for log_file in log_files:
            if os.path.exists(log_file):
                shutil.copy2(log_file, backup_dir)
        
        # Create tar.gz archive
        shutil.make_archive(backup_dir, 'gztar', backup_dir)
        shutil.rmtree(backup_dir)
        
        logger.info(f"Logs backed up to {backup_dir}.tar.gz")
        return f"{backup_dir}.tar.gz"
        
    except Exception as e:
        logger.error(f"Log backup error: {e}")
        return None

def cleanup_old_backups():
    """Clean up old backup files"""
    try:
        # Clean up daily backups older than 7 days
        daily_backup_dir = "backups/daily"
        if os.path.exists(daily_backup_dir):
            for file in os.listdir(daily_backup_dir):
                file_path = os.path.join(daily_backup_dir, file)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if datetime.now() - file_time > timedelta(days=7):
                        os.remove(file_path)
                        logger.info(f"Removed old backup: {file}")
        
        # Clean up weekly backups older than 4 weeks
        weekly_backup_dir = "backups/weekly"
        if os.path.exists(weekly_backup_dir):
            for file in os.listdir(weekly_backup_dir):
                file_path = os.path.join(weekly_backup_dir, file)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if datetime.now() - file_time > timedelta(weeks=4):
                        os.remove(file_path)
                        logger.info(f"Removed old weekly backup: {file}")
        
        logger.info("Old backups cleaned up")
        
    except Exception as e:
        logger.error(f"Backup cleanup error: {e}")

def main():
    """Main backup function"""
    logger.info("Starting Nobel Hedge Fund System Backup")
    
    try:
        # Backup database
        db_backup = backup_database()
        
        # Backup logs
        log_backup = backup_logs()
        
        # Clean up old backups
        cleanup_old_backups()
        
        logger.info("Backup completed successfully")
        
    except Exception as e:
        logger.error(f"Backup error: {e}")

if __name__ == "__main__":
    main()
EOF

chmod +x backup_nobel_system.py
print_success "Backup script created"

# Create startup script
print_header "Creating startup script..."
cat > start_nobel_system.sh << 'EOF'
#!/bin/bash

# Nobel Hedge Fund System Startup Script
# =====================================

set -e

echo "🏆 Starting Nobel Prize Hedge Fund System..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Please run deploy_nobel_system.sh first."
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Check if Nobel system file exists
if [ ! -f "nobel_hedge_fund_system.py" ]; then
    print_error "Nobel Hedge Fund System file not found!"
    exit 1
fi

# Check system health
print_status "Checking system health..."
python3 monitor_nobel_system.py &
MONITOR_PID=$!

# Start Nobel system
print_status "Starting Nobel Hedge Fund System..."
python3 nobel_hedge_fund_system.py &
NOBEL_PID=$!

# Wait for Nobel system to start
sleep 5

# Check if Nobel system is running
if ps -p $NOBEL_PID > /dev/null; then
    print_status "Nobel Hedge Fund System started successfully (PID: $NOBEL_PID)"
    print_status "Monitor started (PID: $MONITOR_PID)"
    print_status "System is now running and monitoring markets..."
    
    # Save PIDs
    echo $NOBEL_PID > nobel_system.pid
    echo $MONITOR_PID > monitor.pid
    
    print_status "PIDs saved to nobel_system.pid and monitor.pid"
    print_status "To stop the system, run: ./stop_nobel_system.sh"
else
    print_error "Failed to start Nobel Hedge Fund System"
    kill $MONITOR_PID 2>/dev/null || true
    exit 1
fi
EOF

chmod +x start_nobel_system.sh
print_success "Startup script created"

# Create stop script
print_header "Creating stop script..."
cat > stop_nobel_system.sh << 'EOF'
#!/bin/bash

# Nobel Hedge Fund System Stop Script
# ===================================

echo "🛑 Stopping Nobel Prize Hedge Fund System..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Stop Nobel system
if [ -f "nobel_system.pid" ]; then
    NOBEL_PID=$(cat nobel_system.pid)
    if ps -p $NOBEL_PID > /dev/null; then
        print_status "Stopping Nobel Hedge Fund System (PID: $NOBEL_PID)..."
        kill $NOBEL_PID
        sleep 2
        
        # Force kill if still running
        if ps -p $NOBEL_PID > /dev/null; then
            print_warning "Force killing Nobel system..."
            kill -9 $NOBEL_PID
        fi
        
        print_status "Nobel Hedge Fund System stopped"
    else
        print_warning "Nobel system not running"
    fi
    rm -f nobel_system.pid
else
    print_warning "No Nobel system PID file found"
fi

# Stop monitor
if [ -f "monitor.pid" ]; then
    MONITOR_PID=$(cat monitor.pid)
    if ps -p $MONITOR_PID > /dev/null; then
        print_status "Stopping monitor (PID: $MONITOR_PID)..."
        kill $MONITOR_PID
        print_status "Monitor stopped"
    else
        print_warning "Monitor not running"
    fi
    rm -f monitor.pid
else
    print_warning "No monitor PID file found"
fi

# Stop systemd service if running
if systemctl is-active --quiet nobel-hedge-fund; then
    print_status "Stopping systemd service..."
    sudo systemctl stop nobel-hedge-fund
    print_status "Systemd service stopped"
fi

print_status "Nobel Hedge Fund System stopped successfully"
EOF

chmod +x stop_nobel_system.sh
print_success "Stop script created"

# Create status script
print_header "Creating status script..."
cat > status_nobel_system.sh << 'EOF'
#!/bin/bash

# Nobel Hedge Fund System Status Script
# =====================================

echo "📊 Nobel Prize Hedge Fund System Status"
echo "======================================="

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

# Check if Nobel system is running
print_header "System Status"
if [ -f "nobel_system.pid" ]; then
    NOBEL_PID=$(cat nobel_system.pid)
    if ps -p $NOBEL_PID > /dev/null; then
        print_status "Nobel Hedge Fund System: RUNNING (PID: $NOBEL_PID)"
    else
        print_error "Nobel Hedge Fund System: NOT RUNNING (stale PID file)"
    fi
else
    print_error "Nobel Hedge Fund System: NOT RUNNING (no PID file)"
fi

# Check monitor
if [ -f "monitor.pid" ]; then
    MONITOR_PID=$(cat monitor.pid)
    if ps -p $MONITOR_PID > /dev/null; then
        print_status "Monitor: RUNNING (PID: $MONITOR_PID)"
    else
        print_warning "Monitor: NOT RUNNING"
    fi
else
    print_warning "Monitor: NOT RUNNING"
fi

# Check systemd service
print_header "Systemd Service Status"
if systemctl is-active --quiet nobel-hedge-fund; then
    print_status "Systemd service: ACTIVE"
else
    print_warning "Systemd service: INACTIVE"
fi

# Check system resources
print_header "System Resources"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"

# Check log files
print_header "Recent Log Activity"
if [ -f "nobel_hedge_fund.log" ]; then
    echo "Last 5 log entries:"
    tail -5 nobel_hedge_fund.log
else
    print_warning "No log file found"
fi

# Check database
print_header "Database Status"
if [ -f "nobel_hedge_fund.db" ]; then
    print_status "Database: EXISTS"
    echo "Database size: $(du -h nobel_hedge_fund.db | cut -f1)"
else
    print_error "Database: NOT FOUND"
fi

# Check configuration
print_header "Configuration Status"
if [ -f "configs/nobel_config.json" ]; then
    print_status "Configuration: EXISTS"
else
    print_error "Configuration: NOT FOUND"
fi

print_header "Status Check Complete"
EOF

chmod +x status_nobel_system.sh
print_success "Status script created"

# Set up cron jobs
print_header "Setting up cron jobs..."
(crontab -l 2>/dev/null; echo "0 0 * * * /workspace/backup_nobel_system.py") | crontab -
(crontab -l 2>/dev/null; echo "0 2 * * 0 /workspace/backup_nobel_system.py && tar -czf /workspace/backups/weekly/nobel_system_$(date +\%Y\%m\%d).tar.gz /workspace/data /workspace/models /workspace/logs") | crontab -
print_success "Cron jobs set up"

# Reload systemd
print_header "Reloading systemd..."
sudo systemctl daemon-reload
print_success "Systemd reloaded"

# Enable service
print_header "Enabling Nobel service..."
sudo systemctl enable nobel-hedge-fund
print_success "Service enabled"

# Final system check
print_header "Running final system check..."
python3 -c "
import sys
import importlib

required_modules = [
    'numpy', 'pandas', 'ccxt', 'sklearn', 'talib', 
    'telegram', 'asyncio', 'websockets', 'aiohttp'
]

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
echo "✅ Virtual environment created and activated"
echo "✅ All Python dependencies installed"
echo "✅ System directories created"
echo "✅ Configuration files created"
echo "✅ Systemd service configured"
echo "✅ Monitoring system set up"
echo "✅ Backup system configured"
echo "✅ Management scripts created"
echo "✅ Cron jobs scheduled"
echo ""
echo "🏆 NOBEL PRIZE HEDGE FUND SYSTEM IS READY!"
echo ""
echo "To start the system:"
echo "  ./start_nobel_system.sh"
echo ""
echo "To check status:"
echo "  ./status_nobel_system.sh"
echo ""
echo "To stop the system:"
echo "  ./stop_nobel_system.sh"
echo ""
echo "To start as service:"
echo "  sudo systemctl start nobel-hedge-fund"
echo ""
echo "To view logs:"
echo "  tail -f nobel_hedge_fund.log"
echo ""
echo "Configuration file: configs/nobel_config.json"
echo "Database: nobel_hedge_fund.db"
echo "Logs: nobel_hedge_fund.log"
echo ""
print_success "Deployment completed successfully!"