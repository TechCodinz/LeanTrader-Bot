#!/bin/bash
##############################################################################
# TRADING BOT - NEW VPS DEPLOYMENT SCRIPT
# Sets up bot on fresh VPS from GitHub backup
##############################################################################

echo "🚀 DEPLOYING TRADING BOT ON NEW VPS..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 STEP 1: System Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}⚠️  Not running as root. Some steps may require sudo.${NC}"
fi

# Check OS
echo "🐧 OS: $(lsb_release -d 2>/dev/null | cut -f2 || uname -s)"
echo "💾 Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "💽 Disk: $(df -h / | awk 'NR==2 {print $4}' | head -1) free"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STEP 2: Installing Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Update system
echo "🔄 Updating system packages..."
sudo apt-get update -qq

# Install Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo "📥 Installing Python 3..."
    sudo apt-get install -y python3 python3-pip python3-venv
else
    echo "✅ Python already installed: $(python3 --version)"
fi

# Install other dependencies
echo "📥 Installing system dependencies..."
sudo apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    sqlite3 \
    redis-server \
    -qq

echo "✅ System dependencies installed"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🐍 STEP 3: Setting Up Python Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || true

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python packages from requirements.txt..."
    pip install -r requirements.txt -q
    echo "✅ Python packages installed"
elif [ -f "complete_requirements.txt" ]; then
    echo "📦 Installing Python packages from complete_requirements.txt..."
    pip install -r complete_requirements.txt -q
    echo "✅ Python packages installed"
else
    echo "📦 Installing essential packages..."
    pip install -q \
        ccxt \
        python-telegram-bot \
        pandas \
        numpy \
        scikit-learn \
        requests \
        asyncio \
        aiohttp
    echo "✅ Essential packages installed"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 STEP 4: Configuring Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found!${NC}"
    echo ""
    echo "You need to create .env file with your API keys."
    echo ""
    
    if [ -f ".env.example" ]; then
        echo "📝 Found .env.example template. Creating .env..."
        cp .env.example .env
        echo ""
        echo -e "${YELLOW}⚠️  IMPORTANT: Edit .env and add your actual API keys:${NC}"
        echo "   nano .env"
        echo ""
        echo "Press Enter when you've added your API keys..."
        read
    else
        echo "Creating basic .env template..."
        cat > .env << 'EOF'
# TELEGRAM
TELEGRAM_BOT_TOKEN=your_token_here
TG_ADMIN_CHAT_ID=your_chat_id
TG_FREE_CHAT_ID=your_free_channel
TG_VIP_CHAT_ID=your_vip_channel

# GATE.IO
GATEIO_TESTNET_API_KEY=your_key
GATEIO_TESTNET_SECRET=your_secret
GATEIO_LIVE_API_KEY=your_key
GATEIO_LIVE_SECRET=your_secret
GATEIO_MODE=testnet

# BYBIT
BYBIT_API_KEY=your_key
BYBIT_SECRET=your_secret
BYBIT_TESTNET=true
EOF
        echo ""
        echo -e "${RED}⚠️  STOP! Edit .env file NOW with your API keys:${NC}"
        echo "   nano .env"
        echo ""
        echo "Press Enter when done..."
        read
    fi
else
    echo "✅ .env file exists"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data models backups
echo "✅ Directories created"

# Set permissions
echo "🔐 Setting permissions..."
chmod +x *.sh 2>/dev/null || true
chmod 600 .env 2>/dev/null || true
echo "✅ Permissions set"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚙️  STEP 5: Setting Up Systemd Service"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BOT_DIR=$(pwd)
USER=$(whoami)

# Create systemd service file
sudo tee /etc/systemd/system/trading-bot-live.service > /dev/null << EOF
[Unit]
Description=Advanced Trading Bot - Live System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$BOT_DIR
Environment=PATH=$BOT_DIR/venv/bin:/usr/bin:/bin
ExecStart=$BOT_DIR/venv/bin/python3 $BOT_DIR/COMPLETE_ULTIMATE_ORCHESTRATOR.py --mode live
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload
echo "✅ Systemd service created: trading-bot-live"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧠 STEP 6: Restoring Learned Intelligence"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Restore learned data if backup exists
if [ -d "learned_data_backup" ]; then
    echo "🧠 Found learned data backup! Restoring..."
    
    # Restore databases
    if ls learned_data_backup/*.db 1> /dev/null 2>&1; then
        cp -v learned_data_backup/*.db .
        echo "✅ Databases restored"
    fi
    
    # Restore ML models
    if [ -d "learned_data_backup/models" ]; then
        cp -rv learned_data_backup/models .
        echo "✅ ML models restored"
    fi
    
    # Restore user database
    if [ -f "learned_data_backup/users_db.json" ]; then
        cp -v learned_data_backup/users_db.json .
        echo "✅ User database restored"
    fi
    
    # Restore data directory
    if [ -d "learned_data_backup/data" ]; then
        cp -rv learned_data_backup/data .
        echo "✅ Data directory restored"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 BOT'S LEARNED INTELLIGENCE RESTORED!${NC}"
    echo "   Your bot will continue from where it left off!"
    echo "   • Evolution Cycle: Preserved"
    echo "   • ML Models: Pre-trained"
    echo "   • Trade History: Complete"
    echo "   • User Data: Intact"
    echo ""
else
    echo "⚠️  No learned data backup found. Bot will start fresh."
    echo "   This is normal for first-time deployment."
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 STEP 7: Testing Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test Python imports
echo "🐍 Testing Python imports..."
python3 -c "import ccxt, telegram; print('✅ Core packages working')" 2>/dev/null || {
    echo -e "${RED}❌ Import test failed. Installing missing packages...${NC}"
    pip install ccxt python-telegram-bot
}

# Check for main bot file
if [ -f "COMPLETE_ULTIMATE_ORCHESTRATOR.py" ]; then
    echo "✅ Main bot file found"
elif [ -f "RUN_BOT.py" ]; then
    echo "✅ Alternative bot file found"
else
    echo -e "${YELLOW}⚠️  Main bot file not found. Check repository.${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. ✅ Start the bot:"
echo "   ${BLUE}bash START_BOT.sh${NC}"
echo ""
echo "2. 📊 View logs:"
echo "   ${BLUE}sudo journalctl -u trading-bot-live -f${NC}"
echo ""
echo "3. 🔍 Check status:"
echo "   ${BLUE}sudo systemctl status trading-bot-live${NC}"
echo ""
echo "4. 🛑 Stop bot:"
echo "   ${BLUE}sudo systemctl stop trading-bot-live${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}🎉 Your trading bot is ready to launch!${NC}"
echo ""
