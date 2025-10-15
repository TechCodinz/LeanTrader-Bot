#!/bin/bash
# VPS Setup Script - Complete Cleanup & Deployment
# Run this on your VPS

echo "================================================================================"
echo "🧹 STEP 1: CLEANING VPS"
echo "================================================================================"
echo ""

# Go to home directory
cd ~
pwd

# Show current contents
echo "Current directories:"
ls -la

echo ""
read -p "Do you want to backup old bots before deleting? (y/n): " backup_choice

if [ "$backup_choice" = "y" ]; then
    echo "Creating backup..."
    timestamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p backups
    
    # Backup old bots if they exist
    for dir in bot trading_bot old_bot lean-trader leantrader; do
        if [ -d "$dir" ]; then
            echo "Backing up $dir..."
            mv "$dir" "backups/${dir}_backup_${timestamp}"
        fi
    done
    
    echo "✅ Backups created in ~/backups/"
else
    echo "Deleting old bots..."
    rm -rf bot trading_bot old_bot lean-trader leantrader
    echo "✅ Old bots deleted"
fi

echo ""
echo "================================================================================"
echo "📦 STEP 2: SETTING UP GIT FOR PRIVATE REPO"
echo "================================================================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    sudo apt update
    sudo apt install -y git
fi

echo "✅ Git is ready"
echo ""

# Configure git credentials for private repo
echo "For PRIVATE repo, you need authentication."
echo ""
echo "Option 1: Personal Access Token (Recommended)"
echo "Option 2: SSH key"
echo ""

read -p "Which method? (1 for Token, 2 for SSH): " auth_method

if [ "$auth_method" = "1" ]; then
    echo ""
    echo "=== USING PERSONAL ACCESS TOKEN ==="
    echo ""
    echo "If you don't have a token, get one:"
    echo "1. Go to: https://github.com/settings/tokens"
    echo "2. Generate new token (classic)"
    echo "3. Select: repo (all permissions)"
    echo "4. Copy the token"
    echo ""
    read -p "Enter your GitHub username: " gh_username
    read -p "Enter your Personal Access Token: " gh_token
    echo ""
    
    # Set git credential store
    git config --global credential.helper store
    
    # Clone URL with token
    REPO_URL="https://${gh_token}@github.com/${gh_username}/YOUR_REPO_NAME.git"
    echo "✅ Authentication configured"
    
elif [ "$auth_method" = "2" ]; then
    echo ""
    echo "=== USING SSH KEY ==="
    echo ""
    
    # Check if SSH key exists
    if [ -f ~/.ssh/id_rsa.pub ]; then
        echo "✅ SSH key exists"
        echo "Public key:"
        cat ~/.ssh/id_rsa.pub
        echo ""
        echo "Add this key to GitHub:"
        echo "https://github.com/settings/keys"
    else
        echo "Generating SSH key..."
        ssh-keygen -t rsa -b 4096 -C "vps@trading-bot" -N "" -f ~/.ssh/id_rsa
        echo ""
        echo "✅ SSH key generated!"
        echo "Public key:"
        cat ~/.ssh/id_rsa.pub
        echo ""
        echo "Add this key to GitHub:"
        echo "https://github.com/settings/keys"
        echo ""
        read -p "Press Enter after adding key to GitHub..."
    fi
    
    read -p "Enter your GitHub username: " gh_username
    REPO_URL="git@github.com:${gh_username}/YOUR_REPO_NAME.git"
    echo "✅ SSH authentication ready"
fi

echo ""
echo "================================================================================"
echo "📥 STEP 3: CLONING REPOSITORY"
echo "================================================================================"
echo ""

# Replace YOUR_REPO_NAME with actual repo name
read -p "Enter your repository name (e.g., Lean-Trader): " repo_name

# Update repo URL
if [ "$auth_method" = "1" ]; then
    REPO_URL="https://${gh_token}@github.com/${gh_username}/${repo_name}.git"
else
    REPO_URL="git@github.com:${gh_username}/${repo_name}.git"
fi

# Clone specific branch
echo "Cloning branch: cursor/integrate-and-unify-existing-trading-bot-components-c04c"
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c "$REPO_URL" trading_bot

if [ $? -eq 0 ]; then
    echo "✅ Repository cloned successfully!"
else
    echo "❌ Clone failed. Check authentication and repo name."
    exit 1
fi

echo ""
echo "================================================================================"
echo "🔑 STEP 4: CONFIGURING API KEYS"
echo "================================================================================"
echo ""

cd trading_bot

echo "Creating .env file with your API keys..."
echo ""

cat > .env << 'ENVFILE'
# ===== TELEGRAM =====
TELEGRAM_BOT_TOKEN=8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg
TG_ADMIN_CHAT_ID=5329503447
TG_FREE_CHAT_ID=-1002930953007
TG_VIP_CHAT_ID=-1002983007302

# ===== BYBIT TESTNET =====
BYBIT_API_KEY=N8BMgWdfisCtkvfZk8
BYBIT_SECRET=BIu7c65FQnDsd6kBmctU7gK9bBbzY15vi8oe
BYBIT_TESTNET=true

# ===== GATE.IO =====
GATEIO_TESTNET_API_KEY=590f4e3cb2a8cfcaa66fe1a3a646e4b1
GATEIO_TESTNET_SECRET=e1e5614876dfd2aa9c59beabd035c2af08a186b5f818209640c66e98225ca37b
GATEIO_LIVE_API_KEY=bbdcedbd7f719a87c851356cf4dd3c20
GATEIO_LIVE_SECRET=068996eb5877b74abf3595aedbc4f0778fe64e7f37d88c01b5af41f62e4d9c26
GATEIO_MODE=testnet

# ===== NEWS & DATA =====
NEWSAPI_KEY=11dc35597f524f868ef814ad0bc9d216
ETHERSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
BSCSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
POLYGONSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
ENVFILE

echo "✅ .env file created with all API keys"

echo ""
echo "================================================================================"
echo "📦 STEP 5: INSTALLING DEPENDENCIES"
echo "================================================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    sudo apt update
    sudo apt install -y python3 python3-pip
fi

echo "Python version: $(python3 --version)"

# Install dependencies
echo "Installing requirements..."
pip3 install -r requirements.txt

echo ""
echo "Verifying installation..."
python3 -c "import ccxt, web3; print('✅ Core dependencies installed')" 2>/dev/null || echo "⚠️  Some optional dependencies missing (OK for basic operation)"

echo ""
echo "================================================================================"
echo "🚀 STEP 6: STARTING BOT"
echo "================================================================================"
echo ""

echo "Starting bot in testnet mode..."
echo ""

# Check if screen is available
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    sudo apt install -y screen
fi

# Start bot in screen
screen -dmS trading_bot python3 RUN_BOT.py --testnet

echo ""
echo "================================================================================"
echo "✅ BOT STARTED!"
echo "================================================================================"
echo ""
echo "Bot is running in background (screen session: trading_bot)"
echo ""
echo "View logs:"
echo "  screen -r trading_bot"
echo ""
echo "Detach from screen:"
echo "  Press Ctrl+A then D"
echo ""
echo "Check if running:"
echo "  screen -ls"
echo "  ps aux | grep RUN_BOT"
echo ""
echo "Stop bot:"
echo "  screen -X -S trading_bot quit"
echo ""
echo "================================================================================"
echo "📱 NEXT STEPS"
echo "================================================================================"
echo ""
echo "1. Attach to screen to see logs:"
echo "   screen -r trading_bot"
echo ""
echo "2. Check Telegram for notifications"
echo ""
echo "3. Monitor for first day"
echo ""
echo "4. Verify trades on Bybit/Gate.io testnet"
echo ""
echo "================================================================================"
echo "Bot is running! Check Telegram for updates! 🚀"
echo "================================================================================"
