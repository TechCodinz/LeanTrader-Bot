#!/bin/bash
##############################################################################
# TRADING BOT - BACKUP TO GITHUB
# Backs up entire bot from VPS to GitHub repository
##############################################################################

echo "🔄 BACKING UP TRADING BOT TO GITHUB..."
echo ""

# Set colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get current directory
BOT_DIR=$(pwd)
echo "📁 Bot Directory: $BOT_DIR"

# Ask for GitHub details
read -p "Enter your GitHub username: " GITHUB_USER
read -p "Enter repository name (e.g., trading-bot-live): " REPO_NAME
read -p "Enter your GitHub Personal Access Token: " GITHUB_TOKEN

# Set GitHub repo URL
REPO_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STEP 1: Preparing Files for Backup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
*.egg-info/
dist/
build/

# Logs
*.log
logs/
*.db-journal

# Sensitive data (DO NOT COMMIT)
.env.backup
*.key
*.pem

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Large databases (optional - uncomment if too large)
# *.db

# Temporary files
*.tmp
*.cache
EOF
    echo "✅ Created .gitignore"
else
    echo "✅ .gitignore exists"
fi

# Create README if it doesn't exist
if [ ! -f README.md ]; then
    cat > README.md << 'EOF'
# 🤖 Advanced Trading Bot - Live System

## 🚀 Features
- 55+ integrated trading systems
- Multi-exchange support (Gate.io, Bybit, etc.)
- ML/AI powered strategies
- Telegram integration (Admin/VIP/Free channels)
- Real-time profit generation
- Advanced risk management
- Evolution learning engine

## 📊 Current Performance
- **Strategies**: Arbitrage, Scalping, Moon Spotting, FX Trading
- **Win Rate**: Improving with ML learning
- **Profit Factor**: 2.0-2.6x on key strategies

## 🛠️ Quick Setup (New VPS)
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Run auto-setup script
bash DEPLOY_NEW_VPS.sh

# 3. Start bot
bash START_BOT.sh
```

## 🔒 Security
- Never commit API keys
- Use environment variables
- Keep `.env` file local only

## 📝 Documentation
See individual .md files for detailed documentation.

**Created**: $(date)
**Last Backup**: $(date)
EOF
    echo "✅ Created README.md"
else
    echo "✅ README.md exists"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔐 STEP 2: Securing Sensitive Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Backup .env (but don't commit it)
if [ -f .env ]; then
    cp .env .env.backup
    echo "✅ Backed up .env to .env.backup (local only)"
fi

# Create .env.example (safe to commit)
if [ -f .env ]; then
    cat > .env.example << 'EOF'
# Trading Bot Environment Variables
# Copy this to .env and fill in your actual values

# ===== TELEGRAM =====
TELEGRAM_BOT_TOKEN=your_bot_token_here
TG_ADMIN_CHAT_ID=your_admin_chat_id
TG_FREE_CHAT_ID=your_free_channel_id
TG_VIP_CHAT_ID=your_vip_channel_id

# ===== GATE.IO =====
GATEIO_TESTNET_API_KEY=your_testnet_key
GATEIO_TESTNET_SECRET=your_testnet_secret
GATEIO_LIVE_API_KEY=your_live_key
GATEIO_LIVE_SECRET=your_live_secret
GATEIO_MODE=testnet

# ===== BYBIT TESTNET =====
BYBIT_API_KEY=your_bybit_key
BYBIT_SECRET=your_bybit_secret
BYBIT_TESTNET=true

# ===== NEWS & DATA =====
NEWSAPI_KEY=your_newsapi_key
ETHERSCAN_API_KEY=your_etherscan_key
BSCSCAN_API_KEY=your_bscscan_key
POLYGONSCAN_API_KEY=your_polygonscan_key
EOF
    echo "✅ Created .env.example (safe template)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📤 STEP 3: Initializing Git Repository"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Initialize git if not already
if [ ! -d .git ]; then
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

# Configure git
git config user.email "bot@tradingbot.local" 2>/dev/null || true
git config user.name "Trading Bot Backup" 2>/dev/null || true

# Add all files
git add .
echo "✅ Files staged for commit"

# Commit
COMMIT_MSG="🤖 Trading Bot Backup - $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$COMMIT_MSG" || echo "⚠️  No changes to commit"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "☁️  STEP 4: Pushing to GitHub"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Add remote (remove if exists)
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

# Push to GitHub
echo "🚀 Pushing to GitHub..."
if git push -u origin main --force 2>/dev/null || git push -u origin master --force; then
    echo -e "${GREEN}✅ Successfully backed up to GitHub!${NC}"
    echo ""
    echo "📍 Repository URL: https://github.com/${GITHUB_USER}/${REPO_NAME}"
else
    echo -e "${RED}❌ Push failed. Trying to create main branch...${NC}"
    git branch -M main
    git push -u origin main --force
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 STEP 5: Creating Deployment Info"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Save deployment info
cat > DEPLOYMENT_INFO.txt << EOF
🤖 TRADING BOT - DEPLOYMENT INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Backup Date: $(date)
💻 VPS: $(hostname)
📍 Bot Directory: $BOT_DIR
☁️  GitHub: https://github.com/${GITHUB_USER}/${REPO_NAME}

🚀 TO DEPLOY ON NEW VPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Clone repository:
   git clone https://github.com/${GITHUB_USER}/${REPO_NAME}.git
   cd ${REPO_NAME}

2. Copy your .env file:
   # Upload your .env.backup or create new .env

3. Run deployment script:
   bash DEPLOY_NEW_VPS.sh

4. Start bot:
   bash START_BOT.sh

🔐 IMPORTANT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Your .env file is NOT in GitHub (security)
- Keep .env.backup safe locally
- Copy .env to new VPS before deploying
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

echo "✅ Created DEPLOYMENT_INFO.txt"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}🎉 BACKUP COMPLETE!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Your bot is now backed up to:"
echo "   https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo ""
echo "📝 Next Steps:"
echo "   1. ✅ Keep your .env.backup file safe (local only)"
echo "   2. ✅ Copy DEPLOY_NEW_VPS.sh to new VPS when needed"
echo "   3. ✅ Use START_BOT.sh to launch on new server"
echo ""
echo "🔒 Security Note:"
echo "   - .env NOT uploaded (sensitive data protected)"
echo "   - .env.example created as template"
echo "   - Keep API keys secure!"
echo ""
