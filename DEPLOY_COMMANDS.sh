#!/bin/bash
# Complete deployment commands for VPS

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         VPS DEPLOYMENT COMMANDS                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ==============================================================================
# STEP 1: PUSH TO GIT (Save your work!)
# ==============================================================================
echo "📦 STEP 1: Push to Git Repository"
echo "────────────────────────────────────────────────────────────────"
echo ""
echo "# Save all this work to git:"
echo "cd /workspace"
echo "git push origin cursor/restore-bot-venv-and-fix-errors-d71f"
echo ""
echo "# Or force push if needed:"
echo "git push -f origin cursor/restore-bot-venv-and-fix-errors-d71f"
echo ""
echo "# Create a new branch for clean history:"
echo "git checkout -b complete-integration-v1"
echo "git push origin complete-integration-v1"
echo ""
echo "Press Enter when done..."
read

# ==============================================================================
# STEP 2: ON YOUR VPS - Clone Repository
# ==============================================================================
echo ""
echo "🖥️  STEP 2: On Your VPS - Clone Repository"
echo "────────────────────────────────────────────────────────────────"
echo ""
echo "# SSH into your VPS first, then run:"
echo ""
echo "cd ~"
echo "git clone YOUR_GITHUB_REPO_URL trading-bot"
echo "cd trading-bot"
echo ""
echo "# Checkout the branch with all systems:"
echo "git checkout cursor/restore-bot-venv-and-fix-errors-d71f"
echo ""
echo "# Or if you created new branch:"
echo "git checkout complete-integration-v1"
echo ""

# ==============================================================================
# STEP 3: Install Dependencies
# ==============================================================================
echo ""
echo "📚 STEP 3: Install Dependencies on VPS"
echo "────────────────────────────────────────────────────────────────"
echo ""
cat << 'EOF'
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.13 if not available
sudo apt install -y python3.13 python3.13-venv python3.13-dev

# Create virtual environment
python3.13 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (use the Python 3.13 compatible ones)
pip install -r py313_requirements.txt

# This will take 5-15 minutes depending on VPS specs
EOF
echo ""

# ==============================================================================
# STEP 4: Configure .env File
# ==============================================================================
echo ""
echo "⚙️  STEP 4: Configure .env File"
echo "────────────────────────────────────────────────────────────────"
echo ""
cat << 'EOF'
# Edit .env file
nano .env

# IMPORTANT SETTINGS:
# ──────────────────────────────────────────────────────────────

# 1. Start in TESTNET mode (SAFE!):
TRADING_MODE=testnet
BYBIT_TESTNET=true
ENABLE_LIVE=false

# 2. Get Bybit TESTNET keys (testnet.bybit.com):
BYBIT_TESTNET_API_KEY=your_testnet_key_here
BYBIT_TESTNET_API_SECRET=your_testnet_secret_here

# 3. Your LIVE keys (won't be used until approved):
BYBIT_API_KEY=mMHs7rDC72TvHs4oQG
BYBIT_API_SECRET=NwTa6UOgczdmZI2Kn2WBcFfh5r6VkVGnvGEI

# 4. Telegram (already configured):
TELEGRAM_BOT_TOKEN=8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg

# 5. Infinite limits (already set):
MAX_DAILY_TRADES=999999
MAX_OPEN_POSITIONS=999999

# Save: Ctrl+O, Enter, Ctrl+X
EOF
echo ""

# ==============================================================================
# STEP 5: Test Import
# ==============================================================================
echo ""
echo "🧪 STEP 5: Test System"
echo "────────────────────────────────────────────────────────────────"
echo ""
cat << 'EOF'
# Make sure virtual environment is active
source venv/bin/activate

# Test import
python -c "from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator; print('✅ Import successful!')"

# If errors, check:
python -c "import ccxt; print('ccxt OK')"
python -c "import pandas; print('pandas OK')"
python -c "import numpy; print('numpy OK')"
EOF
echo ""

# ==============================================================================
# STEP 6: Run Bot
# ==============================================================================
echo ""
echo "🚀 STEP 6: Run Bot (TESTNET First!)"
echo "────────────────────────────────────────────────────────────────"
echo ""
cat << 'EOF'
# Option A: Run in foreground (see logs live):
python COMPLETE_ULTIMATE_ORCHESTRATOR.py

# Option B: Run in background:
nohup python COMPLETE_ULTIMATE_ORCHESTRATOR.py > bot.log 2>&1 &

# Get process ID:
echo $! > bot.pid

# Watch logs:
tail -f bot.log

# Stop bot:
kill $(cat bot.pid)
EOF
echo ""

# ==============================================================================
# STEP 7: Monitor
# ==============================================================================
echo ""
echo "📊 STEP 7: Monitor Bot"
echo "────────────────────────────────────────────────────────────────"
echo ""
cat << 'EOF'
# Watch logs live:
tail -f bot.log

# Check if running:
ps aux | grep COMPLETE_ULTIMATE_ORCHESTRATOR

# Check learned data:
ls -lh data/*.db
cat data/history.csv | wc -l

# Check auto-commits:
git log --oneline | head -10

# Monitor Telegram for alerts
EOF
echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    DEPLOYMENT SUMMARY                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  1. Push to git: Save your work                             ║"
echo "║  2. Clone on VPS: Get the code                              ║"
echo "║  3. Install deps: Setup environment                         ║"
echo "║  4. Configure .env: Add API keys (TESTNET!)                 ║"
echo "║  5. Test import: Verify setup                               ║"
echo "║  6. Run bot: Start trading (TESTNET first!)                 ║"
echo "║  7. Monitor: Watch logs & Telegram                          ║"
echo "║                                                              ║"
echo "║  Expected Timeline:                                         ║"
echo "║  - Setup: 1-2 hours                                         ║"
echo "║  - Testnet: 2-3 weeks                                       ║"
echo "║  - Live: After 60%+ win rate                                ║"
echo "║                                                              ║"
echo "║  Status: Ready to deploy! 🚀                                ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
