#!/bin/bash
# Commands to stop existing bot on VPS and launch new one

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         STOP OLD BOT & LAUNCH NEW BOT                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ==============================================================================
# STEP 1: Find and Stop Running Bots
# ==============================================================================
echo "🔍 STEP 1: Find Running Bots"
echo "────────────────────────────────────────────────────────────────"
echo ""

echo "# Check what's running:"
echo "ps aux | grep -E 'python.*bot|python.*trader|python.*ORCHESTRATOR' | grep -v grep"
echo ""

echo "# Find process IDs:"
echo "pgrep -f 'python.*bot'"
echo "pgrep -f 'python.*ORCHESTRATOR'"
echo ""

# ==============================================================================
# STEP 2: Stop All Bot Processes
# ==============================================================================
echo ""
echo "🛑 STEP 2: Stop All Bot Processes"
echo "────────────────────────────────────────────────────────────────"
echo ""

cat << 'EOF'
# Method 1: Kill by process name (safest)
pkill -f 'python.*bot'
pkill -f 'python.*trader'
pkill -f 'python.*ORCHESTRATOR'

# Method 2: Kill specific PIDs if you see them
# Replace XXXX with actual PID from ps aux
# kill XXXX

# Method 3: Force kill if needed (use carefully!)
# pkill -9 -f 'python.*bot'

# Verify stopped:
ps aux | grep python | grep -v grep

# Should see no bot processes
EOF
echo ""

# ==============================================================================
# STEP 3: Clean Up Old Files (Optional)
# ==============================================================================
echo ""
echo "🧹 STEP 3: Clean Up (Optional)"
echo "────────────────────────────────────────────────────────────────"
echo ""

cat << 'EOF'
# Find old bot directories:
ls -la ~/ | grep -i bot
ls -la ~/ | grep -i trade

# Remove old bot (CAREFUL! Make sure it's the right one):
# cd ~/old-bot-directory
# rm -rf old-bot-directory

# Or just rename it:
# mv old-bot-directory old-bot-directory.backup
EOF
echo ""

# ==============================================================================
# STEP 4: Deploy New Bot
# ==============================================================================
echo ""
echo "🚀 STEP 4: Deploy New Bot"
echo "────────────────────────────────────────────────────────────────"
echo ""

cat << 'EOF'
# Clone the complete integration:
cd ~
git clone https://github.com/TechCodinz/Lean-Trader trading-bot-complete
cd trading-bot-complete

# Checkout the complete integration branch:
git checkout cursor/restore-bot-venv-and-fix-errors-d71f

# Install dependencies:
python3.13 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r py313_requirements.txt

# Configure .env:
nano .env
# Set TRADING_MODE=testnet (start safe!)
# Add your testnet API keys

# Test import:
python -c "from COMPLETE_ULTIMATE_ORCHESTRATOR import CompleteUltimateOrchestrator; print('OK')"

# Run bot:
nohup python COMPLETE_ULTIMATE_ORCHESTRATOR.py > bot.log 2>&1 &

# Save PID:
echo $! > bot.pid

# Watch logs:
tail -f bot.log
EOF
echo ""

# ==============================================================================
# STEP 5: Verify New Bot Running
# ==============================================================================
echo ""
echo "✅ STEP 5: Verify New Bot"
echo "────────────────────────────────────────────────────────────────"
echo ""

cat << 'EOF'
# Check if running:
ps aux | grep COMPLETE_ULTIMATE_ORCHESTRATOR

# Check logs:
tail -50 bot.log

# You should see:
# "✅ ALL 116+ SYSTEMS INITIALIZED!"
# "🚀 AUTO-STARTING ALL SYSTEMS..."
# "🤖 AUTO LIVE TRIGGER ACTIVE"

# Check system count in logs:
grep "ALL.*SYSTEMS" bot.log

# Stop if needed:
kill $(cat bot.pid)
EOF
echo ""

# ==============================================================================
# QUICK REFERENCE
# ==============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    QUICK REFERENCE                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Stop old bot:                                               ║"
echo "║    pkill -f 'python.*bot'                                    ║"
echo "║                                                              ║"
echo "║  Clone new bot:                                              ║"
echo "║    git clone <repo> trading-bot-complete                     ║"
echo "║    cd trading-bot-complete                                   ║"
echo "║    git checkout cursor/restore-bot-venv-and-fix-errors-d71f  ║"
echo "║                                                              ║"
echo "║  Setup:                                                      ║"
echo "║    python3.13 -m venv venv                                   ║"
echo "║    source venv/bin/activate                                  ║"
echo "║    pip install -r py313_requirements.txt                     ║"
echo "║                                                              ║"
echo "║  Run:                                                        ║"
echo "║    nohup python COMPLETE_ULTIMATE_ORCHESTRATOR.py > bot.log &║"
echo "║    tail -f bot.log                                           ║"
echo "║                                                              ║"
echo "║  Stop:                                                       ║"
echo "║    kill $(cat bot.pid)                                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
