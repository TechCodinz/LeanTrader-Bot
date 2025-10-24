#!/bin/bash
#
# COMPLETE VPS BOT DIAGNOSTICS
# Run this on your VPS to get full bot status
#

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                   COMPLETE BOT DIAGNOSTICS - VPS                             ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

cd /root/trading_bot 2>/dev/null || cd ~/trading_bot || { 
    echo "❌ ERROR: trading_bot directory not found!"
    echo "   Tried: /root/trading_bot and ~/trading_bot"
    exit 1
}

echo "📍 Working directory: $(pwd)"
echo ""

# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  GIT STATUS & HISTORY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Current Branch:"
BRANCH=$(git branch --show-current)
echo "   $BRANCH"
echo ""

echo "Current Commit:"
git log --oneline -1
echo ""

echo "Last 5 Commits:"
git log --oneline -5
echo ""

echo "Modified Files:"
git status --short
if [ $? -eq 0 ] && [ -z "$(git status --short)" ]; then
    echo "   ✅ No modified files (clean)"
fi
echo ""

echo "Remote Status:"
git fetch origin 2>&1 | grep -v "^From"
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null)
if [ "$LOCAL" = "$REMOTE" ]; then
    echo "   ✅ Up to date with origin/$BRANCH"
else
    BEHIND=$(git rev-list --count HEAD..@{u} 2>/dev/null)
    AHEAD=$(git rev-list --count @{u}..HEAD 2>/dev/null)
    if [ -n "$BEHIND" ] && [ "$BEHIND" -gt 0 ]; then
        echo "   ⚠️  Behind by $BEHIND commits - NEED TO PULL!"
    fi
    if [ -n "$AHEAD" ] && [ "$AHEAD" -gt 0 ]; then
        echo "   ⚠️  Ahead by $AHEAD commits - LOCAL CHANGES"
    fi
fi
echo ""

# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  CRITICAL FILES CHECK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check key files
FILES=(
    "RUN_BOT.py"
    "COMPLETE_ULTIMATE_ORCHESTRATOR.py"
    "COMPLETE_UNIFIED_ORCHESTRATOR.py"
    "ULTIMATE_ORCHESTRATOR.py"
    "DYNAMIC_PAIR_DISCOVERY.py"
    "start_bot.sh"
    ".env"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(wc -l < "$file" 2>/dev/null)
        echo "   ✅ $file ($SIZE lines)"
    else
        echo "   ❌ $file (MISSING)"
    fi
done
echo ""

# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  BOT VERSION & FEATURES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "System Count:"
if grep -q "ALL 8 ADVANCED SYSTEMS" COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>/dev/null; then
    echo "   ✅ 8 Systems (WITH Dynamic Pair Discovery)"
elif grep -q "ALL 7 NEW SYSTEMS DONE" COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>/dev/null; then
    echo "   ⚠️  7 Systems (WITHOUT Dynamic Pair Discovery)"
elif grep -q "ALL 40 SYSTEMS" COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>/dev/null; then
    echo "   ✅ 40 Systems (Extended version)"
else
    echo "   ❓ Unknown system count"
fi
echo ""

echo "Feature Detection:"
FEATURES=(
    "DYNAMIC_PAIR_DISCOVERY:Dynamic Pair Discovery"
    "run_dynamic_pair_discovery:Pair Discovery Loop"
    "SmartScalpingEngine:Smart Scalping"
    "UltraGodMode:God Mode"
    "UltraMoonSystem:Moon Spotter"
    "IBMQuantumEngine:Quantum Engine"
    "TelegramOrchestrator:Telegram Bot"
    "CRITICAL_FEATURES_AVAILABLE:Critical Profit Features"
    "ULTRA_FEATURES_AVAILABLE:Ultra Goldmine Features"
    "DIVINE_FEATURES_AVAILABLE:Divine Intelligence"
)

for feature in "${FEATURES[@]}"; do
    PATTERN="${feature%%:*}"
    NAME="${feature##*:}"
    if grep -q "$PATTERN" COMPLETE_ULTIMATE_ORCHESTRATOR.py 2>/dev/null; then
        echo "   ✅ $NAME"
    else
        echo "   ❌ $NAME (missing)"
    fi
done
echo ""

# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  RUNNING PROCESSES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

BOT_RUNNING=false
if pgrep -f "RUN_BOT.py" > /dev/null 2>&1; then
    echo "   ✅ Bot is RUNNING"
    BOT_RUNNING=true
    echo ""
    echo "   Process Details:"
    ps aux | grep -E "RUN_BOT.py|COMPLETE_ULTIMATE_ORCHESTRATOR" | grep -v grep | while read line; do
        echo "      $line"
    done
    echo ""
    echo "   PID(s):"
    pgrep -f "RUN_BOT.py" | while read pid; do
        echo "      PID: $pid"
        echo "      Started: $(ps -p $pid -o lstart=)"
        echo "      CPU: $(ps -p $pid -o %cpu=)%"
        echo "      Memory: $(ps -p $pid -o %mem=)%"
    done
else
    echo "   ❌ Bot is NOT running"
fi
echo ""

# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5️⃣  SYSTEMD SERVICE STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

SERVICES=("trading-bot-live" "trading-bot-testnet" "trading-bot" "ultrabot")

for service in "${SERVICES[@]}"; do
    if systemctl list-units --type=service --all | grep -q "$service"; then
        echo "Service: $service"
        systemctl status "$service" --no-pager -n 0 | head -5
        echo ""
    fi
done

# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6️⃣  RECENT LOGS (Last 30 lines)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "bot.log" ]; then
    echo "From bot.log:"
    tail -30 bot.log | grep -E "INFO|ERROR|WARNING|SYSTEMS|Phase|initialized|Discovery" || tail -30 bot.log
elif systemctl is-active --quiet trading-bot-live 2>/dev/null; then
    echo "From systemd (trading-bot-live):"
    journalctl -u trading-bot-live -n 30 --no-pager | grep -E "INFO|ERROR|WARNING|SYSTEMS|Phase|initialized|Discovery" || journalctl -u trading-bot-live -n 30 --no-pager
else
    echo "   ⚠️  No logs found (bot.log missing and no systemd service)"
fi
echo ""

# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7️⃣  ENVIRONMENT & API KEYS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f ".env" ]; then
    echo "Checking .env keys (not showing values):"
    KEYS=(
        "BYBIT_API_KEY"
        "BYBIT_API_SECRET"
        "TELEGRAM_BOT_TOKEN"
        "TELEGRAM_CHAT_ID"
        "PRIVATE_KEY"
        "OPENAI_API_KEY"
    )
    
    for key in "${KEYS[@]}"; do
        if grep -q "^${key}=" .env 2>/dev/null; then
            VALUE=$(grep "^${key}=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
            if [ -n "$VALUE" ] && [ "$VALUE" != "your_key_here" ]; then
                echo "   ✅ $key (set)"
            else
                echo "   ⚠️  $key (empty or default)"
            fi
        else
            echo "   ❌ $key (not found)"
        fi
    done
else
    echo "   ⚠️  No .env file found"
fi
echo ""

# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8️⃣  PYTHON ENVIRONMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -d "venv" ]; then
    echo "   ✅ Virtual environment exists"
    if [ -f "venv/bin/python" ]; then
        echo "   Python version: $(venv/bin/python --version 2>&1)"
    fi
else
    echo "   ⚠️  No venv directory found"
fi
echo ""

# Check key packages
if [ -f "venv/bin/pip" ]; then
    echo "Key packages:"
    PACKAGES=("ccxt" "pandas" "numpy" "tensorflow" "qiskit" "telegram")
    for pkg in "${PACKAGES[@]}"; do
        if venv/bin/pip show "$pkg" > /dev/null 2>&1; then
            VERSION=$(venv/bin/pip show "$pkg" | grep Version | cut -d' ' -f2)
            echo "   ✅ $pkg ($VERSION)"
        else
            echo "   ❌ $pkg (not installed)"
        fi
    done
fi
echo ""

# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                              DIAGNOSIS                                       ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Determine overall status
ISSUES=0

# Check if bot is running
if [ "$BOT_RUNNING" = false ]; then
    echo "⚠️  Issue #$((++ISSUES)): Bot is not running"
fi

# Check if up to date
if [ "$LOCAL" != "$REMOTE" ] 2>/dev/null; then
    echo "⚠️  Issue #$((++ISSUES)): Not up to date with git remote"
fi

# Check if key files exist
if [ ! -f "COMPLETE_ULTIMATE_ORCHESTRATOR.py" ]; then
    echo "❌ Issue #$((++ISSUES)): Main orchestrator file missing!"
fi

if [ ! -f "RUN_BOT.py" ]; then
    echo "❌ Issue #$((++ISSUES)): RUN_BOT.py missing!"
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Issue #$((++ISSUES)): .env file missing (API keys not configured)"
fi

if [ $ISSUES -eq 0 ]; then
    echo "✅ No issues detected! Bot setup looks good."
    echo ""
    if [ "$BOT_RUNNING" = false ]; then
        echo "📋 To start the bot:"
        echo "   ./start_bot.sh"
        echo "   tail -f bot.log"
    else
        echo "📋 Bot is running! To view logs:"
        echo "   tail -f bot.log"
    fi
else
    echo "📋 Found $ISSUES issue(s) that need attention."
fi

echo ""
echo "══════════════════════════════════════════════════════════════════════════════"
echo "Diagnostics complete! $(date)"
echo "══════════════════════════════════════════════════════════════════════════════"
