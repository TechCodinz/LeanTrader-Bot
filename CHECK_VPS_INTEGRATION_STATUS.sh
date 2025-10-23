#!/bin/bash
#
# Run this on your VPS to check where the integration stopped
#

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CHECKING INTEGRATION STATUS ON VPS                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd /root/trading_bot 2>/dev/null || cd ~/trading_bot || { echo "❌ trading_bot directory not found"; exit 1; }

echo "1️⃣  Checking if DYNAMIC_PAIR_DISCOVERY.py exists..."
if [ -f "DYNAMIC_PAIR_DISCOVERY.py" ]; then
    echo "   ✅ DYNAMIC_PAIR_DISCOVERY.py EXISTS"
    lines=$(wc -l < DYNAMIC_PAIR_DISCOVERY.py)
    echo "   📄 File size: $lines lines"
else
    echo "   ❌ DYNAMIC_PAIR_DISCOVERY.py NOT FOUND"
fi
echo ""

echo "2️⃣  Checking COMPLETE_ULTIMATE_ORCHESTRATOR.py integration..."
if grep -q "DYNAMIC_PAIR_DISCOVERY" COMPLETE_ULTIMATE_ORCHESTRATOR.py; then
    echo "   ✅ DYNAMIC_PAIR_DISCOVERY import found"
else
    echo "   ❌ DYNAMIC_PAIR_DISCOVERY import NOT found"
fi

if grep -q "pair_discovery" COMPLETE_ULTIMATE_ORCHESTRATOR.py; then
    echo "   ✅ pair_discovery references found"
    count=$(grep -c "pair_discovery" COMPLETE_ULTIMATE_ORCHESTRATOR.py)
    echo "   📊 Found $count references"
else
    echo "   ❌ pair_discovery references NOT found"
fi

if grep -q "run_dynamic_pair_discovery" COMPLETE_ULTIMATE_ORCHESTRATOR.py; then
    echo "   ✅ run_dynamic_pair_discovery method found"
else
    echo "   ❌ run_dynamic_pair_discovery method NOT found"
fi
echo ""

echo "3️⃣  Checking what version is running..."
if grep -q "ALL 8 ADVANCED SYSTEMS" COMPLETE_ULTIMATE_ORCHESTRATOR.py; then
    echo "   ✅ Code shows 8 systems (WITH pair discovery)"
elif grep -q "ALL 7 NEW SYSTEMS DONE" COMPLETE_ULTIMATE_ORCHESTRATOR.py; then
    echo "   ⚠️  Code shows 7 systems (WITHOUT pair discovery)"
else
    echo "   ❓ Can't determine system count"
fi
echo ""

echo "4️⃣  Checking git status..."
git status --short | head -10
echo ""

echo "5️⃣  Checking current branch..."
BRANCH=$(git branch --show-current)
echo "   Current branch: $BRANCH"
echo ""

echo "6️⃣  Checking if there are unpulled changes..."
git fetch origin 2>/dev/null
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "   ✅ Up to date with remote"
else
    echo "   ⚠️  Remote has changes - need to pull!"
    echo ""
    echo "   Run: git pull origin $BRANCH"
fi
echo ""

echo "7️⃣  Checking last 3 commits..."
git log --oneline -3
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DIAGNOSIS                                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Determine status
HAS_FILE=false
HAS_INTEGRATION=false

if [ -f "DYNAMIC_PAIR_DISCOVERY.py" ]; then
    HAS_FILE=true
fi

if grep -q "run_dynamic_pair_discovery" COMPLETE_ULTIMATE_ORCHESTRATOR.py; then
    HAS_INTEGRATION=true
fi

if [ "$HAS_FILE" = true ] && [ "$HAS_INTEGRATION" = true ]; then
    echo "✅ INTEGRATION IS COMPLETE!"
    echo ""
    echo "   Status: Both files exist and are integrated"
    echo "   Action: Restart bot to activate pair discovery"
    echo ""
    echo "   Commands:"
    echo "   pkill -9 -f RUN_BOT.py"
    echo "   ./start_bot.sh"
    echo "   tail -f bot.log"
elif [ "$HAS_FILE" = true ] && [ "$HAS_INTEGRATION" = false ]; then
    echo "⚠️  INTEGRATION IS INCOMPLETE"
    echo ""
    echo "   Status: DYNAMIC_PAIR_DISCOVERY.py exists"
    echo "   Problem: Not integrated into COMPLETE_ULTIMATE_ORCHESTRATOR.py"
    echo "   Action: Need to pull latest code from git"
    echo ""
    echo "   Commands:"
    echo "   git pull origin $BRANCH"
    echo "   pkill -9 -f RUN_BOT.py"
    echo "   ./start_bot.sh"
elif [ "$HAS_FILE" = false ]; then
    echo "❌ INTEGRATION NOT STARTED"
    echo ""
    echo "   Status: DYNAMIC_PAIR_DISCOVERY.py doesn't exist"
    echo "   Problem: Need to pull code from git"
    echo "   Action: Pull latest changes"
    echo ""
    echo "   Commands:"
    echo "   git pull origin $BRANCH"
else
    echo "❓ UNKNOWN STATUS"
    echo ""
    echo "   Manually check files"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
