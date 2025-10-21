#!/bin/bash
###############################################################################
# VPS DIAGNOSTICS - Understand what's running before making ANY changes
# Run this on your VPS to show the complete system architecture
###############################################################################

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           VPS TRADING BOT - FULL DIAGNOSTICS              ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

cd ~/trading_bot 2>/dev/null || cd /root/trading_bot 2>/dev/null || cd /workspace 2>/dev/null

echo "📁 Working Directory: $(pwd)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. RUNNING PROCESSES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
ps aux | grep -E "python|RUN_BOT|trading|bot" | grep -v grep
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. MAIN BOT FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -f "RUN_BOT.py" ]; then
    echo "✅ RUN_BOT.py found"
    echo ""
    echo "First 50 lines:"
    head -50 RUN_BOT.py | cat -n
else
    echo "❌ RUN_BOT.py not found"
    echo ""
    echo "Python files in directory:"
    ls -la *.py 2>/dev/null | head -20
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. ORCHESTRATOR FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
ORCHESTRATOR=""
for file in COMPLETE_ULTIMATE_ORCHESTRATOR.py ULTIMATE_ORCHESTRATOR.py UNIFIED_ORCHESTRATOR.py; do
    if [ -f "$file" ]; then
        ORCHESTRATOR="$file"
        break
    fi
done

if [ -n "$ORCHESTRATOR" ]; then
    echo "✅ Found: $ORCHESTRATOR"
    echo ""
    echo "Imports and engines (first 100 lines):"
    head -100 "$ORCHESTRATOR" | grep -E "^import|^from|class.*Engine|class.*Orchestrator|class.*Scout" | cat -n
else
    echo "❌ No orchestrator file found"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. ENGINE FILES IN SYSTEM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "All *ENGINE*.py files:"
find . -maxdepth 1 -name "*ENGINE*.py" -o -name "*SCOUT*.py" -o -name "*ORCHESTRATOR*.py" | sort
echo ""
echo "All *engine*.py files:"
find . -maxdepth 1 -name "*engine*.py" -o -name "*scout*.py" | sort
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. CURRENT LOG ACTIVITY (Last 50 lines)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -f "bot.log" ]; then
    tail -50 bot.log
else
    echo "❌ bot.log not found"
    echo ""
    echo "Available log files:"
    find . -maxdepth 1 -name "*.log" -type f
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6. WHAT ENGINES ARE LOADED (from logs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -f "bot.log" ]; then
    echo "Engines/Systems mentioned in logs:"
    grep -iE "engine|scout|orchestrator|loaded|initialized|starting" bot.log | grep -v "grep" | tail -30
else
    echo "❌ No log file to analyze"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7. PAIR DISCOVERY STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -f "bot.log" ]; then
    echo "Pair discovery mentions:"
    grep -iE "discovered|pairs|scanning|markets" bot.log | tail -20
    echo ""
    echo "Total discovered pairs:"
    grep "TOTAL DISCOVERED" bot.log | tail -3
    echo ""
    echo "Unique pairs currently trading:"
    UNIQUE_COUNT=$(grep "Decision:" bot.log 2>/dev/null | grep -oE "[A-Z]{2,5}/[A-Z]{2,5}" | sort -u | wc -l)
    echo "  $UNIQUE_COUNT unique pairs"
    echo ""
    echo "Sample of pairs:"
    grep "Decision:" bot.log 2>/dev/null | grep -oE "[A-Z]{2,5}/[A-Z]{2,5}" | sort -u | head -20
else
    echo "❌ No log file to analyze"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "8. TELEGRAM SIGNAL STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -f "bot.log" ]; then
    VIP_COUNT=$(grep -c "✅ VIP #" bot.log 2>/dev/null || echo 0)
    FREE_COUNT=$(grep -c "✅ FREE #" bot.log 2>/dev/null || echo 0)
    
    echo "  VIP Signals Sent: $VIP_COUNT"
    echo "  FREE Signals Sent: $FREE_COUNT"
    echo ""
    echo "Latest 5 signals:"
    grep "✅ VIP #\|✅ FREE #" bot.log | tail -5
else
    echo "❌ No log file"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "9. KEY FILES PRESENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Core systems:"
for file in \
    "COMPLETE_ULTIMATE_ORCHESTRATOR.py" \
    "DYNAMIC_PAIR_DISCOVERY.py" \
    "ultra_scout.py" \
    "SMART_SCALPING_ENGINE.py" \
    "EXECUTION_ORCHESTRATOR.py" \
    "TELEGRAM_ORCHESTRATOR.py" \
    "LEARNING_ORCHESTRATOR.py" \
    "IBM_QUANTUM_ENGINE.py"
do
    if [ -f "$file" ]; then
        echo "  ✅ $file ($(wc -l < "$file") lines)"
    else
        echo "  ❌ $file (missing)"
    fi
done
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "10. ENVIRONMENT VARIABLES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -f ".env" ]; then
    echo "✅ .env file exists"
    echo ""
    echo "Variables loaded (values hidden for security):"
    grep -E "^[A-Z_]+" .env | sed 's/=.*/=***/' | head -20
else
    echo "❌ No .env file"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "11. DATA FLOW ANALYSIS (from code)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
if [ -n "$ORCHESTRATOR" ]; then
    echo "Checking data flow in $ORCHESTRATOR:"
    echo ""
    echo "Scout engines:"
    grep -n "scout" "$ORCHESTRATOR" -i | head -10
    echo ""
    echo "Learning/Analysis:"
    grep -n "learn\|analyz" "$ORCHESTRATOR" -i | head -10
    echo ""
    echo "Decision/Trading:"
    grep -n "decision\|trade\|signal" "$ORCHESTRATOR" -i | head -10
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "12. SYSTEM RESOURCE USAGE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Memory usage:"
free -h
echo ""
echo "Disk usage:"
df -h . | tail -1
echo ""
echo "CPU load:"
uptime
echo ""

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                 DIAGNOSTICS COMPLETE                      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Save this output and share it to understand the system"
echo ""
