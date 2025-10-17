#!/bin/bash
# CHECK WHAT'S RUNNING ON VPS
# Run this to see all active bots

echo "================================================================================"
echo "🔍 CHECKING WHAT'S RUNNING ON YOUR VPS"
echo "================================================================================"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. PYTHON PROCESSES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python_procs=$(ps aux | grep python | grep -v grep | grep -v "ps aux")

if [ -z "$python_procs" ]; then
    echo "✅ No Python bot processes running"
else
    echo "⚠️  FOUND RUNNING PYTHON PROCESSES:"
    echo ""
    ps aux | grep python | grep -v grep | grep -v "ps aux"
    echo ""
    echo "To kill these, note the PID (2nd column) and run:"
    ps aux | grep python | grep -v grep | grep -v "ps aux" | awk '{print "kill -9 " $2}'
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. SCREEN SESSIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

screen_output=$(screen -ls 2>&1)

if echo "$screen_output" | grep -q "No Sockets found"; then
    echo "✅ No screen sessions running"
else
    echo "⚠️  FOUND ACTIVE SCREEN SESSIONS:"
    echo ""
    screen -ls
    echo ""
    echo "To kill these, run:"
    screen -ls | grep -E "[0-9]+\." | awk '{print $1}' | cut -d. -f1 | while read pid; do
        echo "screen -X -S $pid quit"
    done
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. BOT DIRECTORIES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cd ~

bot_dirs=$(ls -d *bot* *trade* *lean* 2>/dev/null)

if [ -z "$bot_dirs" ]; then
    echo "✅ No bot directories found"
else
    echo "⚠️  FOUND BOT DIRECTORIES:"
    echo ""
    ls -lhd *bot* *trade* *lean* 2>/dev/null
    echo ""
    echo "To remove these:"
    echo "rm -rf $(ls -d *bot* *trade* *lean* 2>/dev/null | tr '\n' ' ')"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. SYSTEMD SERVICES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bot_services=$(systemctl list-units --type=service --state=running | grep -i "bot\|trade" 2>/dev/null)

if [ -z "$bot_services" ]; then
    echo "✅ No bot services running"
else
    echo "⚠️  FOUND SYSTEMD SERVICES:"
    echo ""
    systemctl list-units --type=service --state=running | grep -i "bot\|trade"
    echo ""
    echo "To stop these:"
    systemctl list-units --type=service --state=running | grep -i "bot\|trade" | awk '{print "sudo systemctl stop " $1}'
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "5. NOHUP PROCESSES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

nohup_procs=$(ps aux | grep nohup | grep python | grep -v grep)

if [ -z "$nohup_procs" ]; then
    echo "✅ No nohup processes"
else
    echo "⚠️  FOUND NOHUP PROCESSES:"
    echo ""
    ps aux | grep nohup | grep python | grep -v grep
    echo ""
    echo "To kill:"
    ps aux | grep nohup | grep python | grep -v grep | awk '{print "kill -9 " $2}'
fi

echo ""
echo "================================================================================"
echo "📊 SUMMARY"
echo "================================================================================"
echo ""

# Count issues
issues=0

if ps aux | grep python | grep -v grep | grep -v "ps aux" > /dev/null; then
    echo "❌ Python processes running"
    issues=$((issues+1))
else
    echo "✅ No Python processes"
fi

if ! screen -ls 2>&1 | grep -q "No Sockets found"; then
    echo "❌ Screen sessions active"
    issues=$((issues+1))
else
    echo "✅ No screen sessions"
fi

if systemctl list-units --type=service --state=running | grep -qi "bot\|trade"; then
    echo "❌ Systemd services running"
    issues=$((issues+1))
else
    echo "✅ No systemd services"
fi

echo ""

if [ $issues -eq 0 ]; then
    echo "✅✅✅ VPS IS CLEAN! READY TO DEPLOY NEW BOT! ✅✅✅"
else
    echo "⚠️  Found $issues active bot(s)"
    echo ""
    echo "Run these to kill everything:"
    echo "  pkill -9 -f python"
    echo "  screen -wipe"
    echo "  sudo systemctl stop trading-bot 2>/dev/null"
fi

echo ""
echo "================================================================================"
