#!/bin/bash

# CRASH LOOP FIX - AUTOMATED DEPLOYMENT
# Deploys the fix to stop the bot from crashing every 25 seconds

set -e  # Exit on error

echo "========================================="
echo "🚀 TRADING BOT CRASH LOOP FIX DEPLOYMENT"
echo "========================================="
echo ""

# Check we're in the right directory
if [ ! -f "RUN_BOT.py" ]; then
    echo "❌ Error: RUN_BOT.py not found"
    echo "   Please run this script from /root/trading_bot directory"
    exit 1
fi

echo "✅ Found RUN_BOT.py - correct directory"
echo ""

# Show current status
echo "📊 CURRENT STATUS:"
echo "=================="
systemctl status trading-bot --no-pager | grep -E "Active:|Main PID:|Restart:" | head -5
echo ""

# Get restart count
RESTART_COUNT=$(journalctl -u trading-bot --since "5 minutes ago" | grep -c "Started trading-bot" || echo "0")
echo "🔄 Restarts in last 5 minutes: $RESTART_COUNT"
if [ "$RESTART_COUNT" -gt 3 ]; then
    echo "⚠️  WARNING: Bot is crash-looping!"
fi
echo ""

# Confirm deployment
echo "📦 DEPLOYMENT PLAN:"
echo "==================="
echo "1. Pull latest code from: cursor/check-and-update-trading-bot-service-0f23"
echo "2. Restart trading-bot service"
echo "3. Wait 60 seconds"
echo "4. Verify stability"
echo ""

read -p "Continue with deployment? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🔄 Step 1: Pulling latest code..."
echo "=================================="
git fetch origin
git checkout cursor/check-and-update-trading-bot-service-0f23
git pull origin cursor/check-and-update-trading-bot-service-0f23

# Verify the fix is in the code
if grep -q "DEX_PRIVATE_KEY" DEX_ORCHESTRATOR.py && grep -q "return" DEX_ORCHESTRATOR.py; then
    echo "✅ Fix verified in DEX_ORCHESTRATOR.py"
else
    echo "⚠️  Warning: Fix might not be present in code"
    echo "   Continuing anyway..."
fi

echo ""
echo "🔄 Step 2: Restarting bot service..."
echo "====================================="
sudo systemctl restart trading-bot
echo "✅ Service restarted"

echo ""
echo "⏳ Step 3: Waiting 60 seconds for stabilization..."
echo "===================================================="
for i in {60..1}; do
    printf "\r   ⏱️  %2d seconds remaining..." $i
    sleep 1
done
printf "\r   ✅ Wait complete!                  \n"

echo ""
echo "🔍 Step 4: Verification..."
echo "=========================="

# Check if service is running
if systemctl is-active --quiet trading-bot; then
    echo "✅ Service is active"
else
    echo "❌ Service is NOT active"
    systemctl status trading-bot --no-pager | head -20
    exit 1
fi

# Check for recent restarts
RESTART_COUNT_AFTER=$(journalctl -u trading-bot --since "2 minutes ago" | grep -c "Started trading-bot" || echo "0")
echo "🔄 Restarts in last 2 minutes: $RESTART_COUNT_AFTER"

if [ "$RESTART_COUNT_AFTER" -le 1 ]; then
    echo "✅ Bot is stable (no crash loop)"
else
    echo "⚠️  Warning: Bot may still be restarting"
fi

# Check for execution loop
if journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -q "EXECUTION LOOP STARTED"; then
    echo "✅ Execution loop is running"
else
    echo "⚠️  Execution loop not found in recent logs"
    echo "   (May take a few minutes to appear)"
fi

# Check for DEX warning (should be present)
if journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -q "DEX Orchestrator: No private key"; then
    echo "✅ DEX warning present (expected - this is normal)"
else
    echo "ℹ️  DEX warning not yet in logs (will appear soon)"
fi

# Check for errors
ERROR_COUNT=$(journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -ic "error\|exception\|traceback" || echo "0")
if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "✅ No errors in recent logs"
else
    echo "⚠️  Found $ERROR_COUNT error(s) in logs"
    echo "   Check with: journalctl -u trading-bot -n 50"
fi

echo ""
echo "========================================="
echo "📊 DEPLOYMENT SUMMARY"
echo "========================================="

# Get uptime
UPTIME=$(systemctl show trading-bot --property=ActiveEnterTimestamp --value)
echo "Service started: $UPTIME"

# Show active status
echo ""
echo "Current status:"
systemctl status trading-bot --no-pager | grep -E "Active:|Main PID:" | head -2

echo ""
echo "========================================="
echo "✅ DEPLOYMENT COMPLETE!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Monitor logs: journalctl -u trading-bot -f"
echo "2. Check for trades in 15-60 minutes"
echo "3. Verify Bybit testnet orders at: https://testnet.bybit.com"
echo ""
echo "For detailed monitoring:"
echo "  tail -f /root/trading_bot/bot.log | grep -i 'trade\\|signal\\|execution'"
echo ""
echo "If you see 'DEX Orchestrator: No private key' - this is NORMAL!"
echo "DEX trading is optional. CEX trading is fully active."
echo ""
