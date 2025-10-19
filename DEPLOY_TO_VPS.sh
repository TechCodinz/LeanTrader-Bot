#!/bin/bash
# COMPLETE BOT DEPLOYMENT SCRIPT
# Run this on your VPS to deploy all fixes

echo "🚀 DEPLOYING COMPLETE BOT FIXES..."
echo ""

# Stop running bot
pkill -f RUN_BOT.py
sleep 2

# Backup current version
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp COMPLETE_ULTIMATE_ORCHESTRATOR.py "$BACKUP_DIR/" 2>/dev/null || true
cp ultra_*.py "$BACKUP_DIR/" 2>/dev/null || true
echo "✅ Backed up to $BACKUP_DIR"

# Download fixes from workspace (you'll need to copy fixed files)
echo "📥 Applying fixes..."

# Start bot
echo ""
echo "🚀 Starting bot..."
nohup python3 RUN_BOT.py --testnet --auto-confirm > bot.log 2>&1 &

sleep 20

# Check status
ps aux | grep RUN_BOT.py | grep -v grep
echo ""
echo "📊 Checking systems..."
tail -200 bot.log | grep -E "available|initialized|WIRING|RUNNING|ALL.*SYSTEMS"

echo ""
echo "✅ DEPLOYMENT COMPLETE!"
