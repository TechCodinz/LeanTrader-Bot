#!/bin/bash
##############################################################################
# SETUP AUTOMATIC DAILY BACKUPS
# Configures cron to backup learned data to GitHub every day
##############################################################################

echo "⏰ SETTING UP AUTOMATIC DAILY BACKUPS..."
echo ""

BOT_DIR=$(pwd)

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📅 Creating Backup Schedule"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create cron job file
CRON_FILE="/tmp/trading_bot_cron"

# Backup learned data every 6 hours
cat > "$CRON_FILE" << EOF
# Trading Bot - Auto-backup learned data to GitHub
# Runs every 6 hours to preserve ML models and training data

# Every 6 hours: Backup learned data
0 */6 * * * cd $BOT_DIR && bash AUTO_BACKUP_LEARNED_DATA.sh >> $BOT_DIR/logs/auto_backup.log 2>&1

# Daily at 2 AM: Full backup
0 2 * * * cd $BOT_DIR && bash BACKUP_TO_GITHUB.sh >> $BOT_DIR/logs/full_backup.log 2>&1

EOF

# Install cron job
crontab -l 2>/dev/null > /tmp/current_cron || true
cat "$CRON_FILE" >> /tmp/current_cron
crontab /tmp/current_cron
rm "$CRON_FILE" /tmp/current_cron

echo "✅ Cron jobs configured:"
echo ""
echo "   📦 Every 6 hours: Learned data backup"
echo "   📦 Daily at 2 AM: Full backup"
echo ""

# Create log directory
mkdir -p logs
touch logs/auto_backup.log
touch logs/full_backup.log

echo "✅ Log files created"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Testing Backup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Test backup script
echo "🧪 Running test backup..."
bash AUTO_BACKUP_LEARNED_DATA.sh

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ AUTO-BACKUP CONFIGURED!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📅 Backup Schedule:"
echo "   • Every 6 hours: Learned data → GitHub"
echo "   • Daily at 2 AM: Full backup → GitHub"
echo ""
echo "📊 View backup logs:"
echo "   ${BLUE}tail -f logs/auto_backup.log${NC}"
echo ""
echo "🔍 Check cron jobs:"
echo "   ${BLUE}crontab -l${NC}"
echo ""
echo "🎯 Your bot's intelligence will NEVER be lost!"
echo ""
