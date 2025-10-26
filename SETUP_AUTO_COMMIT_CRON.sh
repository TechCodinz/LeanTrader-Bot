#!/bin/bash
###############################################################################
# SETUP AUTO-COMMIT CRON JOB
# Sets up automatic commit of learned data every 10 minutes
###############################################################################

WORKSPACE="/workspace"
CRON_CMD="*/10 * * * * $WORKSPACE/AUTO_COMMIT.sh >> $WORKSPACE/logs/auto_commit.log 2>&1"

echo "════════════════════════════════════════════════════════════════"
echo "Setting up auto-commit cron job"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "AUTO_COMMIT.sh"; then
    echo "✅ Auto-commit cron job already exists!"
    echo ""
    echo "Current cron jobs:"
    crontab -l | grep AUTO_COMMIT
else
    # Add cron job
    echo "📥 Adding auto-commit cron job..."
    
    # Get current crontab, add new job
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    
    if [ $? -eq 0 ]; then
        echo "✅ Auto-commit cron job added successfully!"
        echo ""
        echo "Schedule: Every 10 minutes"
        echo "Command:  $CRON_CMD"
        echo ""
        echo "Current cron jobs:"
        crontab -l | grep AUTO_COMMIT
    else
        echo "❌ Failed to add cron job"
        echo ""
        echo "Manual setup:"
        echo "1. Run: crontab -e"
        echo "2. Add this line:"
        echo "   $CRON_CMD"
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Auto-commit will save learned data every 10 minutes"
echo "This ensures knowledge persists across VPS restarts/deployments"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To check logs: tail -f $WORKSPACE/logs/auto_commit.log"
echo "To manually commit: $WORKSPACE/AUTO_COMMIT.sh"
echo ""
