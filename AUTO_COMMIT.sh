#!/bin/bash
###############################################################################
# AUTO COMMIT - Automatically commit learned data and improvements
# Runs every 10 minutes to save bot's learning progress
###############################################################################

WORKSPACE="/workspace"
LOG_FILE="$WORKSPACE/logs/auto_commit.log"

mkdir -p "$WORKSPACE/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

cd "$WORKSPACE" || exit 1

log "════════════════════════════════════════════════════════════════"
log "AUTO-COMMIT: Saving learned data and improvements"
log "════════════════════════════════════════════════════════════════"

# Check if there are changes
if git diff --quiet && git diff --cached --quiet; then
    log "✅ No changes to commit - bot learning state unchanged"
    exit 0
fi

# Show what changed
log ""
log "📊 Changes detected:"
git status --short | tee -a "$LOG_FILE"
log ""

# Add all learned data
log "📥 Adding learned data to git..."

# Databases (learned knowledge)
git add *.db 2>/dev/null

# Data directory (history, patterns, etc.)
git add data/*.csv data/*.json 2>/dev/null

# Runtime memory
git add runtime/*.json 2>/dev/null

# Model weights
git add models/*.pkl 2>/dev/null

# Best parameters
git add best_params.json 2>/dev/null

# Configuration updates
git add *.json *.yml *.yaml 2>/dev/null

# Code improvements (if any)
git add *.py 2>/dev/null

# Show what will be committed
log ""
log "📦 Staging changes:"
git diff --cached --stat | tee -a "$LOG_FILE"
log ""

# Create commit message
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
COMMIT_MSG="🤖 Auto-commit: Saved learned data @ $TIMESTAMP

Automated commit of bot learning progress:
- Database updates (learned patterns, strategies, knowledge)
- Trading history updates
- Pattern memory updates
- Model weight updates
- Configuration updates

This ensures learned knowledge is preserved across VPS deployments.
"

# Commit
log "💾 Committing changes..."
if git commit -m "$COMMIT_MSG"; then
    log "✅ Successfully committed learned data!"
    
    # Show commit details
    log ""
    log "📝 Commit details:"
    git log -1 --stat | tee -a "$LOG_FILE"
    log ""
    
    # Optionally push (commented out for safety - enable if desired)
    # log "🚀 Pushing to remote..."
    # if git push; then
    #     log "✅ Successfully pushed to remote!"
    # else
    #     log "⚠️  Failed to push to remote (will retry next time)"
    # fi
    
else
    log "⚠️  Failed to commit (nothing to commit or error)"
fi

log "════════════════════════════════════════════════════════════════"
log "AUTO-COMMIT: Complete"
log "════════════════════════════════════════════════════════════════"
log ""

exit 0
