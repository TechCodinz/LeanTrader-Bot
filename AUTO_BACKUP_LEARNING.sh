#!/bin/bash
# Auto-backup learning data to Git
# Run this hourly via cron to preserve all learning across deployments

cd /workspace || exit 1

# Backup database files (learning data)
git add *.db evolution_*.json divine_*.json ultimate_*.db 2>/dev/null || true

# Check if there are changes to commit
if git diff --cached --quiet; then
    # No changes, silent exit
    exit 0
fi

# Commit with timestamp
git commit -m "Auto-backup: Learning data $(date '+%Y-%m-%d %H:%M:%S')" 2>&1 | \
    grep -v "nothing to commit\|up to date" || true

echo "$(date '+%Y-%m-%d %H:%M:%S') - Learning data backed up successfully"
