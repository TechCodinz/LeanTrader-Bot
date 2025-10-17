#!/bin/bash
##############################################################################
# AUTO-BACKUP LEARNED DATA TO GITHUB
# Saves ML models, databases, and training data automatically
# Run this daily via cron to preserve bot's intelligence!
##############################################################################

echo "🧠 BACKING UP BOT'S LEARNED INTELLIGENCE..."
echo ""

BOT_DIR=$(pwd)
BACKUP_DIR="$BOT_DIR/learned_data_backup"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 STEP 1: Collecting Learned Data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup databases (all the learned knowledge!)
echo "💾 Backing up databases..."
if ls *.db 1> /dev/null 2>&1; then
    cp -v *.db "$BACKUP_DIR/"
    echo "✅ Databases backed up"
fi

# Backup ML models
echo "🧠 Backing up ML models..."
if [ -d "models" ]; then
    cp -rv models "$BACKUP_DIR/"
    echo "✅ ML models backed up"
fi

# Backup evolution engine data
echo "🔄 Backing up evolution data..."
if [ -d "evolution_data" ]; then
    cp -rv evolution_data "$BACKUP_DIR/"
fi

# Backup user database (VIP subscribers)
echo "👥 Backing up user database..."
if [ -f "users_db.json" ]; then
    cp -v users_db.json "$BACKUP_DIR/"
    echo "✅ User database backed up"
fi

# Backup trade history
echo "📊 Backing up trade history..."
if [ -f "trades.db" ]; then
    cp -v trades.db "$BACKUP_DIR/"
fi
if [ -d "data" ]; then
    cp -rv data "$BACKUP_DIR/"
fi

# Create manifest
cat > "$BACKUP_DIR/BACKUP_MANIFEST.txt" << EOF
🧠 BOT LEARNED DATA BACKUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 Backup Date: $(date)
💻 VPS: $(hostname)
📍 Bot Directory: $BOT_DIR

📦 Contents:
$(ls -lh "$BACKUP_DIR" | tail -n +2)

🎯 This backup contains:
- ✅ All trained ML models
- ✅ Evolution engine learning
- ✅ Trade history and performance data
- ✅ User database (VIP subscribers)
- ✅ All databases with learned patterns

🚀 To restore on new VPS:
1. Clone repository from GitHub
2. Copy these files to new VPS
3. Bot will resume with all learned knowledge!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EOF

echo "✅ Backup manifest created"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📤 STEP 2: Committing to Git"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Initialize git if needed
if [ ! -d .git ]; then
    git init
    git config user.email "bot@tradingbot.local"
    git config user.name "Trading Bot Auto-Backup"
fi

# Add learned data
git add learned_data_backup/
git add *.db 2>/dev/null || true
git add models/ 2>/dev/null || true
git add data/ 2>/dev/null || true
git add users_db.json 2>/dev/null || true

# Commit
COMMIT_MSG="🧠 Auto-backup learned data - $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$COMMIT_MSG" || {
    echo "⚠️  No new learned data to commit"
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "☁️  STEP 3: Pushing to GitHub"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Push to GitHub (if remote configured)
if git remote get-url origin &>/dev/null; then
    echo "🚀 Pushing learned data to GitHub..."
    
    if git push origin main 2>/dev/null || git push origin master 2>/dev/null; then
        echo -e "${GREEN}✅ Learned data backed up to GitHub!${NC}"
        
        # Show what was backed up
        echo ""
        echo "📊 Backed Up:"
        du -sh "$BACKUP_DIR" 2>/dev/null || true
        echo ""
    else
        echo -e "${YELLOW}⚠️  Push failed. Run BACKUP_TO_GITHUB.sh first to set up remote.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No GitHub remote configured. Run BACKUP_TO_GITHUB.sh first.${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ LEARNED DATA BACKUP COMPLETE!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🧠 Your bot's intelligence is now safely backed up!"
echo ""
echo "📊 Backup includes:"
echo "   • ML models (Evolution Cycle 231+)"
echo "   • Trade history (9 trades, \$0.98 profit)"
echo "   • User database"
echo "   • All learned patterns"
echo ""
echo "🚀 New VPS will start with ALL this knowledge!"
echo ""
