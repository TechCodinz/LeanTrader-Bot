#!/bin/bash
###############################################################################
# SYNC WORKSPACE TO VPS - Deploy updated bot with 3000+ pair discovery
# Run this FROM THE WORKSPACE to deploy to VPS
###############################################################################

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     DEPLOY UPDATED BOT TO VPS (3000+ Pairs Discovery)     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if VPS IP is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <VPS_IP> [user]"
    echo ""
    echo "Example:"
    echo "  $0 vmi2817884.contaboserver.net root"
    echo "  $0 123.45.67.89"
    echo ""
    exit 1
fi

VPS_IP="$1"
VPS_USER="${2:-root}"
VPS_DIR="~/trading_bot"

echo "🎯 Target VPS: $VPS_USER@$VPS_IP:$VPS_DIR"
echo ""

# Create essential files list
echo "📦 Creating deployment package..."

# Create a temporary deployment directory
DEPLOY_DIR="/tmp/bot_deploy_$(date +%s)"
mkdir -p "$DEPLOY_DIR"

# Copy essential files
echo "  Copying core files..."
cp -r /workspace/*.py "$DEPLOY_DIR/" 2>/dev/null || true
cp -r /workspace/*.sh "$DEPLOY_DIR/" 2>/dev/null || true
cp /workspace/.env "$DEPLOY_DIR/" 2>/dev/null || true
cp /workspace/requirements.txt "$DEPLOY_DIR/" 2>/dev/null || true

# Copy directories if they exist
for dir in core systems engines brokers allocators analytics brain cli; do
    if [ -d "/workspace/$dir" ]; then
        echo "  Copying $dir/..."
        cp -r "/workspace/$dir" "$DEPLOY_DIR/"
    fi
done

echo "✅ Package created: $DEPLOY_DIR"
echo ""

# Count files
FILE_COUNT=$(find "$DEPLOY_DIR" -type f | wc -l)
SIZE=$(du -sh "$DEPLOY_DIR" | cut -f1)
echo "  Files: $FILE_COUNT"
echo "  Size: $SIZE"
echo ""

# Sync to VPS
echo "🚀 Deploying to VPS..."
echo ""

# Create backup script on VPS first
echo "  Creating backup on VPS..."
ssh "$VPS_USER@$VPS_IP" "cd $VPS_DIR && mkdir -p backups && cp -r . backups/backup_\$(date +%Y%m%d_%H%M%S) 2>/dev/null || true"

# Rsync the files (preserves running bot)
echo "  Syncing files..."
rsync -avz --exclude='bot.log' --exclude='*.log' --exclude='__pycache__' \
    "$DEPLOY_DIR/" "$VPS_USER@$VPS_IP:$VPS_DIR/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Files synced successfully!"
    echo ""
    
    # Make scripts executable
    echo "  Setting permissions..."
    ssh "$VPS_USER@$VPS_IP" "cd $VPS_DIR && chmod +x *.sh 2>/dev/null || true"
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                 DEPLOYMENT SUCCESSFUL!                    ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    echo "📋 NEXT STEPS:"
    echo ""
    echo "1. SSH into your VPS:"
    echo "   ssh $VPS_USER@$VPS_IP"
    echo ""
    echo "2. Restart the bot to use new code:"
    echo "   cd $VPS_DIR"
    echo "   ./stop_bot.sh && sleep 2 && ./start_bot.sh"
    echo ""
    echo "3. Verify it's discovering pairs:"
    echo "   tail -f bot.log | grep 'TOTAL DISCOVERED'"
    echo ""
    echo "4. Monitor signals:"
    echo "   tail -f bot.log | grep '✅'"
    echo ""
    
    # Cleanup
    rm -rf "$DEPLOY_DIR"
    
else
    echo ""
    echo "❌ Deployment failed!"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check SSH connection: ssh $VPS_USER@$VPS_IP"
    echo "  2. Check VPS directory exists: $VPS_DIR"
    echo "  3. Check SSH key is configured"
    echo ""
    
    # Cleanup
    rm -rf "$DEPLOY_DIR"
    exit 1
fi
