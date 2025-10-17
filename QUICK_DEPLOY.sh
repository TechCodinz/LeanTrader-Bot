#!/bin/bash
cd /root/trading_bot
echo "🔄 Pulling latest code with all new features..."
git pull origin cursor/check-and-update-trading-bot-service-0f23
echo ""
echo "✅ Code pulled! Making deployment script executable..."
chmod +x DEPLOY_COMPLETE_UPGRADE.sh
echo ""
echo "🚀 Running deployment..."
bash DEPLOY_COMPLETE_UPGRADE.sh
