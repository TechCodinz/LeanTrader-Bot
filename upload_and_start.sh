#!/bin/bash

# Upload and Start Bybit Trading Bot
# This script will upload all files and start the bot

set -e

VPS_IP="75.119.149.117"
VPS_USER="root"

echo "🚀 Uploading and Starting Bybit Trading Bot"
echo "==========================================="
echo "📍 VPS: $VPS_IP"
echo "👤 User: $VPS_USER"
echo "🔑 API: Bybit Testnet"
echo ""

# Create a simple tar archive
echo "📦 Creating deployment package..."
tar -czf trading-bot-deploy.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='venv' \
    --exclude='data' \
    --exclude='logs' \
    --exclude='models' \
    .

echo "📤 Uploading to VPS..."
scp trading-bot-deploy.tar.gz $VPS_USER@$VPS_IP:/tmp/

echo "🔧 Setting up on VPS..."
ssh $VPS_USER@$VPS_IP << 'EOF'
    # Create directory and extract files
    mkdir -p /home/root/trading-bot
    cd /home/root/trading-bot
    tar -xzf /tmp/trading-bot-deploy.tar.gz
    rm /tmp/trading-bot-deploy.tar.gz

    # Make scripts executable
    chmod +x deploy.sh
    chmod +x start_dashboard.sh
    chmod +x monitor.sh

    echo "✅ Files extracted successfully"

    # Run the deployment script
    echo "🚀 Running deployment..."
    ./deploy.sh

    echo "✅ Deployment complete!"
    echo ""
    echo "🎯 Starting the trading bot..."
    cd /home/root/trading-bot
    source venv/bin/activate
    nohup python main.py > bot.log 2>&1 &

    # Start dashboard
    echo "📊 Starting dashboard..."
    nohup streamlit run dashboard_app.py --server.port 8501 --server.address 0.0.0.0 > dashboard.log 2>&1 &

    echo "✅ Trading bot started!"
    echo "📊 Dashboard: http://75.119.149.117:8501"
    echo "📋 Bot logs: tail -f /home/root/trading-bot/bot.log"
EOF

# Clean up
rm trading-bot-deploy.tar.gz

echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo "✅ Trading bot deployed and started"
echo "✅ Bybit testnet API configured"
echo "✅ Dashboard running"
echo ""
echo "📊 Access your bot:"
echo "• Dashboard: http://75.119.149.117:8501"
echo "• SSH: ssh $VPS_USER@$VPS_IP"
echo "• Bot logs: ssh $VPS_USER@$VPS_IP 'tail -f /home/root/trading-bot/bot.log'"
echo ""
echo "🔧 Management commands:"
echo "• Check status: ssh $VPS_USER@$VPS_IP 'ps aux | grep python'"
echo "• Stop bot: ssh $VPS_USER@$VPS_IP 'pkill -f main.py'"
echo "• Restart: ssh $VPS_USER@$VPS_IP 'cd /home/root/trading-bot && python main.py &'"
echo ""
echo "🚀 Your Bybit trading bot is now live and learning!"