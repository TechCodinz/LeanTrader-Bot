#!/bin/bash
##############################################################################
# TRADING BOT - START SCRIPT
# Starts the trading bot as systemd service
##############################################################################

echo "🚀 STARTING TRADING BOT..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if service exists
if ! sudo systemctl list-unit-files | grep -q "trading-bot-live.service"; then
    echo -e "${YELLOW}⚠️  Service not found. Running deployment first...${NC}"
    bash DEPLOY_NEW_VPS.sh
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Starting Trading Bot Service"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Enable service
sudo systemctl enable trading-bot-live

# Start service
sudo systemctl start trading-bot-live

# Wait a moment
sleep 2

# Check status
if sudo systemctl is-active --quiet trading-bot-live; then
    echo -e "${GREEN}✅ Trading bot started successfully!${NC}"
    echo ""
    
    # Show status
    sudo systemctl status trading-bot-live --no-pager -l
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Monitoring Commands"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📈 View live logs:"
    echo "   ${BLUE}sudo journalctl -u trading-bot-live -f${NC}"
    echo ""
    echo "📊 View last 100 lines:"
    echo "   ${BLUE}sudo journalctl -u trading-bot-live -n 100${NC}"
    echo ""
    echo "🔍 Check status:"
    echo "   ${BLUE}sudo systemctl status trading-bot-live${NC}"
    echo ""
    echo "🛑 Stop bot:"
    echo "   ${BLUE}sudo systemctl stop trading-bot-live${NC}"
    echo ""
    echo "🔄 Restart bot:"
    echo "   ${BLUE}sudo systemctl restart trading-bot-live${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${GREEN}🤖 Trading bot is now running!${NC}"
    echo ""
    
    # Auto-show logs
    read -p "Show live logs now? (y/n): " show_logs
    if [ "$show_logs" = "y" ] || [ "$show_logs" = "Y" ]; then
        sudo journalctl -u trading-bot-live -f
    fi
else
    echo -e "${RED}❌ Failed to start trading bot${NC}"
    echo ""
    echo "Checking logs..."
    sudo journalctl -u trading-bot-live -n 50
fi
