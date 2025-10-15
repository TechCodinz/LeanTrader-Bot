#!/bin/bash

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                  🔥 DEPLOYING EXECUTION FIX 🔥                              ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Stopping bot...${NC}"
sudo systemctl stop trading-bot
sleep 2
echo -e "${GREEN}✅ Bot stopped${NC}"
echo ""

echo -e "${YELLOW}Step 2: Pulling latest fixes...${NC}"
git pull origin cursor/integrate-and-unify-existing-trading-bot-components-c04c
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Fixes pulled successfully${NC}"
else
    echo -e "${RED}❌ Git pull failed! Fix manually.${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}Step 3: Restarting bot...${NC}"
sudo systemctl start trading-bot
sleep 3
echo -e "${GREEN}✅ Bot restarted${NC}"
echo ""

echo -e "${YELLOW}Step 4: Checking status...${NC}"
sudo systemctl status trading-bot --no-pager | head -15
echo ""

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                              ║"
echo "║                     ✅ DEPLOYMENT COMPLETE! ✅                              ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

echo -e "${GREEN}🔍 Monitoring logs for execution activity...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop monitoring${NC}"
echo ""
sleep 2

# Monitor logs
tail -f /root/trading_bot/bot.log | grep --line-buffered -iE "signal|decision|executing|trade executed|published"
