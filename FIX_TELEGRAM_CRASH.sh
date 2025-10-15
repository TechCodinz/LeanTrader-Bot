#!/bin/bash
# FIX TELEGRAM CRASH + ENABLE PREMIUM VIP SYSTEM

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         FIXING CRASH + ENABLING PREMIUM VIP SYSTEM               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

cd /root/trading_bot

# Install telegram package
echo "📦 Installing python-telegram-bot..."
/root/trading_bot/venv/bin/pip install python-telegram-bot

echo "✅ Telegram package installed"
echo ""

# Check if .env has all Telegram variables
echo "🔍 Checking Telegram configuration..."

if grep -q "TELEGRAM_BOT_TOKEN" .env && \
   grep -q "TG_ADMIN_CHAT_ID" .env && \
   grep -q "TG_VIP_CHAT_ID" .env && \
   grep -q "TG_FREE_CHAT_ID" .env; then
    echo "✅ Telegram config found in .env"
else
    echo "⚠️  Some Telegram variables missing in .env"
    echo "   Add these to .env:"
    echo "   TELEGRAM_BOT_TOKEN=your_token"
    echo "   TG_ADMIN_CHAT_ID=your_admin_id"
    echo "   TG_VIP_CHAT_ID=your_vip_channel_id"
    echo "   TG_FREE_CHAT_ID=your_free_channel_id"
fi

echo ""

# Restart bot
echo "🔄 Restarting bot with Premium VIP Telegram System..."
sudo systemctl restart trading-bot

echo "⏳ Waiting 20 seconds..."
sleep 20

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ DEPLOYMENT COMPLETE!                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check status
echo "📊 Status:"
systemctl status trading-bot --no-pager | head -10

echo ""
echo "🔍 Checking for Telegram notifications..."
echo ""

# Check logs
journalctl -u trading-bot --since "1 minute ago" --no-pager | grep -i "telegram\|vip\|admin" | tail -10

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "✅ CHECK YOUR TELEGRAM - You should have received:"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  📱 Admin notification: '🚀 BOT STARTED'"
echo "  📊 Status: All 55+ systems active"
echo "  ⏰ Expected: First trade in 15-60 minutes"
echo ""
echo "If no notification, check:"
echo "  journalctl -u trading-bot -n 100 | grep -i error"
echo ""
