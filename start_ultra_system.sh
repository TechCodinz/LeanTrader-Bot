#!/bin/bash
echo "🚀 Starting Ultra Trading System..."
echo "=================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root: sudo $0"
    exit 1
fi

# Check if systemd services exist
if [ ! -f "/etc/systemd/system/leantrader.service" ]; then
    echo "❌ LeanTrader service not found. Please run deployment first."
    exit 1
fi

echo "📊 Checking system status..."

# Start the auto environment router (this will choose live/testnet based on balance)
echo "🔄 Starting auto environment router..."
systemctl start leantrader-router
if [ $? -eq 0 ]; then
    echo "✅ Router started successfully"
else
    echo "❌ Failed to start router"
    systemctl status leantrader-router --no-pager
fi

# Start the main orchestrator
echo "🧠 Starting main orchestrator..."
systemctl start leantrader
if [ $? -eq 0 ]; then
    echo "✅ Main orchestrator started successfully"
else
    echo "❌ Failed to start main orchestrator"
    systemctl status leantrader --no-pager
fi

echo ""
echo "🎉 Ultra Trading System is starting!"
echo "=================================="
echo ""
echo "📊 Monitor with these commands:"
echo "  systemctl status leantrader"
echo "  journalctl -u leantrader -f"
echo "  curl http://localhost:9300/metrics"
echo ""
echo "🔍 Check logs:"
echo "  tail -f /var/log/leantrader/orchestrator.log"
echo ""
echo "💰 Ready to make profits! 🚀"
