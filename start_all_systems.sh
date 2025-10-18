#!/bin/bash
echo "🚀 Starting All Trading Systems..."

# Start Redis
systemctl start redis-server

# Start brain systems
/workspace/activate_brain.sh

# Start ultra features
/workspace/activate_ultra.sh

# Start main trading bot
systemctl start trading-bot-production.service

echo "✅ All systems started"
echo "Check status with: systemctl status trading-bot-production.service"
echo "View logs with: journalctl -u trading-bot-production.service -f"