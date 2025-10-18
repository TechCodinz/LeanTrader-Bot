#!/bin/bash
echo "🔍 System Health Check - $(date)"

# Check Python processes
echo "Python processes:"
ps aux | grep python3 | grep -v grep

# Check system resources
echo "Memory usage:"
free -h

echo "Disk usage:"
df -h

# Check logs
echo "Recent logs:"
tail -n 20 /workspace/logs/*.log 2>/dev/null || echo "No log files found"

# Check services
echo "Systemd services:"
systemctl status trading-bot-production.service --no-pager

echo "✅ Health check completed"