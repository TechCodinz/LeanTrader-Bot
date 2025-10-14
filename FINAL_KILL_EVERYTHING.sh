#!/bin/bash
# FINAL NUCLEAR OPTION - KILL ABSOLUTELY EVERYTHING
# Run this on VPS if still getting notifications

echo "================================================================================"
echo "🔥 NUCLEAR KILL - STOPPING ABSOLUTELY EVERYTHING"
echo "================================================================================"
echo ""

# 1. Kill all Python (with sudo for all users)
echo "1. Killing all Python processes..."
sudo pkill -9 python
sudo pkill -9 -f python
sudo pkill -9 -f "bot"
sudo pkill -9 -f "trade"
sudo pkill -9 -f "venv"
sleep 2

# 2. Kill all screen sessions
echo "2. Killing screen sessions..."
sudo screen -wipe
sudo killall screen 2>/dev/null
sleep 1

# 3. Kill tmux
echo "3. Killing tmux..."
tmux kill-server 2>/dev/null
sudo tmux kill-server 2>/dev/null
sleep 1

# 4. Kill Docker containers
echo "4. Stopping Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null
sleep 2

# 5. Stop systemd services
echo "5. Stopping systemd services..."
for service in trading-bot bot crypto-bot lean-trader; do
    sudo systemctl stop $service 2>/dev/null
    sudo systemctl disable $service 2>/dev/null
done
sleep 2

# 6. Kill PM2
echo "6. Killing PM2..."
pm2 stop all 2>/dev/null
pm2 delete all 2>/dev/null
pm2 kill 2>/dev/null
sudo pm2 kill 2>/dev/null
sleep 1

# 7. Kill supervisor
echo "7. Stopping supervisor..."
sudo supervisorctl stop all 2>/dev/null
sleep 1

# 8. Remove cron jobs
echo "8. Removing cron jobs..."
crontab -r 2>/dev/null
sudo crontab -r 2>/dev/null

# 9. Kill by port
echo "9. Killing processes on common ports..."
sudo lsof -ti:8000 | xargs sudo kill -9 2>/dev/null
sudo lsof -ti:8080 | xargs sudo kill -9 2>/dev/null
sudo lsof -ti:5000 | xargs sudo kill -9 2>/dev/null
sleep 1

# 10. Final Python kill
echo "10. Final Python kill..."
sudo killall -9 python3 2>/dev/null
sudo killall -9 python 2>/dev/null
sleep 3

echo ""
echo "================================================================================"
echo "✅ EVERYTHING KILLED"
echo "================================================================================"
echo ""

# Verification
echo "Verification:"
echo ""

echo "Python processes:"
ps aux | grep python | grep -v grep
echo ""

echo "Screen sessions:"
screen -ls 2>&1
echo ""

echo "Docker containers:"
docker ps 2>/dev/null
echo ""

echo "Systemd bot services:"
systemctl list-units --type=service | grep -i bot
echo ""

# Check if truly empty
if ! ps aux | grep python | grep -v grep | grep -q .; then
    echo "✅✅✅ ALL PYTHON PROCESSES STOPPED! ✅✅✅"
    echo ""
    echo "Now wait 10 minutes and check Telegram."
    echo "If NO new messages arrive = Success! ✅"
    echo ""
    echo "Then you can deploy your new bot!"
else
    echo "⚠️  Some processes still found:"
    ps aux | grep python | grep -v grep
    echo ""
    echo "Manual kill needed:"
    ps aux | grep python | grep -v grep | awk '{print "sudo kill -9 " $2}'
fi

echo ""
echo "================================================================================"
