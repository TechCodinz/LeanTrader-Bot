# 🎮 ULTRA TRADING BOT - MANAGEMENT COMMANDS

Complete command reference for managing your trading bot on VPS.

---

## 🚀 DEPLOYMENT (First Time Setup)

### Deploy to New VPS
```bash
# One-command deployment (run as root):
curl -sSL https://raw.githubusercontent.com/TechCodinz/Lean-Trader/main/DEPLOY_TO_NEW_VPS.sh | bash

# Or manual download:
wget https://raw.githubusercontent.com/TechCodinz/Lean-Trader/main/DEPLOY_TO_NEW_VPS.sh
chmod +x DEPLOY_TO_NEW_VPS.sh
./DEPLOY_TO_NEW_VPS.sh
```

### Configure API Keys
```bash
nano /root/trading_bot/.env
# Edit GATEIO_LIVE_API_KEY and GATEIO_LIVE_SECRET
# Save: Ctrl+O, Enter, Ctrl+X
```

---

## ⚡ BASIC BOT CONTROL

### Start Bot
```bash
# Start live trading:
sudo systemctl start trading-bot-live

# Start testnet (learning mode):
sudo systemctl start trading-bot-testnet

# Start both:
sudo systemctl start trading-bot-live trading-bot-testnet
```

### Stop Bot
```bash
# Stop live trading:
sudo systemctl stop trading-bot-live

# Stop testnet:
sudo systemctl stop trading-bot-testnet

# Stop both:
sudo systemctl stop trading-bot-live trading-bot-testnet
```

### Restart Bot
```bash
# Restart after code updates:
sudo systemctl restart trading-bot-live

# Restart testnet:
sudo systemctl restart trading-bot-testnet
```

### Check Status
```bash
# Quick status:
sudo systemctl status trading-bot-live

# Detailed status with recent logs:
sudo systemctl status trading-bot-live -l --no-pager
```

### Enable/Disable Auto-Start
```bash
# Enable auto-start on server reboot:
sudo systemctl enable trading-bot-live

# Disable auto-start:
sudo systemctl disable trading-bot-live
```

---

## 📊 MONITORING & LOGS

### View Live Logs (Real-Time)
```bash
# Follow live logs:
sudo journalctl -u trading-bot-live -f

# Follow with color:
sudo journalctl -u trading-bot-live -f | grep --color=always -E "ERROR|WARNING|✅|🔮|💰|$"

# Follow last 100 lines:
sudo journalctl -u trading-bot-live -n 100 -f
```

### View Recent Activity
```bash
# Last 50 lines:
sudo journalctl -u trading-bot-live -n 50

# Last 5 minutes:
sudo journalctl -u trading-bot-live --since "5 min ago"

# Last 1 hour:
sudo journalctl -u trading-bot-live --since "1 hour ago"

# Today's logs:
sudo journalctl -u trading-bot-live --since today
```

### Check Specific Systems
```bash
# Check all 4 Ultra AI systems:
sudo journalctl -u trading-bot-live --since "5 min ago" | grep -E "🔮 Quantum:|🌌 ULTRASONIC:|💎 GOLDMINE:|🔮 DIVINE:"

# Check trading activity:
sudo journalctl -u trading-bot-live --since "10 min ago" | grep -E "BUY|SELL|Trade executed|Profit"

# Check for errors:
sudo journalctl -u trading-bot-live --since "10 min ago" | grep -i error

# Check system cycles:
sudo journalctl -u trading-bot-live --since "5 min ago" | grep "COMPLETE CYCLE"
```

### Export Logs
```bash
# Save logs to file:
sudo journalctl -u trading-bot-live --since "1 hour ago" > bot_logs.txt

# Save with timestamps:
sudo journalctl -u trading-bot-live --since today --output short-iso > bot_logs_$(date +%Y%m%d).txt
```

---

## 🔄 CODE UPDATES

### Update from GitHub
```bash
# Navigate to bot directory:
cd /root/trading_bot

# Pull latest code:
git fetch origin main
git pull origin main

# Restart bot with new code:
sudo systemctl restart trading-bot-live

# Verify new code is running:
sudo journalctl -u trading-bot-live --since "1 min ago" | head -30
```

### Update Specific File
```bash
cd /root/trading_bot

# Pull specific file:
git fetch origin main
git checkout origin/main -- FILENAME.py

# Restart:
sudo systemctl restart trading-bot-live
```

### Rollback to Previous Version
```bash
cd /root/trading_bot

# View commit history:
git log --oneline -n 10

# Rollback to specific commit:
git reset --hard COMMIT_HASH

# Restart:
sudo systemctl restart trading-bot-live
```

---

## 🔧 TROUBLESHOOTING

### Bot Not Starting
```bash
# Check for errors in startup:
sudo journalctl -u trading-bot-live --since "2 min ago" | grep -i "error\|exception\|failed"

# Check if Python process is running:
ps aux | grep RUN_BOT.py

# Check API keys are set:
grep "GATEIO_LIVE_API_KEY" /root/trading_bot/.env
```

### High Memory Usage
```bash
# Check current memory:
free -h

# Check bot memory:
sudo systemctl status trading-bot-live | grep Memory

# Stop testnet to free memory:
sudo systemctl stop trading-bot-testnet

# Clear old logs:
sudo journalctl --vacuum-time=2d
```

### Bot Keeps Restarting
```bash
# Check restart count:
sudo systemctl status trading-bot-live | grep "Loaded\|Active"

# Check crash logs:
sudo journalctl -u trading-bot-live --since "10 min ago" | grep -A 10 "Traceback\|Exception"

# Disable auto-restart temporarily:
sudo systemctl stop trading-bot-live
```

### API Errors
```bash
# Check API connection:
sudo journalctl -u trading-bot-live --since "5 min ago" | grep -i "gate\|api\|auth"

# Test API keys manually:
cd /root/trading_bot
source venv/bin/activate
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', os.getenv('GATEIO_LIVE_API_KEY')[:10]+'...')"
```

---

## 📈 PERFORMANCE MONITORING

### Trading Statistics
```bash
# View execution stats:
sudo journalctl -u trading-bot-live --since "1 hour ago" | grep "Execution Stats" -A 5

# Count trades today:
sudo journalctl -u trading-bot-live --since today | grep -c "Trade executed"

# Check profit/loss:
sudo journalctl -u trading-bot-live --since today | grep "Profit:"

# Check open positions:
sudo journalctl -u trading-bot-live -n 200 | grep "Open Positions:" | tail -1
```

### System Performance
```bash
# Check all systems are active:
sudo journalctl -u trading-bot-live --since "2 min ago" | grep "Active Systems:" -A 3

# Count signals generated today:
sudo journalctl -u trading-bot-live --since today | grep -c "signals generated"

# Check cycle times:
sudo journalctl -u trading-bot-live --since "10 min ago" | grep "cycle.*finished"
```

---

## 🔐 SECURITY & MAINTENANCE

### Backup Configuration
```bash
# Backup .env file:
cp /root/trading_bot/.env /root/trading_bot/.env.backup.$(date +%Y%m%d)

# Backup entire bot:
tar -czf /root/trading_bot_backup_$(date +%Y%m%d).tar.gz -C /root trading_bot --exclude='*.pyc' --exclude='__pycache__' --exclude='.git'
```

### Clear Old Data
```bash
# Clear old logs (keep 3 days):
sudo journalctl --vacuum-time=3d

# Clear Python cache:
cd /root/trading_bot
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Clear pip cache:
source venv/bin/activate
pip cache purge
```

### Update Python Dependencies
```bash
cd /root/trading_bot
source venv/bin/activate

# Update all packages:
pip install --upgrade -r requirements.txt

# Update specific package:
pip install --upgrade ccxt

# Restart bot:
sudo systemctl restart trading-bot-live
```

---

## 🎯 QUICK REFERENCE

### Most Used Commands
```bash
# Start bot:
sudo systemctl start trading-bot-live

# Check status:
sudo systemctl status trading-bot-live

# View logs:
sudo journalctl -u trading-bot-live -f

# Restart after update:
cd /root/trading_bot && git pull && sudo systemctl restart trading-bot-live

# Check Ultra AI systems:
sudo journalctl -u trading-bot-live --since "2 min ago" | grep -E "Quantum|ULTRASONIC|GOLDMINE|DIVINE"

# Stop bot:
sudo systemctl stop trading-bot-live
```

---

## 📞 GETTING HELP

If bot is not working as expected:

1. **Check logs for errors:**
   ```bash
   sudo journalctl -u trading-bot-live --since "10 min ago" | grep -i error
   ```

2. **Verify API keys are correct:**
   ```bash
   nano /root/trading_bot/.env
   ```

3. **Ensure bot is running:**
   ```bash
   sudo systemctl status trading-bot-live
   ```

4. **Check memory isn't full:**
   ```bash
   free -h
   ```

5. **Update to latest version:**
   ```bash
   cd /root/trading_bot && git pull && sudo systemctl restart trading-bot-live
   ```

---

**For full bot capabilities, see:** `BOT_OVERVIEW.md`  
**For deployment guide, see:** `DEPLOY_TO_NEW_VPS.sh`
