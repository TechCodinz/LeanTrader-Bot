# 🚀 VPS Trading Bot - Quick Reference Guide

## 📋 Copy This Script to Your VPS

Run this command on your VPS to check and restart the bot:

```bash
./VPS_AUTO_RESTART.sh
```

Or for the interactive version:

```bash
./VPS_CHECK_AND_RESTART.sh
```

---

## 🔧 Essential Commands

### Check if Bot is Running
```bash
ps aux | grep RUN_BOT.py | grep -v grep
```

### View Live Signals as They Happen
```bash
tail -f ~/trading_bot/bot.log | grep "✅"
```

### View All Logs (Live)
```bash
tail -f ~/trading_bot/bot.log
```

### Check Signal Statistics
```bash
cd ~/trading_bot

# Total VIP signals sent
grep -c "✅ VIP #" bot.log

# Total FREE signals sent
grep -c "✅ FREE #" bot.log

# Show last 20 VIP signals
grep "✅ VIP #" bot.log | tail -20

# Count unique trading pairs
grep "Decision:" bot.log | grep -oE "[A-Z]{2,5}/[A-Z]{2,5}" | sort -u | wc -l

# Show total discovered pairs
grep "TOTAL DISCOVERED" bot.log | tail -1
```

### Quick Status Check
```bash
cd ~/trading_bot && ./status.sh
```

### Restart Bot
```bash
cd ~/trading_bot
./stop_bot.sh
sleep 2
./start_bot.sh
```

Or use the all-in-one script:
```bash
cd ~/trading_bot && ./VPS_AUTO_RESTART.sh
```

### Attach to Running Bot Screen Session
```bash
screen -r trading_bot
# Press Ctrl+A then D to detach without stopping
```

### View Last 50 Lines of Log
```bash
tail -50 ~/trading_bot/bot.log
```

### Search for Specific Pair (e.g., BTC/USDT)
```bash
grep "BTC/USDT" ~/trading_bot/bot.log
```

### Check if Telegram Bot is Working
```bash
grep "Telegram" ~/trading_bot/bot.log | tail -10
```

---

## 🐛 Troubleshooting

### Bot Won't Start?

1. **Check Python is installed:**
   ```bash
   python3 --version
   ```

2. **Check if another instance is running:**
   ```bash
   ps aux | grep RUN_BOT.py
   # If found, kill it:
   pkill -9 -f RUN_BOT.py
   ```

3. **Check for errors in log:**
   ```bash
   tail -100 ~/trading_bot/bot.log
   ```

4. **Verify environment variables:**
   ```bash
   cd ~/trading_bot
   cat .env | grep -E "TELEGRAM|BYBIT"
   ```

5. **Check dependencies:**
   ```bash
   cd ~/trading_bot
   pip3 install -r requirements.txt
   ```

### No Signals Being Sent?

```bash
# Check recent decision logs
grep "Decision:" ~/trading_bot/bot.log | tail -20

# Check for errors
grep -i "error\|exception" ~/trading_bot/bot.log | tail -20

# Check Telegram connection
grep -i "telegram" ~/trading_bot/bot.log | tail -10
```

### Bot Keeps Crashing?

```bash
# View crash logs
tail -100 ~/trading_bot/bot.log | grep -A 10 -i "error\|traceback"

# Check system resources
free -h
df -h
```

---

## 📊 Monitoring Commands

### Show Live Trading Activity (Real-time)
```bash
cd ~/trading_bot
tail -f bot.log | grep -E "✅|Decision:|SIGNAL|TRADE"
```

### Generate Quick Report
```bash
cd ~/trading_bot

echo "=== BOT STATUS REPORT ==="
echo ""
echo "Bot Process:"
ps aux | grep RUN_BOT.py | grep -v grep || echo "NOT RUNNING"
echo ""
echo "Total VIP Signals: $(grep -c '✅ VIP #' bot.log 2>/dev/null || echo 0)"
echo "Total FREE Signals: $(grep -c '✅ FREE #' bot.log 2>/dev/null || echo 0)"
echo "Unique Pairs: $(grep 'Decision:' bot.log 2>/dev/null | grep -oE '[A-Z]{2,5}/[A-Z]{2,5}' | sort -u | wc -l)"
echo ""
echo "Latest 5 Signals:"
grep "✅ VIP #\|✅ FREE #" bot.log 2>/dev/null | tail -5
```

---

## 🔄 One-Line Quick Commands

**Quick restart:**
```bash
cd ~/trading_bot && pkill -9 -f RUN_BOT.py && sleep 2 && ./start_bot.sh
```

**Quick status:**
```bash
cd ~/trading_bot && echo "Process: $(pgrep -f RUN_BOT.py || echo 'NOT RUNNING')" && echo "VIP: $(grep -c '✅ VIP' bot.log)" && echo "FREE: $(grep -c '✅ FREE' bot.log)"
```

**Watch for new signals:**
```bash
cd ~/trading_bot && tail -f bot.log | grep --line-buffered "✅"
```

**Count signals in last hour:**
```bash
cd ~/trading_bot && grep "✅" bot.log | grep "$(date +'%Y-%m-%d %H')" | wc -l
```

---

## 📝 Notes

- The bot runs in a `screen` session named `trading_bot`
- Logs are written to `~/trading_bot/bot.log`
- Environment variables are loaded from `~/trading_bot/.env`
- To stop gracefully: `./stop_bot.sh`
- To start: `./start_bot.sh`
- The bot will auto-restart on errors (unless killed with stop script)

---

## 🆘 Emergency Commands

**Nuclear option - kill everything and restart:**
```bash
cd ~/trading_bot
pkill -9 -f RUN_BOT.py
screen -S trading_bot -X quit
rm -rf __pycache__ */__pycache__
sleep 3
./start_bot.sh
```

**Backup logs before restart:**
```bash
cd ~/trading_bot
cp bot.log bot.log.backup.$(date +%Y%m%d_%H%M%S)
./VPS_AUTO_RESTART.sh
```
