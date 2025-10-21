# 🚀 FIX YOUR VPS BOT RIGHT NOW

## ⚡ FASTEST FIX (30 seconds)

On your VPS, run these commands:

```bash
cd ~/trading_bot
pkill -9 -f RUN_BOT.py
sleep 2
./start_bot.sh
```

Then monitor:
```bash
tail -f bot.log
```

---

## 🔧 PROPER FIX (2 minutes)

### Option 1: Copy the entire fix script (RECOMMENDED)

Copy the contents of `COPY_PASTE_TO_VPS.sh` and paste it directly into your VPS terminal.

It will:
1. Create a fix script
2. Stop the old bot
3. Start a new one
4. Show you live output

### Option 2: Upload the scripts

From your local machine:
```bash
scp VPS_AUTO_RESTART.sh root@YOUR_VPS_IP:~/trading_bot/
ssh root@YOUR_VPS_IP
cd ~/trading_bot
chmod +x VPS_AUTO_RESTART.sh
./VPS_AUTO_RESTART.sh
```

---

## 📊 WHAT YOU ASKED FOR

You wanted to:
1. ✅ Check bot status → Use: `./status.sh`
2. ✅ View live signals → Use: `tail -f bot.log | grep "✅"`
3. ✅ See discovered pairs → Use: `grep "TOTAL DISCOVERED" bot.log | tail -1`
4. ✅ Count unique pairs → Use: `grep "Decision:" bot.log | grep -oE "[A-Z]{2,5}/[A-Z]{2,5}" | sort -u | wc -l`
5. ✅ View VIP signals → Use: `grep "✅ VIP #" bot.log | tail -20`
6. ✅ Restart bot → Use: `./VPS_AUTO_RESTART.sh`

---

## 🎯 THE ACTUAL ISSUE

The error `./status.sh: No such file or directory` happened because:
- You might not be in the `~/trading_bot` directory
- Or the file isn't executable

**Fix it:**
```bash
cd ~/trading_bot
chmod +x status.sh
./status.sh
```

---

## 📝 KEY FILES CREATED FOR YOU

1. **VPS_AUTO_RESTART.sh** - Quick restart, no questions
2. **VPS_CHECK_AND_RESTART.sh** - Full status check + restart option
3. **COPY_PASTE_TO_VPS.sh** - One command to fix everything
4. **VPS_QUICK_GUIDE.md** - All monitoring commands
5. **UPLOAD_TO_VPS.txt** - Detailed upload instructions

---

## 🆘 IF NOTHING WORKS

Run this diagnostic:

```bash
cd ~/trading_bot

echo "=== DIAGNOSTIC INFO ==="
echo ""
echo "Current directory:"
pwd
echo ""
echo "Python files:"
ls -la *.py | head -5
echo ""
echo "Shell scripts:"
ls -la *.sh | head -5
echo ""
echo "Bot process:"
ps aux | grep RUN_BOT.py | grep -v grep
echo ""
echo "Log file:"
ls -la bot.log
echo ""
echo "Last 10 lines of log:"
tail -10 bot.log 2>/dev/null
```

Share this output if you need more help.

---

## ✅ QUICK COMMANDS CHEAT SHEET

**Start bot:**
```bash
cd ~/trading_bot && ./start_bot.sh
```

**Stop bot:**
```bash
cd ~/trading_bot && ./stop_bot.sh
```

**Check status:**
```bash
cd ~/trading_bot && ./status.sh
```

**Watch signals live:**
```bash
cd ~/trading_bot && tail -f bot.log | grep "✅"
```

**Count signals:**
```bash
cd ~/trading_bot && echo "VIP: $(grep -c '✅ VIP' bot.log) | FREE: $(grep -c '✅ FREE' bot.log)"
```

**Restart (nuclear option):**
```bash
cd ~/trading_bot && pkill -9 -f RUN_BOT.py && sleep 2 && ./start_bot.sh && tail -f bot.log
```

---

## 🎉 YOU'RE DONE!

Pick one of the options above and run it on your VPS.
Your bot will be running and sending signals within 1 minute.

Good luck! 🚀
