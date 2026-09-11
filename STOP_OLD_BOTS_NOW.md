# 🛑 STOP OLD BOTS IMMEDIATELY!

**You're getting Telegram notifications from old bots = They're still running!**

---

## 🚨 CRITICAL: KILL OLD BOTS NOW

### Quick Kill Commands (Run on VPS):

```bash
# SSH to VPS
ssh your_user@your_vps_ip

# Kill ALL Python bots
pkill -9 -f python

# Kill ALL screen sessions
screen -wipe
screen -ls | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit

# Verify stopped
ps aux | grep python | grep -v grep
```

**If you still see processes, kill them manually:**
```bash
ps aux | grep python
# Note the PID (second column)
kill -9 PID_NUMBER
```

---

## 🧹 COMPLETE CLEANUP (Automated)

### Option 1: Use My Script

**Run this ONE command on VPS:**
```bash
# Copy the script content and save it:
nano kill_bots.sh

# Paste the script content (I'll give you)
# Save: Ctrl+O, Enter, Ctrl+X

# Make executable and run:
chmod +x kill_bots.sh
bash kill_bots.sh
```

### Option 2: Manual Commands

**Run these on VPS one by one:**

```bash
# 1. SSH to VPS
ssh your_user@your_vps_ip

# 2. Kill ALL Python processes
pkill -9 -f "python"
sleep 2

# 3. Kill ALL screen sessions
screen -ls
# For each session:
screen -X -S SESSION_NAME quit

# 4. Double-check
ps aux | grep bot
ps aux | grep python

# 5. If anything still running, kill by PID:
kill -9 PID_NUMBER

# 6. Clean directories
cd ~
rm -rf *bot* *trading* *trade* 2>/dev/null

# 7. Verify clean
ls -la
ps aux | grep python
```

---

## ✅ VERIFY OLD BOTS ARE STOPPED

### Check These:

```bash
# 1. No Python processes:
ps aux | grep python | grep -v grep
# Should show NOTHING or just system Python

# 2. No screen sessions:
screen -ls
# Should show "No Sockets found"

# 3. No Telegram messages:
# Wait 5 minutes
# If no new Telegram messages = Old bots stopped ✅
```

---

## 🚀 AFTER STOPPING OLD BOTS

### Then Deploy New Bot:

```bash
# Get GitHub token from:
# https://github.com/settings/tokens

# Clone (replace TOKEN, USER, REPO):
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  https://YOUR_TOKEN@github.com/YOUR_USER/YOUR_REPO.git trading_bot

cd trading_bot

# Create .env
cat > .env << 'EOF'
TELEGRAM_BOT_TOKEN=REDACTED_ROTATED_TELEGRAM_BOT_TOKEN__SET_VIA_MOUNTED_SECRET_FILE
TG_ADMIN_CHAT_ID=5329503447
TG_FREE_CHAT_ID=-1002930953007
TG_VIP_CHAT_ID=-1002983007302
BYBIT_API_KEY=REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
BYBIT_SECRET=REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
BYBIT_TESTNET=true
GATEIO_TESTNET_API_KEY=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_TESTNET_SECRET=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_LIVE_API_KEY=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_LIVE_SECRET=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_MODE=testnet
NEWSAPI_KEY=REDACTED_ROTATED_NEWSAPI_KEY__SET_VIA_MOUNTED_SECRET_FILE
ETHERSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
BSCSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
POLYGONSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
EOF

# Install & Run
pip3 install -r requirements.txt
screen -dmS trading_bot python3 RUN_BOT.py --testnet

# View logs
screen -r trading_bot
```

---

## 📋 COMPLETE KILL SCRIPT

**Copy this entire script to VPS:**

```bash
#!/bin/bash
echo "🛑 STOPPING ALL OLD BOTS..."

# Kill Python
pkill -9 -f python
sleep 2

# Kill screen
screen -ls | cut -d. -f1 | awk '{print $1}' | xargs -I {} screen -X -S {} quit 2>/dev/null
sleep 2

# Clean directories
cd ~
rm -rf *bot* *trading* *trade* 2>/dev/null

# Verify
echo ""
echo "Checking..."
ps aux | grep python | grep -v grep
screen -ls

echo ""
echo "If empty above, all stopped! ✅"
echo "Wait 5 minutes and check Telegram."
echo "If no messages = Success! ✅"
```

---

## ⚠️ IMPORTANT

### Why Old Bots Still Send Messages:

**Your Telegram token is in old bot code!**
- Old bot is running
- Uses same token
- Sends messages to same channels
- Conflicts with new bot

**Solution:**
- 🔥 **MUST kill old bots completely**
- Verify no Python processes
- Verify no screen sessions
- Wait 5 min, check Telegram
- If silent = Old bots stopped ✅

---

## 🎯 QUICK FIX (FASTEST)

**Run these 3 commands on VPS:**

```bash
# 1. Kill everything
pkill -9 -f python && screen -wipe

# 2. Wait and verify
sleep 5 && ps aux | grep python

# 3. If empty, you're good!
echo "Old bots stopped!"
```

**Then wait 5-10 minutes and check Telegram.**

**No messages = Success!** ✅

---

## 📄 FULL SCRIPT

**I created**: `KILL_ALL_OLD_BOTS.sh`

**To use:**
```bash
# On VPS
wget https://YOUR_REPO/KILL_ALL_OLD_BOTS.sh
bash KILL_ALL_OLD_BOTS.sh
```

**Or copy-paste the script content manually!**

---

## ✅ READY TO DEPLOY NEW BOT

**After old bots stopped:**
1. Clone new bot from private repo
2. Create .env
3. Run new bot
4. Watch Telegram (should be from new bot only)

**Read**: `PRIVATE_REPO_GUIDE.md` for complete instructions!

**STOP OLD BOTS FIRST! 🛑**