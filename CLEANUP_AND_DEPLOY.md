# ✅ FOUND YOUR OLD BOT - CLEANUP AND DEPLOY

**Location**: `/opt/Lean-Trader` (with venv and Docker)  
**Status**: Docker container stopped ✅  
**Next**: Clean up and deploy new bot

---

## 🧹 STEP 1: COMPLETE CLEANUP

**Copy these to VPS:**

```bash
# Go to home
cd ~

# Remove old bot directory
sudo rm -rf /opt/Lean-Trader

# Remove Docker container
docker rm 994920290b67

# Remove any other bot stuff
sudo rm -rf ~/venv ~/env ~/bot* ~/trade* ~/lean*

# Optional: Clean Docker completely
docker system prune -af

# Verify clean
echo "Checking cleanup..."
ls -la /opt
ls -la ~
docker ps -a
ps aux | grep python | grep -v grep
```

**All should be empty!** ✅

---

## ⏰ VERIFY BOT IS STOPPED

**Wait 10 minutes, then:**

1. Check Telegram
2. **If NO new messages** = Old bot stopped! ✅
3. **If messages still coming** = Something else running

**Before deploying new bot, WAIT 10 minutes to confirm silence!**

---

## 🚀 STEP 2: DEPLOY NEW BOT (After Verification)

**Once you confirm no more Telegram messages:**

```bash
# Go to home directory
cd ~

# Get GitHub token from:
# https://github.com/settings/tokens
# Click "Generate new token (classic)"
# Select "repo" permission
# Copy the token

# Clone new bot (replace YOUR_TOKEN, YOUR_USERNAME, YOUR_REPO)
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO_NAME.git trading_bot

# Enter directory
cd trading_bot

# Create .env file
cat > .env << 'EOF'
# ===== TELEGRAM =====
TELEGRAM_BOT_TOKEN=8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg
TG_ADMIN_CHAT_ID=5329503447
TG_FREE_CHAT_ID=-1002930953007
TG_VIP_CHAT_ID=-1002983007302

# ===== BYBIT TESTNET =====
BYBIT_API_KEY=N8BMgWdfisCtkvfZk8
BYBIT_SECRET=BIu7c65FQnDsd6kBmctU7gK9bBbzY15vi8oe
BYBIT_TESTNET=true

# ===== GATE.IO =====
GATEIO_TESTNET_API_KEY=590f4e3cb2a8cfcaa66fe1a3a646e4b1
GATEIO_TESTNET_SECRET=e1e5614876dfd2aa9c59beabd035c2af08a186b5f818209640c66e98225ca37b
GATEIO_LIVE_API_KEY=bbdcedbd7f719a87c851356cf4dd3c20
GATEIO_LIVE_SECRET=068996eb5877b74abf3595aedbc4f0778fe64e7f37d88c01b5af41f62e4d9c26
GATEIO_MODE=testnet

# ===== NEWS & DATA =====
NEWSAPI_KEY=11dc35597f524f868ef814ad0bc9d216
ETHERSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
BSCSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
POLYGONSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
EOF

# Install dependencies (NO venv this time!)
pip3 install -r requirements.txt

# Run bot (in screen, not Docker)
screen -dmS trading_bot python3 RUN_BOT.py --testnet

echo ""
echo "✅ NEW BOT STARTED!"
echo ""
echo "View logs: screen -r trading_bot"
echo "Detach: Ctrl+A then D"
```

---

## 📱 NEW BOT WILL SEND DIFFERENT MESSAGES

**Old bot messages looked like:**
- Generic trading updates
- Maybe from REAL_PROFIT_BOT

**New bot messages will say:**
- "🚀 BOT STARTED IN TESTNET MODE"
- "✅ ALL 40 SYSTEMS INITIALIZED"
- "⚡ EXECUTION LOOP STARTED"

**You'll know it's the new bot!** ✅

---

## 🎯 DEPLOYMENT SUMMARY

### Location of Old Bot:
```
Directory: /opt/Lean-Trader
Environment: venv (virtual environment)
Container: Docker (ID: 994920290b67)
Status: ✅ Stopped and being removed
```

### New Bot Will Be:
```
Directory: ~/trading_bot
Environment: System Python (no venv)
Container: None (runs in screen)
Status: Fresh deployment
```

---

## ⚠️ CRITICAL: WAIT 10 MINUTES

**Before deploying new bot:**

1. ✅ Run cleanup commands (above)
2. ⏰ **WAIT 10 MINUTES**
3. 📱 Check Telegram (should be silent)
4. ✅ If silent, deploy new bot
5. 📱 New bot will send "40 SYSTEMS INITIALIZED"

**Don't rush! Wait to confirm old bot is dead!**

---

## 📋 COMPLETE SEQUENCE

```bash
# 1. Cleanup (NOW)
sudo rm -rf /opt/Lean-Trader
docker rm 994920290b67
sudo rm -rf ~/venv ~/env ~/bot* ~/trade*

# 2. Wait (10 MINUTES)
# Check Telegram - should be silent

# 3. Deploy (AFTER SILENCE CONFIRMED)
cd ~
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  https://TOKEN@github.com/USER/REPO.git trading_bot
cd trading_bot
# Create .env (see above)
pip3 install -r requirements.txt
screen -dmS trading_bot python3 RUN_BOT.py --testnet

# 4. Verify (NEW BOT MESSAGES)
screen -r trading_bot
# Should see "40 SYSTEMS INITIALIZED"
```

---

## ✅ SUMMARY

**Found**: Old bot in /opt/Lean-Trader + Docker ✅  
**Stopped**: Docker container killed ✅  
**Next**: Clean up directory and wait 10 min ⏰  
**Then**: Deploy new bot to ~/trading_bot 🚀

**Run cleanup commands now, then WAIT before deploying!** ⏰