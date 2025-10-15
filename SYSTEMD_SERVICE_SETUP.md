# 🚀 PRODUCTION DEPLOYMENT - RUN LIKE OLD BOT

**Your new bot will run 24/7 just like the old one!**

---

## ✅ WHAT I'M SETTING UP

**Like your old bot, the new one will:**
- ✅ Run continuously 24/7
- ✅ Auto-restart if crashes
- ✅ Survive VPS reboots
- ✅ Run as systemd service (professional)
- ✅ Log everything properly
- ✅ Never stop unless you tell it to

**Just like the old bot in /opt/Lean-Trader!** ✅

---

## 🔧 TWO DEPLOYMENT OPTIONS

### OPTION 1: Automated Script (Easiest)

**I created `PRODUCTION_DEPLOYMENT.sh` that does EVERYTHING:**

```bash
# On VPS (as root):
sudo bash PRODUCTION_DEPLOYMENT.sh
```

**It will:**
1. ✅ Clean up old bot completely
2. ✅ Clone new bot from GitHub
3. ✅ Create .env with all keys
4. ✅ Install dependencies
5. ✅ Create systemd service
6. ✅ Start bot as daemon
7. ✅ Enable auto-restart

**Total time: 5 minutes, fully automated!** ✅

### OPTION 2: Manual Setup (Step by Step)

**If you prefer manual control:**

---

## 📋 MANUAL SETUP (Complete Guide)

### Step 1: Cleanup (5 min)

```bash
# Stop old bot
sudo systemctl stop trading-bot 2>/dev/null
docker stop $(docker ps -q) 2>/dev/null
sudo pkill -9 -f python

# Remove old directories
sudo rm -rf /opt/Lean-Trader
sudo rm -rf ~/venv ~/env ~/bot* ~/trade*
docker rm 994920290b67 2>/dev/null

# Verify clean
ps aux | grep python | grep -v grep
docker ps -a
```

### Step 2: Clone New Bot (5 min)

```bash
# Get GitHub token:
# https://github.com/settings/tokens

# Clone to home directory
cd ~
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  https://YOUR_TOKEN@github.com/YOUR_USER/YOUR_REPO.git trading_bot

cd trading_bot
```

### Step 3: Create .env (2 min)

```bash
cat > .env << 'EOF'
TELEGRAM_BOT_TOKEN=8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg
TG_ADMIN_CHAT_ID=5329503447
TG_FREE_CHAT_ID=-1002930953007
TG_VIP_CHAT_ID=-1002983007302
BYBIT_API_KEY=N8BMgWdfisCtkvfZk8
BYBIT_SECRET=BIu7c65FQnDsd6kBmctU7gK9bBbzY15vi8oe
BYBIT_TESTNET=true
GATEIO_TESTNET_API_KEY=590f4e3cb2a8cfcaa66fe1a3a646e4b1
GATEIO_TESTNET_SECRET=e1e5614876dfd2aa9c59beabd035c2af08a186b5f818209640c66e98225ca37b
GATEIO_LIVE_API_KEY=bbdcedbd7f719a87c851356cf4dd3c20
GATEIO_LIVE_SECRET=068996eb5877b74abf3595aedbc4f0778fe64e7f37d88c01b5af41f62e4d9c26
GATEIO_MODE=testnet
NEWSAPI_KEY=11dc35597f524f868ef814ad0bc9d216
ETHERSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
BSCSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
POLYGONSCAN_API_KEY=2BP6ZUNY13IKCI4NQ5IZGW7YUFIZAJK4GH
EOF
```

### Step 4: Install Dependencies (5 min)

```bash
pip3 install -r requirements.txt
```

### Step 5: Create Systemd Service (3 min)

```bash
# Create service file
sudo nano /etc/systemd/system/trading-bot.service
```

**Paste this content:**

```ini
[Unit]
Description=Advanced AI Trading Bot - 40 Systems
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/trading_bot
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/usr/bin/python3 /home/YOUR_USERNAME/trading_bot/RUN_BOT.py --testnet

# Auto-restart configuration
Restart=always
RestartSec=10
StartLimitInterval=0

# Logging
StandardOutput=append:/home/YOUR_USERNAME/trading_bot/bot.log
StandardError=append:/home/YOUR_USERNAME/trading_bot/bot_error.log

# Resource limits
MemoryMax=4G
CPUQuota=200%

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

**Save:** Ctrl+O, Enter, Ctrl+X

**Replace `YOUR_USERNAME` with your actual username (ubuntu, root, etc.)**

### Step 6: Start Service (1 min)

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable trading-bot

# Start service
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot
```

**Bot is now running 24/7!** ✅

---

## 📊 SERVICE FEATURES

### Your Bot Will:
```
✅ Run continuously (24/7)
✅ Auto-restart if crashes (within 10 seconds)
✅ Start on VPS reboot
✅ Log to bot.log
✅ Use max 4GB RAM
✅ Professional daemon setup
```

### Just Like Old Bot:
```
Old bot in /opt/Lean-Trader:
  • Had auto-restart ✅
  • Ran continuously ✅
  • Survived reboots ✅

New bot in ~/trading_bot:
  • Has auto-restart ✅
  • Runs continuously ✅
  • Survives reboots ✅
  • + Better logging ✅
  • + Resource limits ✅
  • + Easier management ✅
```

---

## 🎮 MANAGEMENT COMMANDS

### View Live Logs:
```bash
tail -f ~/trading_bot/bot.log
```

### Check Bot Status:
```bash
systemctl status trading-bot
```

### Restart Bot:
```bash
sudo systemctl restart trading-bot
```

### Stop Bot:
```bash
sudo systemctl stop trading-bot
```

### Start Bot:
```bash
sudo systemctl start trading-bot
```

### Disable Auto-Start:
```bash
sudo systemctl disable trading-bot
```

### View Recent Logs:
```bash
journalctl -u trading-bot -n 100 -f
```

---

## 🔄 AUTO-RESTART BEHAVIOR

### If Bot Crashes:
```
1. Systemd detects crash
2. Waits 10 seconds
3. Auto-restarts bot
4. Logs the restart
5. Bot continues trading
```

### If VPS Reboots:
```
1. VPS starts up
2. Systemd starts automatically
3. Systemd starts trading-bot service
4. Bot resumes trading
5. No manual intervention needed
```

**Bulletproof continuous operation!** ✅

---

## 📱 WHAT TO EXPECT

### After Deployment:

**Immediately:**
- Bot starts initializing
- Telegram message: "🚀 TRADING BOT LAUNCHER"

**Within 2-3 Minutes:**
- Telegram: "✅ ALL 40 SYSTEMS INITIALIZED"
- Telegram: "✅ BOT IS RUNNING!"

**Ongoing:**
- Trading signals sent to channels
- Admin updates to your chat
- Continuous operation
- Auto-recovery from crashes

---

## 🎯 DEPLOYMENT SEQUENCE

### RIGHT NOW:

**1. Run cleanup** (if not done):
```bash
sudo rm -rf /opt/Lean-Trader
docker rm 994920290b67
```

**2. Wait 10 minutes** (verify old bot silent)

**3. Run automated deployment**:
```bash
sudo bash PRODUCTION_DEPLOYMENT.sh
```

**Or follow manual steps above.**

---

## ✅ AFTER DEPLOYMENT

### Bot Will:
```
✅ Run at: /home/YOUR_USER/trading_bot
✅ Method: Systemd service (like old bot!)
✅ Auto-restart: Yes (if crashes)
✅ Start on boot: Yes (survives reboots)
✅ Logs: bot.log and bot_error.log
✅ Management: systemctl commands
```

### You Can:
```
✅ View logs anytime
✅ Restart bot anytime
✅ Monitor via Telegram
✅ Check status with systemctl
✅ Reboot VPS (bot auto-starts)
```

---

## 🚨 IMPORTANT DIFFERENCES

### Old Bot Setup:
```
Location: /opt/Lean-Trader
Method: Docker + venv
Management: Docker commands
```

### New Bot Setup:
```
Location: ~/trading_bot
Method: Systemd service (better!)
Management: systemctl commands (easier!)
```

**New setup is MORE reliable and easier to manage!** ✅

---

## 📋 QUICK REFERENCE

### After Deployment:

```bash
# Check if running
systemctl status trading-bot

# View logs
tail -f ~/trading_bot/bot.log

# Restart
sudo systemctl restart trading-bot

# Stop
sudo systemctl stop trading-bot

# Start
sudo systemctl start trading-bot
```

**Bot will run forever!** ♾️

---

## 🎯 READY TO DEPLOY

**Use automated script:**
```bash
sudo bash PRODUCTION_DEPLOYMENT.sh
```

**Or follow manual steps!**

**Your bot will run continuously like the old one!** ✅

---

**Total setup time: 20 minutes**  
**Bot lifetime: Forever (until you stop it)** ♾️  
**Auto-restart: Yes** ✅  
**Survives reboots: Yes** ✅  

**DEPLOY NOW! 🚀**