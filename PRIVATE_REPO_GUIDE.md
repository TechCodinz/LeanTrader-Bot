# 🔐 PRIVATE REPO DEPLOYMENT GUIDE

**How to clone private GitHub repo to VPS**

---

## 🔑 METHOD 1: PERSONAL ACCESS TOKEN (Easiest)

### Step 1: Create Token on GitHub

1. Go to: https://github.com/settings/tokens
2. Click: "Generate new token" → "Generate new token (classic)"
3. Name: "VPS Trading Bot"
4. Expiration: 90 days (or No expiration)
5. Select scopes:
   - ✅ **repo** (all repo permissions)
6. Click: "Generate token"
7. **COPY THE TOKEN** (shows only once!)

### Step 2: Clone on VPS

```bash
# SSH to VPS
ssh your_user@your_vps_ip

# Clean old bots
cd ~
rm -rf old_bot trading_bot  # Or backup: mv old_bot old_bot_backup

# Clone with token in URL
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git trading_bot

# Example:
# git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
#   https://ghp_xxxxxxxxxxxx@github.com/johndoe/Lean-Trader.git trading_bot
```

### Step 3: Setup & Run

```bash
cd trading_bot

# Create .env file
nano .env
# Paste all your API keys (provided below)

# Install & Run
bash START_BOT_NOW.sh
```

---

## 🔑 METHOD 2: SSH KEY (More Secure)

### Step 1: Generate SSH Key on VPS

```bash
# On VPS
ssh-keygen -t rsa -b 4096 -C "vps-trading-bot"
# Press Enter 3 times (default location, no passphrase)

# Show public key
cat ~/.ssh/id_rsa.pub
# Copy this entire output
```

### Step 2: Add to GitHub

1. Go to: https://github.com/settings/keys
2. Click: "New SSH key"
3. Title: "VPS Trading Bot"
4. Paste your public key
5. Click: "Add SSH key"

### Step 3: Clone on VPS

```bash
# Clean old bots
cd ~
rm -rf old_bot trading_bot

# Clone with SSH
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  git@github.com:YOUR_USERNAME/YOUR_REPO.git trading_bot

# Setup & Run
cd trading_bot
bash START_BOT_NOW.sh
```

---

## 🧹 VPS CLEANUP COMMANDS

### Option A: Delete Everything (Fresh Start)

```bash
# SSH to VPS
ssh your_user@your_vps_ip

# Go to home
cd ~

# Delete all old bots
rm -rf bot trading_bot old_bot lean-trader leantrader

# Delete old Python environments
rm -rf venv env .venv

# Delete old logs
rm -rf logs *.log

# Delete old databases
rm -rf *.db *.sqlite

echo "✅ VPS cleaned!"
```

### Option B: Backup First (Safer)

```bash
# SSH to VPS
ssh your_user@your_vps_ip

# Go to home
cd ~

# Create backup directory
mkdir -p backups/$(date +%Y%m%d_%H%M%S)

# Move old bots to backup
mv bot backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null
mv trading_bot backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null
mv old_bot backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null

echo "✅ Old bots backed up to ~/backups/"
```

### Option C: Kill Running Processes

```bash
# Find running bots
ps aux | grep python | grep bot

# Kill specific process
kill -9 PROCESS_ID

# Or kill all Python bots
pkill -f "python.*bot"

# Kill screen sessions
screen -X -S trading_bot quit
screen -X -S old_bot quit

echo "✅ All bot processes killed"
```

---

## 🚀 COMPLETE SETUP SCRIPT

### Use the automated script I created:

```bash
# On VPS, download and run:
bash VPS_SETUP_COMMANDS.sh
```

**Or manual commands below:**

---

## 📋 MANUAL SETUP (STEP-BY-STEP)

### Complete Commands for VPS:

```bash
# =====================================
# STEP 1: CLEAN VPS
# =====================================

ssh your_user@your_vps_ip

cd ~

# Kill old processes
pkill -f "python.*bot"
screen -ls | grep bot | cut -d. -f1 | xargs -I {} screen -X -S {} quit

# Backup old bots
mkdir -p backups/$(date +%Y%m%d)
mv *bot* backups/$(date +%Y%m%d)/ 2>/dev/null

# Or just delete
rm -rf bot trading_bot old_bot lean-trader leantrader

echo "✅ VPS cleaned"


# =====================================
# STEP 2: CLONE PRIVATE REPO
# =====================================

# Get your GitHub Personal Access Token first!
# https://github.com/settings/tokens

# Clone (replace YOUR_TOKEN, YOUR_USERNAME, YOUR_REPO):
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  https://YOUR_TOKEN@github.com/YOUR_USERNAME/YOUR_REPO.git trading_bot

# Example:
# git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
#   https://ghp_abc123xyz@github.com/johndoe/Lean-Trader.git trading_bot

cd trading_bot


# =====================================
# STEP 3: CREATE .env FILE
# =====================================

cat > .env << 'EOF'
# ===== TELEGRAM =====
TELEGRAM_BOT_TOKEN=REDACTED_ROTATED_TELEGRAM_BOT_TOKEN__SET_VIA_MOUNTED_SECRET_FILE
TG_ADMIN_CHAT_ID=5329503447
TG_FREE_CHAT_ID=-1002930953007
TG_VIP_CHAT_ID=-1002983007302

# ===== BYBIT TESTNET =====
BYBIT_API_KEY=REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
BYBIT_SECRET=REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
BYBIT_TESTNET=true

# ===== GATE.IO =====
GATEIO_TESTNET_API_KEY=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_TESTNET_SECRET=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_LIVE_API_KEY=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_LIVE_SECRET=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_MODE=testnet

# ===== NEWS & DATA =====
NEWSAPI_KEY=REDACTED_ROTATED_NEWSAPI_KEY__SET_VIA_MOUNTED_SECRET_FILE
ETHERSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
BSCSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
POLYGONSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
EOF

echo "✅ .env file created"


# =====================================
# STEP 4: INSTALL DEPENDENCIES
# =====================================

# Update system
sudo apt update
sudo apt install -y python3 python3-pip screen

# Upgrade pip
pip3 install --upgrade pip

# Install requirements
pip3 install -r requirements.txt

echo "✅ Dependencies installed"


# =====================================
# STEP 5: START BOT
# =====================================

# Start in screen session
screen -dmS trading_bot python3 RUN_BOT.py --testnet

echo ""
echo "================================================================================"
echo "✅ BOT STARTED IN TESTNET MODE!"
echo "================================================================================"
echo ""
echo "View logs:"
echo "  screen -r trading_bot"
echo ""
echo "Detach: Ctrl+A then D"
echo ""
echo "Check status:"
echo "  screen -ls"
echo ""
echo "================================================================================"
```

---

## 📄 YOUR .env FILE CONTENT

**Copy this if creating manually:**

```bash
# ===== TELEGRAM =====
TELEGRAM_BOT_TOKEN=REDACTED_ROTATED_TELEGRAM_BOT_TOKEN__SET_VIA_MOUNTED_SECRET_FILE
TG_ADMIN_CHAT_ID=5329503447
TG_FREE_CHAT_ID=-1002930953007
TG_VIP_CHAT_ID=-1002983007302

# ===== BYBIT TESTNET =====
BYBIT_API_KEY=REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
BYBIT_SECRET=REDACTED_ROTATED_BYBIT_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
BYBIT_TESTNET=true

# ===== GATE.IO =====
GATEIO_TESTNET_API_KEY=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_TESTNET_SECRET=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_LIVE_API_KEY=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_LIVE_SECRET=REDACTED_ROTATED_GATEIO_CREDENTIAL__SET_VIA_MOUNTED_SECRET_FILE
GATEIO_MODE=testnet

# ===== NEWS & DATA =====
NEWSAPI_KEY=REDACTED_ROTATED_NEWSAPI_KEY__SET_VIA_MOUNTED_SECRET_FILE
ETHERSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
BSCSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
POLYGONSCAN_API_KEY=REDACTED_ROTATED_BLOCK_EXPLORER_API_KEY__SET_VIA_MOUNTED_SECRET_FILE
```

---

## 🎯 QUICK REFERENCE

### Kill Old Bots:
```bash
pkill -f "python.*bot"
screen -X -S trading_bot quit
```

### Clean VPS:
```bash
cd ~
rm -rf trading_bot old_bot
```

### Clone Private Repo:
```bash
# Get token from: https://github.com/settings/tokens
git clone -b cursor/integrate-and-unify-existing-trading-bot-components-c04c \
  https://YOUR_TOKEN@github.com/USER/REPO.git trading_bot
```

### Start Bot:
```bash
cd trading_bot
bash START_BOT_NOW.sh
```

### View Logs:
```bash
screen -r trading_bot
# or
tail -f bot.log
```

---

## 📱 AFTER DEPLOYMENT

### What to Watch:

**First 10 Minutes:**
- Bot initializes all 40 systems
- Connects to exchanges
- Telegram bot starts
- You get admin notification

**First Hour:**
- Signals generated
- Orders attempted (check if they execute!)
- Positions tracked
- Telegram messages sent

**First Day:**
- 5-15 trades on testnet
- ML models collecting data
- Verify execution works
- Check for errors

---

## ✅ READY TO DEPLOY!

**Use either:**
- Automated script: `bash VPS_SETUP_COMMANDS.sh`
- Manual commands above

**Your bot will start in testnet mode (safe)** ✅

**Monitor closely first day!** 👀

---

**Total time: 20-30 minutes**

**Let's deploy! 🚀**
