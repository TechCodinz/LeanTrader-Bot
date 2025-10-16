# 🚀 VPS MIGRATION CHEATSHEET - Copy & Paste Commands

## 📤 ON CURRENT VPS (Backup Everything)

### **Option 1: Complete Setup (Recommended)**
```bash
cd ~/trading_bot
bash ONE_COMMAND_SETUP.sh
```
✅ Does everything in one command!

### **Option 2: Step-by-Step**
```bash
cd ~/trading_bot

# Backup to GitHub
bash BACKUP_TO_GITHUB.sh

# Setup auto-backup
bash SETUP_AUTO_BACKUP_CRON.sh

# Save .env locally
cp .env ~/.env.trading_bot.backup
```

---

## 📥 ON NEW VPS (Deploy Bot)

### **Full Deployment (5 Minutes)**
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Add API keys
nano .env
# Paste all your API keys from .env.backup

# 3. Deploy (auto-installs + restores intelligence)
bash DEPLOY_NEW_VPS.sh

# 4. Start bot
bash START_BOT.sh

# 5. Watch it run!
sudo journalctl -u trading-bot-live -f
```

---

## 📊 MANAGEMENT COMMANDS

```bash
# View live logs
sudo journalctl -u trading-bot-live -f

# Check bot status
sudo systemctl status trading-bot-live

# Restart bot
sudo systemctl restart trading-bot-live

# Stop bot
sudo systemctl stop trading-bot-live

# Check backup logs
tail -f logs/auto_backup.log

# Manual backup now
bash AUTO_BACKUP_LEARNED_DATA.sh

# View cron jobs
crontab -l
```

---

## 🧠 VERIFY LEARNED DATA RESTORED

```bash
# Check databases (should have data, not empty)
ls -lh *.db

# Check ML models (should exist)
ls -lh models/

# Check trade history
sqlite3 ultra_trading_system.db "SELECT COUNT(*) FROM trades;"

# Should show 9+ trades, not 0!
```

---

## 🚨 EMERGENCY MIGRATION (VPS Down)

```bash
# On new VPS - FAST recovery:
git clone https://github.com/YOU/REPO.git && \
cd REPO && \
nano .env && \
bash DEPLOY_NEW_VPS.sh && \
bash START_BOT.sh

# Bot running in 5 minutes with ALL intelligence intact!
```

---

## 🔐 SECURITY CHECKLIST

### ✅ Before Backup:
- [x] `.env` excluded from GitHub
- [x] `.env.backup` saved locally
- [x] GitHub token kept secure

### ✅ On New VPS:
- [x] Create new `.env` file
- [x] Add all API keys
- [x] Don't commit `.env` to git

---

## 💡 PRO TIPS

**Backup .env Multiple Places**:
```bash
# Save to your computer
scp root@vps:/root/trading_bot/.env ~/trading_bot_env_backup.txt

# Save to cloud (encrypted)
# Upload .env.backup to Google Drive/Dropbox
```

**Test New VPS Before Switching**:
```bash
# Deploy on test VPS first
# Let it run for 1 hour
# Verify it continues from Evolution Cycle 231+
# Then switch production
```

**Multiple VPS Strategy**:
```bash
# VPS 1: Primary (current)
# VPS 2: Backup (same bot, different VPS)
# If VPS 1 fails, VPS 2 continues trading
# Both share learned intelligence from GitHub
```

---

## 📞 QUICK HELP

**Bot won't start on new VPS?**
```bash
# Check Python version
python3 --version  # Need 3.8+

# Check .env file
cat .env | grep API_KEY  # Should show keys

# Reinstall packages
pip install -r requirements.txt

# Check logs
sudo journalctl -u trading-bot-live -n 100
```

**Learned data not restored?**
```bash
# Check if backup exists in repo
ls -lh learned_data_backup/

# Manually restore
cp -rv learned_data_backup/* .

# Restart bot
sudo systemctl restart trading-bot-live
```

---

**Save this file!** Keep it handy for quick reference! 📋
