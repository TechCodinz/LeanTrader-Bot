# 🚀 QUICK DEPLOYMENT GUIDE - New VPS Setup

## 📋 What You Need
1. ✅ New VPS (Ubuntu 20.04+ recommended)
2. ✅ Your `.env.backup` file (API keys)
3. ✅ GitHub repository (from backup)

---

## ⚡ FAST SETUP (5 Minutes)

### **Step 1: On Current VPS - Backup to GitHub**
```bash
cd ~/trading_bot
bash BACKUP_TO_GITHUB.sh
```
**You'll need**:
- GitHub username
- Repository name (e.g., `trading-bot-live`)
- GitHub Personal Access Token ([create one here](https://github.com/settings/tokens))

### **Step 2: On New VPS - Clone & Deploy**
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Copy your .env file
nano .env
# Paste your API keys from .env.backup

# Run auto-deployment
bash DEPLOY_NEW_VPS.sh

# Start bot
bash START_BOT.sh
```

**That's it!** ✅ Bot is running on new VPS!

---

## 📊 Management Commands

### Start Bot
```bash
bash START_BOT.sh
```

### View Live Logs
```bash
sudo journalctl -u trading-bot-live -f
```

### Check Status
```bash
sudo systemctl status trading-bot-live
```

### Stop Bot
```bash
sudo systemctl stop trading-bot-live
```

### Restart Bot
```bash
sudo systemctl restart trading-bot-live
```

---

## 🔐 Security Checklist

### ✅ Before Backing Up
- [x] `.env` file excluded (in `.gitignore`)
- [x] API keys NOT in GitHub
- [x] `.env.backup` saved locally

### ✅ Before New Deployment
- [x] .env file created on new VPS
- [x] All API keys added
- [x] Telegram bot token configured

---

## 📦 What Gets Backed Up

**Included in GitHub** ✅:
- All Python files (`.py`)
- All scripts (`.sh`)
- Documentation (`.md`)
- Requirements (`requirements.txt`)
- Databases (`.db`) - optional
- Configuration templates (`.env.example`)

**Excluded from GitHub** 🔒:
- `.env` (sensitive API keys)
- `__pycache__/` (Python cache)
- `*.log` (large log files)
- `.env.backup` (local only)

---

## 🆘 Troubleshooting

### Issue: Bot won't start
```bash
# Check Python version
python3 --version  # Need 3.8+

# Reinstall packages
pip install -r requirements.txt

# Check .env file
cat .env  # Make sure API keys are filled in
```

### Issue: Permission denied
```bash
# Fix permissions
chmod +x *.sh
chmod 600 .env
```

### Issue: Service not found
```bash
# Recreate service
bash DEPLOY_NEW_VPS.sh
```

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Backup to GitHub | `bash BACKUP_TO_GITHUB.sh` |
| Deploy on new VPS | `bash DEPLOY_NEW_VPS.sh` |
| Start bot | `bash START_BOT.sh` |
| View logs | `sudo journalctl -u trading-bot-live -f` |
| Stop bot | `sudo systemctl stop trading-bot-live` |
| Restart bot | `sudo systemctl restart trading-bot-live` |

---

## 🎯 Migration Checklist

### On Old VPS:
- [ ] Run `bash BACKUP_TO_GITHUB.sh`
- [ ] Save `.env.backup` to local computer
- [ ] Note GitHub repository URL

### On New VPS:
- [ ] Clone repository from GitHub
- [ ] Create `.env` file with API keys
- [ ] Run `bash DEPLOY_NEW_VPS.sh`
- [ ] Run `bash START_BOT.sh`
- [ ] Verify bot is trading: `sudo journalctl -u trading-bot-live -f`

### Cleanup Old VPS:
- [ ] Stop old bot: `sudo systemctl stop trading-bot-live`
- [ ] Optional: Keep as backup for 24 hours

---

**Created**: $(date)
**For**: Emergency VPS migration & deployment
