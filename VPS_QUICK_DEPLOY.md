# 🚀 VPS QUICK DEPLOY - Ultra Trading System

## 📋 **ONE-LINER DEPLOYMENT**

Run this single command on your VPS to deploy everything:

```bash
curl -sL https://raw.githubusercontent.com/TechCodinz/Lean-Trader/main/vps_deploy_commands.sh | bash
```

## 🔧 **MANUAL DEPLOYMENT**

If you prefer manual steps:

```bash
# 1. Clone the repository
git clone https://github.com/TechCodinz/Lean-Trader.git /opt/leantrader
cd /opt/leantrader

# 2. Run the deployment script
chmod +x vps_deploy_commands.sh
./vps_deploy_commands.sh
```

## ⚙️ **CONFIGURATION**

After deployment, configure your API keys:

```bash
# Edit environment file
nano /opt/leantrader/.env

# For Gate.io live trading, update:
ENABLE_LIVE=true
ALLOW_LIVE=true
LIVE_CONFIRM=YES
EXCHANGE_ID=gateio
GATEIO_API_KEY=your_api_key_here
GATEIO_SECRET=your_secret_here
```

## 🚀 **START THE SYSTEM**

```bash
# Start the Ultra Trading System
/opt/leantrader/start_ultra_system.sh

# Monitor the system
systemctl status leantrader
journalctl -u leantrader -f
```

## 📊 **MONITORING**

```bash
# Check status
systemctl status leantrader

# View logs
journalctl -u leantrader -f

# Check metrics
curl http://localhost:9300/metrics

# View performance data
tail -f /opt/leantrader/out/meta/meta_weights.jsonl
```

## 🎉 **READY TO MAKE PROFITS!**

Your Ultra Trading System is now deployed with:
- ✅ Order safety guardrails
- ✅ Meta-brain ensemble learning
- ✅ Copy signals processing
- ✅ Multi-exchange swarm training
- ✅ Complete monitoring setup

**Time to scale and make those profits! 💰🚀**
