# 🔧 ULTRA TRADING SYSTEM - ENVIRONMENT CONFIGURATION GUIDE

## 📋 **ENVIRONMENT FILES OVERVIEW**

### **Available Environment Files:**
- **`env.example`** - Complete configuration template with all features
- **`env.paper`** - Paper trading mode (safe testing)
- **`env.live`** - Live trading mode (real money)
- **`env.testnet`** - Testnet mode (exchange testnet)

## 🎯 **QUICK CONFIGURATION**

### **For Paper Trading (Recommended Start):**
```bash
cp env.paper .env
# Edit API keys if needed
nano .env
```

### **For Live Trading (After Testing):**
```bash
cp env.live .env
# Update with your real API keys
nano .env
```

### **For Testnet Trading:**
```bash
cp env.testnet .env
# Update with testnet API keys
nano .env
```

## 🔑 **API KEY CONFIGURATION**

### **Gate.io (Live Trading):**
```bash
GATEIO_API_KEY=your_gateio_api_key_here
GATEIO_SECRET=your_gateio_secret_here
GATEIO_SANDBOX=false
```

### **Bybit (Testnet):**
```bash
BYBIT_API_KEY=your_bybit_testnet_api_key_here
BYBIT_SECRET=your_bybit_testnet_secret_here
BYBIT_TESTNET=true
```

### **Binance (Backup):**
```bash
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here
BINANCE_TESTNET=true
```

## ⚙️ **FEATURE CONFIGURATION**

### **Meta-Brain & Ensemble Learning:**
```bash
META_BRAIN_ENABLED=true
ENSEMBLE_LEARNING=true
PERFORMANCE_TRACKING=true
MODEL_UPDATE_INTERVAL=86400
REBALANCE_INTERVAL=3600
```

### **Copy Signals & External Integration:**
```bash
COPY_SIGNALS_ENABLED=true
SIGNALS_INBOX_DIR=/opt/leantrader/inbox_signals
SIGNALS_PROCESSING_INTERVAL=300
EXTERNAL_SIGNALS_ENABLED=true
```

### **Multi-Exchange Swarm Training:**
```bash
SWARM_ENABLED=true
PARALLEL_TRAINING=true
EXCHANGE_ISOLATION=true
SWARM_AGENTS=100
TRAINING_INTERVAL=3600
```

### **Ultra God Mode Features:**
```bash
GOD_MODE_ENABLED=true
QUANTUM_PREDICTION=true
FRACTAL_ANALYSIS=true
SMART_MONEY_TRACKING=true
MOON_SPOTTER_ENABLED=true
AUTO_SNIPE_ENABLED=true
MAX_SNIPE_AMOUNT=100
```

### **Forex & Metals Master:**
```bash
FOREX_MASTER_ENABLED=true
TRADE_FOREX=true
TRADE_METALS=true
TRADE_COMMODITIES=true
FOREX_SYMBOLS=EURUSD,GBPUSD,USDJPY
METALS_SYMBOLS=XAUUSD,XAGUSD
COMMODITIES_SYMBOLS=USOIL,NGAS
```

### **Telegram Notifications:**
```bash
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
TELEGRAM_VIP_CHANNEL=your_vip_channel_id_here
TELEGRAM_HEARTBEAT=true
TELEGRAM_SIGNALS=true
HEARTBEAT_SECS=1800
```

## 🛡️ **SAFETY CONFIGURATION**

### **Live Trading Safety:**
```bash
ENABLE_LIVE=true
ALLOW_LIVE=true
LIVE_CONFIRM=YES
RISK_PER_TRADE=0.01
MAX_POSITIONS=3
STOP_LOSS_PCT=0.03
TAKE_PROFIT_PCT=0.06
```

### **Paper Trading Safety:**
```bash
ENABLE_LIVE=false
ALLOW_LIVE=false
LIVE_CONFIRM=NO
RISK_PER_TRADE=0.02
MAX_POSITIONS=5
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.10
```

## 📊 **MONITORING CONFIGURATION**

### **Metrics & Monitoring:**
```bash
METRICS_ENABLED=true
METRICS_PORT=9300
PROMETHEUS_ENABLED=true
DASHBOARD_ENABLED=true
```

### **Logging:**
```bash
LOG_LEVEL=INFO
DEBUG_MODE=false
VERBOSE_LOGGING=false
LOG_RETENTION_DAYS=30
```

## 🔄 **ENVIRONMENT SWITCHING**

### **Switch to Paper Trading:**
```bash
cp .env.paper .env
systemctl restart leantrader
```

### **Switch to Live Trading:**
```bash
cp .env.live .env
# Update API keys
nano .env
systemctl restart leantrader
```

### **Switch to Testnet:**
```bash
cp .env.testnet .env
# Update testnet API keys
nano .env
systemctl restart leantrader
```

## 🚨 **IMPORTANT SAFETY NOTES**

### **Before Live Trading:**
1. **Test with Paper Trading First** - Use `env.paper` for initial testing
2. **Verify API Keys** - Ensure your API keys are correct
3. **Set Conservative Risk** - Start with small position sizes
4. **Enable Notifications** - Set up Telegram for monitoring
5. **Test Stop Losses** - Verify risk management works

### **API Key Security:**
- Never commit API keys to git
- Use environment files for sensitive data
- Regularly rotate API keys
- Use read-only keys when possible

## 🎯 **RECOMMENDED CONFIGURATIONS**

### **For Beginners:**
- Start with `env.paper`
- Enable all features for learning
- Use conservative risk settings
- Enable verbose logging

### **For Experienced Traders:**
- Use `env.live` with real API keys
- Enable all advanced features
- Use moderate risk settings
- Enable notifications

### **For Testing:**
- Use `env.testnet` with testnet API keys
- Enable aggressive settings for testing
- Use high risk for stress testing
- Enable debug logging

## 🚀 **QUICK START COMMANDS**

```bash
# 1. Deploy the system
curl -sL https://raw.githubusercontent.com/TechCodinz/Lean-Trader/main/vps_deploy_commands.sh | bash

# 2. Configure for paper trading
cd /opt/leantrader
cp env.paper .env

# 3. Start the system
./start_ultra_system.sh

# 4. Monitor
systemctl status leantrader
journalctl -u leantrader -f
```

## 🎉 **READY TO CONFIGURE!**

Your Ultra Trading System now has comprehensive environment configuration files that match all the advanced features we've integrated:

- ✅ **Guard Hook Integration** - Order safety settings
- ✅ **Meta-Brain Learning** - Ensemble learning configuration
- ✅ **Copy Signals** - External signal processing settings
- ✅ **Swarm Training** - Multi-exchange parallel training
- ✅ **God Mode Features** - All advanced trading features
- ✅ **Safety Settings** - Risk management and safety guards

**Choose your environment file and start making profits! 💰🚀**
