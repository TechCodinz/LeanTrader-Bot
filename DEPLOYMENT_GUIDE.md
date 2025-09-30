# 🚀 ULTRA TRADING SYSTEM - VPS DEPLOYMENT GUIDE

## 📋 **QUICK DEPLOYMENT STEPS**

### **Step 1: Prepare Your VPS**
```bash
# SSH into your VPS
ssh root@your-vps-ip

# Clean and prepare directories
rm -rf /opt/leantrader
mkdir -p /opt/leantrader
mkdir -p /var/log/leantrader
mkdir -p /opt/leantrader/inbox_signals
mkdir -p /opt/leantrader/out/meta
mkdir -p /opt/leantrader/data
```

### **Step 2: Upload Project Files**
```bash
# From your local machine, upload the project
# Option A: Using SCP (if you have SSH access)
scp -r . root@your-vps-ip:/opt/leantrader/

# Option B: Using SFTP
sftp root@your-vps-ip
put -r . /opt/leantrader/
quit

# Option C: Using Git (if your repo is on GitHub)
ssh root@your-vps-ip
cd /opt/leantrader
git clone https://github.com/yourusername/your-repo.git .
```

### **Step 3: Set Up Python Environment**
```bash
# On your VPS
cd /opt/leantrader
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ccxt pyyaml feedparser prometheus-client requests beautifulsoup4 numpy pandas
```

### **Step 4: Create Systemd Services**
```bash
# Create main orchestrator service
cat > /etc/systemd/system/leantrader.service << 'EOF'
[Unit]
Description=LeanTrader Orchestrator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/leantrader
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/opt/leantrader
EnvironmentFile=-/opt/leantrader/.env
ExecStart=/opt/leantrader/venv/bin/python /opt/leantrader/runtime/unified_runner.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/leantrader/orchestrator.log
StandardError=append:/var/log/leantrader/orchestrator.err

[Install]
WantedBy=multi-user.target
EOF

# Create auto router service
cat > /etc/systemd/system/leantrader-router.service << 'EOF'
[Unit]
Description=LeanTrader Auto Env Router
After=network.target

[Service]
User=root
WorkingDirectory=/opt/leantrader
ExecStart=/opt/leantrader/venv/bin/python /opt/leantrader/tools/auto_env_router.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

### **Step 5: Set Permissions**
```bash
chmod +x /opt/leantrader/tools/*.py
chmod +x /opt/leantrader/scripts/*.sh
chown -R root:root /opt/leantrader
chown -R root:root /var/log/leantrader
```

### **Step 6: Configure Environment**
```bash
# Create environment file
cat > /opt/leantrader/.env << 'EOF'
# Trading Mode
ENABLE_LIVE=false
ALLOW_LIVE=false
LIVE_CONFIRM=NO

# Exchange Configuration
EXCHANGE_ID=paper
PAPER_START_CASH=5000

# Risk Management
RISK_PER_TRADE=0.02
MAX_POSITIONS=5
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.10

# Meta-Brain Settings
META_BRAIN_ENABLED=true
ENSEMBLE_LEARNING=true
PERFORMANCE_TRACKING=true

# Copy Signals
COPY_SIGNALS_ENABLED=true
SIGNALS_INBOX_DIR=/opt/leantrader/inbox_signals

# Multi-Exchange Swarm
SWARM_ENABLED=true
PARALLEL_TRAINING=true
EXCHANGE_ISOLATION=true

# Logging
LOG_LEVEL=INFO
EOF
```

### **Step 7: Install Additional Tools**
```bash
# Install yq for YAML processing
curl -sL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /usr/local/bin/yq
chmod +x /usr/local/bin/yq

# Install tmux for session management
apt-get update && apt-get install -y tmux
```

### **Step 8: Enable and Start Services**
```bash
# Reload systemd
systemctl daemon-reload

# Enable services
systemctl enable leantrader
systemctl enable leantrader-router

# Start the system
systemctl start leantrader
```

### **Step 9: Monitor the System**
```bash
# Check status
systemctl status leantrader

# View logs
journalctl -u leantrader -f

# Check metrics
curl http://localhost:9300/metrics
```

## 🎯 **CONFIGURATION FOR LIVE TRADING**

### **For Gate.io Live Trading:**
```bash
# Edit environment file
nano /opt/leantrader/.env

# Update these values:
ENABLE_LIVE=true
ALLOW_LIVE=true
LIVE_CONFIRM=YES
EXCHANGE_ID=gateio
GATEIO_API_KEY=your_api_key_here
GATEIO_SECRET=your_secret_here
```

### **For Bybit Testnet:**
```bash
# Update environment file
EXCHANGE_ID=bybit
BYBIT_TESTNET=true
BYBIT_API_KEY=your_testnet_api_key
BYBIT_SECRET=your_testnet_secret
```

## 📊 **MONITORING COMMANDS**

```bash
# System status
systemctl status leantrader leantrader-router

# Real-time logs
journalctl -u leantrader -f

# Check metrics endpoint
curl http://localhost:9300/metrics

# View performance data
ls /opt/leantrader/out/*/reports/metrics.json

# Check meta-brain weights
tail -f /opt/leantrader/out/meta/meta_weights.jsonl
```

## 🚨 **TROUBLESHOOTING**

### **Service won't start:**
```bash
systemctl status leantrader
journalctl -u leantrader -n 50
```

### **Permission errors:**
```bash
chown -R root:root /opt/leantrader
chmod +x /opt/leantrader/tools/*.py
```

### **Missing dependencies:**
```bash
cd /opt/leantrader
source venv/bin/activate
pip install -r requirements.txt
```

## 🎉 **SUCCESS!**

Your Ultra Trading System is now deployed and ready to make profits! 

The system will:
- ✅ Automatically choose between live/testnet based on balance
- ✅ Apply order guardrails for safety
- ✅ Process copy signals from external sources
- ✅ Run parallel training across exchanges
- ✅ Use meta-brain for ensemble learning
- ✅ Evolve and adapt in real-time

**Ready to scale and make those profits! 💰🚀**