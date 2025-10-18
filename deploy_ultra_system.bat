@echo off
echo 🚀 DEPLOYING ULTRA TRADING SYSTEM TO VPS
echo ========================================

REM Configuration - UPDATE THESE VALUES
set VPS_USER=root
set VPS_HOST=your-vps-ip-here
set REPO_DIR=/opt/leantrader

echo.
echo ⚠️  IMPORTANT: Update VPS_HOST in this script before running!
echo Current VPS_HOST: %VPS_HOST%
echo.
pause

echo 📦 Preparing deployment package...

REM Create deployment package
tar -czf ultra_trading_system.tar.gz --exclude=venv --exclude=__pycache__ --exclude=.git --exclude=*.pyc --exclude=.env .

echo ✅ Package created: ultra_trading_system.tar.gz
echo.
echo 📤 Uploading to VPS...
scp ultra_trading_system.tar.gz %VPS_USER%@%VPS_HOST%:/tmp/

echo 📋 Running deployment commands on VPS...
ssh %VPS_USER%@%VPS_HOST% "
echo '🚀 Starting VPS deployment...'
cd /tmp
rm -rf %REPO_DIR%
mkdir -p %REPO_DIR%
tar -xzf ultra_trading_system.tar.gz -C %REPO_DIR%
cd %REPO_DIR%

echo '🐍 Setting up Python environment...'
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ccxt pyyaml feedparser prometheus-client requests beautifulsoup4 numpy pandas

echo '📁 Creating directories...'
mkdir -p /var/log/leantrader
mkdir -p /opt/leantrader/inbox_signals
mkdir -p /opt/leantrader/out/meta
mkdir -p /opt/leantrader/data

echo '⚙️ Setting up systemd services...'
cat > /etc/systemd/system/leantrader.service << 'EOF'
[Unit]
Description=LeanTrader Orchestrator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=%REPO_DIR%
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=%REPO_DIR%
EnvironmentFile=-%REPO_DIR%/.env
ExecStart=%REPO_DIR%/venv/bin/python %REPO_DIR%/runtime/unified_runner.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/leantrader/orchestrator.log
StandardError=append:/var/log/leantrader/orchestrator.err

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/leantrader-router.service << 'EOF'
[Unit]
Description=LeanTrader Auto Env Router
After=network.target

[Service]
User=root
WorkingDirectory=%REPO_DIR%
ExecStart=%REPO_DIR%/venv/bin/python %REPO_DIR%/tools/auto_env_router.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo '🔧 Setting permissions...'
chmod +x %REPO_DIR%/tools/*.py
chmod +x %REPO_DIR%/scripts/*.sh
chown -R root:root %REPO_DIR%
chown -R root:root /var/log/leantrader

echo '🔄 Enabling services...'
systemctl daemon-reload
systemctl enable leantrader
systemctl enable leantrader-router

echo '📊 Installing yq for YAML processing...'
curl -sL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /usr/local/bin/yq
chmod +x /usr/local/bin/yq

echo '✅ Deployment complete!'
echo '📋 Next steps:'
echo '1. Configure API keys: nano %REPO_DIR%/.env'
echo '2. Start system: systemctl start leantrader'
echo '3. Monitor: systemctl status leantrader'
echo '4. View logs: journalctl -u leantrader -f'
"

echo.
echo 🎉 DEPLOYMENT COMPLETE!
echo ======================
echo.
echo 📋 Next steps:
echo 1. SSH into your VPS: ssh %VPS_USER%@%VPS_HOST%
echo 2. Configure API keys: nano %REPO_DIR%/.env
echo 3. Start the system: systemctl start leantrader
echo 4. Monitor: systemctl status leantrader
echo 5. View logs: journalctl -u leantrader -f
echo.
echo 🚀 Ready to make profits! 💰
pause
