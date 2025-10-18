#!/bin/bash
set -euo pipefail

echo "🚀 DEPLOYING ULTRA TRADING SYSTEM TO VPS"
echo "========================================"

# Configuration
VPS_USER="root"
VPS_HOST="your-vps-ip"
REPO_DIR="/opt/leantrader"
VENV_DIR="$REPO_DIR/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run commands on VPS
run_vps() {
    ssh "$VPS_USER@$VPS_HOST" "$@"
}

# Function to copy files to VPS
copy_to_vps() {
    scp -r "$1" "$VPS_USER@$VPS_HOST:$2"
}

print_status "Starting Ultra Trading System deployment..."

# Step 1: Clean VPS and prepare directories
print_status "Step 1: Cleaning VPS and preparing directories..."
run_vps "rm -rf $REPO_DIR && mkdir -p $REPO_DIR"
run_vps "mkdir -p /var/log/leantrader"
run_vps "mkdir -p /opt/leantrader/inbox_signals"
run_vps "mkdir -p /opt/leantrader/out/meta"
run_vps "mkdir -p /opt/leantrader/data"
print_success "VPS cleaned and directories prepared"

# Step 2: Copy project files
print_status "Step 2: Copying project files to VPS..."
copy_to_vps "." "$REPO_DIR/"
print_success "Project files copied"

# Step 3: Set up Python environment
print_status "Step 3: Setting up Python environment..."
run_vps "cd $REPO_DIR && python3 -m venv venv"
run_vps "cd $REPO_DIR && source venv/bin/activate && pip install --upgrade pip"
run_vps "cd $REPO_DIR && source venv/bin/activate && pip install -r requirements.txt"
print_success "Python environment set up"

# Step 4: Install additional dependencies
print_status "Step 4: Installing additional dependencies..."
run_vps "cd $REPO_DIR && source venv/bin/activate && pip install pyyaml feedparser prometheus-client"
print_success "Additional dependencies installed"

# Step 5: Set up environment files
print_status "Step 5: Setting up environment configuration..."
run_vps "cd $REPO_DIR && cp .env.example .env || echo 'No .env.example found'"
print_success "Environment files set up"

# Step 6: Set up systemd services
print_status "Step 6: Setting up systemd services..."

# Create leantrader.service
run_vps "cat > /etc/systemd/system/leantrader.service << 'EOF'
[Unit]
Description=LeanTrader Orchestrator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$REPO_DIR
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$REPO_DIR
EnvironmentFile=-$REPO_DIR/.env
ExecStart=$VENV_DIR/bin/python $REPO_DIR/runtime/unified_runner.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/leantrader/orchestrator.log
StandardError=append:/var/log/leantrader/orchestrator.err

[Install]
WantedBy=multi-user.target
EOF"

# Create leantrader-router.service
run_vps "cat > /etc/systemd/system/leantrader-router.service << 'EOF'
[Unit]
Description=LeanTrader Auto Env Router (live or testnet)
After=network.target

[Service]
User=root
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_DIR/bin/python $REPO_DIR/tools/auto_env_router.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"

# Create leantrader-swarms.service
run_vps "cat > /etc/systemd/system/leantrader-swarms.service << 'EOF'
[Unit]
Description=LeanTrader Multi-Exchange Swarm Trainers
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
User=root
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_DIR/bin/python $REPO_DIR/tools/swarm_manager.py

[Install]
WantedBy=multi-user.target
EOF"

print_success "Systemd services created"

# Step 7: Set permissions
print_status "Step 7: Setting permissions..."
run_vps "chmod +x $REPO_DIR/scripts/*.sh"
run_vps "chmod +x $REPO_DIR/tools/*.py"
run_vps "chown -R root:root $REPO_DIR"
run_vps "chown -R root:root /var/log/leantrader"
print_success "Permissions set"

# Step 8: Reload systemd and enable services
print_status "Step 8: Enabling services..."
run_vps "systemctl daemon-reload"
run_vps "systemctl enable leantrader"
run_vps "systemctl enable leantrader-router"
run_vps "systemctl enable leantrader-swarms"
print_success "Services enabled"

# Step 9: Install yq for YAML processing
print_status "Step 9: Installing yq for YAML processing..."
run_vps "curl -sL https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o /usr/local/bin/yq && chmod +x /usr/local/bin/yq"
print_success "yq installed"

# Step 10: Create startup script
print_status "Step 10: Creating startup script..."
run_vps "cat > $REPO_DIR/start_ultra_system.sh << 'EOF'
#!/bin/bash
echo '🚀 Starting Ultra Trading System...'

# Start the auto environment router (this will choose live/testnet based on balance)
systemctl start leantrader-router

# Start the swarm trainers for parallel learning
systemctl start leantrader-swarms

# Start the main orchestrator
systemctl start leantrader

echo '✅ Ultra Trading System started!'
echo '📊 Check status with: systemctl status leantrader'
echo '📈 View logs with: journalctl -u leantrader -f'
echo '🔍 Check metrics at: http://localhost:9300'
EOF"
run_vps "chmod +x $REPO_DIR/start_ultra_system.sh"
print_success "Startup script created"

# Step 11: Final verification
print_status "Step 11: Verifying installation..."
run_vps "cd $REPO_DIR && source venv/bin/activate && python -c 'import ccxt, yaml, feedparser; print(\"✅ All dependencies available\")'"
run_vps "systemctl daemon-reload"
print_success "Installation verified"

echo ""
echo "🎉 ULTRA TRADING SYSTEM DEPLOYMENT COMPLETE!"
echo "============================================="
echo ""
echo "📋 Next steps:"
echo "1. SSH into your VPS: ssh $VPS_USER@$VPS_HOST"
echo "2. Configure your API keys in $REPO_DIR/.env"
echo "3. Start the system: $REPO_DIR/start_ultra_system.sh"
echo "4. Monitor with: systemctl status leantrader"
echo "5. View logs: journalctl -u leantrader -f"
echo "6. Check metrics: curl http://localhost:9300"
echo ""
echo "🔧 Configuration files:"
echo "- Main config: $REPO_DIR/.env"
echo "- Exchange configs: $REPO_DIR/configs/exchanges.yml"
echo "- Exchange profiles: $REPO_DIR/configs/exchange_profiles.yml"
echo ""
echo "🚀 The system will automatically:"
echo "- Choose between live/testnet based on account balance"
echo "- Apply order guardrails for safety"
echo "- Process copy signals from external sources"
echo "- Run parallel training across multiple exchanges"
echo "- Use meta-brain for ensemble learning"
echo ""
print_success "Deployment complete! Ready to evolve and make profits! 💰"
