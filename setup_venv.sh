#!/bin/bash
# SETUP VIRTUAL ENVIRONMENT FOR TRADING BOT
# Creates venv and installs all dependencies properly

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        SETTING UP VIRTUAL ENVIRONMENT FOR TRADING BOT            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/venv"

# Install python3-venv if not already installed
echo "📦 Installing python3-venv..."
apt-get update -qq
apt-get install -y python3-venv python3-full

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (yes/no): " -r
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "🗑️  Removing old venv..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing venv..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "🔧 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created at $VENV_DIR"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip in venv
echo ""
echo "📦 Upgrading pip in venv..."
pip install --upgrade pip

# Install all dependencies
echo ""
echo "📦 Installing ALL dependencies in venv..."
echo "   This will take a few minutes..."
echo ""

# Core data science
echo "📊 Installing core data science packages..."
pip install numpy pandas scipy scikit-learn

# Async and networking
echo "🌐 Installing async and networking..."
pip install aiohttp websockets requests

# Blockchain/Web3
echo "⛓️  Installing blockchain packages..."
pip install web3 eth-account eth-utils

# Trading
echo "💹 Installing trading packages..."
pip install ccxt pandas-ta

# NLP
echo "💬 Installing NLP packages..."
pip install nltk textblob vaderSentiment

# ML/AI (these are large, may take time)
echo "🤖 Installing ML/AI packages (this may take a while)..."
pip install tensorflow || echo "⚠️  TensorFlow install failed (optional)"
pip install torch || echo "⚠️  PyTorch install failed (optional)"
pip install transformers || echo "⚠️  Transformers install failed (optional)"

# Quantum
echo "⚛️  Installing quantum computing..."
pip install qiskit qiskit-ibm-runtime || echo "⚠️  Qiskit install failed (optional)"

# Utilities
echo "🔧 Installing utilities..."
pip install python-dotenv pyyaml toml

# Visualization (optional)
echo "📈 Installing visualization (optional)..."
pip install matplotlib seaborn plotly || echo "⚠️  Visualization packages failed (optional)"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              ✅ VIRTUAL ENVIRONMENT SETUP COMPLETE!               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Virtual environment location: $VENV_DIR"
echo "Python executable: $VENV_DIR/bin/python3"
echo "Pip executable: $VENV_DIR/bin/pip"
echo ""
echo "Next steps:"
echo "1. Update systemd service to use venv"
echo "2. Restart the bot"
echo ""
echo "Run: bash update_systemd_service.sh"
