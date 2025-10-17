#!/bin/bash
# INSTALL ALL DEPENDENCIES FOR FULL FEATURE SET
# Run this on your VPS to enable ALL ultra-rare goldmine features

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     INSTALLING ALL DEPENDENCIES FOR ULTRA GOLDMINE FEATURES      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Update pip
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip

# Install core dependencies
echo ""
echo "📦 Installing core Python packages..."
pip3 install numpy pandas scikit-learn scipy

# Install async and networking
echo ""
echo "📦 Installing async and networking packages..."
pip3 install aiohttp asyncio websockets

# Install blockchain/web3 (for whale tracking, DEX, on-chain)
echo ""
echo "📦 Installing blockchain packages..."
pip3 install web3 eth-account eth-utils

# Install ML/AI packages
echo ""
echo "📦 Installing ML/AI packages..."
pip3 install tensorflow keras torch transformers

# Install data analysis
echo ""
echo "📦 Installing data analysis packages..."
pip3 install matplotlib seaborn plotly

# Install trading packages
echo ""
echo "📦 Installing trading packages..."
pip3 install ccxt ta-lib pandas-ta

# Install NLP for social media analysis
echo ""
echo "📦 Installing NLP packages..."
pip3 install nltk textblob vaderSentiment

# Install API clients
echo ""
echo "📦 Installing API clients..."
pip3 install requests tweepy praw discord.py

# Install quantum computing (IBM Qiskit)
echo ""
echo "📦 Installing quantum computing..."
pip3 install qiskit qiskit-ibm-runtime

# Install additional utilities
echo ""
echo "📦 Installing utilities..."
pip3 install python-dotenv pyyaml toml

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ ALL DEPENDENCIES INSTALLED!                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Now restart the bot:"
echo "  sudo systemctl restart trading-bot"
echo ""
echo "All 50+ systems and ultra-rare features will be FULLY ACTIVE! 🚀"
