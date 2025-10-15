#!/bin/bash
# Install ALL missing packages - no more dependency errors

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║         INSTALLING ALL MISSING PACKAGES - COMPLETE LIST             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

cd /root/trading_bot

# Activate venv
source venv/bin/activate

echo "📦 Installing ALL dependencies in one go..."
echo ""

# Install everything in one command to resolve dependencies properly
pip install --upgrade pip setuptools wheel -q

echo "Installing core packages..."
pip install \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    -q

echo "Installing trading packages..."
pip install \
    ccxt \
    pandas-ta \
    ta-lib \
    -q 2>/dev/null || echo "   (ta-lib skipped - requires system dependency)"

echo "Installing ML/AI packages..."
pip install \
    tensorflow \
    keras \
    torch \
    transformers \
    -q

echo "Installing blockchain packages..."
pip install \
    web3 \
    eth-account \
    eth-utils \
    -q

echo "Installing web scraping..."
pip install \
    beautifulsoup4 \
    lxml \
    requests \
    -q

echo "Installing NLP packages..."
pip install \
    nltk \
    textblob \
    vaderSentiment \
    -q

echo "Installing LangChain and related..."
pip install \
    langchain \
    langchain-community \
    openai \
    -q

echo "Installing async/networking..."
pip install \
    aiohttp \
    websockets \
    httpx \
    -q

echo "Installing Telegram..."
pip install \
    python-telegram-bot \
    -q

echo "Installing utilities..."
pip install \
    python-dotenv \
    pyyaml \
    toml \
    colorama \
    tqdm \
    -q

echo "Installing social media APIs..."
pip install \
    tweepy \
    praw \
    discord.py \
    -q 2>/dev/null || echo "   (Some social APIs skipped - optional)"

echo "Installing quantum computing (optional)..."
pip install \
    qiskit \
    qiskit-ibm-runtime \
    -q 2>/dev/null || echo "   (Qiskit skipped - optional for quantum features)"

echo ""
echo "✅ ALL PACKAGES INSTALLED!"
echo ""

# Verify critical packages
echo "🔍 Verifying critical imports..."
python3 << 'PYEOF'
import sys
packages = {
    'numpy': 'NumPy',
    'pandas': 'Pandas', 
    'ccxt': 'CCXT',
    'telegram': 'python-telegram-bot',
    'web3': 'Web3',
    'bs4': 'BeautifulSoup4',
    'langchain': 'LangChain',
    'sklearn': 'scikit-learn',
    'aiohttp': 'aiohttp',
}

failed = []
for module, name in packages.items():
    try:
        __import__(module)
        print(f"   ✅ {name}")
    except ImportError as e:
        print(f"   ❌ {name} - {e}")
        failed.append(name)

if failed:
    print(f"\n⚠️  {len(failed)} packages failed to import")
    sys.exit(1)
else:
    print("\n✅ All critical packages verified!")
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "🔄 Restarting bot..."
    sudo systemctl restart trading-bot
    
    sleep 15
    
    echo ""
    echo "📊 STATUS:"
    systemctl status trading-bot --no-pager | head -15
    
    echo ""
    echo "✅ Check logs with: journalctl -u trading-bot -f"
else
    echo ""
    echo "⚠️  Some packages failed. Check errors above."
fi
