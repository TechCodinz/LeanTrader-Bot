#!/bin/bash
# FINAL FIX - Install every single package needed

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║              FINAL DEPENDENCY FIX - ALL PACKAGES                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

cd /root/trading_bot
source venv/bin/activate

echo "📦 Installing ALL packages in one comprehensive installation..."
echo ""

# Install everything
pip install --quiet --upgrade pip setuptools wheel

# Core scientific computing
pip install --quiet numpy pandas scipy scikit-learn matplotlib seaborn

# Trading
pip install --quiet ccxt pandas-ta

# ML/AI
pip install --quiet tensorflow keras torch transformers

# Blockchain
pip install --quiet web3 eth-account eth-utils

# Web scraping
pip install --quiet beautifulsoup4 lxml requests

# NLP
pip install --quiet nltk textblob vaderSentiment

# LangChain
pip install --quiet langchain langchain-community openai

# Telegram (BOTH libraries)
pip install --quiet python-telegram-bot aiogram

# Async/networking
pip install --quiet aiohttp websockets httpx

# Utilities
pip install --quiet python-dotenv pyyaml toml colorama tqdm

# Social APIs (optional)
pip install --quiet tweepy praw discord.py 2>/dev/null || true

echo "✅ All packages installed!"
echo ""

# Verify critical imports
echo "🔍 Verifying imports..."
python3 << 'PYEOF'
critical = ['numpy', 'pandas', 'ccxt', 'telegram', 'aiogram', 'web3', 'bs4', 'langchain', 'sklearn', 'aiohttp']
failed = []
for pkg in critical:
    try:
        __import__(pkg)
        print(f"   ✅ {pkg}")
    except ImportError:
        print(f"   ❌ {pkg}")
        failed.append(pkg)

if not failed:
    print("\n✅ All critical packages verified!")
else:
    print(f"\n⚠️  {len(failed)} packages failed")
    exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "🔄 Restarting bot..."
    sudo systemctl restart trading-bot
    sleep 10
    echo ""
    systemctl status trading-bot --no-pager | head -15
    echo ""
    echo "✅ Check if running! If still issues: journalctl -u trading-bot -n 50"
fi
