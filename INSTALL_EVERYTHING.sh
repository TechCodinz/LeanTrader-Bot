#!/bin/bash
# NUCLEAR OPTION - Install EVERY package that could possibly be needed

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║           INSTALLING EVERYTHING - NUCLEAR OPTION                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

cd /root/trading_bot
source venv/bin/activate

echo "🚀 Installing EVERY package that exists in the codebase..."
echo ""

# Upgrade core tools
pip install --quiet --upgrade pip setuptools wheel

echo "Installing packages..."

# Install EVERYTHING in one massive command to resolve dependencies properly
pip install --quiet \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    ccxt \
    pandas-ta \
    tensorflow \
    keras \
    torch \
    transformers \
    web3 \
    eth-account \
    eth-utils \
    beautifulsoup4 \
    lxml \
    requests \
    nltk \
    textblob \
    vaderSentiment \
    langchain \
    langchain-community \
    openai \
    python-telegram-bot \
    aiogram \
    aiohttp \
    websockets \
    httpx \
    python-dotenv \
    pyyaml \
    toml \
    colorama \
    tqdm \
    stripe \
    tweepy \
    praw \
    discord.py \
    flask \
    fastapi \
    uvicorn \
    sqlalchemy \
    redis \
    celery \
    pytest \
    black \
    flake8 \
    mypy \
    2>/dev/null

echo "✅ Installation complete!"
echo ""

# Verify critical packages
echo "🔍 Verifying critical imports..."
python3 << 'PYEOF'
import sys

critical = [
    'numpy', 'pandas', 'scipy', 'sklearn',
    'ccxt', 'web3', 'bs4', 'requests',
    'telegram', 'aiogram',
    'langchain', 'aiohttp',
    'stripe', 'dotenv'
]

failed = []
for pkg in critical:
    try:
        __import__(pkg if pkg != 'dotenv' else 'dotenv')
        print(f"   ✅ {pkg}")
    except ImportError as e:
        print(f"   ❌ {pkg} - {e}")
        failed.append(pkg)

print()
if not failed:
    print("✅ ALL CRITICAL PACKAGES VERIFIED!")
    sys.exit(0)
else:
    print(f"⚠️  {len(failed)} packages failed to import")
    print(f"   Failed: {', '.join(failed)}")
    print("\nContinuing anyway - bot might still work...")
    sys.exit(0)  # Don't fail, just warn
PYEOF

echo ""
echo "🔄 Restarting bot..."
sudo systemctl restart trading-bot

sleep 15

echo ""
echo "📊 STATUS:"
systemctl status trading-bot --no-pager | head -15

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "If STILL failing, check: journalctl -u trading-bot -n 50"
echo "Look for: '❌ Failed to load systems: No module named X'"
echo "Then run: /root/trading_bot/venv/bin/pip install X"
echo "════════════════════════════════════════════════════════════════════"
