# 📋 COMPLETE API REQUIREMENTS GUIDE

**What you need for full bot functionality**

---

## ✅ ALREADY CONFIGURED (You Have These)

### 1. Telegram Bot
- **What**: Send notifications, signals, alerts
- **Status**: ✅ **CONFIGURED**
- **Token**: You provided this
- **Cost**: FREE

### 2. Bybit Testnet
- **What**: Test trading with fake money
- **Status**: ✅ **CONFIGURED**
- **Keys**: You provided these
- **Cost**: FREE

---

## 🔥 CRITICAL APIs (Needed for Full Functionality)

### 1. **Bybit LIVE API** (When Ready for Real Trading)
**What it does:**
- Execute real trades
- Get live price data
- Manage positions
- Track balance

**How to get:**
1. Go to: https://www.bybit.com/
2. Sign up / Login
3. Go to API Management
4. Create API key
5. Enable: "Trade" permission
6. Save API Key + Secret

**Cost**: FREE (just need account)

**When to get**: After testnet testing (1-2 weeks)

**Priority**: 🔥 HIGH (for live trading)

---

### 2. **OHLCV Data Provider** (FOR TRAINING MODELS)
**This is CRITICAL for ML training!**

**Option A: Use Bybit API (RECOMMENDED - FREE)**
```python
# Already included in ccxt!
import ccxt

exchange = ccxt.bybit()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
# Returns: [[timestamp, open, high, low, close, volume], ...]
```

**How to get historical data:**
```python
# Get last 1000 candles (free)
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)

# For more history, loop:
all_data = []
since = exchange.parse8601('2023-01-01T00:00:00Z')
while since < exchange.milliseconds():
    data = exchange.fetch_ohlcv('BTC/USDT', '1h', since, 1000)
    if not data:
        break
    all_data.extend(data)
    since = data[-1][0] + 1
```

**Cost**: FREE ✅

**Option B: CryptoCompare API (More History)**
- Website: https://www.cryptocompare.com/
- Free tier: 100,000 calls/month
- Get API key: Sign up → API Keys
- Good for: Historical data going back years

**Option C: Alpha Vantage (Stocks + Crypto)**
- Website: https://www.alphavantage.co/
- Free tier: 500 calls/day
- Get API key: Sign up → Get Free API Key

**RECOMMENDATION**: 
✅ **Use Bybit via ccxt** (already working, free, sufficient)

**Priority**: 🔥 **CRITICAL** (for ML training)

---

## 📰 RECOMMENDED APIs (Enhanced Features)

### 3. **News APIs**

**Option A: NewsAPI.org (RECOMMENDED)**
- **What**: Get crypto news for sentiment analysis
- **Free tier**: 100 requests/day
- **Cost**: FREE (limited) or $449/month (unlimited)
- **How to get**:
  1. Go to: https://newsapi.org/
  2. Sign up
  3. Get API key
  4. Add to .env: `NEWSAPI_KEY=your_key`

**Option B: CryptoPanic**
- **What**: Crypto-specific news aggregator
- **Free tier**: Limited
- **Website**: https://cryptopanic.com/developers/api/
- **Cost**: FREE (basic) or $10/month (pro)

**Option C: Finnhub**
- **What**: Financial news + data
- **Free tier**: 60 calls/minute
- **Website**: https://finnhub.io/
- **Cost**: FREE

**RECOMMENDATION**: 
✅ **NewsAPI.org** (best for crypto news)

**Priority**: 🟡 MEDIUM (nice to have, not critical)

---

### 4. **Blockchain Scanner APIs** (For DEX Trading)

**Etherscan (Ethereum)**
- **What**: Track Ethereum transactions, tokens
- **Free tier**: 5 calls/second
- **How to get**:
  1. Go to: https://etherscan.io/
  2. Sign up
  3. API → Get API Key
  4. Add to .env: `ETHERSCAN_API_KEY=your_key`
- **Cost**: FREE ✅

**BSCScan (Binance Smart Chain)**
- **What**: Track BSC transactions, tokens
- **Free tier**: 5 calls/second
- **Website**: https://bscscan.com/
- **Cost**: FREE ✅

**PolygonScan**
- **Website**: https://polygonscan.com/
- **Cost**: FREE ✅

**RECOMMENDATION**: Get all three (they're free)

**Priority**: 🟡 MEDIUM (only if doing DEX trading)

---

### 5. **Safety Checker APIs** (For DEX/New Tokens)

**GoPlus Labs (RECOMMENDED)**
- **What**: Token security checks, honeypot detection
- **Free tier**: 100 calls/day
- **Website**: https://gopluslabs.io/
- **How to get**:
  1. Sign up at https://gopluslabs.io/
  2. Get API key
  3. Add to .env: `GOPLUS_API_KEY=your_key`
- **Cost**: FREE (limited) or paid tiers

**Honeypot.is**
- **What**: Detect honeypot scam tokens
- **Website**: https://honeypot.is/
- **Cost**: FREE (rate limited)

**RECOMMENDATION**: 
✅ **GoPlus Labs** (essential for DEX safety)

**Priority**: 🟡 MEDIUM (critical if trading new tokens on DEX)

---

## 🌟 OPTIONAL APIs (Advanced Features)

### 6. **Twitter API** (Social Sentiment)
- **What**: Track crypto mentions, sentiment
- **Free tier**: NONE (recently removed)
- **Paid**: $100/month minimum
- **Website**: https://developer.twitter.com/
- **Priority**: 🟢 LOW (expensive, not critical)

### 7. **IBM Quantum** (Quantum Computing)
- **What**: Real quantum hardware for predictions
- **Free tier**: Yes! Limited quantum time
- **How to get**:
  1. Go to: https://quantum.ibm.com/
  2. Sign up (free)
  3. Get API token
  4. Add to .env: `QISKIT_IBM_TOKEN=your_token`
- **Cost**: FREE ✅
- **Priority**: 🟢 LOW (cool feature, works without it)

### 8. **On-Chain Data APIs**

**The Graph (Uniswap, DeFi data)**
- **Website**: https://thegraph.com/
- **Cost**: FREE (limited)
- **What**: Query DeFi protocol data

**Dune Analytics API**
- **Website**: https://dune.com/
- **Cost**: Paid plans
- **What**: On-chain analytics

**Priority**: 🟢 LOW (advanced use cases)

---

## 📊 HOW TO GET OHLCV DATA (CRITICAL!)

### Method 1: From Bybit (RECOMMENDED - Already Working!)

**Your bot already has this!** Just use ccxt:

```python
import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Initialize exchange
exchange = ccxt.bybit()

# Get last 1000 1-hour candles
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Save to CSV
df.to_csv('BTC_USDT_1h.csv', index=False)

print(f"Downloaded {len(df)} candles")
print(df.head())
```

**Timeframes available:**
- 1m, 3m, 5m, 15m, 30m
- 1h, 2h, 4h, 6h, 12h
- 1d, 1w, 1M

**Symbols available:**
- Any trading pair on Bybit
- BTC/USDT, ETH/USDT, etc.

### Method 2: Bulk Historical Download

**Create this script:**

```python
# download_training_data.py
import ccxt
import pandas as pd
import time
from datetime import datetime

exchange = ccxt.bybit()

symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
timeframes = ['1h', '4h', '1d']

for symbol in symbols:
    for timeframe in timeframes:
        print(f"Downloading {symbol} {timeframe}...")
        
        # Get max history (1000 candles at a time)
        all_data = []
        since = exchange.parse8601('2023-01-01T00:00:00Z')
        
        while since < exchange.milliseconds():
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, 1000)
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                print(f"  Downloaded {len(all_data)} candles")
                time.sleep(1)  # Rate limit
                
            except Exception as e:
                print(f"  Error: {e}")
                break
        
        # Save to CSV
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        filename = f"data/{symbol.replace('/', '_')}_{timeframe}.csv"
        df.to_csv(filename, index=False)
        
        print(f"  Saved {len(df)} candles to {filename}")

print("✅ All data downloaded!")
```

**Run it:**
```bash
mkdir data
python3 download_training_data.py
```

**You'll get:**
- Free historical data
- Multiple timeframes
- Multiple symbols
- Ready for ML training

### Method 3: Live Data Collection

**Your bot is ALREADY collecting this!**

The orchestrator saves data as it runs. You can:
1. Run bot for a week
2. Collect live data
3. Use for training

---

## 🎯 PRIORITY ORDER (What to Get First)

### **NOW (Before Running Bot):**
1. ✅ **Bybit API** - You have testnet, get LIVE when ready
2. ✅ **OHLCV Data** - Use ccxt (already working!)

### **THIS WEEK (For Better Performance):**
3. 🔥 **NewsAPI** - For news sentiment
   - Sign up: https://newsapi.org/
   - Free tier sufficient for testing
   
4. 🔥 **Etherscan API** (if doing DEX)
   - Sign up: https://etherscan.io/
   - FREE, takes 2 minutes

### **THIS MONTH (For Full Features):**
5. 🟡 **GoPlus Labs** - For token safety
   - Sign up: https://gopluslabs.io/
   
6. 🟡 **BSCScan API** (if doing BSC DEX)
   - Sign up: https://bscscan.com/

### **OPTIONAL (Nice to Have):**
7. 🟢 **IBM Quantum** - For quantum features
   - Sign up: https://quantum.ibm.com/
   
8. 🟢 **CryptoCompare** - For more historical data
   - Sign up: https://www.cryptocompare.com/

### **SKIP (Too Expensive/Not Worth It):**
- ❌ Twitter API ($100/month - too expensive)
- ❌ Premium news services (free ones sufficient)
- ❌ Paid on-chain analytics (free tools work)

---

## 💰 COST SUMMARY

| API | Cost | Priority | Status |
|-----|------|----------|--------|
| Bybit Testnet | FREE | ✅ Critical | ✅ You have |
| Bybit Live | FREE | 🔥 High | Get after testing |
| OHLCV (via ccxt) | FREE | ✅ Critical | ✅ Already working |
| Telegram | FREE | ✅ Critical | ✅ You have |
| NewsAPI | FREE* | 🟡 Medium | Get this week |
| Etherscan | FREE | 🟡 Medium | Get this week |
| BSCScan | FREE | 🟡 Medium | Get this week |
| GoPlus Labs | FREE* | 🟡 Medium | Get this month |
| IBM Quantum | FREE* | 🟢 Low | Optional |
| Twitter | $100/mo | 🟢 Low | Skip |

*Limited free tier

**Total Cost for Full Bot**: $0 - $10/month (all free tiers!)

---

## 📝 RECOMMENDED SETUP

### Minimum (Works Now):
```bash
# You already have these:
TELEGRAM_BOT_TOKEN=...
BYBIT_API_KEY=...
BYBIT_SECRET=...
```

### Recommended (This Week):
```bash
# Add these to .env:
NEWSAPI_KEY=...          # Free from newsapi.org
ETHERSCAN_API_KEY=...    # Free from etherscan.io
BSCSCAN_API_KEY=...      # Free from bscscan.com
```

### Full Featured (This Month):
```bash
# Add these for complete functionality:
GOPLUS_API_KEY=...       # Free from gopluslabs.io
QISKIT_IBM_TOKEN=...     # Free from quantum.ibm.com
```

---

## 🚀 ACTION PLAN

### Today:
1. ✅ Run bot with what you have
2. ✅ Bot will use ccxt for OHLCV (already working!)
3. ✅ Test Telegram notifications
4. ✅ Verify Bybit testnet trades

### This Week:
1. Sign up for NewsAPI (5 minutes)
2. Sign up for Etherscan (5 minutes)
3. Add keys to .env
4. Download historical OHLCV data (run script above)

### This Month:
1. Get GoPlus API for safety checks
2. Get live Bybit API (when ready)
3. Train ML models on collected data
4. Go live with small amounts

---

## 🎓 TRAINING YOUR MODELS

### Step 1: Collect Data (Now)
```bash
# Use the script above to download historical data
python3 download_training_data.py
```

### Step 2: Your Bot Trains Automatically
Your bot already has:
- ✅ Divine Intelligence (trains on trades)
- ✅ Evolution Engine (83 models learning)
- ✅ Online Learner (continuous learning)

**They train as the bot runs!**

### Step 3: Pre-train with Historical Data (Optional)
```python
# The bot's AI systems will use any OHLCV data you provide
# Just run the download script and the models will train on it
```

---

## ✅ WHAT YOU HAVE RIGHT NOW

**Good news!** You already have enough to:

✅ Trade on testnet  
✅ Get OHLCV data (via ccxt)  
✅ Send Telegram notifications  
✅ Train ML models (on live data)  
✅ Generate signals  
✅ Execute trades  

**What you're missing:**
- News sentiment (get NewsAPI this week)
- DEX safety checks (get GoPlus when doing DEX)
- Live Bybit (get when ready to trade real money)

---

## 🎯 BOTTOM LINE

**You can run the bot NOW with what you have!**

**Critical APIs you need:**
1. ✅ Bybit - You have testnet
2. ✅ OHLCV - Already working via ccxt
3. ✅ Telegram - You have it

**Recommended to add this week:**
1. NewsAPI (free, 5 min signup)
2. Etherscan (free, 5 min signup)

**Get later:**
1. GoPlus (when doing DEX)
2. Live Bybit (when ready for real money)

**Total cost**: $0 (everything has free tier!)

---

**START TRADING NOW, ADD APIs GRADUALLY!** 🚀
