# 🚨 BYBIT TESTNET FUND TRANSFER GUIDE

## Problem
Your $17,055 USDT is in the **Funding Account** but needs to be in the **Unified Trading Account** to trade.

## Solution (Takes 30 seconds)

### Step 1: Login to Bybit Testnet
1. Go to: **https://testnet.bybit.com**
2. Login with your Bybit testnet credentials

### Step 2: Navigate to Assets
1. Click on **"Assets"** in the top menu
2. Or go directly to: **https://testnet.bybit.com/user/assets/home**

### Step 3: Transfer Funds
1. Look for **"Transfer"** or **"Internal Transfer"** button
2. Select:
   - **From:** Funding Account
   - **To:** Unified Trading Account (or SPOT Account)
   - **Coin:** USDT
   - **Amount:** 17055 (or "All")
3. Click **"Confirm Transfer"**

### Step 4: Verify
After transfer, run this command on VPS:

```bash
cd /root/trading_bot
source venv/bin/activate

# Check if bot can now trade
sudo journalctl -u trading-bot-testnet -f | grep -E 'TRADE EXECUTED|Insufficient'
```

You should see:
- ✅ `TRADE EXECUTED` messages
- ❌ NO MORE "Insufficient balance" errors

---

## Alternative: If No Transfer Button Visible

If you don't see a transfer option, the testnet may auto-allocate. In that case:

1. Go to **Unified Trading** section
2. Look for **"Activate Unified Trading"** or similar button
3. Enable it

---

## After Transfer

The bot will:
1. ✅ Immediately start executing trades
2. ✅ Learn from every trade
3. ✅ Build performance history
4. ✅ Auto-trigger live trading when profitable (60%+ win rate, 10+ trades)

**Expected:** Within 30-60 minutes, you should see 5-10 test trades and performance metrics building up!
