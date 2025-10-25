# 🧪 SETUP BYBIT TESTNET TRADING

Your bot will trade on **Bybit Testnet with FAKE MONEY** to prove profitability before risking real funds!

---

## 📋 STEP 1: Get Bybit Testnet API Keys

1. Go to: **https://testnet.bybit.com**
2. Sign up / Login (free!)
3. You'll get **100,000 USDT fake money** automatically! 💰
4. Go to **API Management** → **Create New Key**
5. Name it "Trading Bot"
6. Enable **Trading** permissions
7. Copy your:
   - **API Key** (32 characters)
   - **API Secret** (32 characters)

---

## 📝 STEP 2: Update .env File

Run this command on your VPS:

```bash
cd ~/trading_bot
nano .env
```

Replace these lines:

```bash
BYBIT_TESTNET_API_KEY=YOUR_TESTNET_KEY_HERE
BYBIT_TESTNET_API_SECRET=YOUR_TESTNET_SECRET_HERE
```

With your REAL testnet keys from Step 1!

Then press:
- `CTRL + O` (save)
- `ENTER` (confirm)
- `CTRL + X` (exit)

---

## 🚀 STEP 3: Restart Bot in Testnet Mode

```bash
cd ~/trading_bot
pkill -9 -f RUN_BOT.py
./start_bot.sh
sleep 10
tail -f bot.log | grep -E "TESTNET|Trade executed|profit"
```

You should see:
```
🧪 TESTNET MODE - Using fake money for training!
✅ Bybit Testnet connected!
```

---

## 📊 STEP 4: Monitor Testnet Trades

### Check bot logs:
```bash
tail -f ~/trading_bot/bot.log | grep -E "executed|profit|balance"
```

### Check testnet balance:
Go to: **https://testnet.bybit.com/user/assets/wallet**

You'll see:
- Your testnet USDT balance
- All open positions
- Profit/Loss per trade
- Order history

---

## ✅ WHEN YOU SEE CONSISTENT PROFITS

After the bot proves profitability on testnet (e.g., **+10% over 3 days**), then we can:
1. Switch to `TRADING_MODE=live` in .env
2. Start with small real money ($20-50)
3. Let it grow! 📈

---

## 🆘 TROUBLESHOOTING

**"API key invalid"**
- Make sure you copied keys from testnet.bybit.com (NOT bybit.com)
- Keys should be 32 characters each

**"Insufficient balance"**
- Testnet should give you 100K USDT automatically
- If not, contact Bybit support to reset testnet balance

**No trades happening**
- Check `tail -f bot.log` for errors
- Make sure `TRADING_MODE=testnet` in .env
- Restart bot: `pkill -9 -f RUN_BOT.py && ./start_bot.sh`
