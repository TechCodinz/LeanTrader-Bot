# Trading Bot Management Guide

## Quick Reference Commands

### Check Bot Status
```bash
# View live logs
sudo journalctl -u trading-bot-live -f

# Check if bot is running
sudo systemctl status trading-bot-live

# Restart bot
sudo systemctl restart trading-bot-live

# Stop bot
sudo systemctl stop trading-bot-live
```

### View Recent Logs (Last 100 Lines)
```bash
sudo journalctl -u trading-bot-live -n 100
```

### Search Logs for Errors
```bash
sudo journalctl -u trading-bot-live | grep -i error
sudo journalctl -u trading-bot-live | grep -i "❌"
```

### View Signal Activity
```bash
sudo journalctl -u trading-bot-live | grep -i "signal"
sudo journalctl -u trading-bot-live | grep -i "VIP"
```

## What to Monitor

### 1. **Telegram Channels**
- **Admin Channel** (5329503447): All bot activity, trades, errors
- **VIP Channel** (-1002983007302): Premium signals (80%+ confidence)
- **Free Channel** (-1002930953007): Basic signals (65%+ confidence)

### 2. **Key Log Messages**

#### ✅ Good Signs
- `✅✅✅ VIP channel SUCCESS` - Signal sent successfully
- `✅ Trade executed` - Trade placed successfully  
- `💰 TRADE EXECUTED` - Execution confirmed
- `🔄 COMPLETE CYCLE X` - System cycles running

#### ⚠️ Warning Signs
- `⚠️ Trade blocked: Already in position` - Risk management working (GOOD)
- `⚠️ Low confidence` - Signal filtered (GOOD)

#### 🚨 Error Signs (Need Attention)
- `❌ Failed to` - Something failed, check details
- `🚨 EMERGENCY STOP` - Bot halted for safety
- `Connection refused` - API/Exchange issues

### 3. **Performance Metrics**
From logs, look for:
```
Execution Stats:
   • Total Trades: X
   • Win Rate: XX.X%
   • Total Profit: $XX.XX
   • Open Positions: X
   • Daily P&L: $XX.XX
```

### 4. **Active Systems Check**
Should see:
```
Active Systems:
   • Core: 26 systems ✅
   • Advanced: X systems ✅
   • Orchestrators: XX running ✅
```

## Common Issues & Solutions

### Issue: "No signals being sent"
**Check:**
1. Is bot running? `sudo systemctl status trading-bot-live`
2. Check Telegram token: `grep TELEGRAM_BOT_TOKEN /workspace/.env`
3. View logs for Telegram errors: `sudo journalctl -u trading-bot-live | grep -i telegram`

**Solution:**
- Bot needs to be admin in Telegram channels
- Check bot token is correct
- Restart bot: `sudo systemctl restart trading-bot-live`

### Issue: "Exchange connection errors"
**Check:**
1. View Gate.io errors: `sudo journalctl -u trading-bot-live | grep -i gateio`
2. Check API keys: `grep GATEIO /workspace/.env`

**Solution:**
- Verify API keys are correct
- Check if in correct mode (testnet vs live)
- Ensure exchange API has trading permissions

### Issue: "No trades executing"
**Check:**
1. View execution logs: `sudo journalctl -u trading-bot-live | grep -i execution`
2. Check risk management: `sudo journalctl -u trading-bot-live | grep -i "blocked"`

**Possible Reasons:**
- Risk manager blocking (already in position) - **WORKING AS INTENDED**
- Confidence too low (< 65%) - **WORKING AS INTENDED**
- Insufficient balance
- API key lacks trading permission

### Issue: "Emergency stop triggered"
**What Happened:**
- Bot detected dangerous conditions:
  - Max loss exceeded (10% of capital)
  - Too many trades per minute (>10)

**What Bot Did:**
- Closed all positions
- Stopped trading
- Sent Telegram alert

**What to Do:**
1. Review what went wrong
2. Check logs before stop: `sudo journalctl -u trading-bot-live -n 500`
3. Fix issue
4. Restart when ready: `sudo systemctl restart trading-bot-live`

## Understanding the Bot's Behavior

### Signal Generation
- Generates 2-50 signals per minute
- Filters by confidence (>65% for free, >80% for VIP)
- Applies session-aware adjustments (Asia/Europe/US sessions)

### Trade Execution
- Only executes signals >70% confidence (default)
- Applies position sizing based on confidence
- Risk management blocks duplicates
- Emergency stop protection

### Learning & Evolution
- Learns from every trade (win or loss)
- Updates ML models every cycle
- Adapts to market conditions
- Evolution cycles improve strategies

## Monitoring Schedule

### Every Hour
- Quick log check: `sudo journalctl -u trading-bot-live -n 50`
- Check Telegram for signals
- Verify bot is running

### Every 4 Hours  
- Review execution stats
- Check open positions
- Monitor profit/loss

### Daily
- Full performance review
- Update strategy if needed
- Check for any errors
- Monitor VIP subscription revenue

### Weekly
- Analyze win rate trends
- Optimize parameters if needed
- Scale up if profitable
- Back up configuration

## Performance Expectations

### Testnet Mode (Current)
- **Purpose**: Validate strategies without real money
- **Expected**: 50-200 signals/day
- **Trades**: 5-20 per day
- **Win Rate Target**: >55%
- **Daily Profit Target**: $5-20 (virtual)

### Live Mode (When Ready)
- **Capital**: $40 (Gate.io live account)
- **Expected**: Same signal volume
- **Trades**: Conservative (2-10/day to start)
- **Win Rate Target**: >60%
- **Daily Profit Target**: $2-10 (0.25%-1% daily)

## Optimization Tips

### Increase Profitability
1. **Add More Exchange APIs**: More arbitrage opportunities
2. **Increase Confidence Threshold**: Trade only highest-quality signals
3. **Enable Compound Mode**: Reinvest profits for exponential growth
4. **Monitor Best Performing Strategies**: Focus on what works

### Reduce Risk
1. **Lower Position Size**: Reduce max_position_pct
2. **Tighter Stop Losses**: Reduce max loss per trade
3. **Fewer Concurrent Positions**: Lower max open positions
4. **Emergency Stop Settings**: Lower max loss threshold

### Scale Up
1. **Start Small**: Prove profitability with $40
2. **Gradual Increase**: Add capital weekly as confidence grows
3. **Diversify**: Add more exchanges and pairs
4. **Automate More**: Let bot handle more decisions

## Files to Never Delete

### Critical System Files
- `/workspace/COMPLETE_ULTIMATE_ORCHESTRATOR.py` - Main orchestrator
- `/workspace/TELEGRAM_ORCHESTRATOR.py` - Telegram integration
- `/workspace/EXECUTION_ORCHESTRATOR.py` - Trade execution
- `/workspace/.env` - Configuration (API keys)

### Important Data Files  
- `/workspace/users_db.json` - VIP subscribers
- `/workspace/trades.db` - Trade history
- `/workspace/models/` - Trained ML models

### Backup Regularly
```bash
# Backup configuration
cp /workspace/.env /workspace/.env.backup

# Backup user database
cp /workspace/users_db.json /workspace/users_db.json.backup

# Backup trade history
cp /workspace/trades.db /workspace/trades.db.backup
```

## Emergency Procedures

### If Bot Goes Wild (Too Many Trades)
1. **STOP IMMEDIATELY**: `sudo systemctl stop trading-bot-live`
2. **Check logs**: `sudo journalctl -u trading-bot-live -n 500`
3. **Close positions manually on exchange**
4. **Review what triggered it**
5. **Fix before restarting**

### If Losing Money Fast
1. **Emergency stop should trigger automatically**
2. **If not, stop manually**: `sudo systemctl stop trading-bot-live`
3. **Review trades**: Check Telegram and logs
4. **Identify problem strategy**
5. **Disable problematic systems before restart**

### If Telegram Not Working
1. **Check bot is admin in channels**
2. **Verify token**: `grep TELEGRAM_BOT_TOKEN /workspace/.env`
3. **Test bot manually**: `python3 -c "import telegram; bot = telegram.Bot('YOUR_TOKEN'); print(bot.get_me())"`
4. **Restart bot**: `sudo systemctl restart trading-bot-live`

## Getting Help

### Log Analysis
If you need help, provide:
1. Last 500 lines of logs: `sudo journalctl -u trading-bot-live -n 500 > bot_logs.txt`
2. Environment config (remove API keys): `cat /workspace/.env | grep -v API_KEY | grep -v SECRET`
3. Description of the issue
4. Screenshots from Telegram if relevant

### Performance Review
Share:
1. Execution stats from logs
2. Telegram channel activity
3. Time period being analyzed
4. Any unusual behavior observed

---

**Remember**: 
- Bot is designed to be safe (emergency stops, risk management)
- Testnet first, live second
- Start small, scale gradually
- Monitor daily, optimize weekly
- Trust the system, but verify constantly

**Bot Version**: Complete Ultimate Orchestrator v3.0
**Last Updated**: October 16, 2025
