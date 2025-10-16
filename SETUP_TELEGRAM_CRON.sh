#!/bin/bash
# Setup Telegram Status Updates (Cron Job)
# Sends status to admin chat every 30 minutes

echo "================================================================================"
echo "📱 SETTING UP TELEGRAM STATUS UPDATES"
echo "================================================================================"
echo ""

cd /root/trading_bot

# Create the notifier script
cat > telegram_status.py << 'PYEOF'
#!/usr/bin/env python3
import requests
import subprocess
from datetime import datetime

TOKEN = "8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg"
ADMIN_CHAT = "5329503447"

def send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {'chat_id': ADMIN_CHAT, 'text': text, 'parse_mode': 'HTML'}
    requests.post(url, data=data, timeout=10)

# Get status
status = subprocess.run(['systemctl', 'is-active', 'trading-bot'], 
                       capture_output=True, text=True).stdout.strip()

# Get recent activity
logs = subprocess.run(['tail', '-100', '/root/trading_bot/bot.log'],
                     capture_output=True, text=True).stdout

trades = logs.lower().count('trade')
signals = logs.lower().count('signal')

# Send update
message = f"""<b>🤖 BOT UPDATE</b>

⏰ {datetime.now().strftime('%H:%M:%S')}

<b>Status:</b> {status}
<b>Activity:</b>
• Trades: {trades}
• Signals: {signals}

✅ Bot running!"""

send(message)
print("✅ Status sent!")
PYEOF

chmod +x telegram_status.py

echo "✅ Notifier script created"
echo ""

# Add to cron (every 30 minutes)
(crontab -l 2>/dev/null | grep -v telegram_status; echo "*/30 * * * * cd /root/trading_bot && python3 telegram_status.py >> /root/trading_bot/telegram.log 2>&1") | crontab -

echo "✅ Cron job added (runs every 30 minutes)"
echo ""

# Send test message now
python3 telegram_status.py

echo ""
echo "================================================================================"
echo "✅ TELEGRAM NOTIFICATIONS CONFIGURED!"
echo "================================================================================"
echo ""
echo "You will now get status updates every 30 minutes in admin chat!"
echo ""
echo "To check:"
echo "  - Look at your Telegram admin chat"
echo "  - Should see status message"
echo ""
echo "To stop updates:"
echo "  crontab -e"
echo "  (Delete the telegram_status line)"
echo ""
echo "================================================================================"
