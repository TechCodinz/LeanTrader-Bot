#!/bin/bash

cat <<'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              📊 COMPREHENSIVE STATUS CHECK                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "1️⃣ Exchange Status"
echo "═══════════════════════════════════════════════════════════════════════"
journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -E "MEXC|GATEIO|BINANCE|BYBIT" | grep "added to arbitrage"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "2️⃣ Arbitrage Scanner Status"
echo "═══════════════════════════════════════════════════════════════════════"
journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -i "arbitrage.*start\|scanner.*start"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "3️⃣ Signal Generation"
echo "═══════════════════════════════════════════════════════════════════════"
journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -E "signal sent|Scalper generated"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "4️⃣ Arbitrage Opportunities (if any)"
echo "═══════════════════════════════════════════════════════════════════════"
journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -i "arbitrage found\|profit.*%"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "5️⃣ Telegram Channel Messages"
echo "═══════════════════════════════════════════════════════════════════════"
journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -E "FREE channel|VIP channel|msg_id"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "6️⃣ Errors (if any)"
echo "═══════════════════════════════════════════════════════════════════════"
journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -i "error\|failed\|❌" | head -10

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ STATUS SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# Count signals
SIGNALS=$(journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -c "signal sent" || echo 0)
echo "📊 Signals sent to channels: $SIGNALS"

# Check if arbitrage scanner is running
if journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -qi "arbitrage.*start"; then
    echo "✅ Arbitrage scanner: RUNNING"
else
    echo "⚠️  Arbitrage scanner: Not detected in logs"
fi

# Check exchanges
EXCHANGES=$(journalctl -u trading-bot --since "2 minutes ago" --no-pager | grep -c "added to arbitrage" || echo 0)
echo "🔄 Exchanges in arbitrage: $EXCHANGES"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "📱 NEXT: Fix Telegram Channels"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "If signals sent = 0, run:"
echo "  bash FIX_TELEGRAM_CHANNELS_NOW.sh"
echo ""
echo "This will:"
echo "  - Get correct channel IDs"
echo "  - Add bot as admin"
echo "  - Test channels"
echo "  - Start sending signals!"
echo ""
