#!/usr/bin/env bash
set -euo pipefail
LOG="/opt/ultra/reports/daemon_console.log"
: "${TELEGRAM_BOT_TOKEN:=}"
: "${TG_ADMIN_CHAT_ID:=}"
[ -n "${TELEGRAM_BOT_TOKEN}" ] && [ -n "${TG_ADMIN_CHAT_ID}" ] || { echo "TG not configured"; exit 0; }
mkdir -p /opt/ultra/reports
touch "$LOG"
tail -Fn0 "$LOG" | while IFS= read -r line; do
  if echo "$line" | grep -Eiq "(FILLED|Filled|execut|order .* (filled|closed)|realized|unrealized|PnL[:=])" \
    text="[exec/PnL] ${line}"
    curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
      -d chat_id="$TG_ADMIN_CHAT_ID" -d text="$text" >/dev/null || true
  fi
done
