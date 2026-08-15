#!/usr/bin/env bash
set -Eeuo pipefail

readonly APP_DIR="/opt/leantrader/app"

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run this script as root on the LeanTrader VPS." >&2
  exit 1
fi
if [[ ! -f "${APP_DIR}/docker-compose.yml" ]]; then
  echo "ERROR: ${APP_DIR} is not a deployed LeanTrader checkout." >&2
  exit 1
fi

cd "${APP_DIR}"
install -d -m 0750 -o root -g 10001 secrets

echo "Create the bot with @BotFather, then enter its token here."
echo "The bot remains outbound-only and can never place live orders."
read -rsp "Telegram bot token: " telegram_token
echo
read -rp "Admin private chat ID: " admin_chat
read -rp "Free channel/chat ID (optional): " free_chat
read -rp "Paid channel/chat ID (optional): " paid_chat

if [[ "${#telegram_token}" -lt 20 ]]; then
  unset telegram_token
  echo "ERROR: the Telegram token is empty or invalid." >&2
  exit 1
fi
if [[ -z "${admin_chat}" ]]; then
  unset telegram_token
  echo "ERROR: an admin chat ID is required." >&2
  exit 1
fi

umask 027
printf '%s' "${telegram_token}" > secrets/telegram_bot_token
unset telegram_token
chown root:10001 secrets/telegram_bot_token
chmod 0440 secrets/telegram_bot_token

cp -p .env ".env.before-telegram.$(date -u +%Y%m%dT%H%M%SZ)"

set_env() {
  local name="$1"
  local value="$2"
  if grep -qE "^${name}=" .env; then
    sed -i "s|^${name}=.*|${name}=${value}|" .env
  else
    printf '%s=%s\n' "${name}" "${value}" >> .env
  fi
}

set_env TELEGRAM_BOT_TOKEN_FILE /run/secrets/telegram_bot_token
set_env TELEGRAM_ADMIN_CHAT_ID "${admin_chat}"
set_env TELEGRAM_FREE_CHAT_ID "${free_chat}"
set_env TELEGRAM_PAID_CHAT_ID "${paid_chat}"
set_env TELEGRAM_FREE_MIN_CONFIDENCE 0.85
set_env TELEGRAM_PAID_MIN_CONFIDENCE 0.70
set_env TELEGRAM_MOON_MIN_SCORE 1.0
set_env TELEGRAM_SIGNAL_COOLDOWN_SECONDS 900
set_env TELEGRAM_MONITOR_INTERVAL_CYCLES 60
set_env TELEGRAM_TESTNET_TRADE_URL https://testnet.bybit.com/
chmod 0600 .env

python3 - "${admin_chat}" "${free_chat}" "${paid_chat}" <<'PY'
import json
import pathlib
import sys
import urllib.parse
import urllib.request

token = pathlib.Path("secrets/telegram_bot_token").read_text(encoding="utf-8").strip()
admin_chat = sys.argv[1]
free_chat = sys.argv[2]
paid_chat = sys.argv[3]

def call(method, payload=None):
    data = urllib.parse.urlencode(payload or {}).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=data,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("ok") is not True:
        raise RuntimeError(f"Telegram {method} attestation failed")
    return result

identity = call("getMe")["result"]
destinations = {
    "admin": admin_chat,
    "free": free_chat,
    "paid": paid_chat,
}
for audience, chat_id in destinations.items():
    if not chat_id:
        continue
    call(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": (
                f"LeanTrader {audience} alert-channel attestation successful. "
                "Paper/Testnet intelligence only; live authority is disabled."
            ),
        },
    )
print(f"Telegram bot verified: @{identity.get('username', 'unknown')}")
print("Every configured Telegram destination accepted its attestation message.")
PY

docker compose config --quiet
docker compose up -d --build --force-recreate

for _ in $(seq 1 36); do
  if [[ -r runtime/vps_heartbeat.json ]] && jq -e '
    .engines.operations_safety.telegram.configured == true and
    .engines.operations_safety.telegram.outbound_only == true and
    .engines.operations_safety.telegram.inbound_commands == false and
    .engines.operations_safety.telegram.execution_authority == false and
    .engines.operations_safety.telegram.audiences.admin > 0
  ' runtime/vps_heartbeat.json >/dev/null; then
    echo "LEANTRADER_TELEGRAM_READY"
    echo "Admin monitoring: active"
    echo "Free channel configured: $([[ -n "${free_chat}" ]] && echo yes || echo no)"
    echo "Paid channel configured: $([[ -n "${paid_chat}" ]] && echo yes || echo no)"
    echo "One-tap trading link: Bybit Testnet only"
    echo "Direct Telegram execution: disabled"
    echo "Live authority: disabled"
    exit 0
  fi
  sleep 5
done

echo "ERROR: Telegram heartbeat verification failed." >&2
jq '{errors, telegram: .engines.operations_safety.telegram}' runtime/vps_heartbeat.json >&2 || true
exit 1
