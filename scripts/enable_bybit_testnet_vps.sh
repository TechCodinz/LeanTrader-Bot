#!/usr/bin/env bash
set -Eeuo pipefail

readonly APP_DIR="/opt/leantrader/app"
readonly CONFIRMATION="I_UNDERSTAND_TESTNET_ONLY"

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run this script as root on the LeanTrader VPS." >&2
  exit 1
fi
if [[ ! -f "${APP_DIR}/docker-compose.yml" ]]; then
  echo "ERROR: ${APP_DIR} is not a deployed LeanTrader checkout." >&2
  exit 1
fi
if ! grep -q 'verified-multi-engine-v6-attested-routing' "${APP_DIR}/src/leantrader/production/runner.py"; then
  echo "ERROR: upgrade the VPS to the verified testnet release before enabling testnet." >&2
  exit 1
fi

cd "${APP_DIR}"
install -d -m 0750 -o root -g 10001 secrets
touch runtime/TESTNET_HALT
chown 10001:10001 runtime/TESTNET_HALT

echo "This accepts BYBIT TESTNET credentials only. Never enter production credentials."
read -rsp "Bybit Testnet API key: " testnet_key
echo
read -rsp "Bybit Testnet API secret: " testnet_secret
echo
if [[ "${#testnet_key}" -lt 8 || "${#testnet_secret}" -lt 8 ]]; then
  unset testnet_key testnet_secret
  echo "ERROR: the testnet key or secret is empty/invalid." >&2
  exit 1
fi

umask 027
printf '%s' "${testnet_key}" > secrets/bybit_testnet_api_key
printf '%s' "${testnet_secret}" > secrets/bybit_testnet_api_secret
unset testnet_key testnet_secret
chown root:10001 secrets/bybit_testnet_api_key secrets/bybit_testnet_api_secret
chmod 0440 secrets/bybit_testnet_api_key secrets/bybit_testnet_api_secret

cp -p .env ".env.before-testnet.$(date -u +%Y%m%dT%H%M%SZ)"

set_env() {
  local name="$1"
  local value="$2"
  if grep -qE "^${name}=" .env; then
    sed -i "s|^${name}=.*|${name}=${value}|" .env
  else
    printf '%s=%s\n' "${name}" "${value}" >> .env
  fi
}

set_env BYBIT_TESTNET_ENABLED true
set_env BYBIT_TESTNET_CONFIRM "${CONFIRMATION}"
set_env PAPER_SYMBOLS AUTO
set_env MARKET_QUOTE USDT
set_env MARKET_UNIVERSE_STATE_PATH runtime/vps_market_universe.json
set_env MARKET_SCAN_BATCH_SIZE 18
set_env MARKET_REFRESH_SECONDS 3600
set_env MARKET_MIN_QUOTE_VOLUME_USD 250000
set_env MARKET_MAX_SPREAD_BPS 75
set_env MARKET_EVIDENCE_MIN_SAMPLES 8
set_env MARKET_EVIDENCE_WINDOW 50
set_env ROUTER_MIN_ADVANCED_CONFIDENCE 0.20
set_env ROUTER_MIN_COMBINED_SCORE 0.20
set_env ROUTER_NEGATIVE_CONSENSUS_VETO -0.25
set_env BYBIT_TESTNET_MAX_ORDER_USD 10
set_env BYBIT_TESTNET_MAX_POSITION_USD 20
set_env BYBIT_TESTNET_MAX_DAILY_SUBMITTED_USD 50
set_env BYBIT_TESTNET_MAX_ORDERS_PER_DAY 20
chmod 0600 .env

docker compose config --quiet
docker compose up -d --build --force-recreate

container_id="$(docker compose ps -q leantrader)"
health="starting"
for _ in $(seq 1 36); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}' "${container_id}")"
  [[ "${health}" == "healthy" ]] && break
  [[ "${health}" == "unhealthy" || "${health}" == "missing" ]] && break
  sleep 5
done

if [[ "${health}" != "healthy" ]]; then
  echo "ERROR: the testnet runtime is ${health}; new entries remain halted." >&2
  docker compose logs --tail=120 leantrader >&2
  exit 1
fi

heartbeat="runtime/vps_heartbeat.json"
if ! jq -e '
  .healthy == true and
  .engines.bybit_testnet_execution.required == true and
  .engines.bybit_testnet_execution.healthy == true and
  .engines.bybit_testnet_execution.environment == "testnet" and
  .engines.bybit_testnet_execution.authenticated == true and
  .engines.bybit_testnet_execution.sandbox_endpoint_verified == true and
  .engines.bybit_testnet_execution.live_authority == false and
  .engines.bybit_testnet_execution.api_attestation.verified == true and
  .engines.bybit_testnet_execution.api_attestation.environment == "testnet" and
  .engines.bybit_testnet_execution.api_attestation.spot_trade == true and
  .engines.bybit_testnet_execution.api_attestation.withdrawal_permission == false and
  .engines.market_universe.healthy == true and
  .engines.market_universe.mode == "dynamic" and
  .engines.market_universe.eligible_symbols > 0 and
  .engines.market_universe.all_eligible_markets_rotate == true and
  .engines.advanced_shadow_suite.healthy == true and
  .engines.advanced_shadow_suite.activity.smart_scalping.successes > 0 and
  .engines.advanced_shadow_suite.activity.technical_structure.successes > 0 and
  .engines.advanced_shadow_suite.activity.spectral_harmonics.successes > 0 and
  .engines.advanced_shadow_suite.activity.news_awareness.successes > 0 and
  .engines.advanced_shadow_suite.activity.pattern_memory.successes > 0 and
  .engines.advanced_shadow_suite.activity.swarm_hivemind.successes > 0 and
  .engines.research_governor.activity.distribution_drift.calls > 0 and
  .engines.research_governor.activity.capital_preservation.calls > 0 and
  .engines.decision_router.healthy == true and
  .engines.decision_router.routes > 0 and
  .engines.decision_router.live_authority == false and
  ([.engines | to_entries[] | select(.value.required == true and .value.healthy != true)] | length) == 0
' "${heartbeat}" >/dev/null; then
  echo "ERROR: heartbeat verification failed; new entries remain halted." >&2
  jq '{healthy, errors, engines, testnet_execution}' "${heartbeat}" >&2
  exit 1
fi

echo "LEANTRADER_TESTNET_READY"
echo "Authenticated Bybit Testnet: yes"
echo "Dynamic eligible-market universe: yes"
echo "API environment and permissions attested: yes"
echo "Adaptive + ultra decision router: yes"
echo "Live authority: no"
echo "New testnet entries: HALTED for manual acceptance"
echo "Review runtime/vps_heartbeat.json and the Bybit Testnet account."
echo "Only then remove ${APP_DIR}/runtime/TESTNET_HALT to begin mirroring new paper entries."
