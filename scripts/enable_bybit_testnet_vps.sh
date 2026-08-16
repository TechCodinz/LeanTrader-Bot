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
if ! grep -q 'verified-multi-engine-v12.8.2-deep-flow-self-model-hygiene' "${APP_DIR}/src/leantrader/production/runner.py"; then
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
set_env DATA_EXCHANGE bybit
set_env PAPER_SYMBOLS AUTO
set_env MARKET_QUOTE USDT
set_env MARKET_UNIVERSE_STATE_PATH runtime/vps_market_universe.json
set_env EXCHANGE_INTELLIGENCE_STATE_PATH runtime/vps_exchange_intelligence.json
set_env MARKET_SCAN_BATCH_SIZE 18
set_env MARKET_REFRESH_SECONDS 3600
set_env MARKET_MIN_QUOTE_VOLUME_USD 250000
set_env MARKET_MAX_SPREAD_BPS 75
set_env PAPER_TIMEFRAME 15m
set_env CONFIRM_TIMEFRAMES 1m,3m,5m,15m,30m,1h,2h,4h,6h,12h,1d,1w,1M
set_env PUBLIC_CONTEXT_REFRESH_SECONDS 900
set_env PUBLIC_CONTEXT_ENABLED true
set_env NEWS_MAX_AGE_SECONDS 86400
set_env NEWS_MAX_FUTURE_SKEW_SECONDS 300
set_env MAX_CLOCK_OFFSET_MS 5000
set_env CLOCK_SYNC_SECONDS 300
set_env CANDLE_STALE_MULTIPLIER 2.5
set_env ARBITRAGE_MONITOR_ENABLED true
set_env ARBITRAGE_VENUES bybit,okx
set_env ARBITRAGE_REFRESH_SECONDS 60
set_env ARBITRAGE_SLIPPAGE_BPS 3
set_env MARKET_EVIDENCE_MIN_SAMPLES 8
set_env MARKET_EVIDENCE_WINDOW 50
set_env ROUTER_MIN_ADVANCED_CONFIDENCE 0.20
set_env ROUTER_MIN_COMBINED_SCORE 0.20
set_env ROUTER_NEGATIVE_CONSENSUS_VETO -0.25
set_env CNS_STATE_PATH runtime/vps_cns_state.json
set_env BRAIN_STATE_PATH runtime/vps_brain_state.json
set_env MEMORY_RETENTION_STATE_PATH runtime/vps_memory_retention.json
set_env CAPITAL_GROWTH_STATE_PATH runtime/vps_capital_growth.json
set_env MEMORY_MAX_EPISODES 5000
set_env MEMORY_HALF_LIFE_HOURS 720
set_env BRAIN_MIN_STRATEGY_SAMPLES 50
set_env BRAIN_NEGATIVE_EXPECTANCY_FLOOR -0.001
set_env BRAIN_QUARANTINE_MIN_SAMPLES 100
set_env BRAIN_QUARANTINE_EXPECTANCY_FLOOR -0.004
set_env BRAIN_RECOVERY_EXPECTANCY_FLOOR 0.0005
set_env CAPITAL_PRINCIPAL_FLOOR_FRACTION 0.70
set_env CAPITAL_PROFIT_REINVEST_FRACTION 0.50
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
  .engines.bybit_testnet_execution.api_attestation.ip_bound == true and
  .engines.bybit_testnet_execution.api_attestation.spot_trade == true and
  .engines.bybit_testnet_execution.api_attestation.withdrawal_permission == false and
  ([.engines.bybit_testnet_execution.protection_contract | to_entries[] | select(.value != true)] | length) == 0 and
  .engines.exchange_intelligence.required == true and
  .engines.exchange_intelligence.healthy == true and
  .engines.exchange_intelligence.exchange_id == "bybit" and
  .engines.exchange_intelligence.credentials_loaded == false and
  .engines.exchange_intelligence.provider_rules_dynamic == true and
  .engines.exchange_intelligence.execution_authority == false and
  (.engines.exchange_intelligence.resolved_timeframes | length) == 13 and
  .engines.market_temporal_guard.required == true and
  .engines.market_temporal_guard.healthy == true and
  .engines.market_temporal_guard.utc_internal == true and
  .engines.market_temporal_guard.closed_candles_only == true and
  .engines.market_temporal_guard.stale_candle_rejection == true and
  .engines.market_temporal_guard.clock.supported == true and
  .engines.market_temporal_guard.clock.verified == true and
  .engines.market_temporal_guard.clock.safe == true and
  .engines.exchange_protection.healthy == true and
  .engines.exchange_protection.capability_driven == true and
  .engines.exchange_protection.fail_closed == true and
  .engines.exchange_protection.authenticated_executor == "bybit_testnet_spot_only" and
  (.engines.market_temporal_guard.candle_timeframes | length) == 13 and
  ([.engines.market_temporal_guard.candle_timeframes[] | select(.stale == true)] | length) == 0 and
  .engines.cross_venue_arbitrage.healthy == true and
  .engines.cross_venue_arbitrage.enabled == true and
  .engines.cross_venue_arbitrage.collections > 0 and
  (.engines.cross_venue_arbitrage.successful_venues | length) >= 2 and
  .engines.cross_venue_arbitrage.quote_count > 0 and
  .engines.market_universe.healthy == true and
  .engines.market_universe.mode == "dynamic" and
  .engines.market_universe.eligible_symbols > 0 and
  .engines.market_universe.all_eligible_markets_rotate == true and
  .engines.advanced_shadow_suite.healthy == true and
  .engines.advanced_shadow_suite.activity.smart_scalping.successes > 0 and
  .engines.advanced_shadow_suite.activity.multi_timeframe_matrix.successes > 0 and
  .engines.advanced_shadow_suite.activity.fundamental_market_context.successes > 0 and
  .engines.public_market_context.attempts > 0 and
  .engines.public_market_context.markets > 0 and
  .engines.public_market_context.market_data_fresh == true and
  (.engines.public_market_context.successful_sources | index("coingecko_markets")) != null and
  (.engines.public_market_context.news_sources_successful | length) > 0 and
  .engines.public_market_context.news_fresh == true and
  .engines.advanced_shadow_suite.capabilities.news_awareness.items > 0 and
  .engines.advanced_shadow_suite.capabilities.news_awareness.fresh == true and
  .engines.advanced_shadow_suite.activity.technical_structure.successes > 0 and
  .engines.advanced_shadow_suite.activity.spectral_harmonics.successes > 0 and
  .engines.advanced_shadow_suite.activity.news_awareness.successes > 0 and
  .engines.advanced_shadow_suite.activity.pattern_memory.successes > 0 and
  .engines.advanced_shadow_suite.activity.swarm_hivemind.successes > 0 and
  .engines.advanced_shadow_suite.activity.fluid_liquidity.successes > 0 and
  .engines.advanced_shadow_suite.activity.moon_scout_dynamic_scanner.successes > 0 and
  .engines.advanced_shadow_suite.activity.portfolio_risk.successes > 0 and
  .engines.advanced_shadow_suite.activity.arbitrage.successes > 0 and
  .engines.advanced_shadow_suite.activity.business_performance.successes > 0 and
  .engines.research_governor.activity.distribution_drift.calls > 0 and
  .engines.research_governor.activity.capital_preservation.calls > 0 and
  .engines.decision_router.healthy == true and
  .engines.decision_router.routes > 0 and
  .engines.strategy_observatory.calls > 0 and
  .engines.strategy_observatory.router_gates_applied == false and
  .engines.memory_retention.required == true and
  .engines.memory_retention.healthy == true and
  .engines.memory_retention.causal_closed_outcomes_only == true and
  .engines.memory_retention.execution_authority == false and
  .engines.central_nervous_system.required == true and
  .engines.central_nervous_system.healthy == true and
  .engines.central_nervous_system.execution_authority == false and
  .engines.trading_brain.required == true and
  .engines.trading_brain.healthy == true and
  .engines.trading_brain.can_increase_upstream_risk == false and
  .engines.trading_brain.can_enable_live == false and
  .engines.capital_growth.required == true and
  .engines.capital_growth.healthy == true and
  .engines.capital_growth.martingale == false and
  .engines.capital_growth.can_increase_upstream_risk == false and
  .engines.model_research.automatic_live_promotion == false and
  .engines.model_research.execution_authority == false and
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
echo "CNS + Brain + causal persistent memory: yes"
echo "Principal-protecting capital growth governor: yes"
echo "Exchange capability and market-rule intelligence: yes"
echo "UTC/DST, exchange-clock and closed-candle integrity: yes"
echo "Fresh RSS news provenance: yes"
echo "Moon Scout and cross-venue arbitrage observation: yes"
echo "Structured model research boundary: yes"
echo "Live authority: no"
echo "New testnet entries: HALTED for manual acceptance"
echo "Review runtime/vps_heartbeat.json and the Bybit Testnet account."
echo "Only then remove ${APP_DIR}/runtime/TESTNET_HALT to begin mirroring new paper entries."
