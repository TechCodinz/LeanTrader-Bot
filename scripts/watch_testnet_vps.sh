#!/usr/bin/env bash
set -Eeuo pipefail

readonly APP_DIR="/opt/leantrader/app"
readonly HEARTBEAT="${APP_DIR}/runtime/vps_heartbeat.json"
readonly TESTNET_STATE="${APP_DIR}/runtime/vps_testnet_execution.json"

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq is required." >&2
  exit 1
fi

while true; do
  [[ -t 1 ]] && clear
  echo "LeanTrader paper + Bybit Testnet monitor — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  if [[ ! -r "${HEARTBEAT}" ]]; then
    echo "Heartbeat is not available: ${HEARTBEAT}"
  else
    jq '{
      heartbeat_time: (.timestamp | todate),
      healthy,
      runtime,
      paper: {
        equity,
        cash,
        realized_pnl,
        open_positions,
        halt_reason
      },
      adaptive_learning: {
        promotions: .engines.adaptive_intelligence.promotions,
        pending_samples: .engines.adaptive_intelligence.pending_samples,
        weights: .engines.adaptive_intelligence.weights
      },
      exchange_intelligence: .engines.exchange_intelligence,
      exchange_protection: .engines.exchange_protection,
      market_time_integrity: .engines.market_temporal_guard,
      cross_venue_arbitrage: .engines.cross_venue_arbitrage,
      decision_router: {
        routes: .engines.decision_router.routes,
        approved: .engines.decision_router.approved,
        blocked: .engines.decision_router.blocked,
        market_evidence: .engines.decision_router.market_evidence,
        live_authority: .engines.decision_router.live_authority
      },
      ultra_engine_activity: .engines.advanced_shadow_suite.activity,
      timeframe_matrix: .advanced_shadow.symbols | with_entries(.value |= .timeframe_matrix),
      research_engine_activity: .engines.research_governor.activity,
      research_drift: .research_governor.distribution_drift,
      model_research: .engines.model_research,
      latest_model_research_observation: .model_research_observation,
      public_market_context: .engines.public_market_context,
      news_awareness: .engines.advanced_shadow_suite.capabilities.news_awareness,
      moon_scout: .engines.advanced_shadow_suite.capabilities.moon_scout_dynamic_scanner,
      arbitrage_engine: .engines.advanced_shadow_suite.capabilities.arbitrage,
      telegram: .engines.operations_safety.telegram,
      latest_telegram_notifications: .telegram_notifications,
      ungated_strategy_observatory: .engines.strategy_observatory,
      market_universe: {
        mode: .engines.market_universe.mode,
        quote: .engines.market_universe.quote,
        eligible_symbols: .engines.market_universe.eligible_symbols,
        scan_batch_size: .engines.market_universe.scan_batch_size,
        last_scan: .engines.market_universe.last_scan,
        full_sweeps: .engines.market_universe.full_sweeps,
        rejection_counts: .engines.market_universe.rejection_counts
      },
      testnet: {
        authenticated: .engines.bybit_testnet_execution.authenticated,
        api_attestation: .engines.bybit_testnet_execution.api_attestation,
        protection_contract: .engines.bybit_testnet_execution.protection_contract,
        risk_limits: .engines.bybit_testnet_execution.risk_limits,
        kill_switch_active: .engines.bybit_testnet_execution.kill_switch_active,
        exchange_capabilities: .engines.bybit_testnet_execution.exchange_capabilities,
        sandbox_verified: .engines.bybit_testnet_execution.sandbox_endpoint_verified,
        live_authority: .engines.bybit_testnet_execution.live_authority,
        orders: .engines.bybit_testnet_execution.orders,
        open_orders: .engines.bybit_testnet_execution.open_orders,
        positions: .engines.bybit_testnet_execution.positions,
        position_cost_usd: .engines.bybit_testnet_execution.position_cost_usd,
        account_balance: .engines.bybit_testnet_execution.account_balance,
        performance: .engines.bybit_testnet_execution.performance,
        reconciliation_errors: .engines.bybit_testnet_execution.last_reconciliation_errors
      },
      required_engine_failures: [
        .engines | to_entries[] |
        select(.value.required == true and .value.healthy != true) |
        {engine: .key, state: .value.state, error: .value.last_error}
      ],
      cycle_errors: .errors
    }' "${HEARTBEAT}"
  fi

  if [[ -r "${TESTNET_STATE}" ]]; then
    echo
    echo "Latest testnet order records:"
    jq '[
      .orders | to_entries[] | .value |
      {
        time: .paper_event_timestamp,
        symbol,
        side,
        status,
        filled,
        average,
        fee,
        submitted_usd,
        skip_reason
      }
    ] | sort_by(.time) | reverse | .[:10]' "${TESTNET_STATE}"
  fi

  [[ "${WATCH_ONCE:-false}" == "true" ]] && exit 0
  sleep "${WATCH_SECONDS:-10}"
done
