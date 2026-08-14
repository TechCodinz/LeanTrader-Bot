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
