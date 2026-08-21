#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import signal
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

RUNTIME = Path(os.environ.get("LEANTRADER_RUNTIME", "/opt/leantrader/app/runtime"))
INBOX = RUNTIME / "evolution" / "inbox"
STATE = Path(os.environ.get("EVOLUTION_SIDECAR_STATE", "/opt/leantrader/evolution-sidecars/state.json"))
INTERVAL = max(120, int(os.environ.get("EVOLUTION_SIDECAR_INTERVAL", "300")))
BYBIT = os.environ.get("BYBIT_PUBLIC_BASE", "https://api.bybit.com")
BINANCE_PUBLIC = os.environ.get("BINANCE_PUBLIC_BASE", "https://data-api.binance.vision")
BINANCE_FUTURES_PUBLIC = os.environ.get("BINANCE_FUTURES_PUBLIC_BASE", "https://fapi.binance.com")
OKX_PUBLIC = os.environ.get("OKX_PUBLIC_BASE", "https://www.okx.com")
ECB = os.environ.get("ECB_DATA_BASE", "https://data-api.ecb.europa.eu/service/data")
BLS_ICS = os.environ.get("BLS_CALENDAR_ICS", "https://www.bls.gov/schedule/news_release/bls.ics")
VERSION = "1.34.0"
PATCH_RELEASE = "v1.34"
PRODUCER = "LeanTrader automated MTF regime and horizon discovery research sidecar"
STOP = False

PACK_MICRO_5M = "leantrader-microstructure-5m-challenger"
PACK_MICRO_15M = "leantrader-microstructure-15m-challenger"
PACK_WHALE = "leantrader-whale-concentration-context"
PACK_FX = "leantrader-rates-fx-cross-asset-context"
PACK_LEADLAG = "leantrader-causal-leadlag-challenger"
PACK_CROSS_VENUE = "leantrader-cross-venue-orderbook-challenger"
PACK_VENUE_QUALITY = "leantrader-cross-venue-quality-integrity-context"
PACK_CORE_BRIDGE = "leantrader-core-sensor-capability-bridge"
PACK_MACRO = "leantrader-official-macro-calendar-context"
OLD_PACK = "leantrader-adaptive-research-sidecar"
PACK_REPUTATION = "leantrader-specialist-reputation-context"
PACK_EVIDENCE_GOVERNANCE = "leantrader-specialist-evidence-governance-context"
PACK_DIVERSITY = "leantrader-specialist-diversity-governance-context"
PACK_COUNCIL = "leantrader-specialist-council-advisory-context"
PACK_ROUTING = "leantrader-specialist-routing-governance-context"
PACK_COUNCIL_COUNTERFACTUAL = "leantrader-council-decision-counterfactual-context"
PACK_COUNCIL_ROLE_POLICY = "leantrader-council-role-policy-context"
PACK_RISK_FRACTION_POLICY = "leantrader-council-risk-fraction-policy-context"
PACK_NO_TRADE_FRONTIER = "leantrader-council-no-trade-frontier-context"
PACK_HORIZON_POLICY = "leantrader-net-edge-horizon-policy-context"
PACK_HORIZON_CHALLENGER = "leantrader-horizon-optimized-challenger"
PACK_FALSIFICATION_POLICY = "leantrader-prequential-falsification-policy-context"
PACK_INVERSE_CHALLENGER = "leantrader-falsification-inverse-challenger"
PACK_CONDITIONAL_EDGE_POLICY = "leantrader-conditional-edge-policy-context"
PACK_CONDITIONAL_CHALLENGER = "leantrader-context-selective-alpha-challenger"
PACK_HIERARCHICAL_CONTEXT_POLICY = "leantrader-hierarchical-context-policy-context"
PACK_HIERARCHICAL_CHALLENGER = "leantrader-hierarchical-context-alpha-challenger"
PACK_ORTHOGONAL_SENSOR_POLICY = "leantrader-orthogonal-sensor-policy-context"
PACK_ORTHOGONAL_COVERAGE = "leantrader-orthogonal-sensor-coverage-context"
PACK_ORTHOGONAL_FLOW = "leantrader-orthogonal-flow-positioning-challenger"
PACK_DERIV_LIQ = "leantrader-derivatives-liquidation-squeeze-challenger"
PACK_DERIV_LIQ_POLICY = "leantrader-derivatives-liquidation-policy-context"
PACK_CROSS_DERIV = "leantrader-cross-venue-derivatives-carry-challenger"
PACK_CROSS_DERIV_POLICY = "leantrader-cross-venue-derivatives-carry-policy-context"
PACK_EPISODE_INDEPENDENCE = "leantrader-shadow-evidence-independence-context"
PACK_FLOW_RESIDUAL_CONTEXT = "leantrader-flow-residual-research-context"
PACK_EXTERNAL_AGGRESSIVE_FLOW = "leantrader-external-aggressive-flow-research-context"
PACK_MTF_SCALPING_CONTEXT = "leantrader-multi-timeframe-scalping-orchestration-context"
MTF_TIMEFRAMES = {
    "1m": ("1", 60),
    "3m": ("3", 180),
    "5m": ("5", 300),
    "15m": ("15", 900),
    "30m": ("30", 1800),
    "1h": ("60", 3600),
    "4h": ("240", 14400),
    "1d": ("D", 86400),
}
MTF_MAX_SYMBOLS_PER_CYCLE = max(1, min(8, int(os.environ.get("EVOLUTION_MTF_MAX_SYMBOLS_PER_CYCLE", "4"))))
MTF_KLINE_LIMIT = max(40, min(120, int(os.environ.get("EVOLUTION_MTF_KLINE_LIMIT", "64"))))
MTF_MIN_CLOSED_BARS = max(32, min(80, int(os.environ.get("EVOLUTION_MTF_MIN_CLOSED_BARS", "48"))))
MTF_MAX_SPREAD_BPS = max(1.0, min(50.0, float(os.environ.get("EVOLUTION_MTF_MAX_SPREAD_BPS", "12"))))
MTF_ENTRY_MIN_SCORE = max(0.05, min(0.50, float(os.environ.get("EVOLUTION_MTF_ENTRY_MIN_SCORE", "0.18"))))
MTF_STRUCTURE_MIN_SCORE = max(0.05, min(0.50, float(os.environ.get("EVOLUTION_MTF_STRUCTURE_MIN_SCORE", "0.14"))))
MTF_REGIME_OPPOSITION_LIMIT = max(0.05, min(0.80, float(os.environ.get("EVOLUTION_MTF_REGIME_OPPOSITION_LIMIT", "0.28"))))
MTF_OUTCOME_MAX_RESOLVED = max(200, min(5000, int(os.environ.get("EVOLUTION_MTF_OUTCOME_MAX_RESOLVED", "1600"))))
MTF_OUTCOME_MAX_PENDING = max(8, min(128, int(os.environ.get("EVOLUTION_MTF_OUTCOME_MAX_PENDING", "64"))))
MTF_OUTCOME_MIN_VALIDATION_SAMPLES = max(8, min(200, int(os.environ.get("EVOLUTION_MTF_OUTCOME_MIN_VALIDATION_SAMPLES", "20"))))
MTF_OUTCOME_MIN_HOLDOUT_SAMPLES = max(3, min(50, int(os.environ.get("EVOLUTION_MTF_OUTCOME_MIN_HOLDOUT_SAMPLES", "6"))))
# v1.32 falsification lab: evaluate every shadow-ready setup across a fixed ex-ante
# horizon grid instead of changing the v1.31 signal thresholds after a negative result.
MTF_HORIZON_GRID = tuple(sorted({300, 600, 900, 1200, 1800, 2700, 3600}))
MTF_HORIZON_MAX_RESOLVED = max(500, min(10000, int(os.environ.get("EVOLUTION_MTF_HORIZON_MAX_RESOLVED", "5000"))))
MTF_HORIZON_MAX_PENDING = max(4, min(64, int(os.environ.get("EVOLUTION_MTF_HORIZON_MAX_PENDING", "32"))))
MTF_HORIZON_MIN_VALIDATION_SAMPLES = max(12, min(200, int(os.environ.get("EVOLUTION_MTF_HORIZON_MIN_VALIDATION_SAMPLES", "20"))))
MTF_HORIZON_MIN_HOLDOUT_SAMPLES = max(4, min(50, int(os.environ.get("EVOLUTION_MTF_HORIZON_MIN_HOLDOUT_SAMPLES", "6"))))
MTF_HORIZON_BACKFILL_MAX_QUERIES = max(1, min(32, int(os.environ.get("EVOLUTION_MTF_HORIZON_BACKFILL_MAX_QUERIES", "8"))))
# v1.33 automated regime/horizon discovery is descriptive and prospective only.
# Positive discovery requires chronological train+holdout support and robust/outlier-resistant
# metrics. Early falsification may quarantine a research cell from nomination, but never
# manufactures inverse alpha or changes execution/risk thresholds.
MTF_DISCOVERY_MIN_SAMPLES = max(10, min(200, int(os.environ.get("EVOLUTION_MTF_DISCOVERY_MIN_SAMPLES", "12"))))
MTF_DISCOVERY_MIN_HOLDOUT_SAMPLES = max(3, min(50, int(os.environ.get("EVOLUTION_MTF_DISCOVERY_MIN_HOLDOUT_SAMPLES", "4"))))
MTF_DISCOVERY_MIN_UNIQUE_SYMBOLS = max(2, min(20, int(os.environ.get("EVOLUTION_MTF_DISCOVERY_MIN_UNIQUE_SYMBOLS", "3"))))
MTF_DISCOVERY_EARLY_FALSIFICATION_SAMPLES = max(5, min(50, int(os.environ.get("EVOLUTION_MTF_DISCOVERY_EARLY_FALSIFICATION_SAMPLES", "5"))))
MTF_DISCOVERY_MAX_REPORTED_CELLS = max(8, min(100, int(os.environ.get("EVOLUTION_MTF_DISCOVERY_MAX_REPORTED_CELLS", "24"))))
# v1.34 dedicated prospective replication of the first automatically-discovered research-positive family.
MTF_5M_LONG_60M_SPECIALIST_HORIZON = 3600
MTF_5M_LONG_60M_SPECIALIST_CHECKPOINTS = (300, 600, 900, 1200, 1800, 2700, 3600)
MTF_5M_LONG_60M_SPECIALIST_MIN_SAMPLES = max(12, min(100, int(os.environ.get("EVOLUTION_MTF_5M_LONG_60M_MIN_SAMPLES", "12"))))
MTF_5M_LONG_60M_SPECIALIST_MIN_HOLDOUT = max(4, min(40, int(os.environ.get("EVOLUTION_MTF_5M_LONG_60M_MIN_HOLDOUT", "4"))))
MTF_5M_LONG_60M_SPECIALIST_MIN_SYMBOLS = max(4, min(20, int(os.environ.get("EVOLUTION_MTF_5M_LONG_60M_MIN_SYMBOLS", "4"))))
MTF_5M_LONG_60M_SPECIALIST_MAX_PENDING = max(4, min(32, int(os.environ.get("EVOLUTION_MTF_5M_LONG_60M_MAX_PENDING", "16"))))
MTF_5M_LONG_60M_SPECIALIST_MAX_RESOLVED = max(100, min(5000, int(os.environ.get("EVOLUTION_MTF_5M_LONG_60M_MAX_RESOLVED", "1200"))))
MTF_5M_LONG_60M_SPECIALIST_BACKFILL_BUDGET = max(1, min(16, int(os.environ.get("EVOLUTION_MTF_5M_LONG_60M_BACKFILL_BUDGET", "8"))))

DIRECTIONAL_PACKS = (PACK_MICRO_5M, PACK_MICRO_15M, PACK_LEADLAG, PACK_CROSS_VENUE)
DIRECTIONAL_PACK_SET = frozenset(DIRECTIONAL_PACKS)
SPECIALIST_EPISODE_PACK_SET = frozenset(set(DIRECTIONAL_PACKS) | {PACK_INVERSE_CHALLENGER, PACK_CONDITIONAL_CHALLENGER, PACK_HIERARCHICAL_CHALLENGER, PACK_ORTHOGONAL_FLOW, PACK_DERIV_LIQ, PACK_CROSS_DERIV})
INDEPENDENCE_GOVERNED_PACK_SET = frozenset({PACK_ORTHOGONAL_FLOW, PACK_DERIV_LIQ, PACK_CROSS_DERIV, PACK_INVERSE_CHALLENGER, PACK_CONDITIONAL_CHALLENGER, PACK_HIERARCHICAL_CHALLENGER})
ROUND_TRIP_COST_BPS = max(0.0, float(os.environ.get("EVOLUTION_ROUND_TRIP_COST_BPS", "30")))
RISK_FRACTION_GRID = (0.00, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
MIN_RISK_FRACTION_SAMPLES = max(6, int(os.environ.get("EVOLUTION_MIN_RISK_FRACTION_SAMPLES", "8")))
MIN_NO_TRADE_FRONTIER_SAMPLES = max(12, int(os.environ.get("EVOLUTION_MIN_NO_TRADE_FRONTIER_SAMPLES", "20")))
HORIZON_GRID = tuple(sorted({300, 900, 1800, 3600}))
MIN_HORIZON_POLICY_SAMPLES = max(6, int(os.environ.get("EVOLUTION_MIN_HORIZON_POLICY_SAMPLES", "8")))
HORIZON_BACKFILL_PER_CYCLE = max(1, min(24, int(os.environ.get("EVOLUTION_HORIZON_BACKFILL_PER_CYCLE", "10"))))
MAX_HORIZON_LAB_RESOLVED = max(1000, int(os.environ.get("EVOLUTION_HORIZON_LAB_RESOLVED_MAX", "12000")))
MIN_FALSIFICATION_TOTAL_SAMPLES = max(8, int(os.environ.get("EVOLUTION_MIN_FALSIFICATION_TOTAL_SAMPLES", "8")))
MIN_FALSIFICATION_HOLDOUT_SAMPLES = max(3, int(os.environ.get("EVOLUTION_MIN_FALSIFICATION_HOLDOUT_SAMPLES", "3")))
FALSIFICATION_CONFIDENCE_MULTIPLIER = max(0.05, min(1.0, float(os.environ.get("EVOLUTION_FALSIFICATION_CONFIDENCE_MULTIPLIER", "0.25"))))
INVERSE_CHALLENGER_CONFIDENCE_MULTIPLIER = max(0.05, min(1.0, float(os.environ.get("EVOLUTION_INVERSE_CHALLENGER_CONFIDENCE_MULTIPLIER", "0.50"))))
MIN_CONDITIONAL_EDGE_SAMPLES = max(10, int(os.environ.get("EVOLUTION_MIN_CONDITIONAL_EDGE_SAMPLES", "12")))
MIN_CONDITIONAL_HOLDOUT_SAMPLES = max(4, int(os.environ.get("EVOLUTION_MIN_CONDITIONAL_HOLDOUT_SAMPLES", "4")))
MIN_CONDITIONAL_HOLDOUT_NET = max(0.0, float(os.environ.get("EVOLUTION_MIN_CONDITIONAL_HOLDOUT_NET", "0.0005")))
CONDITIONAL_NEGATIVE_CONFIDENCE_MULTIPLIER = max(0.05, min(1.0, float(os.environ.get("EVOLUTION_CONDITIONAL_NEGATIVE_CONFIDENCE_MULTIPLIER", "0.35"))))
CONDITIONAL_CHALLENGER_CONFIDENCE_MULTIPLIER = max(0.05, min(1.0, float(os.environ.get("EVOLUTION_CONDITIONAL_CHALLENGER_CONFIDENCE_MULTIPLIER", "0.50"))))
MIN_HIERARCHICAL_COARSE_SAMPLES = max(10, int(os.environ.get("EVOLUTION_MIN_HIERARCHICAL_COARSE_SAMPLES", "16")))
MIN_HIERARCHICAL_MID_SAMPLES = max(8, int(os.environ.get("EVOLUTION_MIN_HIERARCHICAL_MID_SAMPLES", "10")))
MIN_HIERARCHICAL_FINE_SAMPLES = max(6, int(os.environ.get("EVOLUTION_MIN_HIERARCHICAL_FINE_SAMPLES", "6")))
MIN_HIERARCHICAL_HOLDOUT_SAMPLES = max(3, int(os.environ.get("EVOLUTION_MIN_HIERARCHICAL_HOLDOUT_SAMPLES", "4")))
MIN_HIERARCHICAL_HOLDOUT_NET = max(0.0, float(os.environ.get("EVOLUTION_MIN_HIERARCHICAL_HOLDOUT_NET", "0.0005")))
HIERARCHICAL_PRIOR_STRENGTH = max(2.0, float(os.environ.get("EVOLUTION_HIERARCHICAL_PRIOR_STRENGTH", "8")))
HIERARCHICAL_NEGATIVE_CONFIDENCE_MULTIPLIER = max(0.05, min(1.0, float(os.environ.get("EVOLUTION_HIERARCHICAL_NEGATIVE_CONFIDENCE_MULTIPLIER", "0.30"))))
HIERARCHICAL_CHALLENGER_CONFIDENCE_MULTIPLIER = max(0.05, min(1.0, float(os.environ.get("EVOLUTION_HIERARCHICAL_CHALLENGER_CONFIDENCE_MULTIPLIER", "0.40"))))
ORTHOGONAL_FLOW_HORIZON_SECONDS = max(900, min(3600, int(os.environ.get("EVOLUTION_ORTHOGONAL_FLOW_HORIZON_SECONDS", "1800"))))
ORTHOGONAL_MIN_FLOW_SCORE = max(0.15, min(0.80, float(os.environ.get("EVOLUTION_ORTHOGONAL_MIN_FLOW_SCORE", "0.35"))))
ORTHOGONAL_MIN_OI_BUILD = max(0.005, min(0.20, float(os.environ.get("EVOLUTION_ORTHOGONAL_MIN_OI_BUILD", "0.02"))))
ORTHOGONAL_MIN_POSITIONING_SKEW = max(0.02, min(0.50, float(os.environ.get("EVOLUTION_ORTHOGONAL_MIN_POSITIONING_SKEW", "0.08"))))
ORTHOGONAL_MIN_LIQUIDATION_IMBALANCE = max(0.20, min(0.95, float(os.environ.get("EVOLUTION_ORTHOGONAL_MIN_LIQUIDATION_IMBALANCE", "0.55"))))
ORTHOGONAL_MAX_SIGNAL_CONFIDENCE = max(0.15, min(0.60, float(os.environ.get("EVOLUTION_ORTHOGONAL_MAX_SIGNAL_CONFIDENCE", "0.50"))))
MIN_ORTHOGONAL_POLICY_SAMPLES = max(8, int(os.environ.get("EVOLUTION_MIN_ORTHOGONAL_POLICY_SAMPLES", "12")))
MIN_ORTHOGONAL_HOLDOUT_SAMPLES = max(3, int(os.environ.get("EVOLUTION_MIN_ORTHOGONAL_HOLDOUT_SAMPLES", "4")))
MIN_ORTHOGONAL_HOLDOUT_NET = max(0.0, float(os.environ.get("EVOLUTION_MIN_ORTHOGONAL_HOLDOUT_NET", "0.0005")))
ORTHOGONAL_NEGATIVE_CONFIDENCE_MULTIPLIER = max(0.05, min(1.0, float(os.environ.get("EVOLUTION_ORTHOGONAL_NEGATIVE_CONFIDENCE_MULTIPLIER", "0.25"))))
ORTHOGONAL_CACHE_DERIVATIVES_MAX_AGE = max(300, min(3600, int(os.environ.get("EVOLUTION_ORTHOGONAL_CACHE_DERIVATIVES_MAX_AGE", "1200"))))
ORTHOGONAL_CACHE_FLOW_MAX_AGE = max(300, min(3600, int(os.environ.get("EVOLUTION_ORTHOGONAL_CACHE_FLOW_MAX_AGE", "1800"))))
ORTHOGONAL_CACHE_LIQUIDATIONS_MAX_AGE = max(180, min(1800, int(os.environ.get("EVOLUTION_ORTHOGONAL_CACHE_LIQUIDATIONS_MAX_AGE", "900"))))
ORTHOGONAL_MAX_SOURCE_SKEW_SECONDS = max(120, min(1800, int(os.environ.get("EVOLUTION_ORTHOGONAL_MAX_SOURCE_SKEW_SECONDS", "900"))))
ORTHOGONAL_RESEARCH_UNIVERSE_LIMIT = max(20, min(64, int(os.environ.get("EVOLUTION_ORTHOGONAL_RESEARCH_UNIVERSE_LIMIT", "48"))))
ORTHOGONAL_PRICE_ANCHOR_MAX_AGE_SECONDS = max(120, min(1800, int(os.environ.get("EVOLUTION_ORTHOGONAL_PRICE_ANCHOR_MAX_AGE_SECONDS", "600"))))
ORTHOGONAL_MAX_PRICE_ANCHOR_FETCHES_PER_CYCLE = max(1, min(24, int(os.environ.get("EVOLUTION_ORTHOGONAL_MAX_PRICE_ANCHOR_FETCHES_PER_CYCLE", "12"))))
ORTHOGONAL_SINGLE_COMPONENT_MIN_CONFIDENCE = max(0.45, min(0.80, float(os.environ.get("EVOLUTION_ORTHOGONAL_SINGLE_COMPONENT_MIN_CONFIDENCE", "0.47"))))
ORTHOGONAL_SINGLE_COMPONENT_MIN_FLOW_SCORE = max(0.45, min(0.90, float(os.environ.get("EVOLUTION_ORTHOGONAL_SINGLE_COMPONENT_MIN_FLOW_SCORE", "0.55"))))
ORTHOGONAL_SINGLE_COMPONENT_MIN_OI_BUILD = max(0.02, min(0.20, float(os.environ.get("EVOLUTION_ORTHOGONAL_SINGLE_COMPONENT_MIN_OI_BUILD", "0.03"))))
ORTHOGONAL_SINGLE_COMPONENT_MIN_POSITIONING_SKEW = max(0.05, min(0.50, float(os.environ.get("EVOLUTION_ORTHOGONAL_SINGLE_COMPONENT_MIN_POSITIONING_SKEW", "0.10"))))
ORTHOGONAL_SINGLE_COMPONENT_MIN_LIQUIDATION_IMBALANCE = max(0.50, min(0.95, float(os.environ.get("EVOLUTION_ORTHOGONAL_SINGLE_COMPONENT_MIN_LIQUIDATION_IMBALANCE", "0.65"))))
ORTHOGONAL_SINGLE_COMPONENT_MAX_SIGNAL_CONFIDENCE = max(0.10, min(0.35, float(os.environ.get("EVOLUTION_ORTHOGONAL_SINGLE_COMPONENT_MAX_SIGNAL_CONFIDENCE", "0.28"))))
ORTHOGONAL_COMMON_FACTOR_MIN_CLUSTER = max(4, min(24, int(os.environ.get("EVOLUTION_ORTHOGONAL_COMMON_FACTOR_MIN_CLUSTER", "8"))))
ORTHOGONAL_COMMON_FACTOR_MIN_SHARE = max(0.15, min(0.75, float(os.environ.get("EVOLUTION_ORTHOGONAL_COMMON_FACTOR_MIN_SHARE", "0.25"))))
FLOW_RESIDUAL_MIN_CURRENT_SINGLE_ROWS = max(8, min(48, int(os.environ.get("EVOLUTION_FLOW_RESIDUAL_MIN_CURRENT_SINGLE_ROWS", "12"))))
FLOW_RESIDUAL_SCALE_FLOOR = max(0.001, min(0.05, float(os.environ.get("EVOLUTION_FLOW_RESIDUAL_SCALE_FLOOR", "0.005"))))
FLOW_RESIDUAL_DIAGNOSTIC_MIN_ABS = max(0.005, min(0.20, float(os.environ.get("EVOLUTION_FLOW_RESIDUAL_DIAGNOSTIC_MIN_ABS", "0.02"))))
FLOW_RESIDUAL_DIAGNOSTIC_MIN_Z = max(2.0, min(8.0, float(os.environ.get("EVOLUTION_FLOW_RESIDUAL_DIAGNOSTIC_MIN_Z", "3.0"))))
FLOW_RESIDUAL_HISTORY_PER_SYMBOL = max(6, min(48, int(os.environ.get("EVOLUTION_FLOW_RESIDUAL_HISTORY_PER_SYMBOL", "24"))))
FLOW_RESIDUAL_MAX_SYMBOLS = max(24, min(128, int(os.environ.get("EVOLUTION_FLOW_RESIDUAL_MAX_SYMBOLS", "64"))))
FLOW_RESIDUAL_PERSISTENCE_MIN_SAMPLES = max(3, min(12, int(os.environ.get("EVOLUTION_FLOW_RESIDUAL_PERSISTENCE_MIN_SAMPLES", "4"))))
EXTERNAL_FLOW_MAX_SYMBOLS_PER_CYCLE = max(4, min(16, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_MAX_SYMBOLS_PER_CYCLE", "8"))))
EXTERNAL_FLOW_CLOSED_BARS = max(2, min(6, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_CLOSED_BARS", "3"))))
EXTERNAL_FLOW_MIN_CURRENT_ROWS = max(4, min(12, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_MIN_CURRENT_ROWS", "5"))))
EXTERNAL_FLOW_MAX_SOURCE_AGE_SECONDS = max(600, min(3600, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_MAX_SOURCE_AGE_SECONDS", "1200"))))
EXTERNAL_FLOW_SCALE_FLOOR = max(0.01, min(0.20, float(os.environ.get("EVOLUTION_EXTERNAL_FLOW_SCALE_FLOOR", "0.03"))))
EXTERNAL_FLOW_DIAGNOSTIC_MIN_ABS = max(0.05, min(0.50, float(os.environ.get("EVOLUTION_EXTERNAL_FLOW_DIAGNOSTIC_MIN_ABS", "0.12"))))
EXTERNAL_FLOW_DIAGNOSTIC_MIN_Z = max(1.5, min(6.0, float(os.environ.get("EVOLUTION_EXTERNAL_FLOW_DIAGNOSTIC_MIN_Z", "2.5"))))
EXTERNAL_FLOW_HISTORY_PER_SYMBOL = max(6, min(48, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_HISTORY_PER_SYMBOL", "24"))))
EXTERNAL_FLOW_PERSISTENCE_MIN_SAMPLES = max(3, min(12, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_PERSISTENCE_MIN_SAMPLES", "3"))))
EXTERNAL_FLOW_MAX_TRACKED_SYMBOLS = max(24, min(128, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_MAX_TRACKED_SYMBOLS", "64"))))
EXTERNAL_FLOW_OUTCOME_HORIZON_SECONDS = max(900, min(7200, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_OUTCOME_HORIZON_SECONDS", "1800"))))
EXTERNAL_FLOW_OUTCOME_MAX_RESOLVED = max(200, min(5000, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_OUTCOME_MAX_RESOLVED", "1200"))))
EXTERNAL_FLOW_OUTCOME_MAX_PENDING = max(12, min(128, int(os.environ.get("EVOLUTION_EXTERNAL_FLOW_OUTCOME_MAX_PENDING", "64"))))
LIQUIDATION_EVENT_WINDOW_SECONDS = max(300, min(1800, int(os.environ.get("EVOLUTION_LIQUIDATION_EVENT_WINDOW_SECONDS", "900"))))
LIQUIDATION_EVENT_WINDOW_MAX_SNAPSHOTS = max(2, min(16, int(os.environ.get("EVOLUTION_LIQUIDATION_EVENT_WINDOW_MAX_SNAPSHOTS", "6"))))
LIQUIDATION_EVENT_WINDOW_MAX_SYMBOLS = max(24, min(128, int(os.environ.get("EVOLUTION_LIQUIDATION_EVENT_WINDOW_MAX_SYMBOLS", "64"))))
DERIV_LIQ_HORIZON_SECONDS = max(600, min(3600, int(os.environ.get("EVOLUTION_DERIV_LIQ_HORIZON_SECONDS", "900"))))
DERIV_LIQ_MIN_DERIV_CONFIDENCE = max(0.55, min(0.95, float(os.environ.get("EVOLUTION_DERIV_LIQ_MIN_DERIV_CONFIDENCE", "0.60"))))
DERIV_LIQ_MIN_LIQ_CONFIDENCE = max(0.50, min(0.95, float(os.environ.get("EVOLUTION_DERIV_LIQ_MIN_LIQ_CONFIDENCE", "0.55"))))
DERIV_LIQ_MIN_POSITIONING_SKEW = max(0.05, min(0.50, float(os.environ.get("EVOLUTION_DERIV_LIQ_MIN_POSITIONING_SKEW", "0.10"))))
DERIV_LIQ_MIN_LIQUIDATION_IMBALANCE = max(0.45, min(0.95, float(os.environ.get("EVOLUTION_DERIV_LIQ_MIN_LIQUIDATION_IMBALANCE", "0.60"))))
DERIV_LIQ_MIN_LIQUIDATION_EVENTS = max(2, min(20, int(os.environ.get("EVOLUTION_DERIV_LIQ_MIN_LIQUIDATION_EVENTS", "3"))))
DERIV_LIQ_MAX_SIGNAL_CONFIDENCE = max(0.15, min(0.45, float(os.environ.get("EVOLUTION_DERIV_LIQ_MAX_SIGNAL_CONFIDENCE", "0.34"))))
DERIV_LIQ_SAME_VENUE_CONFIDENCE_MULTIPLIER = max(0.40, min(1.0, float(os.environ.get("EVOLUTION_DERIV_LIQ_SAME_VENUE_CONFIDENCE_MULTIPLIER", "0.75"))))
MIN_DERIV_LIQ_POLICY_SAMPLES = max(10, int(os.environ.get("EVOLUTION_MIN_DERIV_LIQ_POLICY_SAMPLES", "16")))
MIN_DERIV_LIQ_HOLDOUT_SAMPLES = max(4, int(os.environ.get("EVOLUTION_MIN_DERIV_LIQ_HOLDOUT_SAMPLES", "5")))
MIN_DERIV_LIQ_HOLDOUT_NET = max(0.0, float(os.environ.get("EVOLUTION_MIN_DERIV_LIQ_HOLDOUT_NET", "0.0005")))
DERIV_LIQ_NEGATIVE_CONFIDENCE_MULTIPLIER = max(0.05, min(1.0, float(os.environ.get("EVOLUTION_DERIV_LIQ_NEGATIVE_CONFIDENCE_MULTIPLIER", "0.25"))))
CROSS_DERIV_HORIZON_SECONDS = max(900, min(7200, int(os.environ.get("EVOLUTION_CROSS_DERIV_HORIZON_SECONDS", "1800"))))
CROSS_DERIV_MAX_SYMBOLS_PER_CYCLE = max(2, min(16, int(os.environ.get("EVOLUTION_CROSS_DERIV_MAX_SYMBOLS_PER_CYCLE", "8"))))
CROSS_DERIV_MIN_DERIV_CONFIDENCE = max(0.55, min(0.95, float(os.environ.get("EVOLUTION_CROSS_DERIV_MIN_DERIV_CONFIDENCE", "0.60"))))
CROSS_DERIV_MIN_POSITIONING_SKEW = max(0.04, min(0.40, float(os.environ.get("EVOLUTION_CROSS_DERIV_MIN_POSITIONING_SKEW", "0.08"))))
CROSS_DERIV_MIN_FUNDING_ABS = max(0.00002, min(0.002, float(os.environ.get("EVOLUTION_CROSS_DERIV_MIN_FUNDING_ABS", "0.00008"))))
CROSS_DERIV_MIN_BASIS_ABS = max(0.0002, min(0.01, float(os.environ.get("EVOLUTION_CROSS_DERIV_MIN_BASIS_ABS", "0.0008"))))
CROSS_DERIV_MAX_OI_COLLAPSE = max(0.02, min(0.20, float(os.environ.get("EVOLUTION_CROSS_DERIV_MAX_OI_COLLAPSE", "0.06"))))
CROSS_DERIV_MAX_SIGNAL_CONFIDENCE = max(0.15, min(0.45, float(os.environ.get("EVOLUTION_CROSS_DERIV_MAX_SIGNAL_CONFIDENCE", "0.32"))))
MIN_CROSS_DERIV_POLICY_SAMPLES = max(10, int(os.environ.get("EVOLUTION_MIN_CROSS_DERIV_POLICY_SAMPLES", "18")))
MIN_CROSS_DERIV_HOLDOUT_SAMPLES = max(4, int(os.environ.get("EVOLUTION_MIN_CROSS_DERIV_HOLDOUT_SAMPLES", "5")))
MIN_CROSS_DERIV_HOLDOUT_NET = max(0.0, float(os.environ.get("EVOLUTION_MIN_CROSS_DERIV_HOLDOUT_NET", "0.0005")))
CROSS_DERIV_NEGATIVE_CONFIDENCE_MULTIPLIER = max(0.05, min(1.0, float(os.environ.get("EVOLUTION_CROSS_DERIV_NEGATIVE_CONFIDENCE_MULTIPLIER", "0.25"))))
CROSS_DERIV_GLOBAL_NEGATIVE_CONFIDENCE_MULTIPLIER = max(0.02, min(0.25, float(os.environ.get("EVOLUTION_CROSS_DERIV_GLOBAL_NEGATIVE_CONFIDENCE_MULTIPLIER", "0.10"))))
CROSS_DERIV_QUARANTINE_RELEASE_STREAK = max(2, min(12, int(os.environ.get("EVOLUTION_CROSS_DERIV_QUARANTINE_RELEASE_STREAK", "3"))))
MIN_CROSS_DERIV_GLOBAL_QUARANTINE_SAMPLES = max(MIN_CROSS_DERIV_POLICY_SAMPLES, int(os.environ.get("EVOLUTION_MIN_CROSS_DERIV_GLOBAL_QUARANTINE_SAMPLES", str(MIN_CROSS_DERIV_POLICY_SAMPLES))))
CROSS_DERIV_EXPLOIT_SLOTS = max(1, min(CROSS_DERIV_MAX_SYMBOLS_PER_CYCLE, int(os.environ.get("EVOLUTION_CROSS_DERIV_EXPLOIT_SLOTS", "4"))))
CROSS_DERIV_EXPLORATION_SLOTS = max(1, min(CROSS_DERIV_MAX_SYMBOLS_PER_CYCLE, int(os.environ.get("EVOLUTION_CROSS_DERIV_EXPLORATION_SLOTS", "4"))))
CROSS_DERIV_TWO_VENUE_CONFIDENCE_CAP = max(0.12, min(CROSS_DERIV_MAX_SIGNAL_CONFIDENCE, float(os.environ.get("EVOLUTION_CROSS_DERIV_TWO_VENUE_CONFIDENCE_CAP", "0.24"))))
CROSS_DERIV_QUERY_HISTORY_MAX = max(64, min(512, int(os.environ.get("EVOLUTION_CROSS_DERIV_QUERY_HISTORY_MAX", "256"))))
CROSS_DERIV_OKX_OI_CACHE_MAX_AGE = max(300, min(7200, int(os.environ.get("EVOLUTION_CROSS_DERIV_OKX_OI_CACHE_MAX_AGE", "3600"))))
EPISODE_OVERLAP_GUARD_ENABLED = str(os.environ.get("EVOLUTION_EPISODE_OVERLAP_GUARD", "1")).strip().lower() not in {"0", "false", "no", "off"}
MAX_SPECIALIST_RESOLVED = max(500, int(os.environ.get("EVOLUTION_SPECIALIST_RESOLVED_MAX", "5000")))
EVIDENCE_HALF_LIFE_SECONDS = max(3600.0, float(os.environ.get("EVOLUTION_EVIDENCE_HALF_LIFE_SECONDS", "172800")))
MIN_GLOBAL_VALIDATION_SAMPLES = max(20, int(os.environ.get("EVOLUTION_MIN_GLOBAL_VALIDATION_SAMPLES", "40")))
MIN_CONTEXT_VALIDATION_SAMPLES = max(8, int(os.environ.get("EVOLUTION_MIN_CONTEXT_VALIDATION_SAMPLES", "12")))
MAJOR_ASSETS = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "TRX", "BCH", "DOT", "LTC"}
STABLE_ASSETS = {"USDT", "USDC", "USDE", "USDS", "DAI", "USD1", "USDY", "PYUSD", "FDUSD", "USDD"}

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

# Snapshot from official Federal Reserve published calendar. This is context only;
# BLS events are fetched live from the official BLS calendar feed each sidecar cycle.
FOMC_EVENTS = [
    ("2026-08-19T14:00:00-04:00", "FOMC Minutes — July 28-29 meeting"),
    ("2026-09-16T14:00:00-04:00", "FOMC policy decision — September 15-16 meeting"),
    ("2026-10-28T14:00:00-04:00", "FOMC policy decision — October 27-28 meeting"),
    ("2026-12-09T14:00:00-05:00", "FOMC policy decision — December 8-9 meeting"),
    ("2027-01-27T14:00:00-05:00", "FOMC policy decision — January 26-27 meeting"),
    ("2027-03-17T14:00:00-04:00", "FOMC policy decision — March 16-17 meeting"),
    ("2027-04-28T14:00:00-04:00", "FOMC policy decision — April 27-28 meeting"),
    ("2027-06-09T14:00:00-04:00", "FOMC policy decision — June 8-9 meeting"),
    ("2027-07-28T14:00:00-04:00", "FOMC policy decision — July 27-28 meeting"),
    ("2027-09-15T14:00:00-04:00", "FOMC policy decision — September 14-15 meeting"),
    ("2027-10-27T14:00:00-04:00", "FOMC policy decision — October 26-27 meeting"),
    ("2027-12-08T14:00:00-05:00", "FOMC policy decision — December 7-8 meeting"),
]

HIGH_IMPACT_PATTERNS = (
    "consumer price index",
    "employment situation",
    "producer price index",
    "job openings and labor turnover",
    "employment cost index",
    "productivity and costs",
    "import and export price indexes",
    "real earnings",
)

# Verified fallback snapshot from the official BLS 2026 release schedule.
# This is used only when the live BLS .ics feed rejects automated retrieval.
# It is intentionally bounded to 2026 and treated as calendar context, never alpha.
BLS_VERIFIED_SNAPSHOT_ASOF = "2026-08-17"
BLS_VERIFIED_SNAPSHOT_EXPIRES = "2026-12-31T23:59:59-05:00"
BLS_VERIFIED_EVENTS = [
    ("2026-08-18T08:30:00-04:00", "U.S. Import and Export Price Indexes for July 2026"),
    ("2026-09-01T10:00:00-04:00", "Job Openings and Labor Turnover Survey for July 2026"),
    ("2026-09-03T08:30:00-04:00", "Productivity and Costs (R) for Second Quarter 2026"),
    ("2026-09-04T08:30:00-04:00", "Employment Situation for August 2026"),
    ("2026-09-10T08:30:00-04:00", "Producer Price Index for August 2026"),
    ("2026-09-11T08:30:00-04:00", "Consumer Price Index for August 2026"),
    ("2026-09-11T08:30:00-04:00", "Real Earnings for August 2026"),
    ("2026-09-16T08:30:00-04:00", "U.S. Import and Export Price Indexes for August 2026"),
    ("2026-09-29T10:00:00-04:00", "Job Openings and Labor Turnover Survey for August 2026"),
    ("2026-10-02T08:30:00-04:00", "Employment Situation for September 2026"),
    ("2026-10-14T08:30:00-04:00", "Consumer Price Index for September 2026"),
    ("2026-10-14T08:30:00-04:00", "Real Earnings for September 2026"),
    ("2026-10-15T08:30:00-04:00", "Producer Price Index for September 2026"),
    ("2026-10-16T08:30:00-04:00", "U.S. Import and Export Price Indexes for September 2026"),
    ("2026-10-30T08:30:00-04:00", "Employment Cost Index for Third Quarter 2026"),
    ("2026-11-03T10:00:00-05:00", "Job Openings and Labor Turnover Survey for September 2026"),
    ("2026-11-05T08:30:00-05:00", "Productivity and Costs (P) for Third Quarter 2026"),
    ("2026-11-06T08:30:00-05:00", "Employment Situation for October 2026"),
    ("2026-11-10T08:30:00-05:00", "Consumer Price Index for October 2026"),
    ("2026-11-10T08:30:00-05:00", "Real Earnings for October 2026"),
    ("2026-11-13T08:30:00-05:00", "Producer Price Index for October 2026"),
    ("2026-11-17T08:30:00-05:00", "U.S. Import and Export Price Indexes for October 2026"),
    ("2026-12-01T10:00:00-05:00", "Job Openings and Labor Turnover Survey for October 2026"),
    ("2026-12-04T08:30:00-05:00", "Employment Situation for November 2026"),
    ("2026-12-08T08:30:00-05:00", "Productivity and Costs (R) for Third Quarter 2026"),
    ("2026-12-10T08:30:00-05:00", "Consumer Price Index for November 2026"),
    ("2026-12-10T08:30:00-05:00", "Real Earnings for November 2026"),
    ("2026-12-15T08:30:00-05:00", "Producer Price Index for November 2026"),
    ("2026-12-17T08:30:00-05:00", "U.S. Import and Export Price Indexes for November 2026"),
]


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def finite(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def load_json(path):
    try:
        v = json.loads(Path(path).read_text(encoding="utf-8"))
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def request_json(url, timeout=12):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; LeanTrader-Evolution-Sidecar/1.34.0; +research-only)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


def request_json_any(url, timeout=12):
    """Read a public JSON endpoint that may return an object or an array."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; LeanTrader-Evolution-Sidecar/1.34.0; +research-only)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_text(url, timeout=12):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; LeanTrader-Evolution-Sidecar/1.34.0; +research-only)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def normalize_symbol(symbol):
    s = str(symbol or "").upper().strip()
    if "/" in s:
        return s
    return s[:-4] + "/USDT" if s.endswith("USDT") and len(s) > 4 else s


def bybit_symbol(symbol):
    return normalize_symbol(symbol).replace("/", "")


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def fabric_metrics(pack_id):
    state = load_json(RUNTIME / "vps_evolution_fabric.json")
    metrics = state.get("shadow_metrics") or {}
    row = metrics.get(pack_id) if isinstance(metrics, dict) else None
    return row if isinstance(row, dict) else {}


def calibration(pack_id):
    row = fabric_metrics(pack_id)
    samples = int(row.get("samples") or 0)
    avg = finite(row.get("average_net_return"))
    ewma = finite(row.get("ewma_net_return"))
    win = finite(row.get("win_rate"))
    hard = samples >= 12 and avg < -0.003 and ewma < 0.0
    cap = 0.72
    if samples >= 5:
        cap = clamp(0.50 + 8.0 * ewma + 0.18 * (win - 0.5), 0.25, 0.72)
    if hard:
        cap = min(cap, 0.35)
    return {
        "samples": samples,
        "average_net_return": avg,
        "ewma_net_return": ewma,
        "win_rate": win,
        "hard_negative": hard,
        "supportive": samples >= 20 and avg > 0 and ewma > 0,
        "confidence_cap": cap,
    }



def symbol_class(symbol):
    base = normalize_symbol(symbol).split("/", 1)[0]
    if base in MAJOR_ASSETS:
        return "major_crypto"
    if base in STABLE_ASSETS or base.startswith("USD"):
        return "stablecoin"
    if base == "XAUT":
        return "tokenized_commodity"
    if base.endswith("X") and len(base) >= 4:
        return "tokenized_equity_or_rwa"
    return "long_tail_crypto"


def liquidity_state(row):
    book = finite(row.get("book_notional"))
    trades = finite(row.get("trade_notional"))
    if book <= 0 and trades <= 0:
        return "unknown"
    if book >= 2_000_000 and trades >= 500_000:
        return "high"
    if book >= 250_000 and trades >= 50_000:
        return "medium"
    return "low"


def world_context_map():
    world = load_json(RUNTIME / "vps_market_world_model.json")
    latest = {}

    def walk(x, depth=0):
        if depth > 9:
            return
        if isinstance(x, dict):
            symbol = normalize_symbol(x.get("symbol") or "")
            regime = str(x.get("regime") or "").strip().lower()
            if symbol.endswith("/USDT") and regime:
                ts = finite(x.get("timestamp") or x.get("observed_at") or x.get("updated_at"))
                prior = latest.get(symbol) or {}
                if ts >= finite(prior.get("timestamp"), -1):
                    patterns = x.get("patterns") or x.get("latent_patterns") or []
                    if not isinstance(patterns, list):
                        patterns = [str(patterns)] if patterns else []
                    latest[symbol] = {
                        "regime": regime,
                        "novelty": clamp(finite(x.get("novelty")), 0, 1),
                        "rare_scope_score": clamp(finite(x.get("score")), 0, 1),
                        "patterns": [str(v) for v in patterns[:12]],
                        "timestamp": ts,
                    }
            for value in x.values():
                if isinstance(value, (dict, list)):
                    walk(value, depth + 1)
        elif isinstance(x, list):
            for value in x[-1500:]:
                if isinstance(value, (dict, list)):
                    walk(value, depth + 1)

    walk(world)
    return latest


def market_context(symbol, row, contexts):
    wc = dict((contexts or {}).get(normalize_symbol(symbol)) or {})
    patterns = [str(v) for v in wc.get("patterns") or []]
    novelty = clamp(finite(wc.get("novelty")), 0, 1)
    ood = "out_of_distribution_market_state" in patterns or novelty >= 0.80
    return {
        "regime": str(wc.get("regime") or "unknown"),
        "symbol_class": symbol_class(symbol),
        "liquidity_state": liquidity_state(row or {}),
        "out_of_distribution": bool(ood),
        "novelty": novelty,
        "rare_scope_score": clamp(finite(wc.get("rare_scope_score")), 0, 1),
        "patterns": patterns[:8],
    }


def current_spot_price(symbol):
    q = urllib.parse.urlencode({"category": "spot", "symbol": bybit_symbol(symbol)})
    payload = request_json(f"{BYBIT}/v5/market/tickers?{q}")
    if int(payload.get("retCode", -1)) != 0:
        return None
    rows = (payload.get("result") or {}).get("list") or []
    if not rows:
        return None
    price = finite(rows[0].get("lastPrice"))
    return price if price > 0 else None

def historical_spot_closes(symbol, opened_at, horizons=HORIZON_GRID):
    """Fetch one bounded 1-minute Bybit spot window and map horizon -> close.

    This is research-only historical market data. One request covers the full horizon
    grid, avoiding one API call per horizon. Returned candles are sorted by start time
    by us because Bybit returns kline rows newest-first.
    """
    horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    if not horizons:
        return {}
    start_ms = int(max(0.0, finite(opened_at)) * 1000)
    end_ms = int((max(0.0, finite(opened_at)) + max(horizons) + 180) * 1000)
    q = urllib.parse.urlencode({
        "category": "spot",
        "symbol": bybit_symbol(symbol),
        "interval": "1",
        "start": start_ms,
        "end": end_ms,
        "limit": min(1000, max(20, int(max(horizons) / 60) + 8)),
    })
    payload = request_json(f"{BYBIT}/v5/market/kline?{q}")
    if int(payload.get("retCode", -1)) != 0:
        return {}
    raw = (payload.get("result") or {}).get("list") or []
    candles = []
    for r in raw:
        if not isinstance(r, list) or len(r) < 5:
            continue
        ts = int(finite(r[0]))
        close = finite(r[4])
        if ts > 0 and close > 0:
            candles.append((ts, close))
    candles.sort()
    out = {}
    for horizon in horizons:
        target = int((finite(opened_at) + horizon) * 1000)
        eligible = [(ts, px) for ts, px in candles if ts <= target < ts + 120000]
        if eligible:
            out[horizon] = eligible[-1][1]
            continue
        near = min(candles, key=lambda x: abs(x[0] - target), default=None)
        if near and abs(near[0] - target) <= 120000:
            out[horizon] = near[1]
    return out


def metric_stats(rows):
    rows = list(rows or [])
    if not rows:
        return {
            "samples": 0,
            "wins": 0,
            "win_rate": 0.0,
            "average_net_return": 0.0,
            "ewma_net_return": 0.0,
            "negative_streak": 0,
            "average_inverse_net_return": 0.0,
            "average_edge_vs_inverse": 0.0,
            "falsification_warning": False,
        }
    net = [finite(r.get("net_return")) for r in rows]
    inv = [finite(r.get("inverse_net_return")) for r in rows]
    edge = [finite(r.get("edge_vs_inverse"), n - i) for r, n, i in zip(rows, net, inv)]
    wins = sum(v > 0 for v in net)
    ewma = 0.0
    for i, v in enumerate(net):
        ewma = v if i == 0 else 0.15 * v + 0.85 * ewma
    neg = 0
    for v in reversed(net):
        if v > 0:
            break
        neg += 1
    avg = sum(net) / len(net)
    inv_avg = sum(inv) / len(inv)
    edge_avg = sum(edge) / len(edge)
    return {
        "samples": len(net),
        "wins": wins,
        "win_rate": wins / len(net),
        "average_net_return": avg,
        "ewma_net_return": ewma,
        "negative_streak": neg,
        "average_inverse_net_return": inv_avg,
        "average_edge_vs_inverse": edge_avg,
        "falsification_warning": bool(len(net) >= 8 and avg < 0 and inv_avg > 0 and edge_avg < 0),
    }


def wilson_lower_bound(wins, samples, z=1.2815515655446004):
    """One-sided ~90% Wilson lower bound for Bernoulli win probability."""
    n = max(0, int(samples or 0))
    if n <= 0:
        return 0.0
    w = max(0, min(n, int(wins or 0)))
    phat = w / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    margin = z * math.sqrt(max(0.0, (phat * (1.0 - phat) + z2 / (4.0 * n)) / n))
    return clamp((center - margin) / denom, 0.0, 1.0)


def age_weighted_stats(rows, now=None):
    """Decay old specialist evidence without deleting it; never creates authority."""
    now = finite(now, time.time())
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    base = metric_stats(rows)
    if not rows:
        return {**base, "effective_samples": 0.0, "age_weighted_net_return": 0.0, "age_weighted_edge_vs_inverse": 0.0, "wilson_win_rate_lower": 0.0}
    weighted_net = weighted_edge = weight_sum = 0.0
    for r in rows:
        age = max(0.0, now - finite(r.get("closed_at") or r.get("opened_at"), now))
        weight = 0.5 ** (age / EVIDENCE_HALF_LIFE_SECONDS)
        weighted_net += weight * finite(r.get("net_return"))
        weighted_edge += weight * finite(r.get("edge_vs_inverse"))
        weight_sum += weight
    return {
        **base,
        "effective_samples": weight_sum,
        "age_weighted_net_return": weighted_net / weight_sum if weight_sum else 0.0,
        "age_weighted_edge_vs_inverse": weighted_edge / weight_sum if weight_sum else 0.0,
        "wilson_win_rate_lower": wilson_lower_bound(base.get("wins"), base.get("samples")),
    }


def evidence_lifecycle(rows, *, context=None, now=None, multiple_comparison_buckets=1):
    """Bounded research lifecycle. Positive evidence can nominate review, never auto-promote."""
    context = context or {}
    stats = age_weighted_stats(rows, now=now)
    n = int(stats.get("samples") or 0)
    ood = bool(context.get("out_of_distribution"))
    multiplicity_penalty = max(0, int(math.ceil(math.log2(max(1, int(multiple_comparison_buckets))))))
    required = MIN_GLOBAL_VALIDATION_SAMPLES + 2 * multiplicity_penalty
    if context:
        required = MIN_CONTEXT_VALIDATION_SAMPLES + 2 * multiplicity_penalty
    if ood:
        required = max(required + 8, 20)

    avg = finite(stats.get("average_net_return"))
    ewma = finite(stats.get("ewma_net_return"))
    aged = finite(stats.get("age_weighted_net_return"))
    edge = finite(stats.get("average_edge_vs_inverse"))
    aged_edge = finite(stats.get("age_weighted_edge_vs_inverse"))
    lower = finite(stats.get("wilson_win_rate_lower"))
    falsified = bool(stats.get("falsification_warning"))
    hard_negative = bool(
        falsified
        or (n >= 8 and avg < -0.003 and ewma < 0.0)
        or (n >= 12 and aged < -0.002 and aged_edge < 0.0)
        or int(stats.get("negative_streak") or 0) >= 6
    )

    validated = bool(
        n >= required
        and avg > 0.0
        and ewma > 0.0
        and aged > 0.0
        and edge > 0.0
        and aged_edge > 0.0
        and lower >= 0.45
        and not falsified
    )

    if hard_negative:
        lifecycle = "suppressed"
    elif n < 8:
        lifecycle = "insufficient_evidence"
    elif n < 20:
        lifecycle = "probation"
    elif n < required:
        lifecycle = "watch"
    elif validated:
        lifecycle = "research_validated"
    else:
        lifecycle = "degraded_or_unproven"

    return {
        "lifecycle": lifecycle,
        "research_validated": validated,
        "hard_negative": hard_negative,
        "required_samples": required,
        "multiple_comparison_penalty": multiplicity_penalty,
        "stats": stats,
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
        "promotion_requires_verified_core_release": True,
    }


def specialist_evidence_governance(state, now=None):
    now = finite(now, time.time())
    resolved = state.get("specialist_resolved") if isinstance(state.get("specialist_resolved"), list) else []
    out = {}
    for pack_id in DIRECTIONAL_PACKS:
        rows = [r for r in resolved[-MAX_SPECIALIST_RESOLVED:] if isinstance(r, dict) and r.get("pack_id") == pack_id]
        bucket_keys = {
            (str(r.get("regime") or "unknown"), str(r.get("symbol_class") or "unknown"), str(r.get("liquidity_state") or "unknown"), int(r.get("horizon_seconds") or 0))
            for r in rows
        }
        global_lifecycle = evidence_lifecycle(rows, now=now, multiple_comparison_buckets=max(1, len(bucket_keys)))
        contexts = {}
        for key in sorted(bucket_keys):
            regime, symbol_cls, liquidity, horizon = key
            subset = [
                r for r in rows
                if str(r.get("regime") or "unknown") == regime
                and str(r.get("symbol_class") or "unknown") == symbol_cls
                and str(r.get("liquidity_state") or "unknown") == liquidity
                and int(r.get("horizon_seconds") or 0) == horizon
            ]
            context = {
                "regime": regime,
                "symbol_class": symbol_cls,
                "liquidity_state": liquidity,
                "out_of_distribution": any(bool(r.get("out_of_distribution")) for r in subset),
            }
            contexts["|".join(map(str, key))] = evidence_lifecycle(
                subset,
                context=context,
                now=now,
                multiple_comparison_buckets=max(1, len(bucket_keys)),
            )
        out[pack_id] = {
            "global": global_lifecycle,
            "contexts": contexts,
            "context_bucket_count": len(bucket_keys),
            "execution_authority": False,
            "auto_promotion_allowed": False,
        }
    return out


def evidence_governance_observation(state, now):
    governance = specialist_evidence_governance(state, now=now)
    nominations = []
    suppressed = []
    for pack_id, row in governance.items():
        life = (row.get("global") or {}).get("lifecycle")
        if life == "research_validated":
            nominations.append(pack_id)
        if life == "suppressed":
            suppressed.append(pack_id)
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.98,
        "source": "LeanTrader sidecar evidence governance",
        "provenance": f"leantrader:specialist-evidence-governance:v1.14:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(900, INTERVAL * 4),
        "metadata": {
            "model": "bayesian_shrinkage_evidence_governance_v1.14",
            "directional_claim": False,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "evidence_half_life_seconds": EVIDENCE_HALF_LIFE_SECONDS,
            "research_validation_nominations": nominations,
            "suppressed_specialists": suppressed,
            "governance": governance,
            "execution_authority": False,
            "can_enable_live": False,
            "can_increase_risk": False,
            "auto_promotion_allowed": False,
            "promotion_requires_verified_core_release": True,
        },
    }

def _signal_context_lifecycle(evidence_gov, pack_id, row):
    """Return global/context lifecycle for a signal without granting authority."""
    pack_gov = evidence_gov.get(pack_id) if isinstance(evidence_gov, dict) else {}
    pack_gov = pack_gov if isinstance(pack_gov, dict) else {}
    global_life = str(((pack_gov.get("global") or {}).get("lifecycle")) or "insufficient_evidence")
    meta = row.get("metadata") if isinstance(row, dict) and isinstance(row.get("metadata"), dict) else {}
    ctx = meta.get("market_context") if isinstance(meta.get("market_context"), dict) else {}
    key = "|".join([
        str(ctx.get("regime") or "unknown"),
        str(ctx.get("symbol_class") or symbol_class(row.get("symbol") or "")),
        str(ctx.get("liquidity_state") or "unknown"),
        str(int(row.get("horizon_seconds") or 0)),
    ])
    context_row = (pack_gov.get("contexts") or {}).get(key) if isinstance(pack_gov.get("contexts"), dict) else None
    context_life = str((context_row or {}).get("lifecycle") or "unseen")
    return global_life, context_life, key


def apply_specialist_routing(packs, evidence_gov, now=None):
    """Quarantine suppressed specialist votes from council/diversity.

    Suppressed signals remain low-confidence research probes so the system can
    gather recovery evidence, but they cannot vote in the specialist council.
    This is deliberately asymmetric: routing can only preserve/reduce influence.
    """
    now = finite(now, time.time())
    decisions = []
    counts = {"council_eligible": 0, "recovery_probe": 0, "context_suppressed": 0, "global_suppressed": 0}
    for pack in packs or []:
        pid = str(pack.get("pack_id") or "")
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        for row in pack.get("observations") or []:
            if not isinstance(row, dict) or row.get("kind") != "signal":
                continue
            global_life, context_life, context_key = _signal_context_lifecycle(evidence_gov, pid, row)
            blocked_global = global_life == "suppressed"
            blocked_context = context_life == "suppressed"
            eligible = not (blocked_global or blocked_context)
            reason = "eligible"
            cap = 1.0
            if blocked_global:
                reason = "global_suppressed_recovery_probe"
                cap = 0.10
                counts["global_suppressed"] += 1
            elif blocked_context:
                reason = "context_suppressed_recovery_probe"
                cap = 0.15
                counts["context_suppressed"] += 1
            before = finite(row.get("confidence"))
            row["confidence"] = clamp(before * cap, 0.0, before)
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            meta.update({
                "pre_routing_confidence": before,
                "routing_confidence_cap_multiplier": cap,
                "council_eligible": eligible,
                "diversity_eligible": eligible,
                "recovery_probe": not eligible,
                "routing_reason": reason,
                "global_evidence_lifecycle": global_life,
                "context_evidence_lifecycle": context_life,
                "context_evidence_key": context_key,
                "routing_can_only_preserve_or_reduce_confidence": True,
            })
            row["metadata"] = meta
            counts["council_eligible" if eligible else "recovery_probe"] += 1
            decisions.append({
                "pack_id": pid,
                "symbol": normalize_symbol(row.get("symbol") or ""),
                "horizon_seconds": int(row.get("horizon_seconds") or 0),
                "council_eligible": eligible,
                "recovery_probe": not eligible,
                "global_lifecycle": global_life,
                "context_lifecycle": context_life,
                "reason": reason,
                "confidence_before": before,
                "confidence_after": finite(row.get("confidence")),
            })
    return {
        "model": "contextual_specialist_routing_governance_v1.14",
        "generated_at": now,
        "counts": counts,
        "decisions": decisions,
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "suppressed_signals_can_vote": False,
        "recovery_probes_are_research_only": True,
        "routing_can_only_preserve_or_reduce_confidence": True,
    }


def routing_observation(routing, now):
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.98,
        "source": "LeanTrader evidence lifecycle + contextual specialist router",
        "provenance": f"leantrader:specialist-routing:v1.14:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(900, INTERVAL * 4),
        "metadata": {
            "model": "contextual_specialist_routing_governance_v1.14",
            "directional_claim": False,
            "routing": routing,
            "execution_authority": False,
            "can_increase_risk": False,
        },
    }


def specialist_reputation(state, pack_id, context, horizon_seconds):
    resolved = state.get("specialist_resolved") if isinstance(state.get("specialist_resolved"), list) else []
    raw_rows = [r for r in resolved[-MAX_SPECIALIST_RESOLVED:] if isinstance(r, dict) and r.get("pack_id") == pack_id]
    rows = nonoverlapping_resolved_rows(raw_rows)
    global_stats = metric_stats(rows)
    dimensions = {
        "regime": metric_stats([r for r in rows if r.get("regime") == context.get("regime")]),
        "symbol_class": metric_stats([r for r in rows if r.get("symbol_class") == context.get("symbol_class")]),
        "liquidity": metric_stats([r for r in rows if r.get("liquidity_state") == context.get("liquidity_state")]),
        "horizon": metric_stats([r for r in rows if int(r.get("horizon_seconds") or 0) == int(horizon_seconds)]),
        "composite": metric_stats([
            r for r in rows
            if r.get("regime") == context.get("regime")
            and r.get("symbol_class") == context.get("symbol_class")
            and r.get("liquidity_state") == context.get("liquidity_state")
            and int(r.get("horizon_seconds") or 0) == int(horizon_seconds)
        ]),
    }
    weighted = []
    weights = {"regime": 0.28, "symbol_class": 0.16, "liquidity": 0.12, "horizon": 0.18, "composite": 0.36}
    for name, stats in dimensions.items():
        samples = int(stats.get("samples") or 0)
        if samples < 2:
            continue
        avg = finite(stats.get("average_net_return"))
        ewma = finite(stats.get("ewma_net_return"))
        win = finite(stats.get("win_rate"))
        quality = clamp(28.0 * avg + 22.0 * ewma + 0.55 * (win - 0.50))
        support = min(1.0, samples / 20.0)
        weighted.append((quality, weights[name] * support))
    if int(global_stats.get("samples") or 0) >= 3:
        avg = finite(global_stats.get("average_net_return"))
        ewma = finite(global_stats.get("ewma_net_return"))
        win = finite(global_stats.get("win_rate"))
        quality = clamp(24.0 * avg + 18.0 * ewma + 0.45 * (win - 0.50))
        weighted.append((quality, 0.22 * min(1.0, int(global_stats["samples"]) / 30.0)))
    total_w = sum(w for _, w in weighted)
    quality = sum(q * w for q, w in weighted) / total_w if total_w > 0 else 0.0
    hard_negative = bool(
        (int(global_stats.get("samples") or 0) >= 12 and finite(global_stats.get("average_net_return")) < -0.003 and finite(global_stats.get("ewma_net_return")) < 0)
        or bool(global_stats.get("falsification_warning"))
        or any(
            int(v.get("samples") or 0) >= 5
            and (
                (finite(v.get("average_net_return")) < -0.003 and finite(v.get("ewma_net_return")) < 0)
                or bool(v.get("falsification_warning"))
            )
            for v in dimensions.values()
        )
    )
    lifecycle = evidence_lifecycle(rows, context=context, now=time.time(), multiple_comparison_buckets=max(1, len(dimensions)))
    supportive = bool(lifecycle.get("research_validated"))
    multiplier = clamp(1.0 + 0.55 * min(0.0, quality), 0.45, 1.0)
    if hard_negative:
        multiplier = min(multiplier, 0.60)
    if context.get("out_of_distribution"):
        multiplier = min(multiplier, 0.75)
    return {
        "quality": quality,
        "confidence_multiplier": multiplier,
        "hard_negative": hard_negative,
        "supportive": supportive,
        "evidence_lifecycle": lifecycle,
        "global": global_stats,
        "raw_resolved_samples": len(raw_rows),
        "effective_nonoverlap_samples": len(rows),
        "overlap_discounted_samples": max(0, len(raw_rows) - len(rows)),
        "prequential_policy_uses_nonoverlap_effective_samples": True,
        "dimensions": dimensions,
        "context": dict(context),
        "risk_increase_allowed": False,
        "execution_authority": False,
    }



def _pearson(xs, ys):
    xs = [finite(v) for v in xs]
    ys = [finite(v) for v in ys]
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((v - mx) ** 2 for v in xs)
    vy = sum((v - my) ** 2 for v in ys)
    if vx <= 1e-18 or vy <= 1e-18:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return clamp(cov / math.sqrt(vx * vy))


def specialist_family(pack_id):
    if pack_id in {PACK_MICRO_5M, PACK_MICRO_15M}:
        return "microstructure"
    if pack_id == PACK_LEADLAG:
        return "causal_leadlag"
    if pack_id == PACK_CROSS_VENUE:
        return "cross_venue_microstructure"
    return str(pack_id or "unknown")


def _resolved_outcome_dependence(state):
    rows = state.get("specialist_resolved") if isinstance(state.get("specialist_resolved"), list) else []
    by_pack = {}
    for r in rows[-MAX_SPECIALIST_RESOLVED:]:
        if not isinstance(r, dict):
            continue
        pid = str(r.get("pack_id") or "")
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        # A coarse 15-minute event bucket prevents accidental millisecond matching
        # while keeping comparisons tied to the same symbol/time neighborhood.
        bucket = int(finite(r.get("opened_at")) // 900)
        key = (normalize_symbol(r.get("symbol") or ""), bucket)
        by_pack.setdefault(pid, {})[key] = finite(r.get("net_return"))
    packs = sorted(by_pack)
    out = []
    for i, left in enumerate(packs):
        for right in packs[i + 1:]:
            common = sorted(set(by_pack[left]) & set(by_pack[right]))
            if len(common) < 3:
                continue
            corr = _pearson([by_pack[left][k] for k in common], [by_pack[right][k] for k in common])
            out.append({
                "left": left,
                "right": right,
                "samples": len(common),
                "correlation": corr,
                "absolute_correlation": abs(corr),
                "same_family": specialist_family(left) == specialist_family(right),
            })
    return sorted(out, key=lambda r: (r["absolute_correlation"], r["samples"]), reverse=True)


def specialist_diversity_governance(state, packs, now=None):
    """Measure whether apparent specialist consensus is actually independent.

    This layer can only reduce advisory confidence. It never creates execution
    authority and never turns correlated votes into additional risk.
    """
    now = finite(now, time.time())
    signals = []
    for pack in packs or []:
        pid = str(pack.get("pack_id") or "")
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        for row in pack.get("observations") or []:
            if not isinstance(row, dict) or row.get("kind") != "signal":
                continue
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if meta.get("diversity_eligible") is False or meta.get("council_eligible") is False:
                continue
            score = finite(row.get("score"))
            if abs(score) < 0.05:
                continue
            signals.append({
                "pack_id": pid,
                "family": specialist_family(pid),
                "symbol": normalize_symbol(row.get("symbol") or ""),
                "direction": 1 if score > 0 else -1,
                "score": score,
                "confidence": finite(row.get("confidence")),
                "horizon_seconds": int(row.get("horizon_seconds") or 0),
            })

    caps = {pid: 1.0 for pid in DIRECTIONAL_PACK_SET}
    redundant = []
    conflicts = []
    for i, a in enumerate(signals):
        for b in signals[i + 1:]:
            if a["symbol"] != b["symbol"] or a["pack_id"] == b["pack_id"]:
                continue
            same_dir = a["direction"] == b["direction"]
            same_family = a["family"] == b["family"]
            if same_family and same_dir:
                # Same-family horizons share nearly the same underlying evidence.
                caps[a["pack_id"]] = min(caps[a["pack_id"]], 0.68)
                caps[b["pack_id"]] = min(caps[b["pack_id"]], 0.68)
                redundant.append({
                    "symbol": a["symbol"], "left": a["pack_id"], "right": b["pack_id"],
                    "reason": "same_family_same_direction", "advisory_pair_weight": 1.36,
                })
            elif same_dir and PACK_CROSS_VENUE in {a["pack_id"], b["pack_id"]} and ({a["pack_id"], b["pack_id"]} & {PACK_MICRO_5M, PACK_MICRO_15M}):
                # Cross-venue confirmation adds independent venue evidence, but it still
                # includes Bybit depth, so it is only partially independent from the
                # original Bybit microstructure family. Do not count it as a full extra vote.
                caps[a["pack_id"]] = min(caps[a["pack_id"]], 0.82)
                caps[b["pack_id"]] = min(caps[b["pack_id"]], 0.82)
                redundant.append({
                    "symbol": a["symbol"], "left": a["pack_id"], "right": b["pack_id"],
                    "reason": "shared_bybit_source_cross_venue_confirmation", "advisory_pair_weight": 1.64,
                })
            elif not same_dir:
                # Conflicting directional evidence should reduce confidence, never
                # choose a winner merely because more models repeat one family.
                caps[a["pack_id"]] = min(caps[a["pack_id"]], 0.72)
                caps[b["pack_id"]] = min(caps[b["pack_id"]], 0.72)
                conflicts.append({
                    "symbol": a["symbol"], "left": a["pack_id"], "right": b["pack_id"],
                    "left_direction": a["direction"], "right_direction": b["direction"],
                    "reason": "directional_disagreement",
                })

    dependence = _resolved_outcome_dependence(state)
    for row in dependence:
        if row["samples"] >= 5 and row["absolute_correlation"] >= 0.75:
            penalty = max(0.55, 1.0 - 0.35 * row["absolute_correlation"])
            caps[row["left"]] = min(caps.get(row["left"], 1.0), penalty)
            caps[row["right"]] = min(caps.get(row["right"], 1.0), penalty)

    by_symbol = {}
    for sig in signals:
        by_symbol.setdefault(sig["symbol"], []).append(sig)
    symbol_consensus = {}
    for symbol, rows in sorted(by_symbol.items()):
        weighted_direction = 0.0
        effective = 0.0
        families = set()
        for row in rows:
            cap = clamp(caps.get(row["pack_id"], 1.0), 0.0, 1.0)
            effective += cap
            weighted_direction += row["direction"] * cap * max(0.0, row["confidence"])
            families.add(row["family"])
        symbol_consensus[symbol] = {
            "raw_votes": len(rows),
            "independent_families": len(families),
            "effective_independent_votes": effective,
            "advisory_direction_score": clamp(weighted_direction / max(1e-9, effective)),
            "risk_increase_allowed": False,
        }

    return {
        "model": "specialist_independence_diversity_governance_v1.20",
        "generated_at": now,
        "active_directional_signals": len(signals),
        "active_directional_specialists": sorted({r["pack_id"] for r in signals}),
        "families": sorted({r["family"] for r in signals}),
        "advisory_weight_caps": {k: clamp(v, 0.0, 1.0) for k, v in sorted(caps.items())},
        "redundancy_clusters": redundant,
        "directional_conflicts": conflicts,
        "pairwise_outcome_dependence": dependence[:24],
        "symbol_consensus": symbol_consensus,
        "effective_independent_votes_total": sum(v["effective_independent_votes"] for v in symbol_consensus.values()),
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
        "diversity_can_only_preserve_or_reduce_confidence": True,
    }


def apply_diversity_caps(packs, governance):
    caps = governance.get("advisory_weight_caps") if isinstance(governance, dict) else {}
    caps = caps if isinstance(caps, dict) else {}
    for pack in packs or []:
        pid = str(pack.get("pack_id") or "")
        cap = clamp(caps.get(pid, 1.0), 0.0, 1.0)
        if cap >= 0.999999:
            continue
        for row in pack.get("observations") or []:
            if not isinstance(row, dict) or row.get("kind") != "signal":
                continue
            before = finite(row.get("confidence"))
            row["confidence"] = clamp(before * cap, 0.0, before)
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            meta["pre_diversity_confidence"] = before
            meta["diversity_confidence_cap_multiplier"] = cap
            meta["diversity_can_only_reduce_confidence"] = True
            row["metadata"] = meta
    return packs


def diversity_observation(state, packs, now):
    governance = specialist_diversity_governance(state, packs, now=now)
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.95,
        "source": "LeanTrader sidecar current signals + costed specialist outcome ledger",
        "provenance": f"leantrader:specialist-diversity:v1.20:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(900, INTERVAL * 4),
        "metadata": {
            "model": "specialist_independence_diversity_governance_v1.20",
            "directional_claim": False,
            "risk_increase_allowed": False,
            "governance": governance,
        },
    }


LIFECYCLE_COUNCIL_CAP = {
    "insufficient_evidence": 0.58,
    "probation": 0.68,
    "watch": 0.78,
    "research_validated": 0.92,
    "degraded_or_unproven": 0.48,
    "suppressed": 0.22,
}


def _macro_council_multiplier(macro_obs):
    meta = macro_obs.get("metadata") if isinstance(macro_obs, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    state = str(meta.get("risk_state") or "normal")
    return {
        "imminent": 0.45,
        "near": 0.62,
        "upcoming": 0.82,
        "post_release_absorption": 0.72,
        "normal": 1.0,
    }.get(state, 0.88), state


def _council_directional_alpha_policy(rows):
    """Separate council directional-alpha evidence from its risk-overlay value.

    Negative directional expectancy must not be masked by useful reduce/veto decisions.
    This policy can only disable shadow support; it never creates execution authority.
    """
    stats = metric_stats(rows or [])
    n = int(stats.get("samples") or 0)
    avg = finite(stats.get("average_net_return"))
    ewma = finite(stats.get("ewma_net_return"))
    win = finite(stats.get("win_rate"))
    edge = finite(stats.get("average_edge_vs_inverse"))
    if n < 8:
        lifecycle = "insufficient_evidence"
    elif n >= 10 and avg < 0.0 and ewma < 0.0:
        lifecycle = "directional_degraded"
    elif n < 20:
        lifecycle = "probation"
    elif n < 40:
        lifecycle = "watch"
    elif avg > 0.0 and ewma > 0.0 and edge > 0.0 and win >= 0.45:
        lifecycle = "research_positive"
    else:
        lifecycle = "degraded_or_unproven"
    support_allowed = lifecycle == "research_positive"
    return {
        "lifecycle": lifecycle,
        "stats": stats,
        "support_shadow_research_allowed": support_allowed,
        "required_positive_samples": 40,
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }


def _risk_policy_summary(summary, minimum_samples=8):
    summary = summary if isinstance(summary, dict) else {}
    n = int(summary.get("samples") or 0)
    helpful = finite(summary.get("helpful_rate"))
    value = finite(summary.get("average_decision_value_vs_baseline"))
    if n < int(minimum_samples):
        lifecycle = "insufficient_evidence"
    elif value > 0.0 and helpful >= 0.60:
        lifecycle = "provisional_positive"
    elif value < 0.0 and helpful < 0.50:
        lifecycle = "degraded"
    else:
        lifecycle = "watch"
    return {
        "lifecycle": lifecycle,
        "minimum_samples": int(minimum_samples),
        "stats": dict(summary),
        "execution_authority": False,
        "can_increase_risk": False,
    }


def council_role_policy(state, now=None):
    """Govern council *roles* independently: directional alpha vs risk overlay.

    A council may be poor at predicting direction while still adding value by reducing
    exposure or vetoing bad candidates. The two jobs therefore have separate ledgers
    and lifecycles. No positive result grants execution/live/risk authority.
    """
    _ = finite(now, time.time())
    alpha_rows = state.get("council_resolved") if isinstance(state.get("council_resolved"), list) else []
    cf_rows = state.get("council_counterfactual_resolved") if isinstance(state.get("council_counterfactual_resolved"), list) else []
    cf_metrics = council_counterfactual_metrics(cf_rows)
    alpha = _council_directional_alpha_policy(alpha_rows)
    risk_global = _risk_policy_summary(cf_metrics.get("global") or {}, minimum_samples=8)
    action_minimums = {"reduce": 8, "veto_recommended": 5, "support_shadow": 8, "abstain": 8}
    actions = {}
    for action, summary in (cf_metrics.get("actions") or {}).items():
        actions[action] = _risk_policy_summary(summary, minimum_samples=action_minimums.get(action, 8))
    return {
        "model": "council_role_separation_policy_v1.20",
        "directional_alpha": alpha,
        "risk_overlay": risk_global,
        "actions": actions,
        "directional_and_risk_evidence_are_separate": True,
        "negative_alpha_cannot_be_hidden_by_helpful_risk_actions": True,
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
        "promotion_requires_verified_core_release": True,
    }


def council_role_policy_observation(state, now):
    policy = council_role_policy(state, now=now)
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.98,
        "source": "LeanTrader council directional ledger + council risk counterfactual ledger",
        "provenance": f"leantrader:council-role-policy:v1.14:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(900, INTERVAL * 4),
        "metadata": {
            "model": "council_role_separation_policy_v1.20",
            "directional_claim": False,
            "policy": policy,
            "execution_authority": False,
            "risk_increase_allowed": False,
        },
    }



def _percentile_tail_mean(values, fraction=0.25):
    vals = sorted(finite(v) for v in values)
    if not vals:
        return 0.0
    k = max(1, int(math.ceil(len(vals) * clamp(fraction, 0.05, 1.0))))
    return sum(vals[:k]) / k


def _median(values, default=0.55):
    vals = sorted(finite(v) for v in values if math.isfinite(finite(v)))
    if not vals:
        return float(default)
    n = len(vals)
    m = n // 2
    return vals[m] if n % 2 else 0.5 * (vals[m - 1] + vals[m])


def _risk_fraction_curve(rows):
    """Counterfactual exposure curve for council reduce decisions.

    v1.14 explicitly includes 0% exposure. This closes a research blind spot from
    earlier versions: when every positive exposure fraction has negative after-cost
    expectancy, the optimizer must be able to say "do not deploy capital" rather
    than falsely calling the smallest non-zero fraction optimal.

    This remains research-only. It never increases the current council cap and can
    only tighten a reduce action or convert it to a shadow veto recommendation.
    """
    rows = [r for r in (rows or []) if isinstance(r, dict) and str(r.get('action')) == 'reduce']
    baseline = [finite(r.get('baseline_net_return')) for r in rows]
    current = [clamp(finite(r.get('decision_exposure_fraction'), 0.55), 0.0, 1.0) for r in rows]
    n = len(rows)
    prior = clamp(_median(current, 0.55), 0.15, 1.0)
    curve = {}
    for f in RISK_FRACTION_GRID:
        f = clamp(f, 0.0, 1.0)
        decision = [f * r for r in baseline]
        value = [(f - 1.0) * r for r in baseline]
        wins = sum(v > 0 for v in decision)
        avg = sum(decision) / n if n else 0.0
        avg_value = sum(value) / n if n else 0.0
        cvar25 = _percentile_tail_mean(decision, 0.25) if n else 0.0
        robust = avg + 0.35 * min(0.0, cvar25)
        curve[f'{f:.2f}'] = {
            'exposure_fraction': f,
            'average_net_return': avg,
            'average_value_vs_full_baseline': avg_value,
            'win_rate': (wins / n) if n else 0.0,
            'cvar25_net_return': cvar25,
            'robust_score': robust,
        }

    if curve:
        # Ties always choose less capital. Zero exposure is a legitimate research
        # outcome, not a synthetic profit claim.
        best = max(curve.values(), key=lambda r: (finite(r.get('robust_score')), -finite(r.get('exposure_fraction'))))
        best_fraction = finite(best.get('exposure_fraction'), prior)
    else:
        best_fraction = prior

    base_summary = council_counterfactual_metrics(rows).get('global') if rows else {}
    helpful = finite((base_summary or {}).get('helpful_rate'))
    value = finite((base_summary or {}).get('average_decision_value_vs_baseline'))
    baseline_avg = finite((base_summary or {}).get('average_baseline_net_return'))
    baseline_win = finite((base_summary or {}).get('baseline_win_rate'))

    positive_curve = [r for r in curve.values() if finite(r.get('exposure_fraction')) > 0.0]
    all_positive_avg_nonpositive = bool(positive_curve) and all(finite(r.get('average_net_return')) <= 0.0 for r in positive_curve)
    all_positive_robust_nonpositive = bool(positive_curve) and all(finite(r.get('robust_score')) <= 0.0 for r in positive_curve)
    zero_dominates = bool(curve.get('0.00')) and best_fraction <= 1e-12

    no_trade_candidate = bool(
        n >= MIN_NO_TRADE_FRONTIER_SAMPLES
        and helpful >= 0.65
        and value > 0.0
        and baseline_avg < 0.0
        and baseline_win <= 0.40
        and all_positive_avg_nonpositive
        and all_positive_robust_nonpositive
        and zero_dominates
    )

    eligible = n >= MIN_RISK_FRACTION_SAMPLES and helpful >= 0.60 and value > 0.0
    if no_trade_candidate:
        recommended = 0.0
        lifecycle = 'no_trade_frontier_provisional' if n < 40 else 'no_trade_frontier_evidence_mature'
    elif eligible and best_fraction < prior:
        strength = clamp((n - MIN_RISK_FRACTION_SAMPLES + 1) / 32.0, 0.15, 1.0)
        recommended = prior + strength * (best_fraction - prior)
        recommended = clamp(recommended, 0.0, prior)
        lifecycle = 'calibrating' if n < 20 else ('provisional_positive' if n < 40 else 'evidence_mature')
    else:
        recommended = prior
        lifecycle = 'insufficient_evidence' if n < MIN_RISK_FRACTION_SAMPLES else ('watch' if value >= 0.0 else 'degraded')

    return {
        'samples': n,
        'observed_median_exposure_fraction': prior,
        'best_shadow_fraction': clamp(best_fraction, 0.0, 1.0),
        'recommended_exposure_ceiling': clamp(recommended, 0.0, 1.0),
        'policy_applies': bool((eligible and recommended < prior - 1e-12) or no_trade_candidate),
        'lifecycle': lifecycle,
        'helpful_rate': helpful,
        'average_decision_value_vs_baseline': value,
        'baseline_average_net_return': baseline_avg,
        'baseline_win_rate': baseline_win,
        'all_positive_exposures_negative_after_cost': all_positive_avg_nonpositive,
        'all_positive_exposures_negative_robust_score': all_positive_robust_nonpositive,
        'no_trade_shadow_candidate': no_trade_candidate,
        'zero_exposure_is_capital_preservation_not_profit': True,
        'curve': curve,
        'execution_authority': False,
        'can_enable_live': False,
        'can_increase_risk': False,
        'auto_promotion_allowed': False,
    }

def council_risk_fraction_policy(state, now=None):
    """Learn a conservative ceiling for *reduce* actions from counterfactuals.

    The optimizer is monotone: it may preserve or lower the council's existing
    exposure cap, never raise it. Context policies are permitted only after their
    own minimum sample threshold; otherwise the global policy is used.
    """
    _ = finite(now, time.time())
    rows = state.get('council_counterfactual_resolved') if isinstance(state.get('council_counterfactual_resolved'), list) else []
    reduce_rows = [r for r in rows if isinstance(r, dict) and str(r.get('action')) == 'reduce']
    contexts = {}
    grouped = {}
    for r in reduce_rows:
        macro = str(r.get('macro_risk_state') or 'unknown')
        ood = 'true' if bool(r.get('out_of_distribution')) else 'false'
        grouped.setdefault(f'macro:{macro}', []).append(r)
        grouped.setdefault(f'ood:{ood}', []).append(r)
    for key, subset in sorted(grouped.items()):
        contexts[key] = _risk_fraction_curve(subset)
    global_policy = _risk_fraction_curve(reduce_rows)
    return {
        'model': 'counterfactual_risk_fraction_policy_v1.20',
        'global': global_policy,
        'contexts': contexts,
        'candidate_exposure_grid': list(RISK_FRACTION_GRID),
        'zero_exposure_candidate_enabled': True,
        'minimum_no_trade_frontier_samples': MIN_NO_TRADE_FRONTIER_SAMPLES,
        'policy_is_monotone_nonincreasing': True,
        'execution_authority': False,
        'can_enable_live': False,
        'can_increase_risk': False,
        'auto_promotion_allowed': False,
        'promotion_requires_verified_core_release': True,
    }


def apply_council_risk_fraction_policy(council, policy):
    """Apply only tighter research caps to current reduce actions."""
    council = council if isinstance(council, dict) else {}
    policy = policy if isinstance(policy, dict) else {}
    gp = policy.get('global') if isinstance(policy.get('global'), dict) else {}
    contexts = policy.get('contexts') if isinstance(policy.get('contexts'), dict) else {}
    applied = []
    considered = 0
    for symbol, row in (council.get('symbols') or {}).items():
        if not isinstance(row, dict) or str(row.get('action')) != 'reduce':
            continue
        considered += 1
        before = clamp(finite(row.get('confidence_cap_multiplier'), 1.0), 0.0, 1.0)
        candidate_ceilings = []
        sources = []
        if gp.get('policy_applies'):
            candidate_ceilings.append(clamp(finite(gp.get('recommended_exposure_ceiling'), before), 0.0, 1.0))
            sources.append('global')
        macro_key = f"macro:{str(row.get('macro_risk_state') or 'unknown')}"
        ood_key = f"ood:{'true' if bool(row.get('out_of_distribution')) else 'false'}"
        for key in (macro_key, ood_key):
            cp = contexts.get(key) if isinstance(contexts.get(key), dict) else {}
            if cp.get('policy_applies'):
                candidate_ceilings.append(clamp(finite(cp.get('recommended_exposure_ceiling'), before), 0.0, 1.0))
                sources.append(key)
        learned = min(candidate_ceilings) if candidate_ceilings else before
        after = min(before, learned)  # invariant: never increase risk
        row['risk_fraction_policy'] = {
            'before_cap': before,
            'after_cap': after,
            'policy_sources': sources,
            'monotone_nonincreasing': True,
        }
        if after < before - 1e-12:
            row['confidence_cap_multiplier'] = after
            row.setdefault('reasons', []).append('counterfactual_risk_fraction_tightening')
            applied.append({'symbol': symbol, 'before_cap': before, 'after_cap': after, 'sources': sources})
    council['advisory_symbol_caps'] = {s: clamp(finite(r.get('confidence_cap_multiplier'), 1.0), 0.0, 1.0) for s, r in (council.get('symbols') or {}).items() if isinstance(r, dict)}
    return {
        'model': 'counterfactual_risk_fraction_application_v1.20',
        'reduce_actions_considered': considered,
        'caps_tightened': len(applied),
        'applications': applied,
        'execution_authority': False,
        'can_enable_live': False,
        'can_increase_risk': False,
        'policy_can_only_preserve_or_reduce_exposure': True,
    }



def council_no_trade_frontier_policy(risk_fraction_policy):
    """Extract a strict zero-exposure research frontier from the risk curve.

    A no-trade recommendation is permitted only when the resolved reduce ledger is
    sufficiently large and *every* positive exposure candidate remains negative
    after modeled costs. This does not touch execution authority.
    """
    policy = risk_fraction_policy if isinstance(risk_fraction_policy, dict) else {}
    gp = policy.get('global') if isinstance(policy.get('global'), dict) else {}
    contexts = policy.get('contexts') if isinstance(policy.get('contexts'), dict) else {}
    qualifying_contexts = sorted(k for k,v in contexts.items() if isinstance(v, dict) and v.get('no_trade_shadow_candidate'))
    applies = bool(gp.get('no_trade_shadow_candidate'))
    return {
        'model': 'counterfactual_no_trade_frontier_policy_v1.20',
        'global_no_trade_shadow_candidate': applies,
        'qualifying_contexts': qualifying_contexts,
        'samples': int(finite(gp.get('samples'))),
        'best_shadow_fraction': finite(gp.get('best_shadow_fraction')),
        'recommended_exposure_ceiling': finite(gp.get('recommended_exposure_ceiling'), 1.0),
        'baseline_average_net_return': finite(gp.get('baseline_average_net_return')),
        'baseline_win_rate': finite(gp.get('baseline_win_rate')),
        'all_positive_exposures_negative_after_cost': bool(gp.get('all_positive_exposures_negative_after_cost')),
        'all_positive_exposures_negative_robust_score': bool(gp.get('all_positive_exposures_negative_robust_score')),
        'zero_exposure_is_capital_preservation_not_profit': True,
        'execution_authority': False,
        'can_enable_live': False,
        'can_increase_risk': False,
        'auto_promotion_allowed': False,
        'promotion_requires_verified_core_release': True,
    }


def apply_no_trade_frontier_policy(council, frontier_policy):
    """Convert eligible shadow reduce actions to zero-exposure veto research.

    This transformation is advisory only and strictly decreases exposure. It is
    intentionally performed after the risk-fraction cap application so the audit
    trail contains both the learned cap and the final no-trade decision.
    """
    council = council if isinstance(council, dict) else {}
    frontier_policy = frontier_policy if isinstance(frontier_policy, dict) else {}
    global_applies = bool(frontier_policy.get('global_no_trade_shadow_candidate'))
    qualifying_contexts = set(frontier_policy.get('qualifying_contexts') or [])
    applied = []
    considered = 0
    for symbol,row in (council.get('symbols') or {}).items():
        if not isinstance(row, dict) or str(row.get('action')) != 'reduce':
            continue
        considered += 1
        macro_key = f"macro:{str(row.get('macro_risk_state') or 'unknown')}"
        ood_key = f"ood:{'true' if bool(row.get('out_of_distribution')) else 'false'}"
        context_applies = macro_key in qualifying_contexts or ood_key in qualifying_contexts
        if not (global_applies or context_applies):
            continue
        before = clamp(finite(row.get('confidence_cap_multiplier'), 1.0), 0.0, 1.0)
        row['pre_no_trade_action'] = 'reduce'
        row['action'] = 'veto_recommended'
        row['confidence_cap_multiplier'] = 0.0
        row.setdefault('reasons', []).append('counterfactual_no_trade_frontier')
        row['no_trade_frontier'] = {
            'before_cap': before,
            'after_cap': 0.0,
            'global_applies': global_applies,
            'context_applies': context_applies,
            'zero_exposure_is_capital_preservation_not_profit': True,
        }
        applied.append({'symbol': symbol, 'before_cap': before, 'after_cap': 0.0, 'action_after': 'veto_recommended'})
    council['advisory_symbol_caps'] = {s: clamp(finite(r.get('confidence_cap_multiplier'), 1.0), 0.0, 1.0) for s,r in (council.get('symbols') or {}).items() if isinstance(r, dict)}
    return {
        'model': 'counterfactual_no_trade_frontier_application_v1.20',
        'reduce_actions_considered': considered,
        'reduce_to_veto_shadow': len(applied),
        'applications': applied,
        'execution_authority': False,
        'can_enable_live': False,
        'can_increase_risk': False,
        'policy_can_only_reduce_exposure_to_zero': True,
    }


def council_no_trade_frontier_observation(frontier_policy, application, now):
    return {
        'symbol': 'GLOBAL',
        'kind': 'context',
        'score': 0.0,
        'confidence': 0.99,
        'source': 'LeanTrader zero-exposure counterfactual frontier',
        'provenance': f'leantrader:council-no-trade-frontier:v1.14:{int(now)}',
        'observed_at': now,
        'horizon_seconds': max(900, INTERVAL * 4),
        'metadata': {
            'model': 'counterfactual_no_trade_frontier_policy_v1.20',
            'directional_claim': False,
            'policy': frontier_policy,
            'application': application,
            'execution_authority': False,
            'risk_increase_allowed': False,
        },
    }

def council_risk_fraction_policy_observation(state, now):
    policy = council_risk_fraction_policy(state, now=now)
    return {
        'symbol': 'GLOBAL',
        'kind': 'context',
        'score': 0.0,
        'confidence': 0.98,
        'source': 'LeanTrader council reduce counterfactual exposure curve',
        'provenance': f'leantrader:council-risk-fraction:v1.14:{int(now)}',
        'observed_at': now,
        'horizon_seconds': max(900, INTERVAL * 4),
        'metadata': {
            'model': 'counterfactual_risk_fraction_policy_v1.20',
            'directional_claim': False,
            'policy': policy,
            'execution_authority': False,
            'risk_increase_allowed': False,
        },
    }


def _horizon_lab_episode_id(source_row, horizon_seconds):
    source_id = str(source_row.get('episode_id') or '')
    if not source_id:
        raw = '|'.join([
            str(source_row.get('pack_id') or ''),
            normalize_symbol(source_row.get('symbol') or ''),
            f"{finite(source_row.get('opened_at')):.3f}",
            f"{finite(source_row.get('entry_price')):.12g}",
        ])
        source_id = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]
    return hashlib.sha256(f"{source_id}|horizon|{int(horizon_seconds)}".encode('utf-8')).hexdigest()[:24]


def backfill_horizon_lab(state, now=None):
    """Backfill alternate hold horizons from one bounded Bybit kline request/episode.

    Existing specialist outcomes are never rewritten. This creates a separate,
    research-only counterfactual ledger so a losing 5m/15m signal can be tested at
    30m/60m without changing the deployed strategy or execution semantics.
    """
    now = finite(now, time.time())
    source_rows = state.get('specialist_resolved') if isinstance(state.get('specialist_resolved'), list) else []
    lab = state.get('horizon_lab_resolved') if isinstance(state.get('horizon_lab_resolved'), list) else []
    attempts = state.get('horizon_backfill_attempts') if isinstance(state.get('horizon_backfill_attempts'), dict) else {}
    existing = {str(r.get('horizon_lab_id') or '') for r in lab if isinstance(r, dict)}
    queries = added = unavailable = 0
    errors = []
    cost = ROUND_TRIP_COST_BPS / 10_000.0

    for row in reversed(source_rows[-MAX_SPECIALIST_RESOLVED:]):
        if queries >= HORIZON_BACKFILL_PER_CYCLE:
            break
        if not isinstance(row, dict) or str(row.get('pack_id') or '') not in DIRECTIONAL_PACK_SET:
            continue
        opened = finite(row.get('opened_at'))
        entry = finite(row.get('entry_price'))
        signal_value = finite(row.get('signal'))
        symbol = normalize_symbol(row.get('symbol') or '')
        if opened <= 0 or entry <= 0 or abs(signal_value) < 0.05 or not symbol:
            continue
        eligible_horizons = [h for h in HORIZON_GRID if opened + h <= now - 90]
        missing = [h for h in eligible_horizons if _horizon_lab_episode_id(row, h) not in existing]
        if not missing:
            continue
        source_key = str(row.get('episode_id') or hashlib.sha256(f"{row.get('pack_id')}|{symbol}|{opened}".encode()).hexdigest()[:24])
        if int(attempts.get(source_key) or 0) >= 3:
            continue
        queries += 1
        try:
            closes = historical_spot_closes(symbol, opened, horizons=missing)
        except Exception as exc:
            attempts[source_key] = int(attempts.get(source_key) or 0) + 1
            errors.append(f"{symbol}:{type(exc).__name__}:{exc}")
            continue
        if not closes:
            attempts[source_key] = int(attempts.get(source_key) or 0) + 1
            unavailable += 1
            continue
        # A successful partial query is not a source failure. Future cycles can fill
        # horizons that were not yet available when this query ran.
        attempts[source_key] = 0
        direction = 1.0 if signal_value > 0 else -1.0
        for horizon, exit_price in sorted(closes.items()):
            hid = _horizon_lab_episode_id(row, horizon)
            if hid in existing or finite(exit_price) <= 0:
                continue
            gross = direction * (finite(exit_price) / entry - 1.0)
            net = gross - cost
            lab.append({
                'horizon_lab_id': hid,
                'source_episode_id': str(row.get('episode_id') or ''),
                'pack_id': str(row.get('pack_id') or ''),
                'symbol': symbol,
                'signal': signal_value,
                'confidence': finite(row.get('confidence')),
                'entry_price': entry,
                'exit_price': finite(exit_price),
                'opened_at': opened,
                'closed_at': opened + int(horizon),
                'horizon_seconds': int(horizon),
                'regime': str(row.get('regime') or 'unknown'),
                'symbol_class': str(row.get('symbol_class') or symbol_class(symbol)),
                'liquidity_state': str(row.get('liquidity_state') or 'unknown'),
                'out_of_distribution': bool(row.get('out_of_distribution')),
                'recovery_probe': bool(row.get('recovery_probe')),
                'gross_return': gross,
                'net_return': net,
                'inverse_net_return': -gross - cost,
                'edge_vs_inverse': 2.0 * gross,
                'profitable_after_cost': net > 0,
                'round_trip_cost_bps': ROUND_TRIP_COST_BPS,
                'evidence_authority': 'sidecar_multi_horizon_shadow_only',
                'execution_authority': False,
            })
            existing.add(hid)
            added += 1
    if len(lab) > MAX_HORIZON_LAB_RESOLVED:
        lab = lab[-MAX_HORIZON_LAB_RESOLVED:]
    state['horizon_lab_resolved'] = lab
    state['horizon_backfill_attempts'] = attempts
    return {
        'queries_this_cycle': queries,
        'rows_added_this_cycle': added,
        'source_unavailable_this_cycle': unavailable,
        'errors': errors[:12],
        'resolved_rows': len(lab),
        'execution_authority': False,
    }


def _horizon_metric_stats(rows):
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    n = len(rows)
    base = metric_stats(rows)
    gross = [finite(r.get('gross_return')) for r in rows]
    net = [finite(r.get('net_return')) for r in rows]
    avg_gross = sum(gross) / n if n else 0.0
    cvar25 = _percentile_tail_mean(net, 0.25) if n else 0.0
    cost = ROUND_TRIP_COST_BPS / 10_000.0
    coverage = avg_gross / cost if cost > 1e-12 else 999.0
    robust = finite(base.get('average_net_return')) + 0.35 * min(0.0, cvar25)
    return {
        **base,
        'average_gross_return': avg_gross,
        'cost_coverage_ratio': coverage,
        'cvar25_net_return': cvar25,
        'robust_net_edge_score': robust,
    }



def _prequential_direction_stats(rows, use_inverse=False):
    """Chronological train/holdout evidence for one pack/horizon direction.

    Rows are never shuffled.  The latest observations form a true holdout so a
    hypothesis cannot validate itself on the same outcomes used to discover it.
    """
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    rows = sorted(rows, key=lambda r: (finite(r.get('opened_at')), str(r.get('horizon_lab_id') or '')))
    n = len(rows)
    holdout_n = max(MIN_FALSIFICATION_HOLDOUT_SAMPLES, int(math.ceil(n * 0.30))) if n else 0
    if n >= MIN_FALSIFICATION_TOTAL_SAMPLES:
        holdout_n = min(holdout_n, max(MIN_FALSIFICATION_HOLDOUT_SAMPLES, n - 4))
    else:
        holdout_n = min(holdout_n, n)
    split = max(0, n - holdout_n)
    train_rows = rows[:split]
    hold_rows = rows[split:]

    def project(rs):
        out=[]
        for r in rs:
            orig=finite(r.get('net_return'))
            inv=finite(r.get('inverse_net_return'))
            chosen=inv if use_inverse else orig
            other=orig if use_inverse else inv
            out.append({
                **r,
                'net_return': chosen,
                'inverse_net_return': other,
                'edge_vs_inverse': chosen - other,
            })
        return out

    tr=project(train_rows)
    ho=project(hold_rows)
    allp=project(rows)
    tr_stats=_horizon_metric_stats(tr)
    ho_stats=_horizon_metric_stats(ho)
    all_stats=_horizon_metric_stats(allp)
    return {
        'samples': n,
        'train_samples': len(tr),
        'holdout_samples': len(ho),
        'train': tr_stats,
        'holdout': ho_stats,
        'all': all_stats,
    }


def prequential_falsification_policy(state, now=None):
    """Test whether a directional specialist is persistently useful or inverted.

    This is a falsification lab, not an auto-reversal engine.  A direction may be
    called `inverse_shadow_candidate` only when the inverse is positive on both
    chronological discovery data and the newer holdout, while the original
    direction remains negative on holdout after the same modeled costs.
    """
    _ = finite(now, time.time())
    rows = state.get('horizon_lab_resolved') if isinstance(state.get('horizon_lab_resolved'), list) else []
    by_pack={}
    inverse_candidates=[]
    original_candidates=[]
    watches=[]
    for pid in DIRECTIONAL_PACKS:
        prow=[r for r in rows if isinstance(r, dict) and r.get('pack_id') == pid]
        ph={}
        for h in HORIZON_GRID:
            hrs=[r for r in prow if int(r.get('horizon_seconds') or 0) == h]
            original=_prequential_direction_stats(hrs, use_inverse=False)
            inverse=_prequential_direction_stats(hrs, use_inverse=True)
            n=int(original.get('samples') or 0)
            lifecycle='insufficient_prequential_evidence'
            inverse_ok=False
            original_ok=False
            if n >= MIN_FALSIFICATION_TOTAL_SAMPLES and int(original.get('holdout_samples') or 0) >= MIN_FALSIFICATION_HOLDOUT_SAMPLES:
                ot=original['train']; oh=original['holdout']
                it=inverse['train']; ih=inverse['holdout']
                inverse_ok=bool(
                    finite(it.get('average_net_return')) > 0.0
                    and finite(ih.get('average_net_return')) > 0.0
                    and finite(ih.get('ewma_net_return')) > 0.0
                    and finite(ih.get('robust_net_edge_score')) > 0.0
                    and finite(ih.get('win_rate')) >= 0.50
                    and finite(oh.get('average_net_return')) < 0.0
                )
                original_ok=bool(
                    finite(ot.get('average_net_return')) > 0.0
                    and finite(oh.get('average_net_return')) > 0.0
                    and finite(oh.get('ewma_net_return')) > 0.0
                    and finite(oh.get('robust_net_edge_score')) > 0.0
                    and finite(oh.get('win_rate')) >= 0.50
                )
                if inverse_ok:
                    lifecycle='inverse_shadow_candidate'
                    inverse_candidates.append({'pack_id':pid,'horizon_seconds':h})
                elif original_ok:
                    lifecycle='original_direction_holdout_positive'
                    original_candidates.append({'pack_id':pid,'horizon_seconds':h})
                else:
                    lifecycle='prequentially_unproven_or_negative'
            else:
                # A watch is informative but has zero routing authority.  It exposes
                # patterns like the current lead/lag inversion without prematurely
                # converting them into a challenger.
                inv_all=inverse.get('all') or {}
                orig_all=original.get('all') or {}
                if n >= max(4, MIN_FALSIFICATION_HOLDOUT_SAMPLES) and finite(inv_all.get('average_net_return')) > 0.0 and finite(orig_all.get('average_net_return')) < 0.0:
                    lifecycle='inverse_watch_insufficient_holdout'
                    watches.append({'pack_id':pid,'horizon_seconds':h,'samples':n})
            ph[str(h)]={
                'lifecycle': lifecycle,
                'original': original,
                'inverse': inverse,
                'inverse_shadow_candidate': inverse_ok,
                'original_direction_holdout_positive': original_ok,
                'execution_authority': False,
                'can_increase_risk': False,
                'auto_promotion_allowed': False,
            }
        by_pack[pid]={'horizons':ph,'execution_authority':False,'can_increase_risk':False,'auto_promotion_allowed':False}
    return {
        'model':'prequential_directional_falsification_policy_v1.20',
        'minimum_total_samples':MIN_FALSIFICATION_TOTAL_SAMPLES,
        'minimum_holdout_samples':MIN_FALSIFICATION_HOLDOUT_SAMPLES,
        'by_pack':by_pack,
        'inverse_shadow_candidates':inverse_candidates,
        'original_direction_holdout_positive':original_candidates,
        'inverse_watches':watches,
        'execution_authority':False,
        'can_enable_live':False,
        'can_increase_risk':False,
        'auto_promotion_allowed':False,
        'promotion_requires_verified_core_release':True,
        'same_outcomes_cannot_both_discover_and_validate':True,
    }


def apply_prequential_falsification_gate(packs, policy):
    """Remove a validated-wrong original direction from council/diversity voting.

    Even when the inverse is validated in shadow, the original signal is only
    reduced/blocked here; the inverse never inherits execution or council authority.
    """
    by_pack=policy.get('by_pack') if isinstance(policy,dict) else {}
    by_pack=by_pack if isinstance(by_pack,dict) else {}
    counts={'considered':0,'original_falsified_blocked':0,'inverse_candidate_available':0}
    decisions=[]
    for pack in packs or []:
        pid=str(pack.get('pack_id') or '')
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        pp=by_pack.get(pid) if isinstance(by_pack.get(pid),dict) else {}
        hsmap=pp.get('horizons') if isinstance(pp.get('horizons'),dict) else {}
        for row in pack.get('observations') or []:
            if not isinstance(row,dict) or row.get('kind') != 'signal':
                continue
            meta=row.get('metadata') if isinstance(row.get('metadata'),dict) else {}
            h=int(row.get('horizon_seconds') or 0)
            hp=hsmap.get(str(h)) if isinstance(hsmap.get(str(h)),dict) else {}
            counts['considered'] += 1
            before=finite(row.get('confidence'))
            decision='preserve'
            if hp.get('inverse_shadow_candidate'):
                row['confidence']=clamp(before*FALSIFICATION_CONFIDENCE_MULTIPLIER,0.0,before)
                meta['council_eligible']=False
                meta['diversity_eligible']=False
                meta['prequential_falsified_original']=True
                meta['routing_reason']='prequential_inverse_holdout_positive'
                decision='original_falsified_blocked'
                counts['original_falsified_blocked'] += 1
                counts['inverse_candidate_available'] += 1
            meta['pre_falsification_confidence']=before
            meta['prequential_falsification_decision']=decision
            meta['falsification_gate_can_only_preserve_or_reduce_confidence']=True
            row['metadata']=meta
            decisions.append({'pack_id':pid,'symbol':normalize_symbol(row.get('symbol') or ''),'horizon_seconds':h,'decision':decision,'confidence_before':before,'confidence_after':finite(row.get('confidence'))})
    return {'model':'prequential_falsification_gate_v1.20','counts':counts,'decisions':decisions,'execution_authority':False,'can_increase_risk':False,'gate_can_only_preserve_or_reduce_confidence':True}


def falsification_inverse_observations(packs, policy, now):
    """Emit separately named inverse hypotheses after chronological holdout validation.

    These observations cannot vote in council/diversity and cannot affect execution.
    They exist solely so future outcomes can independently confirm or falsify the
    reversal hypothesis without silently flipping a production signal.
    """
    by_pack=policy.get('by_pack') if isinstance(policy,dict) else {}
    by_pack=by_pack if isinstance(by_pack,dict) else {}
    out=[]
    for pack in packs or []:
        pid=str(pack.get('pack_id') or '')
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        pp=by_pack.get(pid) if isinstance(by_pack.get(pid),dict) else {}
        hsmap=pp.get('horizons') if isinstance(pp.get('horizons'),dict) else {}
        for row in pack.get('observations') or []:
            if not isinstance(row,dict) or row.get('kind') != 'signal':
                continue
            h=int(row.get('horizon_seconds') or 0)
            hp=hsmap.get(str(h)) if isinstance(hsmap.get(str(h)),dict) else {}
            if not hp.get('inverse_shadow_candidate'):
                continue
            score=finite(row.get('score'))
            if abs(score) < 0.05:
                continue
            meta=dict(row.get('metadata') or {})
            base_conf=finite(meta.get('pre_falsification_confidence'),finite(row.get('confidence')))
            meta.update({
                'derived_from_pack':pid,
                'prequential_inverse_hypothesis':True,
                'chronological_holdout_validated':True,
                'council_eligible':False,
                'diversity_eligible':False,
                'execution_authority':False,
                'risk_increase_allowed':False,
                'auto_promotion_allowed':False,
            })
            out.append({
                **row,
                'score': -score,
                'confidence': clamp(base_conf*INVERSE_CHALLENGER_CONFIDENCE_MULTIPLIER,0.0,base_conf),
                'provenance':f"falsification-inverse-v1.14:{pid}:{normalize_symbol(row.get('symbol') or '')}:{int(now)}:{h}",
                'metadata':meta,
            })
    return out


def prequential_falsification_observation(policy, gate, now):
    return {
        'symbol':'GLOBAL','kind':'context','score':0.0,'confidence':0.99,
        'source':'LeanTrader chronological multi-horizon outcome ledger',
        'provenance':f'leantrader:prequential-falsification:v1.20:{int(now)}',
        'observed_at':now,'horizon_seconds':max(HORIZON_GRID),
        'metadata':{
            'model':'prequential_directional_falsification_policy_v1.20',
            'directional_claim':False,'policy':policy,'gate':gate,
            'execution_authority':False,'risk_increase_allowed':False,
        },
    }


def _conditional_strength_bucket(value):
    a = abs(finite(value))
    if a < 0.25:
        return "weak"
    if a < 0.50:
        return "medium"
    return "strong"


def _conditional_context_key(row):
    """One disjoint, ex-ante context bucket per horizon-lab observation.

    The fields already existed when the signal was generated; realized P/L is never
    used to assign a row to a bucket. This keeps discovery interpretable and avoids
    overlapping combinatorial rule mining.
    """
    regime = str(row.get("regime") or "unknown")
    cls = str(row.get("symbol_class") or "unknown")
    liq = str(row.get("liquidity_state") or "unknown")
    ood = "ood" if bool(row.get("out_of_distribution")) else "in_distribution"
    strength = _conditional_strength_bucket(row.get("signal"))
    return f"{regime}|{cls}|{liq}|{ood}|strength:{strength}"


def _conditional_split_stats(rows):
    rows = sorted([r for r in (rows or []) if isinstance(r, dict)], key=lambda r: (finite(r.get("opened_at")), str(r.get("horizon_lab_id") or "")))
    n = len(rows)
    holdout_n = max(MIN_CONDITIONAL_HOLDOUT_SAMPLES, int(math.ceil(n * 0.30))) if n else 0
    holdout_n = min(holdout_n, n)
    split = max(0, n - holdout_n)
    train = rows[:split]
    holdout = rows[split:]
    return {
        "samples": n,
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "train": _horizon_metric_stats(train),
        "holdout": _horizon_metric_stats(holdout),
        "all": _horizon_metric_stats(rows),
    }


def conditional_edge_policy(state, now=None):
    """Discover narrow contexts where an otherwise weak specialist clears costs.

    Contexts are disjoint and defined only from ex-ante regime/class/liquidity/OOD/
    signal-strength metadata. Each candidate must survive a chronological holdout.
    A multiple-comparison sample penalty grows with the number of contexts tested.
    This policy is research-only and can never increase confidence or execution risk.
    """
    _ = finite(now, time.time())
    rows = state.get("horizon_lab_resolved") if isinstance(state.get("horizon_lab_resolved"), list) else []
    by_pack = {}
    validated = []
    negative = []
    watches = []
    cost = ROUND_TRIP_COST_BPS / 10_000.0
    for pid in DIRECTIONAL_PACKS:
        prow = [r for r in rows if isinstance(r, dict) and r.get("pack_id") == pid]
        horizons = {}
        for h in HORIZON_GRID:
            hrs = [r for r in prow if int(r.get("horizon_seconds") or 0) == h]
            grouped = {}
            for row in hrs:
                grouped.setdefault(_conditional_context_key(row), []).append(row)
            tested = max(1, len(grouped))
            multiple_penalty = int(math.ceil(math.log2(tested + 1))) * 2
            required = MIN_CONDITIONAL_EDGE_SAMPLES + multiple_penalty
            contexts = {}
            for key, grows in sorted(grouped.items()):
                split = _conditional_split_stats(grows)
                n = int(split.get("samples") or 0)
                tr = split.get("train") or {}
                ho = split.get("holdout") or {}
                hold_n = int(split.get("holdout_samples") or 0)
                candidate = bool(
                    n >= required
                    and hold_n >= MIN_CONDITIONAL_HOLDOUT_SAMPLES
                    and finite(tr.get("average_net_return")) > 0.0
                    and finite(tr.get("robust_net_edge_score")) > 0.0
                    and finite(ho.get("average_net_return")) >= MIN_CONDITIONAL_HOLDOUT_NET
                    and finite(ho.get("ewma_net_return")) > 0.0
                    and finite(ho.get("robust_net_edge_score")) > 0.0
                    and finite(ho.get("win_rate")) >= 0.40
                    and finite(ho.get("average_gross_return")) >= cost * 1.10
                )
                mature_negative = bool(
                    n >= required
                    and hold_n >= MIN_CONDITIONAL_HOLDOUT_SAMPLES
                    and finite(ho.get("average_net_return")) < 0.0
                    and finite(ho.get("robust_net_edge_score")) < 0.0
                    and finite(ho.get("ewma_net_return")) < 0.0
                )
                if candidate:
                    lifecycle = "conditional_edge_holdout_positive"
                elif mature_negative:
                    lifecycle = "conditional_negative_holdout"
                elif n >= required:
                    lifecycle = "conditional_watch"
                else:
                    lifecycle = "insufficient_context_evidence"
                rowout = {
                    "context_key": key,
                    "lifecycle": lifecycle,
                    "conditional_shadow_candidate": candidate,
                    "mature_negative": mature_negative,
                    "required_samples": required,
                    "contexts_tested": tested,
                    "multiple_comparison_sample_penalty": multiple_penalty,
                    "minimum_holdout_net_return": MIN_CONDITIONAL_HOLDOUT_NET,
                    **split,
                    "execution_authority": False,
                    "can_increase_risk": False,
                    "auto_promotion_allowed": False,
                }
                contexts[key] = rowout
                ref = {"pack_id": pid, "horizon_seconds": h, "context_key": key, "holdout": ho, "samples": n}
                if candidate:
                    validated.append(ref)
                elif mature_negative:
                    negative.append(ref)
                elif n >= MIN_CONDITIONAL_EDGE_SAMPLES:
                    watches.append(ref)
            horizons[str(h)] = {
                "contexts": contexts,
                "contexts_tested": tested,
                "required_samples": required,
                "validated_contexts": sum(1 for v in contexts.values() if v.get("conditional_shadow_candidate")),
                "mature_negative_contexts": sum(1 for v in contexts.values() if v.get("mature_negative")),
            }
        by_pack[pid] = {"horizons": horizons, "execution_authority": False, "can_increase_risk": False, "auto_promotion_allowed": False}
    validated.sort(key=lambda x: finite((x.get("holdout") or {}).get("robust_net_edge_score")), reverse=True)
    return {
        "model": "context_selective_alpha_discovery_policy_v1.20",
        "by_pack": by_pack,
        "validated_contexts": validated[:40],
        "mature_negative_contexts": negative[:80],
        "context_watches": watches[:80],
        "minimum_base_samples": MIN_CONDITIONAL_EDGE_SAMPLES,
        "minimum_holdout_samples": MIN_CONDITIONAL_HOLDOUT_SAMPLES,
        "minimum_holdout_net_return": MIN_CONDITIONAL_HOLDOUT_NET,
        "context_definition": ["regime", "symbol_class", "liquidity_state", "out_of_distribution", "signal_strength_bucket"],
        "contexts_are_disjoint": True,
        "chronological_holdout_required": True,
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
        "promotion_requires_verified_core_release": True,
    }


def _current_conditional_context(row):
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    ctx = meta.get("market_context") if isinstance(meta.get("market_context"), dict) else {}
    proxy = {
        "regime": str(ctx.get("regime") or "unknown"),
        "symbol_class": str(ctx.get("symbol_class") or symbol_class(row.get("symbol") or "")),
        "liquidity_state": str(ctx.get("liquidity_state") or "unknown"),
        "out_of_distribution": bool(ctx.get("out_of_distribution")),
        "signal": finite(row.get("score")),
    }
    return _conditional_context_key(proxy)


def apply_conditional_edge_gate(packs, policy):
    by_pack = policy.get("by_pack") if isinstance(policy, dict) else {}
    by_pack = by_pack if isinstance(by_pack, dict) else {}
    counts = {"considered": 0, "validated_context_current": 0, "mature_negative_blocked": 0}
    decisions = []
    for pack in packs or []:
        pid = str(pack.get("pack_id") or "")
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        ph = (by_pack.get(pid) or {}).get("horizons") if isinstance(by_pack.get(pid), dict) else {}
        ph = ph if isinstance(ph, dict) else {}
        for row in pack.get("observations") or []:
            if not isinstance(row, dict) or row.get("kind") != "signal":
                continue
            counts["considered"] += 1
            h = str(int(row.get("horizon_seconds") or 0))
            key = _current_conditional_context(row)
            crow = (((ph.get(h) or {}).get("contexts") or {}).get(key) or {}) if isinstance(ph.get(h), dict) else {}
            before = finite(row.get("confidence"))
            decision = "preserve_unknown_or_watch"
            if crow.get("conditional_shadow_candidate"):
                counts["validated_context_current"] += 1
                decision = "validated_context_shadow_only"
            elif crow.get("mature_negative"):
                meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                row["confidence"] = clamp(before * CONDITIONAL_NEGATIVE_CONFIDENCE_MULTIPLIER, 0.0, before)
                meta["council_eligible"] = False
                meta["diversity_eligible"] = False
                meta["conditional_edge_negative_context"] = True
                meta["routing_reason"] = "conditional_negative_holdout"
                row["metadata"] = meta
                counts["mature_negative_blocked"] += 1
                decision = "conditional_negative_blocked"
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            meta["conditional_edge_context_key"] = key
            meta["conditional_edge_decision"] = decision
            meta["conditional_edge_gate_can_only_preserve_or_reduce"] = True
            row["metadata"] = meta
            decisions.append({
                "pack_id": pid, "symbol": normalize_symbol(row.get("symbol") or ""), "horizon_seconds": int(row.get("horizon_seconds") or 0),
                "context_key": key, "decision": decision, "confidence_before": before, "confidence_after": finite(row.get("confidence")),
            })
    return {
        "model": "context_selective_alpha_gate_v1.20",
        "counts": counts, "decisions": decisions,
        "execution_authority": False, "can_increase_risk": False,
        "gate_can_only_preserve_or_reduce_confidence": True,
    }


def conditional_challenger_observations(packs, policy, now):
    by_pack = policy.get("by_pack") if isinstance(policy, dict) else {}
    by_pack = by_pack if isinstance(by_pack, dict) else {}
    out = []
    for pack in packs or []:
        pid = str(pack.get("pack_id") or "")
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        ph = (by_pack.get(pid) or {}).get("horizons") if isinstance(by_pack.get(pid), dict) else {}
        ph = ph if isinstance(ph, dict) else {}
        for row in pack.get("observations") or []:
            if not isinstance(row, dict) or row.get("kind") != "signal":
                continue
            h = str(int(row.get("horizon_seconds") or 0))
            key = _current_conditional_context(row)
            crow = (((ph.get(h) or {}).get("contexts") or {}).get(key) or {}) if isinstance(ph.get(h), dict) else {}
            if not crow.get("conditional_shadow_candidate"):
                continue
            meta0 = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            ctx = meta0.get("market_context") if isinstance(meta0.get("market_context"), dict) else {}
            base_conf = finite(row.get("confidence"))
            conf = clamp(min(base_conf * CONDITIONAL_CHALLENGER_CONFIDENCE_MULTIPLIER, 0.45), 0.0, base_conf)
            out.append({
                "symbol": normalize_symbol(row.get("symbol") or ""),
                "kind": "signal",
                "score": finite(row.get("score")),
                "confidence": conf,
                "source": "LeanTrader context-selective alpha discovery from chronological multi-horizon evidence",
                "provenance": f"leantrader:conditional-alpha:v1.20:{pid}:{h}:{hashlib.sha256(key.encode()).hexdigest()[:10]}:{int(now)}",
                "observed_at": finite(row.get("observed_at"), now),
                "horizon_seconds": int(row.get("horizon_seconds") or 0),
                "metadata": {
                    "model": "context_selective_alpha_challenger_v1.20",
                    "source_pack_id": pid,
                    "validated_context_key": key,
                    "conditional_policy_evidence": crow,
                    "market_context": ctx,
                    "mid_price": meta0.get("mid_price"),
                    "council_eligible": False,
                    "diversity_eligible": False,
                    "conditional_shadow_only": True,
                    "execution_authority": False,
                    "risk_increase_allowed": False,
                    "can_enable_live": False,
                },
            })
    return out


def conditional_edge_policy_observation(policy, gate, now):
    return {
        "symbol": "GLOBAL", "kind": "context", "score": 0.0, "confidence": 0.96,
        "source": "LeanTrader disjoint-context multi-horizon outcome ledger",
        "provenance": f"leantrader:conditional-edge-policy:v1.20:{int(now)}",
        "observed_at": now, "horizon_seconds": max(HORIZON_GRID),
        "metadata": {
            "model": "context_selective_alpha_discovery_policy_v1.20",
            "policy": policy, "gate": gate, "directional_claim": False,
            "execution_authority": False, "risk_increase_allowed": False,
        },
    }


def _hierarchical_context_keys(row):
    regime = str(row.get("regime") or "unknown")
    cls = str(row.get("symbol_class") or "unknown")
    liq = str(row.get("liquidity_state") or "unknown")
    ood = "ood" if bool(row.get("out_of_distribution")) else "in_distribution"
    strength = _conditional_strength_bucket(row.get("signal"))
    return {
        "coarse": f"{regime}|{cls}|{ood}",
        "mid": f"{regime}|{cls}|{liq}|{ood}",
        "fine": f"{regime}|{cls}|{liq}|{ood}|strength:{strength}",
    }


def _hierarchical_group_summaries(rows, level):
    grouped = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        key = _hierarchical_context_keys(row)[level]
        grouped.setdefault(key, []).append(row)
    return {key: _conditional_split_stats(grows) for key, grows in grouped.items()}


def _hierarchical_shrink(child_value, child_n, parent_value, prior_strength=None):
    prior = HIERARCHICAL_PRIOR_STRENGTH if prior_strength is None else max(0.0, finite(prior_strength))
    n = max(0.0, finite(child_n))
    if n <= 0.0:
        return finite(parent_value)
    weight = n / (n + prior) if (n + prior) > 0 else 0.0
    return weight * finite(child_value) + (1.0 - weight) * finite(parent_value)


def hierarchical_context_policy(state, now=None):
    """Generalize sparse conditional evidence without pretending sparse cells are proof.

    Exact v1.15 buckets can be too fragmented to learn quickly. v1.20 therefore
    evaluates a fixed ex-ante hierarchy: regime/class/OOD -> +liquidity -> +strength.
    Child holdout evidence is shrunk toward its parent. Positive candidates require
    chronological agreement across all three levels; broad negative agreement may
    reduce a sparse child, but can never create or strengthen a positive vote.
    """
    _ = finite(now, time.time())
    rows = state.get("horizon_lab_resolved") if isinstance(state.get("horizon_lab_resolved"), list) else []
    by_pack = {}
    positive = []
    negative = []
    watches = []
    cost = ROUND_TRIP_COST_BPS / 10_000.0
    for pid in DIRECTIONAL_PACKS:
        prow = [r for r in rows if isinstance(r, dict) and r.get("pack_id") == pid]
        horizons = {}
        for h in HORIZON_GRID:
            hrs = [r for r in prow if int(r.get("horizon_seconds") or 0) == h]
            coarse = _hierarchical_group_summaries(hrs, "coarse")
            mid = _hierarchical_group_summaries(hrs, "mid")
            fine = _hierarchical_group_summaries(hrs, "fine")
            fine_rows = {}
            for row in hrs:
                keys = _hierarchical_context_keys(row)
                fine_rows.setdefault(keys["fine"], {"keys": keys, "rows": []})["rows"].append(row)
            contexts = {}
            for fkey, payload in sorted(fine_rows.items()):
                keys = payload["keys"]
                fs = fine.get(fkey) or _conditional_split_stats(payload["rows"])
                ms = mid.get(keys["mid"]) or _conditional_split_stats([])
                cs = coarse.get(keys["coarse"]) or _conditional_split_stats([])
                fho, mho, cho = fs.get("holdout") or {}, ms.get("holdout") or {}, cs.get("holdout") or {}
                fhn = int(fs.get("holdout_samples") or 0)
                mhn = int(ms.get("holdout_samples") or 0)
                chn = int(cs.get("holdout_samples") or 0)
                mid_shrunk = _hierarchical_shrink(finite(mho.get("average_net_return")), mhn, finite(cho.get("average_net_return")))
                fine_shrunk = _hierarchical_shrink(finite(fho.get("average_net_return")), fhn, mid_shrunk)
                mid_robust_shrunk = _hierarchical_shrink(finite(mho.get("robust_net_edge_score")), mhn, finite(cho.get("robust_net_edge_score")))
                fine_robust_shrunk = _hierarchical_shrink(finite(fho.get("robust_net_edge_score")), fhn, mid_robust_shrunk)
                coarse_ready = int(cs.get("samples") or 0) >= MIN_HIERARCHICAL_COARSE_SAMPLES and chn >= MIN_HIERARCHICAL_HOLDOUT_SAMPLES
                mid_ready = int(ms.get("samples") or 0) >= MIN_HIERARCHICAL_MID_SAMPLES and mhn >= MIN_HIERARCHICAL_HOLDOUT_SAMPLES
                fine_ready = int(fs.get("samples") or 0) >= MIN_HIERARCHICAL_FINE_SAMPLES and fhn >= MIN_HIERARCHICAL_HOLDOUT_SAMPLES
                parent_positive = bool(
                    coarse_ready and mid_ready
                    and finite(cho.get("average_net_return")) > 0.0
                    and finite(cho.get("ewma_net_return")) > 0.0
                    and finite(mho.get("average_net_return")) > 0.0
                    and finite(mho.get("ewma_net_return")) > 0.0
                )
                fine_positive = bool(
                    fine_ready
                    and finite(fho.get("average_net_return")) >= MIN_HIERARCHICAL_HOLDOUT_NET
                    and finite(fho.get("ewma_net_return")) > 0.0
                    and finite(fho.get("win_rate")) >= 0.40
                    and finite(fho.get("average_gross_return")) >= cost * 1.10
                )
                candidate = bool(
                    parent_positive and fine_positive
                    and fine_shrunk >= MIN_HIERARCHICAL_HOLDOUT_NET
                    and fine_robust_shrunk > 0.0
                )
                parent_negative = bool(
                    coarse_ready and mid_ready
                    and finite(cho.get("average_net_return")) < 0.0
                    and finite(cho.get("ewma_net_return")) < 0.0
                    and finite(cho.get("robust_net_edge_score")) < 0.0
                    and finite(mho.get("average_net_return")) < 0.0
                    and finite(mho.get("ewma_net_return")) < 0.0
                    and finite(mho.get("robust_net_edge_score")) < 0.0
                )
                child_has_positive_hint = bool(
                    fhn >= 2
                    and finite(fho.get("average_net_return")) > 0.0
                    and finite(fho.get("ewma_net_return")) > 0.0
                )
                negative_consensus = bool(parent_negative and not child_has_positive_hint)
                if candidate:
                    lifecycle = "hierarchical_holdout_positive"
                elif negative_consensus:
                    lifecycle = "hierarchical_parent_negative_consensus"
                elif parent_positive or fine_shrunk > 0.0:
                    lifecycle = "hierarchical_positive_watch"
                else:
                    lifecycle = "hierarchical_unproven"
                out = {
                    "fine_context_key": fkey,
                    "coarse_context_key": keys["coarse"],
                    "mid_context_key": keys["mid"],
                    "lifecycle": lifecycle,
                    "hierarchical_shadow_candidate": candidate,
                    "hierarchical_negative_consensus": negative_consensus,
                    "parent_positive": parent_positive,
                    "fine_positive": fine_positive,
                    "coarse_ready": coarse_ready,
                    "mid_ready": mid_ready,
                    "fine_ready": fine_ready,
                    "shrunk_holdout_net_return": fine_shrunk,
                    "shrunk_holdout_robust_score": fine_robust_shrunk,
                    "coarse": cs,
                    "mid": ms,
                    "fine": fs,
                    "execution_authority": False,
                    "can_increase_risk": False,
                    "auto_promotion_allowed": False,
                }
                contexts[fkey] = out
                ref = {
                    "pack_id": pid, "horizon_seconds": h, "context_key": fkey,
                    "lifecycle": lifecycle, "shrunk_holdout_net_return": fine_shrunk,
                    "fine_samples": int(fs.get("samples") or 0), "mid_samples": int(ms.get("samples") or 0),
                    "coarse_samples": int(cs.get("samples") or 0),
                }
                if candidate:
                    positive.append(ref)
                elif negative_consensus:
                    negative.append(ref)
                elif lifecycle == "hierarchical_positive_watch":
                    watches.append(ref)
            horizons[str(h)] = {
                "contexts": contexts,
                "fine_contexts_tested": len(fine),
                "mid_contexts_tested": len(mid),
                "coarse_contexts_tested": len(coarse),
                "validated_contexts": sum(1 for v in contexts.values() if v.get("hierarchical_shadow_candidate")),
                "negative_consensus_contexts": sum(1 for v in contexts.values() if v.get("hierarchical_negative_consensus")),
            }
        by_pack[pid] = {"horizons": horizons, "execution_authority": False, "can_increase_risk": False, "auto_promotion_allowed": False}
    positive.sort(key=lambda x: finite(x.get("shrunk_holdout_net_return")), reverse=True)
    watches.sort(key=lambda x: finite(x.get("shrunk_holdout_net_return")), reverse=True)
    return {
        "model": "hierarchical_sparse_context_generalization_v1.20",
        "hierarchy": ["regime+symbol_class+ood", "+liquidity_state", "+signal_strength_bucket"],
        "by_pack": by_pack,
        "validated_contexts": positive[:40],
        "negative_consensus_contexts": negative[:80],
        "positive_watches": watches[:80],
        "minimum_coarse_samples": MIN_HIERARCHICAL_COARSE_SAMPLES,
        "minimum_mid_samples": MIN_HIERARCHICAL_MID_SAMPLES,
        "minimum_fine_samples": MIN_HIERARCHICAL_FINE_SAMPLES,
        "minimum_holdout_samples": MIN_HIERARCHICAL_HOLDOUT_SAMPLES,
        "minimum_holdout_net_return": MIN_HIERARCHICAL_HOLDOUT_NET,
        "prior_strength": HIERARCHICAL_PRIOR_STRENGTH,
        "chronological_holdout_required": True,
        "child_evidence_is_shrunk_to_parent": True,
        "positive_requires_parent_agreement": True,
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
        "promotion_requires_verified_core_release": True,
    }


def apply_hierarchical_context_gate(packs, policy):
    by_pack = policy.get("by_pack") if isinstance(policy, dict) else {}
    by_pack = by_pack if isinstance(by_pack, dict) else {}
    counts = {"considered": 0, "positive_watch": 0, "validated_context_current": 0, "parent_negative_blocked": 0}
    decisions = []
    for pack in packs or []:
        pid = str(pack.get("pack_id") or "")
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        ph = (by_pack.get(pid) or {}).get("horizons") if isinstance(by_pack.get(pid), dict) else {}
        ph = ph if isinstance(ph, dict) else {}
        for row in pack.get("observations") or []:
            if not isinstance(row, dict) or row.get("kind") != "signal":
                continue
            counts["considered"] += 1
            h = str(int(row.get("horizon_seconds") or 0))
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            ctx = meta.get("market_context") if isinstance(meta.get("market_context"), dict) else {}
            proxy = {
                "regime": str(ctx.get("regime") or "unknown"),
                "symbol_class": str(ctx.get("symbol_class") or symbol_class(row.get("symbol") or "")),
                "liquidity_state": str(ctx.get("liquidity_state") or "unknown"),
                "out_of_distribution": bool(ctx.get("out_of_distribution")),
                "signal": finite(row.get("score")),
            }
            key = _hierarchical_context_keys(proxy)["fine"]
            crow = (((ph.get(h) or {}).get("contexts") or {}).get(key) or {}) if isinstance(ph.get(h), dict) else {}
            before = finite(row.get("confidence"))
            decision = "preserve_unknown"
            if crow.get("hierarchical_shadow_candidate"):
                counts["validated_context_current"] += 1
                decision = "validated_shadow_only"
            elif crow.get("hierarchical_negative_consensus"):
                row["confidence"] = clamp(before * HIERARCHICAL_NEGATIVE_CONFIDENCE_MULTIPLIER, 0.0, before)
                meta["council_eligible"] = False
                meta["diversity_eligible"] = False
                meta["hierarchical_context_negative"] = True
                meta["routing_reason"] = "hierarchical_parent_negative_consensus"
                counts["parent_negative_blocked"] += 1
                decision = "parent_negative_blocked"
            elif crow.get("lifecycle") == "hierarchical_positive_watch":
                counts["positive_watch"] += 1
                decision = "positive_watch_no_authority"
            meta["hierarchical_context_key"] = key
            meta["hierarchical_context_decision"] = decision
            meta["hierarchical_context_gate_can_only_preserve_or_reduce"] = True
            row["metadata"] = meta
            decisions.append({
                "pack_id": pid, "symbol": normalize_symbol(row.get("symbol") or ""),
                "horizon_seconds": int(row.get("horizon_seconds") or 0), "context_key": key,
                "decision": decision, "confidence_before": before, "confidence_after": finite(row.get("confidence")),
                "shrunk_holdout_net_return": finite(crow.get("shrunk_holdout_net_return")),
            })
    return {
        "model": "hierarchical_sparse_context_gate_v1.20",
        "counts": counts,
        "decisions": decisions,
        "execution_authority": False,
        "can_increase_risk": False,
        "gate_can_only_preserve_or_reduce_confidence": True,
    }


def hierarchical_challenger_observations(packs, policy, now):
    by_pack = policy.get("by_pack") if isinstance(policy, dict) else {}
    by_pack = by_pack if isinstance(by_pack, dict) else {}
    out = []
    for pack in packs or []:
        pid = str(pack.get("pack_id") or "")
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        ph = (by_pack.get(pid) or {}).get("horizons") if isinstance(by_pack.get(pid), dict) else {}
        ph = ph if isinstance(ph, dict) else {}
        for row in pack.get("observations") or []:
            if not isinstance(row, dict) or row.get("kind") != "signal":
                continue
            h = str(int(row.get("horizon_seconds") or 0))
            meta0 = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            ctx = meta0.get("market_context") if isinstance(meta0.get("market_context"), dict) else {}
            proxy = {
                "regime": str(ctx.get("regime") or "unknown"),
                "symbol_class": str(ctx.get("symbol_class") or symbol_class(row.get("symbol") or "")),
                "liquidity_state": str(ctx.get("liquidity_state") or "unknown"),
                "out_of_distribution": bool(ctx.get("out_of_distribution")),
                "signal": finite(row.get("score")),
            }
            key = _hierarchical_context_keys(proxy)["fine"]
            crow = (((ph.get(h) or {}).get("contexts") or {}).get(key) or {}) if isinstance(ph.get(h), dict) else {}
            if not crow.get("hierarchical_shadow_candidate"):
                continue
            base_conf = finite(meta0.get("pre_net_edge_gate_confidence"), finite(row.get("confidence")))
            conf = clamp(min(base_conf * HIERARCHICAL_CHALLENGER_CONFIDENCE_MULTIPLIER, 0.40), 0.0, base_conf)
            meta = dict(meta0)
            meta.update({
                "model": "hierarchical_context_alpha_challenger_v1.20",
                "source_pack_id": pid,
                "validated_hierarchical_context_key": key,
                "hierarchical_policy_evidence": crow,
                "council_eligible": False,
                "diversity_eligible": False,
                "hierarchical_shadow_only": True,
                "execution_authority": False,
                "risk_increase_allowed": False,
                "can_enable_live": False,
            })
            out.append({
                **row,
                "confidence": conf,
                "source": "LeanTrader hierarchical context generalization with chronological parent-child agreement",
                "provenance": f"leantrader:hierarchical-context-alpha:v1.20:{pid}:{h}:{hashlib.sha256(key.encode()).hexdigest()[:10]}:{int(now)}",
                "metadata": meta,
            })
    return out


def hierarchical_context_policy_observation(policy, gate, now):
    return {
        "symbol": "GLOBAL", "kind": "context", "score": 0.0, "confidence": 0.97,
        "source": "LeanTrader hierarchical context evidence over the multi-horizon ledger",
        "provenance": f"leantrader:hierarchical-context-policy:v1.20:{int(now)}",
        "observed_at": now, "horizon_seconds": max(HORIZON_GRID),
        "metadata": {
            "model": "hierarchical_sparse_context_generalization_v1.20",
            "policy": policy, "gate": gate, "directional_claim": False,
            "execution_authority": False, "risk_increase_allowed": False,
        },
    }

def net_edge_horizon_policy(state, now=None):
    """Learn which holding horizons, if any, clear modeled round-trip costs."""
    _ = finite(now, time.time())
    rows = state.get('horizon_lab_resolved') if isinstance(state.get('horizon_lab_resolved'), list) else []
    by_pack = {}
    positive = []
    for pack_id in DIRECTIONAL_PACKS:
        pack_rows = [r for r in rows if isinstance(r, dict) and r.get('pack_id') == pack_id]
        horizons = {}
        candidates = []
        for h in HORIZON_GRID:
            stats = _horizon_metric_stats([r for r in pack_rows if int(r.get('horizon_seconds') or 0) == h])
            stats['horizon_seconds'] = h
            stats['minimum_samples'] = MIN_HORIZON_POLICY_SAMPLES
            stats['economically_tradeable_shadow'] = bool(
                int(stats.get('samples') or 0) >= MIN_HORIZON_POLICY_SAMPLES
                and finite(stats.get('average_net_return')) > 0.0
                and finite(stats.get('ewma_net_return')) > 0.0
                and finite(stats.get('robust_net_edge_score')) > 0.0
                and finite(stats.get('win_rate')) >= 0.45
                and finite(stats.get('cost_coverage_ratio')) > 1.0
            )
            horizons[str(h)] = stats
            if stats['economically_tradeable_shadow']:
                candidates.append(stats)
        if candidates:
            best = max(candidates, key=lambda x: (finite(x.get('robust_net_edge_score')), finite(x.get('average_net_return')), -int(x.get('horizon_seconds') or 0)))
            recommended = int(best['horizon_seconds'])
            lifecycle = 'research_positive_horizon'
            positive.append(pack_id)
        else:
            all_stats = [v for v in horizons.values() if int(v.get('samples') or 0) >= MIN_HORIZON_POLICY_SAMPLES]
            best = max(all_stats, key=lambda x: finite(x.get('robust_net_edge_score')), default=None)
            recommended = None
            lifecycle = 'sub_cost_or_negative' if all_stats and all(finite(v.get('robust_net_edge_score')) <= 0.0 for v in all_stats) else 'insufficient_multi_horizon_evidence'
        by_pack[pack_id] = {
            'lifecycle': lifecycle,
            'recommended_shadow_horizon_seconds': recommended,
            'best_observed_horizon_seconds': int(best.get('horizon_seconds')) if isinstance(best, dict) else None,
            'horizons': horizons,
            'policy_applies': recommended is not None,
            'execution_authority': False,
            'can_enable_live': False,
            'can_increase_risk': False,
            'auto_promotion_allowed': False,
        }
    return {
        'model': 'cost_aware_multi_horizon_net_edge_policy_v1.20',
        'candidate_horizons_seconds': list(HORIZON_GRID),
        'minimum_samples_per_horizon': MIN_HORIZON_POLICY_SAMPLES,
        'round_trip_cost_bps': ROUND_TRIP_COST_BPS,
        'by_pack': by_pack,
        'research_positive_horizon_packs': positive,
        'execution_authority': False,
        'can_enable_live': False,
        'can_increase_risk': False,
        'auto_promotion_allowed': False,
        'promotion_requires_verified_core_release': True,
    }


def apply_net_edge_horizon_gate(packs, policy):
    """Block council voting at a statistically sub-cost current horizon.

    This gate never creates a positive vote and never increases confidence. Signals
    blocked here remain visible to research and can still be tested by the separate
    horizon-optimized challenger.
    """
    by_pack = policy.get('by_pack') if isinstance(policy, dict) else {}
    by_pack = by_pack if isinstance(by_pack, dict) else {}
    counts = {'considered': 0, 'sub_cost_blocked': 0, 'alternate_horizon_available': 0}
    decisions = []
    for pack in packs or []:
        pid = str(pack.get('pack_id') or '')
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        pp = by_pack.get(pid) if isinstance(by_pack.get(pid), dict) else {}
        horizons = pp.get('horizons') if isinstance(pp.get('horizons'), dict) else {}
        recommended = pp.get('recommended_shadow_horizon_seconds')
        for row in pack.get('observations') or []:
            if not isinstance(row, dict) or row.get('kind') != 'signal':
                continue
            meta = row.get('metadata') if isinstance(row.get('metadata'), dict) else {}
            if meta.get('council_eligible') is False:
                continue
            counts['considered'] += 1
            current_h = int(row.get('horizon_seconds') or 0)
            hs = horizons.get(str(current_h)) if isinstance(horizons.get(str(current_h)), dict) else {}
            samples = int(hs.get('samples') or 0)
            sub_cost = bool(
                samples >= MIN_HORIZON_POLICY_SAMPLES
                and finite(hs.get('average_net_return')) < 0.0
                and finite(hs.get('ewma_net_return')) < 0.0
                and finite(hs.get('cost_coverage_ratio')) < 1.0
            )
            before = finite(row.get('confidence'))
            decision = 'preserve'
            if sub_cost:
                row['confidence'] = clamp(before * 0.35, 0.0, before)
                meta['council_eligible'] = False
                meta['diversity_eligible'] = False
                meta['net_edge_research_only'] = True
                meta['routing_reason'] = 'sub_cost_current_horizon'
                decision = 'sub_cost_blocked'
                counts['sub_cost_blocked'] += 1
            if recommended and int(recommended) != current_h:
                meta['alternate_research_horizon_seconds'] = int(recommended)
                counts['alternate_horizon_available'] += 1
            meta['pre_net_edge_gate_confidence'] = before
            meta['net_edge_gate_decision'] = decision
            meta['net_edge_gate_can_only_preserve_or_reduce_confidence'] = True
            row['metadata'] = meta
            decisions.append({
                'pack_id': pid,
                'symbol': normalize_symbol(row.get('symbol') or ''),
                'current_horizon_seconds': current_h,
                'current_horizon_samples': samples,
                'decision': decision,
                'confidence_before': before,
                'confidence_after': finite(row.get('confidence')),
                'recommended_alternate_horizon_seconds': int(recommended) if recommended else None,
            })
    return {
        'model': 'cost_aware_net_edge_tradeability_gate_v1.20',
        'counts': counts,
        'decisions': decisions,
        'execution_authority': False,
        'can_increase_risk': False,
        'gate_can_only_preserve_or_reduce_confidence': True,
    }


def horizon_optimized_observations(packs, policy, now):
    """Emit a separate non-council challenger only for evidence-positive horizons."""
    by_pack = policy.get('by_pack') if isinstance(policy, dict) else {}
    by_pack = by_pack if isinstance(by_pack, dict) else {}
    out = []
    for pack in packs or []:
        pid = str(pack.get('pack_id') or '')
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        pp = by_pack.get(pid) if isinstance(by_pack.get(pid), dict) else {}
        horizon = pp.get('recommended_shadow_horizon_seconds')
        if not pp.get('policy_applies') or not horizon:
            continue
        for row in pack.get('observations') or []:
            if not isinstance(row, dict) or row.get('kind') != 'signal':
                continue
            if int(row.get('horizon_seconds') or 0) == int(horizon):
                continue
            score = finite(row.get('score'))
            if abs(score) < 0.05:
                continue
            meta = dict(row.get('metadata') or {})
            base_conf = finite(meta.get('pre_net_edge_gate_confidence'), finite(row.get('confidence')))
            meta.update({
                'derived_from_pack': pid,
                'original_horizon_seconds': int(row.get('horizon_seconds') or 0),
                'horizon_optimized_research_only': True,
                'council_eligible': False,
                'diversity_eligible': False,
                'execution_authority': False,
                'risk_increase_allowed': False,
            })
            out.append({
                **row,
                'horizon_seconds': int(horizon),
                'confidence': clamp(base_conf * 0.60, 0.0, base_conf),
                'provenance': f"horizon-optimized-v1.14:{pid}:{normalize_symbol(row.get('symbol') or '')}:{int(now)}:{int(horizon)}",
                'metadata': meta,
            })
    return out


def net_edge_horizon_policy_observation(policy, backfill_status, now):
    return {
        'symbol': 'GLOBAL',
        'kind': 'context',
        'score': 0.0,
        'confidence': 0.98,
        'source': 'LeanTrader costed specialist outcomes + Bybit historical spot klines',
        'provenance': f'leantrader:net-edge-horizon-policy:v1.14:{int(now)}',
        'observed_at': now,
        'horizon_seconds': max(HORIZON_GRID),
        'metadata': {
            'model': 'cost_aware_multi_horizon_net_edge_policy_v1.20',
            'directional_claim': False,
            'policy': policy,
            'backfill_status': backfill_status,
            'historical_kline_source': 'Bybit public V5 spot kline',
            'execution_authority': False,
            'risk_increase_allowed': False,
        },
    }

def specialist_council_advisory(packs, diversity_gov, evidence_gov, macro_obs=None, now=None, role_policy=None):
    """Evidence-aware shadow council.

    The council cannot create authority or increase confidence. It combines only
    already-bounded specialist signals and can preserve, reduce or recommend a
    veto. Its output is context-only and is measured separately before any core
    integration can be considered.
    """
    now = finite(now, time.time())
    macro_mult, macro_state = _macro_council_multiplier(macro_obs)
    div_symbols = diversity_gov.get("symbol_consensus") if isinstance(diversity_gov, dict) else {}
    div_symbols = div_symbols if isinstance(div_symbols, dict) else {}
    div_conflicts = diversity_gov.get("directional_conflicts") if isinstance(diversity_gov, dict) else []
    conflict_symbols = {normalize_symbol(r.get("symbol") or "") for r in div_conflicts if isinstance(r, dict)}
    evidence_gov = evidence_gov if isinstance(evidence_gov, dict) else {}
    role_policy = role_policy if isinstance(role_policy, dict) else {}
    alpha_policy = role_policy.get("directional_alpha") if isinstance(role_policy.get("directional_alpha"), dict) else {}
    support_shadow_allowed = bool(alpha_policy.get("support_shadow_research_allowed", True))
    risk_action_policy = role_policy.get("actions") if isinstance(role_policy.get("actions"), dict) else {}

    by_symbol = {}
    for pack in packs or []:
        pid = str(pack.get("pack_id") or "")
        if pid not in DIRECTIONAL_PACK_SET:
            continue
        lifecycle = str(((evidence_gov.get(pid) or {}).get("global") or {}).get("lifecycle") or "insufficient_evidence")
        evidence_cap = clamp(LIFECYCLE_COUNCIL_CAP.get(lifecycle, 0.55), 0.0, 1.0)
        for row in pack.get("observations") or []:
            if not isinstance(row, dict) or row.get("kind") != "signal":
                continue
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if meta.get("diversity_eligible") is False or meta.get("council_eligible") is False:
                continue
            score = finite(row.get("score"))
            if abs(score) < 0.05:
                continue
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            ctx = meta.get("market_context") if isinstance(meta.get("market_context"), dict) else {}
            by_symbol.setdefault(normalize_symbol(row.get("symbol") or ""), []).append({
                "pack_id": pid,
                "family": specialist_family(pid),
                "direction": 1 if score > 0 else -1,
                "score": score,
                "confidence": finite(row.get("confidence")),
                "evidence_lifecycle": lifecycle,
                "evidence_cap": evidence_cap,
                "out_of_distribution": bool(ctx.get("out_of_distribution")),
                "regime": str(ctx.get("regime") or "unknown"),
            })

    symbols = {}
    advisory_caps = {}
    for symbol, rows in sorted(by_symbol.items()):
        div = div_symbols.get(symbol) if isinstance(div_symbols.get(symbol), dict) else {}
        raw_votes = int(div.get("raw_votes") or len(rows))
        independent_families = int(div.get("independent_families") or len({r["family"] for r in rows}))
        effective_votes = finite(div.get("effective_independent_votes"), float(len(rows)))
        conflict = symbol in conflict_symbols or len({r["direction"] for r in rows}) > 1
        ood = any(r["out_of_distribution"] for r in rows)

        num = den = 0.0
        lifecycles = []
        for r in rows:
            w = max(0.0, r["confidence"]) * r["evidence_cap"]
            num += r["direction"] * w
            den += w
            lifecycles.append(r["evidence_lifecycle"])
        direction_score = clamp(num / max(1e-9, den)) if den > 0 else 0.0

        cap = macro_mult
        reasons = [f"macro:{macro_state}"] if macro_mult < 0.999 else []
        if independent_families < 2 and raw_votes > 1:
            cap *= 0.78
            reasons.append("same_family_consensus_discount")
        if effective_votes < 1.25:
            cap *= 0.82
            reasons.append("low_effective_independent_votes")
        if conflict:
            cap *= 0.55
            reasons.append("directional_conflict")
        if ood:
            cap *= 0.58
            reasons.append("out_of_distribution")
        if any(v == "suppressed" for v in lifecycles):
            cap *= 0.50
            reasons.append("suppressed_specialist_present")
        elif all(v in {"insufficient_evidence", "probation"} for v in lifecycles):
            cap *= 0.82
            reasons.append("immature_evidence")
        cap = clamp(cap, 0.0, 1.0)

        if conflict or cap < 0.42:
            action = "veto_recommended"
        elif cap < 0.78:
            action = "reduce"
        elif independent_families >= 2 and abs(direction_score) >= 0.25:
            action = "support_shadow"
        else:
            action = "abstain"

        pre_role_action = action
        if action == "support_shadow" and not support_shadow_allowed:
            # Role separation is monotone: a council with unvalidated/degraded
            # directional alpha cannot issue a positive shadow-support label.
            action = "abstain"
            cap = min(cap, 0.78)
            reasons.append("directional_council_alpha_not_validated")
        action_policy = risk_action_policy.get(action) if isinstance(risk_action_policy.get(action), dict) else {}
        if action in {"reduce", "veto_recommended"} and action_policy:
            reasons.append(f"counterfactual_{action}_policy:{action_policy.get('lifecycle','unknown')}")

        symbols[symbol] = {
            "action": action,
            "pre_role_policy_action": pre_role_action,
            "directional_council_alpha_lifecycle": str(alpha_policy.get("lifecycle") or "unavailable"),
            "advisory_direction_score": direction_score,
            "confidence_cap_multiplier": cap,
            "raw_votes": raw_votes,
            "effective_independent_votes": effective_votes,
            "independent_families": independent_families,
            "macro_risk_state": macro_state,
            "out_of_distribution": ood,
            "conflict": conflict,
            "evidence_lifecycles": sorted(set(lifecycles)),
            "reasons": reasons,
            "execution_authority": False,
            "risk_increase_allowed": False,
        }
        advisory_caps[symbol] = cap

    counts = {}
    for row in symbols.values():
        counts[row["action"]] = counts.get(row["action"], 0) + 1
    return {
        "model": "evidence_aware_specialist_council_v1.20",
        "generated_at": now,
        "macro_risk_state": macro_state,
        "macro_confidence_multiplier": macro_mult,
        "symbols": symbols,
        "action_counts": counts,
        "advisory_symbol_caps": advisory_caps,
        "directional_council_alpha_lifecycle": str(alpha_policy.get("lifecycle") or "unavailable"),
        "support_shadow_research_allowed": support_shadow_allowed,
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
        "council_can_only_preserve_reduce_or_veto": True,
    }


def apply_council_caps(packs, council):
    caps = council.get("advisory_symbol_caps") if isinstance(council, dict) else {}
    caps = caps if isinstance(caps, dict) else {}
    symbols = council.get("symbols") if isinstance(council, dict) else {}
    symbols = symbols if isinstance(symbols, dict) else {}
    for pack in packs or []:
        if str(pack.get("pack_id") or "") not in DIRECTIONAL_PACK_SET:
            continue
        for row in pack.get("observations") or []:
            if not isinstance(row, dict) or row.get("kind") != "signal":
                continue
            symbol = normalize_symbol(row.get("symbol") or "")
            cap = clamp(caps.get(symbol, 1.0), 0.0, 1.0)
            before = finite(row.get("confidence"))
            row["confidence"] = clamp(before * cap, 0.0, before)
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            meta["pre_council_confidence"] = before
            meta["council_confidence_cap_multiplier"] = cap
            meta["council_action"] = (symbols.get(symbol) or {}).get("action", "abstain")
            meta["council_can_only_preserve_reduce_or_veto"] = True
            row["metadata"] = meta
    return packs


def council_observation(council, state, now):
    resolved = state.get("council_resolved") if isinstance(state.get("council_resolved"), list) else []
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.97,
        "source": "LeanTrader bounded specialist council + costed council shadow ledger",
        "provenance": f"leantrader:specialist-council:v1.20:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(900, INTERVAL * 4),
        "metadata": {
            "model": "evidence_aware_specialist_council_v1.20",
            "directional_claim": False,
            "risk_increase_allowed": False,
            "council": council,
            "shadow_metrics": metric_stats(resolved),
            "pending_episodes": len(state.get("council_pending") or {}),
            "resolved_episodes": len(resolved),
            "execution_authority": False,
        },
    }


def resolve_council_episodes(state, now):
    pending = state.get("council_pending") if isinstance(state.get("council_pending"), dict) else {}
    resolved = state.get("council_resolved") if isinstance(state.get("council_resolved"), list) else []
    closed = 0
    price_cache = {}
    for episode_id, row in list(pending.items()):
        if now < finite(row.get("due_at")):
            continue
        symbol = normalize_symbol(row.get("symbol") or "")
        if symbol not in price_cache:
            try:
                price_cache[symbol] = current_spot_price(symbol)
            except Exception:
                price_cache[symbol] = None
        exit_price = price_cache.get(symbol)
        entry = finite(row.get("entry_price"))
        if not exit_price or entry <= 0:
            if now - finite(row.get("due_at"), now) > 3600:
                pending.pop(episode_id, None)
            continue
        direction = 1.0 if finite(row.get("signal")) > 0 else -1.0
        gross = direction * (exit_price / entry - 1.0)
        cost = ROUND_TRIP_COST_BPS / 10_000.0
        net = gross - cost
        inverse_net = -gross - cost
        resolved.append({
            **row,
            "exit_price": exit_price,
            "closed_at": now,
            "gross_return": gross,
            "net_return": net,
            "inverse_net_return": inverse_net,
            "edge_vs_inverse": net - inverse_net,
            "profitable_after_cost": net > 0,
            "evidence_authority": "sidecar_council_shadow_only",
        })
        pending.pop(episode_id, None)
        closed += 1
    if len(resolved) > MAX_SPECIALIST_RESOLVED:
        resolved = resolved[-MAX_SPECIALIST_RESOLVED:]
    state["council_pending"] = pending
    state["council_resolved"] = resolved
    return closed


def open_council_episodes(state, council, now):
    pending = state.get("council_pending") if isinstance(state.get("council_pending"), dict) else {}
    opened = 0
    for symbol, row in (council.get("symbols") or {}).items():
        if not isinstance(row, dict):
            continue
        action = str(row.get("action") or "abstain")
        score = finite(row.get("advisory_direction_score"))
        if action not in {"support_shadow", "reduce"} or abs(score) < 0.20:
            continue
        try:
            price = finite(current_spot_price(symbol))
        except Exception:
            price = 0.0
        if price <= 0:
            continue
        # One council episode per symbol/time bucket; this is research-only.
        bucket = int(now // max(300, INTERVAL))
        raw_id = f"council|{symbol}|{bucket}|900|{1 if score > 0 else -1}"
        episode_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
        if episode_id in pending:
            continue
        pending[episode_id] = {
            "episode_id": episode_id,
            "pack_id": PACK_COUNCIL,
            "symbol": normalize_symbol(symbol),
            "signal": score,
            "confidence": clamp(abs(score) * finite(row.get("confidence_cap_multiplier"), 1.0), 0.0, 1.0),
            "entry_price": price,
            "opened_at": now,
            "due_at": now + 900,
            "horizon_seconds": 900,
            "regime": "mixed_council",
            "symbol_class": symbol_class(symbol),
            "liquidity_state": "unknown",
            "out_of_distribution": bool(row.get("out_of_distribution")),
            "action": action,
            "execution_authority": False,
        }
        opened += 1
    state["council_pending"] = pending
    return opened

def council_counterfactual_metrics(rows):
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    def summary(subset):
        subset = list(subset)
        if not subset:
            return {
                "samples": 0, "helpful_decisions": 0, "helpful_rate": 0.0,
                "average_baseline_net_return": 0.0, "average_decision_net_return": 0.0,
                "average_decision_value_vs_baseline": 0.0, "total_avoided_loss": 0.0,
                "total_opportunity_cost": 0.0, "baseline_win_rate": 0.0,
            }
        baseline = [finite(r.get("baseline_net_return")) for r in subset]
        decision = [finite(r.get("decision_net_return")) for r in subset]
        value = [finite(r.get("decision_value_vs_baseline")) for r in subset]
        helpful = sum(v > 0 for v in value)
        return {
            "samples": len(subset),
            "helpful_decisions": helpful,
            "helpful_rate": helpful / len(subset),
            "average_baseline_net_return": sum(baseline) / len(subset),
            "average_decision_net_return": sum(decision) / len(subset),
            "average_decision_value_vs_baseline": sum(value) / len(subset),
            "total_avoided_loss": sum(max(0.0, v) for v in value),
            "total_opportunity_cost": sum(max(0.0, -v) for v in value),
            "baseline_win_rate": sum(v > 0 for v in baseline) / len(subset),
        }
    actions = sorted({str(r.get("action") or "unknown") for r in rows})
    out = {"global": summary(rows), "actions": {a: summary([r for r in rows if str(r.get("action") or "unknown") == a]) for a in actions}}
    veto = [r for r in rows if str(r.get("action")) == "veto_recommended"]
    out["veto_precision"] = (sum(finite(r.get("baseline_net_return")) <= 0 for r in veto) / len(veto)) if veto else 0.0
    out["execution_authority"] = False
    out["can_increase_risk"] = False
    return out


def resolve_council_counterfactuals(state, now):
    pending = state.get("council_counterfactual_pending") if isinstance(state.get("council_counterfactual_pending"), dict) else {}
    resolved = state.get("council_counterfactual_resolved") if isinstance(state.get("council_counterfactual_resolved"), list) else []
    price_cache = {}
    closed = 0
    for episode_id, row in list(pending.items()):
        if now < finite(row.get("due_at")):
            continue
        symbol = normalize_symbol(row.get("symbol") or "")
        if symbol not in price_cache:
            try:
                price_cache[symbol] = current_spot_price(symbol)
            except Exception:
                price_cache[symbol] = None
        exit_price = price_cache.get(symbol)
        entry = finite(row.get("entry_price"))
        if not exit_price or entry <= 0:
            if now - finite(row.get("due_at"), now) > 3600:
                pending.pop(episode_id, None)
            continue
        direction = 1.0 if finite(row.get("signal")) > 0 else -1.0
        gross = direction * (exit_price / entry - 1.0)
        cost = ROUND_TRIP_COST_BPS / 10_000.0
        baseline_net = gross - cost
        action = str(row.get("action") or "abstain")
        cap = clamp(finite(row.get("confidence_cap_multiplier"), 1.0), 0.0, 1.0)
        if action == "support_shadow":
            exposure = 1.0
        elif action == "reduce":
            exposure = cap
        else:
            exposure = 0.0
        decision_net = exposure * baseline_net
        decision_value = decision_net - baseline_net
        resolved.append({
            **row,
            "exit_price": exit_price,
            "closed_at": now,
            "gross_return": gross,
            "baseline_net_return": baseline_net,
            "decision_exposure_fraction": exposure,
            "decision_net_return": decision_net,
            "decision_value_vs_baseline": decision_value,
            "decision_helped": decision_value > 0,
            "avoided_loss": max(0.0, decision_value),
            "opportunity_cost": max(0.0, -decision_value),
            "baseline_profitable_after_cost": baseline_net > 0,
            "evidence_authority": "sidecar_council_counterfactual_only",
            "execution_authority": False,
        })
        pending.pop(episode_id, None)
        closed += 1
    if len(resolved) > MAX_SPECIALIST_RESOLVED:
        resolved = resolved[-MAX_SPECIALIST_RESOLVED:]
    state["council_counterfactual_pending"] = pending
    state["council_counterfactual_resolved"] = resolved
    return closed


def open_council_counterfactuals(state, council, now):
    pending = state.get("council_counterfactual_pending") if isinstance(state.get("council_counterfactual_pending"), dict) else {}
    opened = 0
    for symbol, row in (council.get("symbols") or {}).items():
        if not isinstance(row, dict):
            continue
        score = finite(row.get("advisory_direction_score"))
        if abs(score) < 0.20:
            continue
        action = str(row.get("action") or "abstain")
        try:
            price = finite(current_spot_price(symbol))
        except Exception:
            price = 0.0
        if price <= 0:
            continue
        # At most one counterfactual decision sample per symbol per 15-minute bucket.
        bucket = int(now // 900)
        raw_id = f"council-cf|{symbol}|{bucket}|{action}|{1 if score > 0 else -1}"
        episode_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
        if episode_id in pending:
            continue
        pending[episode_id] = {
            "episode_id": episode_id,
            "pack_id": PACK_COUNCIL_COUNTERFACTUAL,
            "symbol": normalize_symbol(symbol),
            "signal": score,
            "action": action,
            "confidence_cap_multiplier": clamp(finite(row.get("confidence_cap_multiplier"), 1.0), 0.0, 1.0),
            "entry_price": price,
            "opened_at": now,
            "due_at": now + 900,
            "horizon_seconds": 900,
            "macro_risk_state": str(row.get("macro_risk_state") or "unknown"),
            "out_of_distribution": bool(row.get("out_of_distribution")),
            "reasons": list(row.get("reasons") or []),
            "execution_authority": False,
        }
        opened += 1
    state["council_counterfactual_pending"] = pending
    return opened


def council_counterfactual_observation(state, now):
    resolved = state.get("council_counterfactual_resolved") if isinstance(state.get("council_counterfactual_resolved"), list) else []
    metrics = council_counterfactual_metrics(resolved)
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.97,
        "source": "LeanTrader council action counterfactual ledger",
        "provenance": f"leantrader:council-counterfactual:v1.14:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(900, INTERVAL * 4),
        "metadata": {
            "model": "council_action_counterfactual_evidence_v1.14",
            "directional_claim": False,
            "metrics": metrics,
            "pending": len(state.get("council_counterfactual_pending") or {}),
            "resolved": len(resolved),
            "execution_authority": False,
            "can_increase_risk": False,
            "vetoes_are_measured_for_avoided_loss_and_opportunity_cost": True,
        },
    }


def resolve_specialist_episodes(state, now):
    pending = state.get("specialist_pending") if isinstance(state.get("specialist_pending"), dict) else {}
    resolved = state.get("specialist_resolved") if isinstance(state.get("specialist_resolved"), list) else []
    closed = 0
    price_cache = {}
    for episode_id, row in list(pending.items()):
        if now < finite(row.get("due_at")):
            continue
        symbol = normalize_symbol(row.get("symbol") or "")
        if symbol not in price_cache:
            try:
                price_cache[symbol] = current_spot_price(symbol)
            except Exception:
                price_cache[symbol] = None
        exit_price = price_cache.get(symbol)
        entry = finite(row.get("entry_price"))
        if not exit_price or entry <= 0:
            if now - finite(row.get("due_at"), now) > 3600:
                pending.pop(episode_id, None)
            continue
        direction = 1.0 if finite(row.get("signal")) > 0 else -1.0
        gross = direction * (exit_price / entry - 1.0)
        cost = ROUND_TRIP_COST_BPS / 10_000.0
        net = gross - cost
        inverse_net = -gross - cost
        resolved.append({
            **row,
            "exit_price": exit_price,
            "closed_at": now,
            "gross_return": gross,
            "net_return": net,
            "inverse_net_return": inverse_net,
            "edge_vs_inverse": net - inverse_net,
            "profitable_after_cost": net > 0,
            "evidence_authority": "sidecar_regime_shadow_only",
        })
        pending.pop(episode_id, None)
        closed += 1
    if len(resolved) > MAX_SPECIALIST_RESOLVED:
        resolved = resolved[-MAX_SPECIALIST_RESOLVED:]
    state["specialist_pending"] = pending
    state["specialist_resolved"] = resolved
    return closed


def _episode_independence_key(pack_id, symbol, hypothesis_id, horizon_seconds):
    """Stable key for one shadow thesis whose forward windows must not overlap.

    Pack + normalized symbol + hypothesis + horizon is intentionally conservative:
    repeated confirmations of the same thesis during an already-open outcome window
    strengthen persistence diagnostics but do not manufacture new independent samples.
    """
    return "|".join([
        str(pack_id or ""),
        normalize_symbol(symbol or ""),
        str(hypothesis_id or "__default__"),
        str(int(horizon_seconds or 0)),
    ])


def _resolved_independence_key(row):
    return _episode_independence_key(
        row.get("pack_id"), row.get("symbol"), row.get("hypothesis_id"), row.get("horizon_seconds")
    )


def nonoverlapping_resolved_rows(rows):
    """Return a chronological non-overlapping subset without deleting raw evidence.

    This is an effective-sample filter.  It prevents six highly overlapping 30-minute
    shadows opened five minutes apart from counting as six independent confirmations.
    Direction is deliberately *not* part of the key: an opposite signal inside an
    already-measured window is correlated evidence, not a fresh independent trial.
    """
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    rows.sort(key=lambda r: (finite(r.get("opened_at")), finite(r.get("closed_at")), str(r.get("episode_id") or "")))
    last_close = {}
    kept = []
    for row in rows:
        key = _resolved_independence_key(row)
        opened = finite(row.get("opened_at"))
        closed = finite(row.get("closed_at"), opened + int(row.get("horizon_seconds") or 0))
        if opened <= 0:
            continue
        prior_close = finite(last_close.get(key))
        if prior_close > 0 and opened < prior_close:
            continue
        kept.append(row)
        last_close[key] = max(opened, closed)
    return kept


def shadow_evidence_independence(state):
    resolved = state.get("specialist_resolved") if isinstance(state.get("specialist_resolved"), list) else []
    by_pack = {}
    for pid in sorted(SPECIALIST_EPISODE_PACK_SET):
        raw = [r for r in resolved if isinstance(r, dict) and r.get("pack_id") == pid]
        effective = nonoverlapping_resolved_rows(raw)
        raw_n = len(raw); eff_n = len(effective)
        by_pack[pid] = {
            "raw_samples": raw_n,
            "effective_nonoverlap_samples": eff_n,
            "overlapping_samples_discounted": max(0, raw_n - eff_n),
            "effective_sample_ratio": (eff_n / raw_n) if raw_n else 1.0,
            "prequential_policy_uses_effective_samples": pid in INDEPENDENCE_GOVERNED_PACK_SET,
            "execution_authority": False,
        }
    return {
        "model": "shadow_evidence_independence_governance_v1.24",
        "by_pack": by_pack,
        "raw_samples_total": sum(v["raw_samples"] for v in by_pack.values()),
        "effective_nonoverlap_samples_total": sum(v["effective_nonoverlap_samples"] for v in by_pack.values()),
        "overlapping_samples_discounted_total": sum(v["overlapping_samples_discounted"] for v in by_pack.values()),
        "raw_evidence_is_preserved": True,
        "overlap_cannot_increase_effective_sample_size": True,
        "execution_authority": False,
        "can_increase_risk": False,
        "can_enable_live": False,
    }


def episode_independence_observation(episode_status, evidence_status, now):
    return {
        "symbol": "GLOBAL", "kind": "context", "score": 0.0, "confidence": 0.99,
        "source": "LeanTrader independent shadow evidence governance",
        "provenance": f"leantrader:shadow-evidence-independence:v1.24:{int(now)}",
        "observed_at": now, "horizon_seconds": max(900, INTERVAL * 4),
        "metadata": {
            "model": "shadow_evidence_independence_governance_v1.24",
            "episode_opening": dict(episode_status or {}),
            "resolved_evidence": dict(evidence_status or {}),
            "directional_claim": False,
            "execution_authority": False,
            "can_increase_risk": False,
            "can_enable_live": False,
        },
    }


def open_specialist_episodes(state, packs, now):
    pending = state.get("specialist_pending") if isinstance(state.get("specialist_pending"), dict) else {}
    opened = 0
    status = {
        "model": "shadow_episode_nonoverlap_guard_v1.24",
        "guard_enabled": bool(EPISODE_OVERLAP_GUARD_ENABLED),
        "candidate_signals_considered": 0,
        "episodes_opened": 0,
        "overlapping_confirmations_merged": 0,
        "opposite_direction_conflicts_suppressed": 0,
        "price_unavailable": 0,
        "independent_keys_pending": 0,
        "execution_authority": False,
        "can_increase_risk": False,
    }
    for pack in packs:
        pack_id = str(pack.get("pack_id") or "")
        if pack_id not in SPECIALIST_EPISODE_PACK_SET:
            continue
        for row in pack.get("observations") or []:
            if row.get("kind") != "signal" or abs(finite(row.get("score"))) < 0.05:
                continue
            status["candidate_signals_considered"] += 1
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            context = meta.get("market_context") if isinstance(meta.get("market_context"), dict) else {}
            symbol = normalize_symbol(row.get("symbol") or "")
            observed = finite(row.get("observed_at"), now)
            horizon = int(row.get("horizon_seconds") or 900)
            hypothesis_id = str(meta.get("hypothesis_id") or "")
            independence_key = _episode_independence_key(pack_id, symbol, hypothesis_id, horizon)
            direction = 1 if finite(row.get("score")) > 0 else -1

            overlapping = None
            if EPISODE_OVERLAP_GUARD_ENABLED:
                for existing in pending.values():
                    if not isinstance(existing, dict):
                        continue
                    existing_key = str(existing.get("independence_key") or _episode_independence_key(
                        existing.get("pack_id"), existing.get("symbol"), existing.get("hypothesis_id"), existing.get("horizon_seconds")
                    ))
                    if existing_key != independence_key:
                        continue
                    # Only an outcome window that is still open at this signal timestamp blocks a new trial.
                    if finite(existing.get("due_at")) > observed:
                        overlapping = existing
                        break
            if overlapping is not None:
                prior_dir = 1 if finite(overlapping.get("signal")) > 0 else -1
                overlapping["last_observed_at"] = observed
                overlapping["observation_count"] = int(overlapping.get("observation_count") or 1) + 1
                overlapping["max_observed_confidence"] = max(
                    finite(overlapping.get("max_observed_confidence"), overlapping.get("confidence")),
                    finite(row.get("confidence")),
                )
                if prior_dir == direction:
                    overlapping["confirmation_count"] = int(overlapping.get("confirmation_count") or 1) + 1
                    overlapping["last_confirmation_at"] = observed
                    overlapping["persistence_confirmations"] = int(overlapping.get("persistence_confirmations") or 0) + 1
                    status["overlapping_confirmations_merged"] += 1
                else:
                    overlapping["direction_conflicts"] = int(overlapping.get("direction_conflicts") or 0) + 1
                    overlapping["last_direction_conflict_at"] = observed
                    status["opposite_direction_conflicts_suppressed"] += 1
                # Crucially, repeat observations never raise the episode's original evidence confidence.
                continue

            price = finite(meta.get("mid_price"))
            if price <= 0:
                try:
                    price = finite(current_spot_price(symbol))
                except Exception:
                    price = 0.0
            if price <= 0:
                status["price_unavailable"] += 1
                continue
            raw_id = f"{pack_id}|{symbol}|{observed:.3f}|{horizon}|{row.get('provenance')}"
            episode_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
            if episode_id in pending:
                continue
            pending[episode_id] = {
                "episode_id": episode_id,
                "independence_key": independence_key,
                "pack_id": pack_id,
                "symbol": symbol,
                "signal": finite(row.get("score")),
                "confidence": finite(row.get("confidence")),
                "max_observed_confidence": finite(row.get("confidence")),
                "observation_count": 1,
                "confirmation_count": 1,
                "persistence_confirmations": 0,
                "direction_conflicts": 0,
                "entry_price": price,
                "opened_at": observed,
                "last_observed_at": observed,
                "last_confirmation_at": observed,
                "due_at": observed + horizon,
                "horizon_seconds": horizon,
                "regime": str(context.get("regime") or "unknown"),
                "symbol_class": str(context.get("symbol_class") or symbol_class(symbol)),
                "liquidity_state": str(context.get("liquidity_state") or "unknown"),
                "out_of_distribution": bool(context.get("out_of_distribution")),
                "recovery_probe": bool(meta.get("recovery_probe")),
                "routing_reason": str(meta.get("routing_reason") or "eligible"),
                "council_eligible_at_open": bool(meta.get("council_eligible", True)),
                "hypothesis_id": hypothesis_id,
                "source_families": [str(v) for v in (meta.get("source_families") or [])[:8]],
                "sensor_snapshot_timestamp": finite(meta.get("sensor_snapshot_timestamp")),
                "execution_authority": False,
            }
            opened += 1
    status["episodes_opened"] = opened
    status["independent_keys_pending"] = len({
        str(v.get("independence_key") or _episode_independence_key(v.get("pack_id"), v.get("symbol"), v.get("hypothesis_id"), v.get("horizon_seconds")))
        for v in pending.values() if isinstance(v, dict)
    })
    state["specialist_pending"] = pending
    state["specialist_episode_independence"] = status
    return opened


def reputation_pack_observation(state, now):
    resolved = state.get("specialist_resolved") if isinstance(state.get("specialist_resolved"), list) else []
    summaries = {}
    for pack_id in DIRECTIONAL_PACKS:
        rows = [r for r in resolved if isinstance(r, dict) and r.get("pack_id") == pack_id]
        if not rows:
            summaries[pack_id] = {"global": metric_stats([]), "regimes": {}, "symbol_classes": {}, "liquidity_states": {}}
            continue
        regimes = sorted({str(r.get("regime") or "unknown") for r in rows})
        classes = sorted({str(r.get("symbol_class") or "unknown") for r in rows})
        liqs = sorted({str(r.get("liquidity_state") or "unknown") for r in rows})
        summaries[pack_id] = {
            "global": metric_stats(rows),
            "regimes": {k: metric_stats([r for r in rows if str(r.get("regime") or "unknown") == k]) for k in regimes},
            "symbol_classes": {k: metric_stats([r for r in rows if str(r.get("symbol_class") or "unknown") == k]) for k in classes},
            "liquidity_states": {k: metric_stats([r for r in rows if str(r.get("liquidity_state") or "unknown") == k]) for k in liqs},
        }
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.95,
        "source": "LeanTrader sidecar-owned costed specialist outcome ledger",
        "provenance": f"leantrader:specialist-reputation:v1.20:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(900, INTERVAL * 4),
        "metadata": {
            "model": "hierarchical_regime_specialist_reputation_v1.14",
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "directional_claim": False,
            "risk_increase_allowed": False,
            "summaries": summaries,
            "pending_episodes": len(state.get("specialist_pending") or {}),
            "resolved_episodes": len(resolved),
        },
    }

def select_symbols(limit=16):
    selected = []
    hb = load_json(RUNTIME / "vps_heartbeat.json")
    for s in hb.get("open_positions") or []:
        s = normalize_symbol(s)
        if s and s not in selected:
            selected.append(s)
    world = load_json(RUNTIME / "vps_market_world_model.json")
    cand = []

    def walk(x, depth=0):
        if depth > 8:
            return
        if isinstance(x, dict):
            s = normalize_symbol(x.get("symbol") or "")
            if s.endswith("/USDT"):
                n = finite(x.get("novelty"))
                sc = abs(finite(x.get("score")))
                ts = finite(x.get("timestamp") or x.get("observed_at"))
                if n > 0 or sc > 0:
                    cand.append((n + sc, ts, s))
            for v in x.values():
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)
        elif isinstance(x, list):
            for v in x[-1200:]:
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)

    walk(world)
    cand.sort(reverse=True)
    for _, _, s in cand:
        if s not in selected:
            selected.append(s)
        if len(selected) >= limit:
            break
    for s in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT"]:
        if s not in selected:
            selected.append(s)
        if len(selected) >= limit:
            break
    return selected[:limit]


def raw_microstructure(symbol, now):
    sym = bybit_symbol(symbol)
    q1 = urllib.parse.urlencode({"category": "spot", "symbol": sym, "limit": 50})
    q2 = urllib.parse.urlencode({"category": "spot", "symbol": sym, "limit": 60})
    ob = request_json(f"{BYBIT}/v5/market/orderbook?{q1}")
    tr = request_json(f"{BYBIT}/v5/market/recent-trade?{q2}")
    if int(ob.get("retCode", -1)) != 0 or int(tr.get("retCode", -1)) != 0:
        return None
    book = ob.get("result") or {}
    bids = book.get("b") or []
    asks = book.get("a") or []

    def notional(level):
        try:
            return float(level[0]) * float(level[1])
        except Exception:
            return 0.0

    bn = [notional(x) for x in bids if isinstance(x, list)]
    an = [notional(x) for x in asks if isinstance(x, list)]
    bt, at = sum(bn), sum(an)
    total = bt + at
    if total <= 0:
        return None
    bi = (bt - at) / total
    dc = max(max(bn or [0.0]) / max(bt, 1e-12), max(an or [0.0]) / max(at, 1e-12))
    trades = (tr.get("result") or {}).get("list") or []
    buys = sells = tn = 0.0
    notionals = []
    for r in trades:
        if not isinstance(r, dict):
            continue
        n = finite(r.get("price")) * finite(r.get("size"))
        if n <= 0:
            continue
        tn += n
        notionals.append(n)
        side = str(r.get("side") or "").lower()
        if side == "buy":
            buys += n
        elif side == "sell":
            sells += n
    fi = (buys - sells) / max(buys + sells, 1e-12) if buys + sells > 0 else 0.0
    lc = sum(sorted(notionals, reverse=True)[:5]) / max(tn, 1e-12) if tn > 0 else 0.0
    score = clamp(0.35 * bi + 0.65 * fi)
    best_bid = finite(bids[0][0]) if bids and isinstance(bids[0], list) else 0.0
    best_ask = finite(asks[0][0]) if asks and isinstance(asks[0], list) else 0.0
    mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
    spread_bps = ((best_ask - best_bid) / mid_price * 10_000.0) if mid_price > 0 and best_ask >= best_bid else 0.0
    return {
        "symbol": normalize_symbol(symbol),
        "score": score,
        "mid_price": mid_price,
        "spread_bps": spread_bps,
        "book_notional": total,
        "trade_notional": tn,
        "book_imbalance": bi,
        "trade_flow_imbalance": fi,
        "depth_concentration": dc,
        "large_trade_concentration": lc,
        "trade_samples": len(trades),
        "book_bid_levels": len(bids),
        "book_ask_levels": len(asks),
        "observed_at": now,
        "provenance": f"bybit:v5:spot:{sym}:{int(now)}",
    }



def _book_pressure(levels_bid, levels_ask, *, received_at=None, remote_ts=None):
    def notionals(levels):
        out=[]
        for level in levels or []:
            try:
                out.append(float(level[0]) * float(level[1]))
            except Exception:
                continue
        return out
    bn, an = notionals(levels_bid), notionals(levels_ask)
    bt, at = sum(bn), sum(an)
    total = bt + at
    if total <= 0:
        return None
    try:
        best_bid = float((levels_bid or [])[0][0])
        best_ask = float((levels_ask or [])[0][0])
    except Exception:
        best_bid = best_ask = 0.0
    mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
    spread_bps = ((best_ask - best_bid) / mid_price * 10_000.0) if mid_price > 0 and best_ask >= best_bid else 0.0
    return {
        "book_imbalance": clamp((bt - at) / total),
        "book_notional": total,
        "bid_levels": len(bn),
        "ask_levels": len(an),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "spread_bps": spread_bps,
        "received_at": finite(received_at, time.time()),
        "remote_ts": finite(remote_ts),
    }


def _binance_orderbook(symbol):
    sym = bybit_symbol(symbol)
    q = urllib.parse.urlencode({"symbol": sym, "limit": 100})
    try:
        payload = request_json(f"{BINANCE_PUBLIC}/api/v3/depth?{q}", timeout=8)
    except urllib.error.HTTPError as exc:
        if int(getattr(exc, "code", 0) or 0) in {400, 404}:
            return None, "symbol_unavailable"
        return None, f"http_{getattr(exc, 'code', 'error')}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    row = _book_pressure(payload.get("bids") or [], payload.get("asks") or [], received_at=time.time())
    if not row or row["bid_levels"] < 20 or row["ask_levels"] < 20:
        return None, "insufficient_depth"
    row.update({"venue": "binance", "symbol": normalize_symbol(symbol)})
    return row, "available"


def _okx_orderbook(symbol):
    inst = normalize_symbol(symbol).replace("/", "-")
    q = urllib.parse.urlencode({"instId": inst, "sz": 100})
    try:
        payload = request_json(f"{OKX_PUBLIC}/api/v5/market/books?{q}", timeout=8)
    except urllib.error.HTTPError as exc:
        if int(getattr(exc, "code", 0) or 0) in {400, 404}:
            return None, "symbol_unavailable"
        return None, f"http_{getattr(exc, 'code', 'error')}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if str(payload.get("code") or "") != "0":
        return None, "symbol_unavailable"
    data = payload.get("data") or []
    if not data or not isinstance(data[0], dict):
        return None, "symbol_unavailable"
    remote_ts = finite(data[0].get("ts")) / 1000.0 if finite(data[0].get("ts")) > 10_000_000_000 else finite(data[0].get("ts"))
    row = _book_pressure(data[0].get("bids") or [], data[0].get("asks") or [], received_at=time.time(), remote_ts=remote_ts)
    if not row or row["bid_levels"] < 20 or row["ask_levels"] < 20:
        return None, "insufficient_depth"
    row.update({"venue": "okx", "symbol": normalize_symbol(symbol)})
    return row, "available"


def _venue_quality(bybit_row, venue_row, now=None):
    """Bounded data-quality score for an external venue snapshot.

    Quality can only reject/downweight evidence. It never increases risk or
    execution authority. Price-basis mismatches are treated as context rather
    than directional confirmation.
    """
    now = finite(now, time.time())
    ref_mid = finite(bybit_row.get("mid_price"))
    mid = finite(venue_row.get("mid_price"))
    spread_bps = max(0.0, finite(venue_row.get("spread_bps")))
    bid_levels = int(venue_row.get("bid_levels") or 0)
    ask_levels = int(venue_row.get("ask_levels") or 0)
    price_dev_bps = abs(mid / ref_mid - 1.0) * 10_000.0 if mid > 0 and ref_mid > 0 else 10_000.0
    remote_ts = finite(venue_row.get("remote_ts"))
    freshness_seconds = max(0.0, now - remote_ts) if remote_ts > 0 else None

    price_q = clamp(1.0 - price_dev_bps / 100.0, 0.0, 1.0)
    spread_q = clamp(1.0 - spread_bps / 60.0, 0.0, 1.0)
    depth_q = clamp(min(bid_levels, ask_levels) / 100.0, 0.0, 1.0)
    freshness_q = 0.85 if freshness_seconds is None else clamp(1.0 - freshness_seconds / 10.0, 0.0, 1.0)
    quality = clamp(0.40 * price_q + 0.30 * spread_q + 0.20 * depth_q + 0.10 * freshness_q, 0.0, 1.0)

    reasons = []
    if mid <= 0 or ref_mid <= 0:
        reasons.append("missing_price_basis")
    if price_dev_bps > 80.0:
        reasons.append("price_basis_mismatch")
    if spread_bps > 50.0:
        reasons.append("wide_spread")
    if min(bid_levels, ask_levels) < 20:
        reasons.append("insufficient_depth")
    if freshness_seconds is not None and freshness_seconds > 8.0:
        reasons.append("stale_remote_timestamp")
    if quality < 0.55:
        reasons.append("low_composite_quality")
    trusted = not reasons
    return {
        "venue": str(venue_row.get("venue") or "unknown"),
        "trusted_confirmation": trusted,
        "quality_score": quality,
        "price_deviation_bps_vs_bybit": price_dev_bps,
        "spread_bps": spread_bps,
        "bid_levels": bid_levels,
        "ask_levels": ask_levels,
        "freshness_seconds": freshness_seconds,
        "rejection_reasons": reasons,
        "execution_authority": False,
        "can_increase_risk": False,
    }


def venue_quality_observation(bybit_row, external_rows, now=None):
    now = finite(now, time.time())
    assessments = [_venue_quality(bybit_row, r, now=now) for r in (external_rows or []) if isinstance(r, dict)]
    trusted = [r for r in assessments if r.get("trusted_confirmation")]
    rejected = [r for r in assessments if not r.get("trusted_confirmation")]
    return {
        "symbol": normalize_symbol(bybit_row.get("symbol") or ""),
        "kind": "context",
        "score": 0.0,
        "confidence": 0.97,
        "source": "cross-venue public order-book integrity checks",
        "provenance": f"cross-venue-quality-v1.14:{normalize_symbol(bybit_row.get('symbol') or '')}:{int(now)}",
        "observed_at": now,
        "horizon_seconds": 900,
        "metadata": {
            "model": "cross_venue_quality_integrity_v1.14",
            "bybit_mid_price": finite(bybit_row.get("mid_price")),
            "bybit_spread_bps": finite(bybit_row.get("spread_bps")),
            "assessments": assessments,
            "trusted_external_venues": [r.get("venue") for r in trusted],
            "rejected_external_venues": [r.get("venue") for r in rejected],
            "trusted_count": len(trusted),
            "rejected_count": len(rejected),
            "price_basis_integrity_required": True,
            "execution_authority": False,
            "risk_increase_allowed": False,
        },
    }


def cross_venue_observation(bybit_row, previous, cal, reputation=None, context=None, now=None, external_rows=None):
    """Research-only cross-venue order-book persistence challenger.

    A signal requires Bybit plus at least one independent public venue to agree,
    and the aggregate direction must persist across sidecar cycles. OOD signals
    require both external venues. This pack has no execution authority.
    """
    now = finite(now, time.time())
    reputation = reputation or {}
    context = context or {}
    rows = []
    statuses = {}
    if external_rows is None:
        for name, fn in (("binance", _binance_orderbook), ("okx", _okx_orderbook)):
            row, status = fn(bybit_row.get("symbol") or "")
            statuses[name] = status
            if row:
                rows.append(row)
    else:
        rows = list(external_rows or [])
        statuses = {str(r.get("venue") or "external"): "available" for r in rows if isinstance(r, dict)}
    quality_assessments = [_venue_quality(bybit_row, r, now=now) for r in rows if isinstance(r, dict)]
    trusted_names = {str(q.get("venue")) for q in quality_assessments if q.get("trusted_confirmation")}
    trusted_external = [r for r in rows if isinstance(r, dict) and str(r.get("venue")) in trusted_names]
    rejected_external = [r for r in rows if isinstance(r, dict) and str(r.get("venue")) not in trusted_names]
    venue_rows = [{
        "venue": "bybit",
        "book_imbalance": finite(bybit_row.get("book_imbalance")),
        "book_notional": finite(bybit_row.get("book_notional")),
        "mid_price": finite(bybit_row.get("mid_price")),
        "spread_bps": finite(bybit_row.get("spread_bps")),
    }] + trusted_external
    imbalances = [finite(r.get("book_imbalance")) for r in venue_rows]
    external_count = len(trusted_external)
    raw_external_count = len(rows)
    same_sign = (bool(imbalances) and all(v > 0 for v in imbalances)) or (bool(imbalances) and all(v < 0 for v in imbalances))
    weights = [1.0]
    for r in trusted_external:
        q = next((finite(x.get("quality_score"), 0.55) for x in quality_assessments if str(x.get("venue")) == str(r.get("venue"))), 0.55)
        weights.append(clamp(q, 0.55, 1.0))
    strength = sum(v*w for v,w in zip(imbalances, weights)) / max(1e-12, sum(weights)) if imbalances else 0.0
    min_abs = min([abs(v) for v in imbalances] or [0.0])
    prior = finite((previous or {}).get("score"))
    persistent = bool(previous) and prior * strength > 0 and abs(prior) >= 0.08
    eligible = external_count >= 1 and same_sign and min_abs >= 0.04 and abs(strength) >= 0.08 and persistent
    reason = "persistent_cross_venue_depth_agreement" if eligible else "cross_venue_context_only"
    if raw_external_count < 1:
        reason = "no_independent_external_venue"
    elif external_count < 1:
        reason = "external_venue_quality_rejected"
    elif not same_sign:
        reason = "venue_direction_disagreement"
    elif min_abs < 0.04 or abs(strength) < 0.08:
        reason = "weak_cross_venue_pressure"
    elif not persistent:
        reason = "not_persistent_across_cycles"
    if context.get("out_of_distribution") and external_count < 2:
        eligible = False
        reason = "ood_requires_two_external_venues"
    if context.get("out_of_distribution") and abs(strength) < 0.16:
        eligible = False
        reason = "ood_requires_stronger_cross_venue_pressure"
    if reputation.get("hard_negative") and abs(strength) < 0.35:
        eligible = False
        reason = "regime_reputation_defensive_probe_mode"
    if cal.get("hard_negative") and abs(strength) < 0.35:
        eligible = False
        reason = "challenger_in_defensive_probe_mode"
    raw_conf = 0.34 + 0.08 * external_count + 0.22 * min(1.0, abs(strength)) + (0.08 if persistent else 0.0)
    raw_conf *= finite(reputation.get("confidence_multiplier"), 1.0)
    if context.get("out_of_distribution"):
        raw_conf *= 0.82
    conf = clamp(min(raw_conf, finite(cal.get("confidence_cap"), 0.68), 0.68), 0.20, 0.68)
    return {
        "symbol": normalize_symbol(bybit_row.get("symbol") or ""),
        "kind": "signal" if eligible else "context",
        "score": strength if eligible else 0.0,
        "confidence": conf if eligible else 0.50,
        "source": "Bybit + Binance + OKX public spot order books",
        "provenance": f"cross-venue-orderbook-v1.14:{normalize_symbol(bybit_row.get('symbol') or '')}:{int(now)}",
        "observed_at": now,
        "horizon_seconds": 900,
        "metadata": {
            "model": "cross_venue_orderbook_persistence_v1.14",
            "directional_eligible": eligible,
            "eligibility_reason": reason,
            "venue_count": len(venue_rows),
            "raw_external_venues": raw_external_count,
            "independent_external_venues": external_count,
            "trusted_external_venues": [str(r.get("venue")) for r in trusted_external],
            "rejected_external_venues": [str(r.get("venue")) for r in rejected_external],
            "venue_quality_assessments": quality_assessments,
            "venue_status": statuses,
            "venue_book_imbalances": {str(r.get("venue")): finite(r.get("book_imbalance")) for r in venue_rows},
            "consensus_strength": strength,
            "minimum_abs_venue_pressure": min_abs,
            "shared_source_with_bybit_microstructure": True,
            "source_lineage": [str(r.get("venue")) + "_orderbook" for r in venue_rows],
            "global_calibration": cal,
            "specialist_reputation": reputation,
            "market_context": context,
            "mid_price": finite(bybit_row.get("mid_price")),
            "risk_increase_allowed": False,
            "execution_authority": False,
        },
    }


def signal_eligible(row, previous, cal, reputation=None, context=None):
    reputation = reputation or {}
    context = context or {}
    book = finite(row.get("book_imbalance"))
    flow = finite(row.get("trade_flow_imbalance"))
    score = finite(row.get("score"))
    if int(row.get("trade_samples") or 0) < 40:
        return False, "insufficient_trade_samples"
    if int(row.get("book_bid_levels") or 0) < 20 or int(row.get("book_ask_levels") or 0) < 20:
        return False, "insufficient_book_depth"
    if book * flow <= 0 or abs(book) < 0.05 or abs(flow) < 0.10:
        return False, "book_flow_disagreement"
    if abs(score) < 0.18:
        return False, "weak_combined_pressure"
    prior = finite((previous or {}).get("score"))
    if not previous or prior * score <= 0 or abs(prior) < 0.12:
        return False, "not_persistent_across_cycles"
    if context.get("out_of_distribution") and (abs(score) < 0.35 or abs(prior) < 0.20):
        return False, "ood_requires_stronger_persistent_evidence"
    if reputation.get("hard_negative") and abs(score) < 0.50:
        return False, "regime_reputation_defensive_probe_mode"
    if cal.get("hard_negative") and abs(score) < 0.45:
        return False, "challenger_in_defensive_probe_mode"
    return True, "persistent_book_flow_agreement"


def confidence_for_signal(row, previous, cal, reputation=None, context=None):
    reputation = reputation or {}
    context = context or {}
    book = abs(finite(row.get("book_imbalance")))
    flow = abs(finite(row.get("trade_flow_imbalance")))
    score = abs(finite(row.get("score")))
    prior = abs(finite((previous or {}).get("score")))
    raw = 0.36 + 0.14 * min(1, book) + 0.18 * min(1, flow) + 0.14 * min(1, score) + 0.10 * min(1, prior)
    if cal.get("hard_negative"):
        raw *= 0.60
    raw *= finite(reputation.get("confidence_multiplier"), 1.0)
    if context.get("out_of_distribution"):
        raw *= 0.85
    cap = min(finite(cal.get("confidence_cap"), 0.72), 0.72)
    return clamp(min(raw, cap), 0.20, 0.72)


def micro_observation(row, previous, cal, horizon, model, reputation=None, context=None):
    reputation = reputation or {}
    context = context or {}
    ok, reason = signal_eligible(row, previous, cal, reputation=reputation, context=context)
    return {
        "symbol": row["symbol"],
        "kind": "signal" if ok else "context",
        "score": finite(row.get("score")) if ok else 0.0,
        "confidence": confidence_for_signal(row, previous, cal, reputation=reputation, context=context) if ok else 0.50,
        "source": "Bybit public V5 orderbook + public recent trades",
        "provenance": f"{row['provenance']}:{model}",
        "observed_at": row["observed_at"],
        "horizon_seconds": horizon,
        "metadata": {
            "model": model,
            "eligibility_reason": reason,
            "directional_eligible": ok,
            "book_imbalance": row["book_imbalance"],
            "trade_flow_imbalance": row["trade_flow_imbalance"],
            "combined_pressure": row["score"],
            "prior_combined_pressure": finite((previous or {}).get("score")),
            "trade_samples": row["trade_samples"],
            "book_bid_levels": row["book_bid_levels"],
            "book_ask_levels": row["book_ask_levels"],
            "book_notional": row.get("book_notional"),
            "trade_notional": row.get("trade_notional"),
            "mid_price": row.get("mid_price"),
            "global_calibration": cal,
            "specialist_reputation": reputation,
            "market_context": context,
            "risk_increase_allowed": False,
        },
    }

def whale_context(row):
    c = clamp(0.55 * finite(row.get("large_trade_concentration")) + 0.45 * finite(row.get("depth_concentration")), 0, 1)
    return {
        "symbol": row["symbol"],
        "kind": "context",
        "score": 0.0,
        "confidence": clamp(0.45 + 0.40 * c, 0.45, 0.85),
        "source": "Bybit public V5 orderbook + public recent trades",
        "provenance": f"{row['provenance']}:whale-context-v1.2",
        "observed_at": row["observed_at"],
        "horizon_seconds": 900,
        "metadata": {
            "model": "exchange_concentration_proxy_v1.2",
            "concentration_score": c,
            "depth_concentration": row["depth_concentration"],
            "large_trade_concentration": row["large_trade_concentration"],
            "proxy_notice": "exchange microstructure concentration only; not an on-chain wallet identity claim",
            "directional_claim": False,
        },
    }


def ecb_fx_context(now):
    params = urllib.parse.urlencode({"lastNObservations": 8, "format": "csvdata"})
    rows = list(csv.DictReader(io.StringIO(request_text(f"{ECB}/EXR/D.USD+GBP+JPY.EUR.SP00.A?{params}"))))
    series = {}
    for row in rows:
        up = {str(k).upper(): v for k, v in row.items()}
        cur = str(up.get("CURRENCY") or up.get("CURRENCY_DENOM") or up.get("KEY_VALUE") or "").upper()
        period = str(up.get("TIME_PERIOD") or up.get("TIME PERIOD") or "")
        val = finite(up.get("OBS_VALUE") or up.get("OBS VALUE"), float("nan"))
        if cur in {"USD", "GBP", "JPY"} and math.isfinite(val) and val > 0:
            series.setdefault(cur, []).append((period, val))
    if not series:
        return None
    snap = {}
    for cur, vals in series.items():
        vals.sort(key=lambda x: x[0])
        latest = vals[-1][1]
        earlier = vals[max(0, len(vals) - 5)][1]
        snap[cur] = {
            "latest_per_eur": latest,
            "change_vs_5obs": latest / earlier - 1 if earlier > 0 else 0.0,
            "latest_period": vals[-1][0],
        }
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.90,
        "source": "European Central Bank Data Portal EXR daily reference rates",
        "provenance": f"ecb:EXR:D:USD+GBP+JPY:EUR:{int(now)}",
        "observed_at": now,
        "horizon_seconds": 86400,
        "metadata": {"model": "official_fx_context_v1.2", "rates": snap, "directional_claim": False},
    }


def leader_return(symbol, lag_steps):
    sym = bybit_symbol(symbol)
    limit = min(20, max(4, lag_steps + 3))
    q = urllib.parse.urlencode({"category": "spot", "symbol": sym, "interval": "15", "limit": limit})
    payload = request_json(f"{BYBIT}/v5/market/kline?{q}")
    if int(payload.get("retCode", -1)) != 0:
        return None
    parsed = []
    for row in (payload.get("result") or {}).get("list") or []:
        try:
            parsed.append((int(row[0]), float(row[4])))
        except Exception:
            pass
    if len(parsed) < 3:
        return None
    parsed.sort()
    latest = parsed[-2][1]
    earlier = parsed[max(0, len(parsed) - 2 - max(1, lag_steps))][1]
    return latest / earlier - 1 if latest > 0 and earlier > 0 else None


def leadlag_observations(now, contexts=None, state=None):
    world = load_json(RUNTIME / "vps_market_world_model.json")
    market = world.get("market") if isinstance(world.get("market"), dict) else {}
    rows = market.get("lead_lag_research_candidates") or []
    cal = calibration(PACK_LEADLAG)
    contexts = contexts or {}
    state = state or {}
    out = []
    history = state.get("leadlag_history") if isinstance(state.get("leadlag_history"), dict) else {}
    # Prune stale candidate memory; this is research persistence, not trading memory.
    history = {k: v for k, v in history.items() if isinstance(v, dict) and now - finite(v.get("last_seen"), now) <= 7200}
    state["leadlag_history"] = history
    for idx, row in enumerate(rows[:8]):
        if not isinstance(row, dict):
            continue
        leader = normalize_symbol(row.get("leader") or "")
        follower = normalize_symbol(row.get("follower") or "")
        corr = clamp(finite(row.get("correlation")))
        strength = clamp(abs(finite(row.get("incremental_strength"))), 0, 1)
        lag = max(1, int(finite(row.get("lag_steps"), 1)))
        if not leader.endswith("/USDT") or not follower.endswith("/USDT") or strength < 0.25 or abs(corr) < 0.25:
            continue
        try:
            move = leader_return(leader, lag)
        except Exception:
            move = None
        if move is None:
            continue
        pred = clamp(math.copysign(min(1.0, abs(move) / 0.015), corr * move) if corr * move != 0 else 0.0)
        sign_bucket = "pos" if corr >= 0 else "neg"
        hkey = f"{leader}>{follower}:{lag}:{sign_bucket}"
        prev_h = history.get(hkey) if isinstance(history.get(hkey), dict) else {}
        prior_count = int(prev_h.get("count") or 0) if now - finite(prev_h.get("last_seen"), 0) <= 1200 else 0
        persistence = prior_count + 1
        history[hkey] = {"count": persistence, "last_seen": now, "correlation": corr, "incremental_strength": strength}
        context = market_context(follower, {}, contexts)
        rep = specialist_reputation(state, PACK_LEADLAG, context, lag * 900)
        ok = abs(move) >= 0.002 and abs(pred) >= 0.12 and persistence >= 2
        if context.get("out_of_distribution") and (strength < 0.40 or abs(pred) < 0.30):
            ok = False
        if rep.get("hard_negative") and abs(pred) < 0.50:
            ok = False
        if cal.get("hard_negative") and abs(pred) < 0.45:
            ok = False
        raw_conf = 0.30 + 0.45 * strength + 0.10 * min(1, abs(move) / 0.01)
        raw_conf *= finite(rep.get("confidence_multiplier"), 1.0)
        if context.get("out_of_distribution"):
            raw_conf *= 0.85
        conf = clamp(min(raw_conf, cal["confidence_cap"]), 0.20, 0.72)
        out.append({
            "symbol": follower,
            "kind": "signal" if ok else "relationship",
            "score": pred if ok else corr,
            "confidence": conf,
            "source": "LeanTrader World Model lead-lag candidate + Bybit public 15m closed candles",
            "provenance": f"world-model:leadlag-v1.20:{leader}>{follower}:{idx}:{int(now)}",
            "observed_at": now,
            "horizon_seconds": lag * 900,
            "metadata": {
                "model": "causal_leadlag_challenger_v1.20",
                "leader": leader,
                "follower": follower,
                "lag_steps": lag,
                "correlation": corr,
                "incremental_strength": strength,
                "leader_closed_return": move,
                "relationship_persistence_cycles": persistence,
                "persistence_required_cycles": 2,
                "directional_eligible": ok,
                "multiple_comparison_unadjusted": True,
                "research_only": True,
                "global_calibration": cal,
                "specialist_reputation": rep,
                "market_context": context,
                "risk_increase_allowed": False,
            },
        })
    return out

def core_sensor_bridge(now):
    """Expose already-healthy native sensor capabilities to the evolution demand layer.

    This deliberately carries no directional score so native data is not double-counted as a
    new alpha source. It fixes the semantic gap where the research planner requested data that
    the core sensor fabric already had available but the Evolution Fabric could not see as a pack.
    """
    sensor = load_json(RUNTIME / "vps_market_sensor_fabric.json")
    latest = sensor.get("latest") if isinstance(sensor.get("latest"), dict) else {}
    source_status = latest.get("source_status") if isinstance(latest.get("source_status"), dict) else sensor.get("source_status") or {}
    status = {str(k): str(v) for k, v in source_status.items()} if isinstance(source_status, dict) else {}
    capabilities = []
    mapping = {
        "onchain_flows": ("onchain_flows",),
        "stablecoin_liquidity": ("stablecoin_liquidity",),
        "open_interest": ("open_interest", "derivatives_positioning"),
        "liquidations": ("liquidations",),
    }
    evidence = {}
    for capability, keys in mapping.items():
        matched = None
        for key in keys:
            value = status.get(key)
            if value in {"available", "partial_or_unavailable", "unconfigured_or_partial"}:
                matched = (key, value)
                if value == "available":
                    break
        if matched and matched[1] == "available":
            capabilities.append(capability)
            evidence[capability] = {"native_source_key": matched[0], "native_status": matched[1]}
    # Carry bounded summaries where available; avoid copying huge state blobs.
    stable = latest.get("stablecoins") if isinstance(latest.get("stablecoins"), dict) else {}
    if stable:
        values = stable.get("values") if isinstance(stable.get("values"), dict) else {}
        evidence.setdefault("stablecoin_liquidity", {}).update({
            "status": stable.get("status"),
            "total_circulating_usd": values.get("total_circulating_usd"),
            "recent_supply_change": values.get("recent_supply_change"),
            "depeg_count_1pct": values.get("depeg_count_1pct"),
        })
    liq = latest.get("liquidations") if isinstance(latest.get("liquidations"), dict) else {}
    if liq:
        evidence.setdefault("liquidations", {}).update({
            "connected": liq.get("connected"),
            "messages": liq.get("messages"),
            "last_error": liq.get("last_error"),
        })
    if not capabilities:
        return None, []
    obs = [{
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.98,
        "source": "LeanTrader native Market Sensor Fabric capability bridge",
        "provenance": f"leantrader:native-sensor-bridge:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(600, INTERVAL * 3),
        "metadata": {
            "model": "native_sensor_capability_bridge_v1.3",
            "directional_claim": False,
            "independent_alpha_source": False,
            "duplicate_signal_suppressed": True,
            "capabilities": evidence,
        },
    }]
    return obs, capabilities


def unfold_ics(text):
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw.startswith((" ", "\t")) and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def parse_ics_dt(key, value):
    value = value.strip()
    tz_name = None
    m = re.search(r"TZID=([^;:]+)", key)
    if m:
        tz_name = m.group(1)
    if re.fullmatch(r"\d{8}", value):
        dt = datetime.strptime(value, "%Y%m%d").replace(hour=8, minute=30)
        return dt.replace(tzinfo=ET).astimezone(UTC)
    if value.endswith("Z"):
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    fmt = "%Y%m%dT%H%M%S" if len(value) >= 15 else "%Y%m%dT%H%M"
    dt = datetime.strptime(value[:15] if fmt.endswith("%S") else value[:13], fmt)
    tz = ZoneInfo(tz_name) if tz_name else ET
    return dt.replace(tzinfo=tz).astimezone(UTC)


def parse_bls_calendar(text, now):
    events = []
    current = None
    for line in unfold_ics(text):
        if line == "BEGIN:VEVENT":
            current = {}
            continue
        if line == "END:VEVENT":
            if current:
                try:
                    dt = current.get("dt")
                    title = current.get("summary") or "BLS scheduled release"
                    if isinstance(dt, datetime):
                        delta = dt.timestamp() - now
                        if -6 * 3600 <= delta <= 14 * 86400:
                            low = title.lower()
                            impact = "high" if any(p in low for p in HIGH_IMPACT_PATTERNS) else "standard"
                            events.append({
                                "title": title,
                                "event_time": dt.isoformat(),
                                "seconds_to_event": delta,
                                "impact": impact,
                                "agency": "BLS",
                                "source": "BLS official online release calendar",
                            })
                except Exception:
                    pass
            current = None
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.startswith("DTSTART"):
            try:
                current["dt"] = parse_ics_dt(key, value)
            except Exception:
                pass
        elif key == "SUMMARY":
            current["summary"] = value.replace("\\,", ",").replace("\\n", " ")
    return sorted(events, key=lambda x: x["event_time"])


def bls_verified_snapshot(now):
    expiry = datetime.fromisoformat(BLS_VERIFIED_SNAPSHOT_EXPIRES).astimezone(UTC).timestamp()
    if now > expiry:
        return []
    out = []
    for raw, title in BLS_VERIFIED_EVENTS:
        dt = datetime.fromisoformat(raw).astimezone(UTC)
        delta = dt.timestamp() - now
        if -6 * 3600 <= delta <= 120 * 86400:
            low = title.lower()
            impact = "high" if any(p in low for p in HIGH_IMPACT_PATTERNS) else "standard"
            out.append({
                "title": title,
                "event_time": dt.isoformat(),
                "seconds_to_event": delta,
                "impact": impact,
                "agency": "BLS",
                "source": "BLS official schedule verified fallback snapshot",
                "snapshot_asof": BLS_VERIFIED_SNAPSHOT_ASOF,
            })
    return sorted(out, key=lambda x: x["event_time"])


def fed_calendar(now):
    out = []
    for raw, title in FOMC_EVENTS:
        dt = datetime.fromisoformat(raw).astimezone(UTC)
        delta = dt.timestamp() - now
        if -6 * 3600 <= delta <= 120 * 86400:
            out.append({
                "title": title,
                "event_time": dt.isoformat(),
                "seconds_to_event": delta,
                "impact": "high",
                "agency": "Federal Reserve",
                "source": "Federal Reserve official FOMC calendar snapshot",
            })
    return sorted(out, key=lambda x: x["event_time"])


def macro_calendar_context(now):
    events = []
    fetch_errors = []
    source_status = {
        "bls_live_ics": "unattempted",
        "bls_verified_snapshot": "not_used",
        "federal_reserve_snapshot": "available",
    }
    try:
        live = parse_bls_calendar(request_text(BLS_ICS), now)
        if live:
            events.extend(live)
            source_status["bls_live_ics"] = "available"
        else:
            source_status["bls_live_ics"] = "empty"
    except Exception as exc:
        source_status["bls_live_ics"] = "unavailable"
        fetch_errors.append(f"BLS live calendar: {type(exc).__name__}: {exc}")

    # Officially verified bounded fallback prevents a provider-side 403 from
    # making the macro risk calendar disappear. It never becomes directional.
    if source_status["bls_live_ics"] != "available":
        fallback = bls_verified_snapshot(now)
        if fallback:
            events.extend(fallback)
            source_status["bls_verified_snapshot"] = "available_fallback"
        else:
            source_status["bls_verified_snapshot"] = "expired_or_empty"

    events.extend(fed_calendar(now))
    dedup = {}
    for row in events:
        dedup[(row.get("agency"), row.get("title"), row.get("event_time"))] = row
    events = sorted(dedup.values(), key=lambda x: x["event_time"])
    if not events:
        return None, fetch_errors, source_status
    nearest = events[0]
    seconds = finite(nearest.get("seconds_to_event"), 999999999)
    risk_state = "normal"
    if 0 <= seconds <= 3600:
        risk_state = "imminent"
    elif 0 <= seconds <= 6 * 3600:
        risk_state = "near"
    elif 0 <= seconds <= 24 * 3600:
        risk_state = "upcoming"
    elif -2 * 3600 <= seconds < 0:
        risk_state = "post_release_absorption"

    live_bls = source_status["bls_live_ics"] == "available"
    fallback_bls = source_status["bls_verified_snapshot"] == "available_fallback"
    confidence = 0.96 if live_bls else (0.90 if fallback_bls else 0.84)
    obs = {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": confidence,
        "source": "Official BLS release calendar + Federal Reserve FOMC calendar",
        "provenance": f"official-macro-calendar-v1.14:{int(now)}",
        "observed_at": now,
        "horizon_seconds": 86400,
        "metadata": {
            "model": "official_macro_event_risk_v1.14",
            "directional_claim": False,
            "risk_state": risk_state,
            "nearest_event": nearest,
            "events": events[:32],
            "source_status": source_status,
            "fetch_errors": fetch_errors,
            "fallback_is_bounded_snapshot": fallback_bls,
            "fallback_snapshot_asof": BLS_VERIFIED_SNAPSHOT_ASOF if fallback_bls else None,
            "fallback_snapshot_expires": BLS_VERIFIED_SNAPSHOT_EXPIRES if fallback_bls else None,
            "calendar_timezone": "UTC-normalized",
        },
    }
    return obs, fetch_errors, source_status


def make_pack(pack_id, capabilities, observations, now):
    if not observations or not capabilities:
        return None
    return {
        "schema_version": 1,
        "pack_id": pack_id,
        "version": VERSION,
        "producer": PRODUCER,
        "generated_at": now,
        "expires_at": now + max(900, INTERVAL * 4),
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "can_add_credentials": False,
        "capabilities": capabilities,
        "observations": observations,
    }



def _sensor_block_values(block):
    block = block if isinstance(block, dict) else {}
    values = block.get("values") if isinstance(block.get("values"), dict) else {}
    return str(block.get("status") or "unknown"), clamp(finite(block.get("confidence")), 0.0, 1.0), values


def _orthogonal_sensor_snapshot():
    """Read the core-owned public sensor snapshot without creating new network load."""
    sensor = load_json(RUNTIME / "vps_market_sensor_fabric.json")
    latest = sensor.get("latest") if isinstance(sensor.get("latest"), dict) else sensor
    latest = latest if isinstance(latest, dict) else {}
    symbols = latest.get("symbols") if isinstance(latest.get("symbols"), dict) else {}
    return latest, symbols


def _sensor_block_timestamp(block, fallback):
    block = block if isinstance(block, dict) else {}
    values = block.get("values") if isinstance(block.get("values"), dict) else {}
    for key in ("observed_at", "timestamp", "updated_at", "received_at", "as_of"):
        value = block.get(key)
        if value is None:
            value = values.get(key)
        ts = finite(value)
        if ts > 0:
            # tolerate millisecond timestamps from exchange/native adapters
            if ts > 10_000_000_000:
                ts /= 1000.0
            return ts
    return finite(fallback)


def _orthogonal_family_max_age(family):
    return {
        "derivatives": ORTHOGONAL_CACHE_DERIVATIVES_MAX_AGE,
        "flow_intelligence": ORTHOGONAL_CACHE_FLOW_MAX_AGE,
        "liquidations": ORTHOGONAL_CACHE_LIQUIDATIONS_MAX_AGE,
    }.get(str(family), ORTHOGONAL_CACHE_DERIVATIVES_MAX_AGE)


def _update_orthogonal_sensor_cache(state, sensor_symbols, snapshot_ts, now):
    """Persist last-known *available* blocks so rotating native batches can be joined safely.

    This cache never upgrades confidence and never invents a source. A cached family remains
    usable only inside its family-specific freshness bound and the pairwise source-skew gate.
    """
    state = state if isinstance(state, dict) else {}
    cache = state.get("orthogonal_sensor_cache") if isinstance(state.get("orthogonal_sensor_cache"), dict) else {}
    cache = dict(cache)
    for raw_symbol, bundle in (sensor_symbols or {}).items():
        symbol = normalize_symbol(raw_symbol)
        if not symbol or not isinstance(bundle, dict):
            continue
        row = cache.get(symbol) if isinstance(cache.get(symbol), dict) else {}
        row = dict(row)
        for family in ("derivatives", "flow_intelligence", "liquidations"):
            block = bundle.get(family) if isinstance(bundle.get(family), dict) else {}
            status, _conf, _values = _sensor_block_values(block)
            if status != "available":
                continue
            source_ts = _sensor_block_timestamp(block, snapshot_ts)
            row[family] = {
                "block": block,
                "seen_at": now,
                "source_timestamp": source_ts,
            }
        if row:
            cache[symbol] = row

    # Bound stale state growth; retain a generous window for diagnostics, not signaling.
    longest = max(ORTHOGONAL_CACHE_DERIVATIVES_MAX_AGE, ORTHOGONAL_CACHE_FLOW_MAX_AGE, ORTHOGONAL_CACHE_LIQUIDATIONS_MAX_AGE)
    pruned = {}
    for symbol, families in cache.items():
        if not isinstance(families, dict):
            continue
        kept = {}
        for family, entry in families.items():
            if not isinstance(entry, dict):
                continue
            seen = finite(entry.get("seen_at"))
            if seen > 0 and now - seen <= longest * 3:
                kept[family] = entry
        if kept:
            pruned[symbol] = kept
    # Hard cap protects the state file from accidental universe explosion.
    if len(pruned) > 64:
        ranked = sorted(pruned.items(), key=lambda kv: max([finite((e or {}).get("seen_at")) for e in kv[1].values()] or [0.0]), reverse=True)
        pruned = dict(ranked[:64])
    state["orthogonal_sensor_cache"] = pruned
    return pruned


def _resolve_orthogonal_family(bundle, cache_row, family, snapshot_ts, now):
    block = bundle.get(family) if isinstance(bundle, dict) and isinstance(bundle.get(family), dict) else {}
    status, conf, values = _sensor_block_values(block)
    max_age = _orthogonal_family_max_age(family)
    if status == "available":
        source_ts = _sensor_block_timestamp(block, snapshot_ts)
        age = max(0.0, now - source_ts) if source_ts > 0 else 0.0
        if source_ts <= 0 or age <= max_age:
            return block, status, conf, values, source_ts, False, age
    entry = cache_row.get(family) if isinstance(cache_row, dict) and isinstance(cache_row.get(family), dict) else {}
    cached_block = entry.get("block") if isinstance(entry.get("block"), dict) else {}
    cstatus, cconf, cvalues = _sensor_block_values(cached_block)
    seen_at = finite(entry.get("seen_at"))
    source_ts = finite(entry.get("source_timestamp") or seen_at)
    age = max(0.0, now - source_ts) if source_ts > 0 else float("inf")
    if cstatus == "available" and seen_at > 0 and now - seen_at <= max_age and age <= max_age:
        return cached_block, "available", cconf, cvalues, source_ts, True, age
    return {}, "unavailable", 0.0, {}, 0.0, False, age


def _orthogonal_pair_aligned(ts_a, ts_b):
    return ts_a > 0 and ts_b > 0 and abs(ts_a - ts_b) <= ORTHOGONAL_MAX_SOURCE_SKEW_SECONDS


def _orthogonal_flow_fingerprint(score):
    # A coarse cross-sectional fingerprint is intentionally used only to detect
    # single-component common factors. It never creates a directional signal.
    return f"{clamp(finite(score)):+.3f}"


def _orthogonal_flow_lineage_profiles(universe, sensor_symbols, cache, snapshot_ts, now):
    """Classify single-component flow without pretending it is internally diversified.

    The core flow synthesizer can legitimately expose one available component. v1.19 rejected
    all such rows, which discarded most of the sensor universe. v1.20 keeps that evidence only
    when it is strong, fresh, cross-sectionally non-common, and paired with an independent
    derivatives or liquidation family. Repeated one-component fingerprints are treated as a
    common/global factor and remain context-only.
    """
    profiles = {}
    fingerprint_symbols = {}
    multi = 0
    single = 0
    available = 0
    for symbol in sorted(universe or []):
        bundle = sensor_symbols.get(symbol) if isinstance(sensor_symbols, dict) else {}
        if not isinstance(bundle, dict):
            bundle = next((v for k, v in (sensor_symbols or {}).items() if normalize_symbol(k) == symbol and isinstance(v, dict)), {})
        cache_row = cache.get(symbol) if isinstance(cache, dict) and isinstance(cache.get(symbol), dict) else {}
        _fb, status, conf, values, ts, cached, age = _resolve_orthogonal_family(bundle, cache_row, "flow_intelligence", snapshot_ts, now)
        if status != "available":
            continue
        available += 1
        components = int(finite(values.get("components_available")))
        score = clamp(finite(values.get("flow_score")))
        fp = _orthogonal_flow_fingerprint(score)
        row = {
            "status": status,
            "confidence": conf,
            "components_available": components,
            "flow_score": score,
            "fingerprint": fp,
            "source_timestamp": ts,
            "cached": bool(cached),
            "age_seconds": age,
            "common_factor_like": False,
            "single_component_strict_quality": False,
        }
        profiles[symbol] = row
        if components >= 2:
            multi += 1
        elif components == 1:
            single += 1
            fingerprint_symbols.setdefault(fp, []).append(symbol)

    common = set()
    dominant_fp = None
    dominant_n = 0
    for fp, symbols in fingerprint_symbols.items():
        n = len(symbols)
        if n > dominant_n:
            dominant_fp, dominant_n = fp, n
        share = n / max(1, single)
        # Zero/near-zero single-component scores have no directional content. A large
        # repeated fingerprint is also likely a global/common factor rather than asset alpha.
        if abs(finite(fp)) < 0.0005 or n >= ORTHOGONAL_COMMON_FACTOR_MIN_CLUSTER or (single >= 8 and n >= 4 and share >= ORTHOGONAL_COMMON_FACTOR_MIN_SHARE):
            common.add(fp)

    asset_specific_eligible = 0
    common_rejected = 0
    strict_magnitude_rejected = 0
    confidence_rejected = 0
    for row in profiles.values():
        if row["components_available"] != 1:
            continue
        row["common_factor_like"] = row["fingerprint"] in common
        if row["common_factor_like"]:
            common_rejected += 1
            continue
        if row["confidence"] < ORTHOGONAL_SINGLE_COMPONENT_MIN_CONFIDENCE:
            confidence_rejected += 1
            continue
        if abs(row["flow_score"]) < ORTHOGONAL_SINGLE_COMPONENT_MIN_FLOW_SCORE:
            strict_magnitude_rejected += 1
            continue
        row["single_component_strict_quality"] = True
        asset_specific_eligible += 1

    summary = {
        "available_flow_blocks": available,
        "multi_component_blocks": multi,
        "single_component_blocks": single,
        "single_component_asset_specific_eligible": asset_specific_eligible,
        "single_component_common_factor_rejected": common_rejected,
        "single_component_below_strict_magnitude": strict_magnitude_rejected,
        "single_component_below_confidence": confidence_rejected,
        "single_component_common_fingerprints": sorted(common),
        "dominant_single_component_fingerprint": dominant_fp,
        "dominant_single_component_cluster_size": dominant_n,
        "dominant_single_component_share": dominant_n / max(1, single),
        "single_component_never_counts_as_internal_multi_source": True,
        "single_component_requires_independent_external_family": True,
        "common_factor_single_component_is_context_only": True,
        "single_component_signal_confidence_cap": ORTHOGONAL_SINGLE_COMPONENT_MAX_SIGNAL_CONFIDENCE,
        "execution_authority": False,
    }
    return profiles, summary, common



def _flow_residual_cross_section(flow_profiles):
    """Measure asset-specific deviations after removing the current common flow factor.

    This is a diagnostic lab only. It never upgrades a common single-component flow row into
    directional alpha. v1.27 requires a sufficiently large *current* (non-cached) cross-section,
    estimates a robust median factor, and records only statistically unusual residuals for
    persistence research.
    """
    current = {
        str(symbol): row for symbol, row in (flow_profiles or {}).items()
        if isinstance(row, dict)
        and int(finite(row.get("components_available"))) == 1
        and not bool(row.get("cached"))
        and str(row.get("status") or "") == "available"
    }
    scores = [clamp(finite(row.get("flow_score"))) for row in current.values()]
    active = len(scores) >= FLOW_RESIDUAL_MIN_CURRENT_SINGLE_ROWS
    factor = _median(scores) if active else 0.0
    abs_dev = [abs(v - factor) for v in scores] if active else []
    mad = _median(abs_dev) if active else 0.0
    robust_scale = max(FLOW_RESIDUAL_SCALE_FLOOR, 1.4826 * mad) if active else FLOW_RESIDUAL_SCALE_FLOOR
    rows = {}
    diagnostics = []
    for symbol, row in sorted(current.items()):
        score = clamp(finite(row.get("flow_score")))
        residual = score - factor if active else 0.0
        robust_z = residual / robust_scale if active and robust_scale > 0 else 0.0
        unusual = bool(
            active
            and abs(residual) >= FLOW_RESIDUAL_DIAGNOSTIC_MIN_ABS
            and abs(robust_z) >= FLOW_RESIDUAL_DIAGNOSTIC_MIN_Z
            and finite(row.get("confidence")) >= ORTHOGONAL_SINGLE_COMPONENT_MIN_CONFIDENCE
        )
        out = {
            "symbol": symbol,
            "source_timestamp": finite(row.get("source_timestamp")),
            "raw_flow_score": score,
            "common_factor_median": factor,
            "residual": residual,
            "robust_z": robust_z,
            "confidence": clamp(finite(row.get("confidence")), 0.0, 1.0),
            "diagnostic_outlier": unusual,
            "execution_authority": False,
        }
        rows[symbol] = out
        if unusual:
            diagnostics.append(out)
    diagnostics = sorted(diagnostics, key=lambda r: (abs(finite(r.get("robust_z"))), abs(finite(r.get("residual")))), reverse=True)
    return rows, {
        "active": active,
        "current_single_component_rows": len(current),
        "minimum_current_rows": FLOW_RESIDUAL_MIN_CURRENT_SINGLE_ROWS,
        "common_factor_median": factor,
        "median_absolute_deviation": mad,
        "robust_scale": robust_scale,
        "diagnostic_min_abs_residual": FLOW_RESIDUAL_DIAGNOSTIC_MIN_ABS,
        "diagnostic_min_robust_z": FLOW_RESIDUAL_DIAGNOSTIC_MIN_Z,
        "diagnostic_outliers": len(diagnostics),
        "max_abs_residual": max([abs(finite(r.get("residual"))) for r in rows.values()] or [0.0]),
        "max_abs_robust_z": max([abs(finite(r.get("robust_z"))) for r in rows.values()] or [0.0]),
        "top_diagnostic_symbols": [r.get("symbol") for r in diagnostics[:8]],
        "common_factor_remains_context_only": True,
        "residuals_do_not_create_alpha": True,
        "execution_authority": False,
        "can_increase_risk": False,
    }


def _update_flow_residual_lab(state, residual_rows, residual_summary, now):
    state = state if isinstance(state, dict) else {}
    lab = state.get("flow_residual_lab") if isinstance(state.get("flow_residual_lab"), dict) else {}
    symbols = lab.get("symbols") if isinstance(lab.get("symbols"), dict) else {}
    symbols = {str(k): list(v) for k, v in symbols.items() if isinstance(v, list)}
    added = 0
    for symbol, row in (residual_rows or {}).items():
        if not isinstance(row, dict) or not residual_summary.get("active"):
            continue
        ts = finite(row.get("source_timestamp"))
        if ts <= 0:
            continue
        history = [r for r in symbols.get(symbol, []) if isinstance(r, dict)]
        if any(abs(finite(r.get("source_timestamp")) - ts) < 0.001 for r in history):
            continue
        history.append({
            "source_timestamp": ts,
            "seen_at": finite(now),
            "residual": finite(row.get("residual")),
            "robust_z": finite(row.get("robust_z")),
            "raw_flow_score": finite(row.get("raw_flow_score")),
            "common_factor_median": finite(row.get("common_factor_median")),
            "diagnostic_outlier": bool(row.get("diagnostic_outlier")),
        })
        symbols[symbol] = history[-FLOW_RESIDUAL_HISTORY_PER_SYMBOL:]
        added += 1
    if len(symbols) > FLOW_RESIDUAL_MAX_SYMBOLS:
        ranked = sorted(
            symbols.items(),
            key=lambda kv: max([finite(r.get("source_timestamp")) for r in kv[1] if isinstance(r, dict)] or [0.0]),
            reverse=True,
        )[:FLOW_RESIDUAL_MAX_SYMBOLS]
        symbols = dict(ranked)

    persistent = []
    for symbol, history in symbols.items():
        recent = [r for r in history[-FLOW_RESIDUAL_HISTORY_PER_SYMBOL:] if isinstance(r, dict)]
        if len(recent) < FLOW_RESIDUAL_PERSISTENCE_MIN_SAMPLES:
            continue
        signs = [1 if finite(r.get("residual")) > 0 else -1 if finite(r.get("residual")) < 0 else 0 for r in recent]
        nonzero = [v for v in signs if v]
        if not nonzero:
            continue
        dominant_sign = 1 if sum(nonzero) >= 0 else -1
        consistency = sum(1 for v in nonzero if v == dominant_sign) / max(1, len(nonzero))
        outlier_hits = sum(1 for r in recent if bool(r.get("diagnostic_outlier")) and (1 if finite(r.get("residual")) > 0 else -1 if finite(r.get("residual")) < 0 else 0) == dominant_sign)
        persistent_flag = bool(consistency >= 0.75 and outlier_hits >= 2)
        if persistent_flag:
            persistent.append({
                "symbol": symbol,
                "samples": len(recent),
                "dominant_sign": dominant_sign,
                "sign_consistency": consistency,
                "outlier_hits": outlier_hits,
                "median_residual": _median([finite(r.get("residual")) for r in recent]),
                "median_abs_robust_z": _median([abs(finite(r.get("robust_z"))) for r in recent]),
            })
    persistent = sorted(persistent, key=lambda r: (r.get("outlier_hits", 0), r.get("sign_consistency", 0.0), abs(finite(r.get("median_residual")))), reverse=True)
    lab = {
        "model": "common_factor_residual_persistence_lab_v1.27",
        "symbols": symbols,
        "samples_added_this_cycle": added,
        "symbols_tracked": len(symbols),
        "persistent_diagnostic_candidates": persistent[:12],
        "persistent_candidate_count": len(persistent),
        "minimum_persistence_samples": FLOW_RESIDUAL_PERSISTENCE_MIN_SAMPLES,
        "diagnostic_only": True,
        "residuals_do_not_create_alpha": True,
        "execution_authority": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }
    state["flow_residual_lab"] = lab
    return lab


def flow_residual_context_observation(source_status, state, now):
    residual = (source_status or {}).get("flow_residual_diagnostics") or {}
    lab = (state or {}).get("flow_residual_lab") or {}
    return {
        "symbol": "MARKET",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.98,
        "source": "LeanTrader common-factor residual persistence lab",
        "provenance": f"leantrader:flow-residual-lab:v1.27:{int(now)}",
        "observed_at": now,
        "horizon_seconds": 0,
        "metadata": {
            "model": "common_factor_residual_persistence_lab_v1.27",
            "cross_section": residual,
            "lab": {k: v for k, v in lab.items() if k != "symbols"},
            "raw_common_factor_is_context_only": True,
            "residuals_do_not_create_alpha": True,
            "execution_authority": False,
            "risk_increase_allowed": False,
            "auto_promotion_allowed": False,
        },
    }



def _aggregate_binance_usdm_taker_klines(payload, now):
    """Aggregate only fully closed USD-M klines into a bounded aggressive-flow diagnostic.

    Binance kline rows expose total quote volume and taker-buy quote volume.  v1.28 uses those
    public market-data fields to measure aggressive buy/sell imbalance from a source independent
    of LeanTrader's native flow synthesizer.  The result is diagnostic research only.
    """
    closed = []
    for row in payload if isinstance(payload, list) else []:
        if not isinstance(row, (list, tuple)) or len(row) < 11:
            continue
        close_ms = finite(row[6])
        close_ts = close_ms / 1000.0 if close_ms > 10_000_000_000 else close_ms
        if close_ts <= 0 or close_ts > finite(now) + 2.0:
            continue
        quote_volume = max(0.0, finite(row[7]))
        taker_buy_quote = max(0.0, finite(row[10]))
        close_price = finite(row[4])
        if quote_volume <= 0.0 or close_price <= 0.0:
            continue
        taker_buy_quote = min(quote_volume, taker_buy_quote)
        closed.append({
            "close_timestamp": close_ts,
            "close_price": close_price,
            "quote_volume": quote_volume,
            "taker_buy_quote_volume": taker_buy_quote,
        })
    closed.sort(key=lambda r: finite(r.get("close_timestamp")))
    closed = closed[-EXTERNAL_FLOW_CLOSED_BARS:]
    if len(closed) < min(2, EXTERNAL_FLOW_CLOSED_BARS):
        return None
    total_quote = sum(finite(r.get("quote_volume")) for r in closed)
    total_taker_buy = sum(finite(r.get("taker_buy_quote_volume")) for r in closed)
    if total_quote <= 0.0:
        return None
    buy_share = clamp(total_taker_buy / total_quote, 0.0, 1.0)
    imbalance = clamp(2.0 * buy_share - 1.0)
    fingerprint_src = [
        [int(finite(r.get("close_timestamp")) * 1000), round(finite(r.get("quote_volume")), 8), round(finite(r.get("taker_buy_quote_volume")), 8)]
        for r in closed
    ]
    fingerprint = hashlib.sha256(json.dumps(fingerprint_src, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]
    return {
        "aggressive_flow_imbalance": imbalance,
        "taker_buy_quote_share": buy_share,
        "quote_volume": total_quote,
        "taker_buy_quote_volume": total_taker_buy,
        "closed_bars": len(closed),
        "source_timestamp": max(finite(r.get("close_timestamp")) for r in closed),
        "close_price": finite(closed[-1].get("close_price")),
        "payload_fingerprint": fingerprint,
    }


def _binance_usdm_aggressive_flow_snapshot(symbol, now=None):
    """Read-only Binance USD-M taker-flow snapshot; never used for order execution."""
    now = finite(now, time.time())
    sym = bybit_symbol(symbol)
    limit = EXTERNAL_FLOW_CLOSED_BARS + 1
    try:
        payload = request_json_any(
            f"{BINANCE_FUTURES_PUBLIC}/fapi/v1/klines?{urllib.parse.urlencode({'symbol': sym, 'interval': '5m', 'limit': limit})}",
            timeout=8,
        )
        if isinstance(payload, dict):
            return None, "symbol_unavailable" if int(finite(payload.get("code"))) in {-1121, -4003} else "unexpected_payload"
        row = _aggregate_binance_usdm_taker_klines(payload, now)
        if not isinstance(row, dict):
            return None, "insufficient_closed_bars"
        age = max(0.0, now - finite(row.get("source_timestamp")))
        if age > EXTERNAL_FLOW_MAX_SOURCE_AGE_SECONDS:
            return None, "stale_source"
        row.update({
            "venue": "binance_usdm",
            "symbol": normalize_symbol(symbol),
            "source_family": "external_aggressive_trade_flow",
            "source_method": "closed_5m_kline_taker_quote_volume",
            "source_age_seconds": age,
            "confidence": 0.90 if int(row.get("closed_bars") or 0) >= 3 else 0.82,
            "execution_authority": False,
        })
        return row, "available"
    except urllib.error.HTTPError as exc:
        code = int(getattr(exc, "code", 0) or 0)
        if code in {400, 404}:
            return None, "symbol_unavailable"
        if code in {403, 451}:
            return None, f"regional_or_access_restriction_{code}"
        return None, f"http_{code or 'error'}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _external_flow_candidate_symbols(state, now):
    """Rotate through current common-factor single-component symbols without strategy bias."""
    latest, sensor_symbols = _orthogonal_sensor_snapshot()
    snapshot_ts = finite(latest.get("timestamp") or latest.get("updated_at"), now)
    cache = state.get("orthogonal_sensor_cache") if isinstance(state.get("orthogonal_sensor_cache"), dict) else {}
    universe = sorted({normalize_symbol(k) for k in sensor_symbols.keys() if normalize_symbol(k)})[:ORTHOGONAL_RESEARCH_UNIVERSE_LIMIT]
    profiles, lineage, _common = _orthogonal_flow_lineage_profiles(set(universe), sensor_symbols, cache, snapshot_ts, now)
    current_common = [
        symbol for symbol, row in profiles.items()
        if isinstance(row, dict)
        and int(finite(row.get("components_available"))) == 1
        and not bool(row.get("cached"))
        and bool(row.get("common_factor_like"))
        and str(row.get("status") or "") == "available"
    ]
    current_single = [
        symbol for symbol, row in profiles.items()
        if isinstance(row, dict)
        and int(finite(row.get("components_available"))) == 1
        and not bool(row.get("cached"))
        and str(row.get("status") or "") == "available"
    ]
    candidates = current_common or current_single
    qhist = state.get("external_aggressive_flow_query_history") if isinstance(state.get("external_aggressive_flow_query_history"), dict) else {}
    def rank(symbol):
        row = qhist.get(symbol) if isinstance(qhist.get(symbol), dict) else {}
        return (finite(row.get("last_queried_at")), int(row.get("attempts") or 0), symbol)
    selected = sorted(set(candidates), key=rank)[:EXTERNAL_FLOW_MAX_SYMBOLS_PER_CYCLE]
    return selected, {
        "sensor_snapshot_timestamp": snapshot_ts,
        "current_common_factor_candidates": len(current_common),
        "current_single_component_candidates": len(current_single),
        "selected_symbols": selected,
        "flow_lineage": lineage,
    }


def _external_flow_cross_section(rows):
    available = {str(k): v for k, v in (rows or {}).items() if isinstance(v, dict)}
    values = [clamp(finite(r.get("aggressive_flow_imbalance"))) for r in available.values()]
    active = len(values) >= EXTERNAL_FLOW_MIN_CURRENT_ROWS
    factor = _median(values) if active else 0.0
    mad = _median([abs(v - factor) for v in values]) if active else 0.0
    scale = max(EXTERNAL_FLOW_SCALE_FLOOR, 1.4826 * mad) if active else EXTERNAL_FLOW_SCALE_FLOOR
    out = {}
    diagnostics = []
    for symbol, row in sorted(available.items()):
        raw = clamp(finite(row.get("aggressive_flow_imbalance")))
        residual = raw - factor if active else 0.0
        z = residual / scale if active and scale > 0 else 0.0
        unusual = bool(active and abs(residual) >= EXTERNAL_FLOW_DIAGNOSTIC_MIN_ABS and abs(z) >= EXTERNAL_FLOW_DIAGNOSTIC_MIN_Z)
        rr = {
            **row,
            "cross_section_common_factor": factor,
            "cross_section_residual": residual,
            "robust_z": z,
            "diagnostic_outlier": unusual,
            "execution_authority": False,
        }
        out[symbol] = rr
        if unusual:
            diagnostics.append(rr)
    diagnostics.sort(key=lambda r: (abs(finite(r.get("robust_z"))), abs(finite(r.get("cross_section_residual")))), reverse=True)
    return out, {
        "active": active,
        "available_current_rows": len(values),
        "minimum_current_rows": EXTERNAL_FLOW_MIN_CURRENT_ROWS,
        "common_factor_median": factor,
        "median_absolute_deviation": mad,
        "robust_scale": scale,
        "diagnostic_outliers": len(diagnostics),
        "max_abs_robust_z": max([abs(finite(r.get("robust_z"))) for r in out.values()] or [0.0]),
        "top_diagnostic_symbols": [r.get("symbol") for r in diagnostics[:8]],
        "independent_from_native_flow_synthesizer": True,
        "same_venue_as_binance_derivatives_research": True,
        "does_not_count_as_independent_council_vote": True,
        "diagnostic_only": True,
        "execution_authority": False,
        "can_increase_risk": False,
    }


def _update_external_flow_lab(state, rows, summary, now):
    lab = state.get("external_aggressive_flow_lab") if isinstance(state.get("external_aggressive_flow_lab"), dict) else {}
    symbols = lab.get("symbols") if isinstance(lab.get("symbols"), dict) else {}
    symbols = {str(k): list(v) for k, v in symbols.items() if isinstance(v, list)}
    added = 0
    replay_suppressed = 0
    for symbol, row in (rows or {}).items():
        if not isinstance(row, dict) or not summary.get("active"):
            continue
        ts = finite(row.get("source_timestamp"))
        fp = str(row.get("payload_fingerprint") or "")
        if ts <= 0:
            continue
        hist = [r for r in symbols.get(symbol, []) if isinstance(r, dict)]
        if any(abs(finite(r.get("source_timestamp")) - ts) < 0.001 or (fp and str(r.get("payload_fingerprint") or "") == fp) for r in hist):
            replay_suppressed += 1
            continue
        hist.append({
            "source_timestamp": ts,
            "seen_at": finite(now),
            "payload_fingerprint": fp,
            "aggressive_flow_imbalance": finite(row.get("aggressive_flow_imbalance")),
            "cross_section_residual": finite(row.get("cross_section_residual")),
            "robust_z": finite(row.get("robust_z")),
            "diagnostic_outlier": bool(row.get("diagnostic_outlier")),
        })
        symbols[symbol] = hist[-EXTERNAL_FLOW_HISTORY_PER_SYMBOL:]
        added += 1
    if len(symbols) > EXTERNAL_FLOW_MAX_TRACKED_SYMBOLS:
        ranked = sorted(symbols.items(), key=lambda kv: max([finite(r.get("source_timestamp")) for r in kv[1] if isinstance(r, dict)] or [0.0]), reverse=True)
        symbols = dict(ranked[:EXTERNAL_FLOW_MAX_TRACKED_SYMBOLS])

    persistent = []
    for symbol, hist in symbols.items():
        recent = [r for r in hist[-EXTERNAL_FLOW_HISTORY_PER_SYMBOL:] if isinstance(r, dict)]
        if len(recent) < EXTERNAL_FLOW_PERSISTENCE_MIN_SAMPLES:
            continue
        signs = [1 if finite(r.get("cross_section_residual")) > 0 else -1 if finite(r.get("cross_section_residual")) < 0 else 0 for r in recent]
        nonzero = [v for v in signs if v]
        if not nonzero:
            continue
        dominant = 1 if sum(nonzero) >= 0 else -1
        consistency = sum(1 for v in nonzero if v == dominant) / max(1, len(nonzero))
        hits = sum(1 for r in recent if bool(r.get("diagnostic_outlier")) and (1 if finite(r.get("cross_section_residual")) > 0 else -1 if finite(r.get("cross_section_residual")) < 0 else 0) == dominant)
        if consistency >= 0.75 and hits >= 2:
            persistent.append({
                "symbol": symbol,
                "samples": len(recent),
                "dominant_sign": dominant,
                "sign_consistency": consistency,
                "outlier_hits": hits,
                "median_residual": _median([finite(r.get("cross_section_residual")) for r in recent]),
                "median_abs_robust_z": _median([abs(finite(r.get("robust_z"))) for r in recent]),
            })
    persistent.sort(key=lambda r: (r.get("outlier_hits", 0), r.get("sign_consistency", 0.0), abs(finite(r.get("median_residual")))), reverse=True)
    lab = {
        "model": "binance_usdm_aggressive_flow_residual_persistence_lab_v1.28",
        "symbols": symbols,
        "samples_added_this_cycle": added,
        "replays_suppressed_this_cycle": replay_suppressed,
        "symbols_tracked": len(symbols),
        "persistent_diagnostic_candidates": persistent[:12],
        "persistent_candidate_count": len(persistent),
        "minimum_persistence_samples": EXTERNAL_FLOW_PERSISTENCE_MIN_SAMPLES,
        "independent_from_native_flow_synthesizer": True,
        "same_venue_as_binance_derivatives_research": True,
        "does_not_count_as_independent_council_vote": True,
        "diagnostic_only": True,
        "execution_authority": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }
    state["external_aggressive_flow_lab"] = lab
    return lab


def _update_external_flow_outcome_lab(state, rows, cross_summary, persistence_lab, now):
    """Track forward outcomes for already-persistent external-flow diagnostics.

    v1.29 deliberately separates *discovery* from *outcome measurement*. A symbol must first
    satisfy the v1.28 persistence rule (including at least two historical diagnostic-outlier
    hits). Only then can this lab open one non-overlapping research watch. The watch has no
    execution/council authority and is resolved from a later closed Binance USD-M candle for
    the same symbol after a fixed horizon. Repeated source timestamps cannot create episodes.
    """
    state = state if isinstance(state, dict) else {}
    pending = state.get("external_aggressive_flow_outcome_pending") if isinstance(state.get("external_aggressive_flow_outcome_pending"), dict) else {}
    pending = {str(k): dict(v) for k, v in pending.items() if isinstance(v, dict)}
    resolved = state.get("external_aggressive_flow_outcome_resolved") if isinstance(state.get("external_aggressive_flow_outcome_resolved"), list) else []
    resolved = [dict(r) for r in resolved if isinstance(r, dict)][-EXTERNAL_FLOW_OUTCOME_MAX_RESOLVED:]
    last_opened = state.get("external_aggressive_flow_outcome_last_opened") if isinstance(state.get("external_aggressive_flow_outcome_last_opened"), dict) else {}
    last_opened = {str(k): finite(v) for k, v in last_opened.items()}
    opened = resolved_now = overlap_suppressed = sign_mismatch_suppressed = 0
    cost = ROUND_TRIP_COST_BPS / 10_000.0

    # Resolve first so a symbol can open a fresh, non-overlapping episode from the current bar.
    for symbol, episode in list(pending.items()):
        row = (rows or {}).get(symbol) if isinstance((rows or {}).get(symbol), dict) else None
        if not row:
            continue
        source_ts = finite(row.get("source_timestamp"))
        if source_ts + 1e-9 < finite(episode.get("target_source_timestamp")):
            continue
        exit_price = finite(row.get("close_price"))
        entry_price = finite(episode.get("entry_price"))
        if exit_price <= 0 or entry_price <= 0:
            continue
        direction = 1.0 if finite(episode.get("direction")) > 0 else -1.0
        gross = direction * (exit_price / entry_price - 1.0)
        net = gross - cost
        resolved.append({
            **episode,
            "resolved_at": finite(now),
            "exit_source_timestamp": source_ts,
            "exit_price": exit_price,
            "actual_horizon_seconds": max(0.0, source_ts - finite(episode.get("entry_source_timestamp"))),
            "gross_return": gross,
            "net_return": net,
            "inverse_net_return": -gross - cost,
            "edge_vs_inverse": 2.0 * gross,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "execution_authority": False,
            "diagnostic_only": True,
        })
        del pending[symbol]
        resolved_now += 1

    resolved = resolved[-EXTERNAL_FLOW_OUTCOME_MAX_RESOLVED:]

    persistent = (persistence_lab or {}).get("persistent_diagnostic_candidates") or []
    current_active = bool((cross_summary or {}).get("active"))
    for cand in persistent:
        if not isinstance(cand, dict) or not current_active:
            continue
        symbol = normalize_symbol(cand.get("symbol") or "")
        row = (rows or {}).get(symbol) if isinstance((rows or {}).get(symbol), dict) else None
        if not symbol or not row:
            continue
        if symbol in pending:
            overlap_suppressed += 1
            continue
        source_ts = finite(row.get("source_timestamp"))
        entry_price = finite(row.get("close_price"))
        if source_ts <= 0 or entry_price <= 0 or source_ts <= finite(last_opened.get(symbol)) + 1e-9:
            overlap_suppressed += 1
            continue
        dominant = 1 if int(cand.get("dominant_sign") or 0) > 0 else -1
        current_residual = finite(row.get("cross_section_residual"))
        current_sign = 1 if current_residual > 0 else -1 if current_residual < 0 else 0
        if current_sign != dominant:
            sign_mismatch_suppressed += 1
            continue
        episode_id = hashlib.sha256(
            f"external-flow-outcome|{symbol}|{source_ts:.3f}|{dominant}|{EXTERNAL_FLOW_OUTCOME_HORIZON_SECONDS}".encode("utf-8")
        ).hexdigest()[:24]
        pending[symbol] = {
            "episode_id": episode_id,
            "symbol": symbol,
            "direction": dominant,
            "opened_at": finite(now),
            "entry_source_timestamp": source_ts,
            "target_source_timestamp": source_ts + EXTERNAL_FLOW_OUTCOME_HORIZON_SECONDS,
            "entry_price": entry_price,
            "entry_residual": current_residual,
            "entry_robust_z": finite(row.get("robust_z")),
            "persistence_samples": int(cand.get("samples") or 0),
            "persistence_outlier_hits": int(cand.get("outlier_hits") or 0),
            "persistence_sign_consistency": finite(cand.get("sign_consistency")),
            "horizon_seconds": EXTERNAL_FLOW_OUTCOME_HORIZON_SECONDS,
            "source_family": "external_aggressive_trade_flow",
            "source_method": "closed_5m_kline_taker_quote_volume",
            "diagnostic_only": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "auto_promotion_allowed": False,
        }
        last_opened[symbol] = source_ts
        opened += 1
        if len(pending) >= EXTERNAL_FLOW_OUTCOME_MAX_PENDING:
            break

    metrics = _horizon_metric_stats(resolved)
    candidate_symbols = [normalize_symbol(r.get("symbol") or "") for r in persistent if isinstance(r, dict) and normalize_symbol(r.get("symbol") or "")][:12]
    lab = {
        "model": "external_aggressive_flow_forward_outcome_lab_v1.29",
        "horizon_seconds": EXTERNAL_FLOW_OUTCOME_HORIZON_SECONDS,
        "opened_this_cycle": opened,
        "resolved_this_cycle": resolved_now,
        "overlap_suppressed_this_cycle": overlap_suppressed,
        "sign_mismatch_suppressed_this_cycle": sign_mismatch_suppressed,
        "pending_count": len(pending),
        "resolved_count": len(resolved),
        "persistent_candidate_symbols": candidate_symbols,
        "metrics": metrics,
        "minimum_source_persistence_samples": EXTERNAL_FLOW_PERSISTENCE_MIN_SAMPLES,
        "requires_prior_persistence_and_outlier_history": True,
        "non_overlapping_per_symbol": True,
        "uses_closed_source_candles_for_entry_and_resolution": True,
        "does_not_create_alpha": True,
        "does_not_count_as_independent_council_vote": True,
        "execution_authority": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }
    state["external_aggressive_flow_outcome_pending"] = pending
    state["external_aggressive_flow_outcome_resolved"] = resolved
    state["external_aggressive_flow_outcome_last_opened"] = last_opened
    state["external_aggressive_flow_outcome_lab"] = lab
    return lab


def external_aggressive_flow_research(state, now):
    """Bounded public acquisition of symbol-specific aggressive-flow evidence."""
    state = state if isinstance(state, dict) else {}
    selected, selection = _external_flow_candidate_symbols(state, now)
    qhist = state.get("external_aggressive_flow_query_history") if isinstance(state.get("external_aggressive_flow_query_history"), dict) else {}
    snapshots = {}
    status_counts = {}
    for symbol in selected:
        row, status = _binance_usdm_aggressive_flow_snapshot(symbol, now=now)
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        prev = qhist.get(symbol) if isinstance(qhist.get(symbol), dict) else {}
        qhist[symbol] = {
            "last_queried_at": finite(now),
            "attempts": int(prev.get("attempts") or 0) + 1,
            "available_count": int(prev.get("available_count") or 0) + (1 if status == "available" else 0),
            "last_status": status,
            "last_source_timestamp": finite((row or {}).get("source_timestamp")),
        }
        if status == "available" and isinstance(row, dict):
            snapshots[symbol] = row
    if len(qhist) > EXTERNAL_FLOW_MAX_TRACKED_SYMBOLS * 2:
        qhist = dict(sorted(qhist.items(), key=lambda kv: finite((kv[1] or {}).get("last_queried_at")), reverse=True)[:EXTERNAL_FLOW_MAX_TRACKED_SYMBOLS * 2])
    state["external_aggressive_flow_query_history"] = qhist
    cross_rows, cross_summary = _external_flow_cross_section(snapshots)
    lab = _update_external_flow_lab(state, cross_rows, cross_summary, now)
    outcome_lab = _update_external_flow_outcome_lab(state, cross_rows, cross_summary, lab, now)
    return {
        "model": "external_aggressive_flow_acquisition_v1.29",
        "venue": "binance_usdm",
        "source_method": "closed_5m_kline_taker_quote_volume",
        "queries_budget": EXTERNAL_FLOW_MAX_SYMBOLS_PER_CYCLE,
        "queries_attempted": len(selected),
        "available": len(snapshots),
        "unavailable": max(0, len(selected) - len(snapshots)),
        "status_counts": status_counts,
        "selection": selection,
        "cross_section": cross_summary,
        "lab": {k: v for k, v in lab.items() if k != "symbols"},
        "outcome_lab": outcome_lab,
        "independent_from_native_flow_synthesizer": True,
        "same_venue_as_binance_derivatives_research": True,
        "does_not_count_as_independent_council_vote": True,
        "diagnostic_only": True,
        "execution_authority": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }


def external_aggressive_flow_context_observation(source_status, now):
    return {
        "symbol": "MARKET",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.98,
        "source": "LeanTrader Binance USD-M aggressive-flow acquisition lab",
        "provenance": f"leantrader:external-aggressive-flow:v1.29:{int(now)}",
        "observed_at": now,
        "horizon_seconds": 0,
        "metadata": {
            "model": "external_aggressive_flow_acquisition_v1.29",
            "source_status": source_status,
            "directional_claim": False,
            "diagnostic_only": True,
            "does_not_count_as_independent_council_vote": True,
            "execution_authority": False,
            "risk_increase_allowed": False,
            "auto_promotion_allowed": False,
        },
    }

def _liquidation_event_window(state, symbol, values, source_ts, now, *, cached=False, confidence=0.0):
    """Persist only unique, eventful liquidation snapshots inside a short research window.

    v1.25 addresses sparse liquidation transport without lowering event thresholds. The ledger
    accepts only fresh, non-cached snapshots with positive event count/notional, suppresses
    exact payload replays even if a producer timestamp changes, and prunes aggressively. The
    aggregate is shadow evidence only; it cannot create execution authority or increase risk.
    """
    state = state if isinstance(state, dict) else {}
    symbol = normalize_symbol(symbol)
    values = values if isinstance(values, dict) else {}
    ledger = state.get("liquidation_event_window") if isinstance(state.get("liquidation_event_window"), dict) else {}
    ledger = {str(k): list(v) for k, v in ledger.items() if isinstance(v, list)}

    cutoff = finite(now) - LIQUIDATION_EVENT_WINDOW_SECONDS
    for sym in list(ledger):
        kept = [r for r in ledger.get(sym, []) if isinstance(r, dict) and finite(r.get("source_timestamp")) >= cutoff]
        if kept:
            ledger[sym] = kept[-LIQUIDATION_EVENT_WINDOW_MAX_SNAPSHOTS:]
        else:
            ledger.pop(sym, None)

    rows = list(ledger.get(symbol, [])) if symbol else []
    events = max(0, int(finite(values.get("events"))))
    notional = max(0.0, finite(values.get("liquidation_notional")))
    imbalance = clamp(finite(values.get("liquidation_imbalance")))
    source_ts = finite(source_ts)
    fresh_enough = source_ts > 0 and source_ts >= cutoff and source_ts <= finite(now) + 30.0
    payload_signature = f"{events}|{notional:.6f}|{imbalance:+.6f}"
    payload_replay = any(str(r.get("payload_signature") or "") == payload_signature for r in rows)
    timestamp_replay = any(abs(finite(r.get("source_timestamp")) - source_ts) < 0.001 for r in rows) if source_ts > 0 else False
    same_cycle_module_reuse = any(
        abs(finite(r.get("seen_at")) - finite(now)) < 0.001
        and (str(r.get("payload_signature") or "") == payload_signature or abs(finite(r.get("source_timestamp")) - source_ts) < 0.001)
        for r in rows
    )
    replay = payload_replay or timestamp_replay
    added = False
    if symbol and not cached and fresh_enough and events > 0 and notional > 0.0 and not replay:
        rows.append({
            "source_timestamp": source_ts,
            "seen_at": finite(now),
            "events": events,
            "liquidation_notional": notional,
            "liquidation_imbalance": imbalance,
            "confidence": clamp(finite(confidence), 0.0, 1.0),
            "payload_signature": payload_signature,
        })
        rows = sorted(rows, key=lambda r: (finite(r.get("source_timestamp")), str(r.get("payload_signature") or "")))[-LIQUIDATION_EVENT_WINDOW_MAX_SNAPSHOTS:]
        ledger[symbol] = rows
        added = True

    # Bound symbol growth by newest retained event snapshot.
    if len(ledger) > LIQUIDATION_EVENT_WINDOW_MAX_SYMBOLS:
        ranked = sorted(
            ledger.items(),
            key=lambda kv: max([finite(r.get("source_timestamp")) for r in kv[1] if isinstance(r, dict)] or [0.0]),
            reverse=True,
        )[:LIQUIDATION_EVENT_WINDOW_MAX_SYMBOLS]
        ledger = dict(ranked)
    state["liquidation_event_window"] = ledger
    rows = list(ledger.get(symbol, [])) if symbol else []

    total_events = sum(max(0, int(finite(r.get("events")))) for r in rows)
    total_notional = sum(max(0.0, finite(r.get("liquidation_notional"))) for r in rows)
    weighted_den = 0.0
    weighted_num = 0.0
    confidences = []
    for r in rows:
        weight = max(0.0, finite(r.get("liquidation_notional")))
        if weight <= 0.0:
            weight = max(1.0, finite(r.get("events")))
        weighted_den += weight
        weighted_num += clamp(finite(r.get("liquidation_imbalance"))) * weight
        c = clamp(finite(r.get("confidence")), 0.0, 1.0)
        if c > 0:
            confidences.append(c)
    agg_imbalance = clamp(weighted_num / weighted_den) if weighted_den > 0 else 0.0
    latest_ts = max([finite(r.get("source_timestamp")) for r in rows] or [0.0])
    oldest_ts = min([finite(r.get("source_timestamp")) for r in rows] or [0.0])
    aggregate_confidence = min(confidences) if confidences else 0.0
    return {
        "events": total_events,
        "liquidation_notional": total_notional,
        "liquidation_imbalance": agg_imbalance,
        "confidence": aggregate_confidence,
        "snapshot_count": len(rows),
        "latest_source_timestamp": latest_ts,
        "oldest_source_timestamp": oldest_ts,
        "oldest_age_seconds": max(0.0, finite(now) - oldest_ts) if oldest_ts > 0 else 0.0,
        "fresh_event_snapshot_added": added,
        "payload_replay_suppressed": bool(payload_replay and not same_cycle_module_reuse and not cached and fresh_enough and events > 0 and notional > 0.0),
        "source_timestamp_replay_suppressed": bool(timestamp_replay and not same_cycle_module_reuse and not cached and fresh_enough and events > 0 and notional > 0.0),
        "event_replay_suppressed": bool(replay and not same_cycle_module_reuse and not cached and fresh_enough and events > 0 and notional > 0.0),
        "same_cycle_module_reuse": bool(same_cycle_module_reuse),
        "cached_snapshot_ignored_for_accumulation": bool(cached and events > 0 and notional > 0.0),
        "window_seconds": LIQUIDATION_EVENT_WINDOW_SECONDS,
        "execution_authority": False,
        "can_increase_risk": False,
    }

def _orthogonal_price_anchor(symbol, raw_by_symbol, state, now):
    """Resolve a price anchor without allowing price to influence signal direction.

    Raw microstructure prices are preferred. A short-lived cache is second. Only when a
    sensor-derived hypothesis already qualifies do we query Bybit's public ticker endpoint.
    The returned price is used solely to open/resolve a shadow outcome episode.
    """
    raw = raw_by_symbol.get(symbol) if isinstance(raw_by_symbol, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    price = finite(raw.get("mid_price"))
    if price > 0:
        return price, "raw_microstructure", None

    state = state if isinstance(state, dict) else {}
    cache = state.get("orthogonal_price_anchor_cache") if isinstance(state.get("orthogonal_price_anchor_cache"), dict) else {}
    entry = cache.get(symbol) if isinstance(cache.get(symbol), dict) else {}
    cached_price = finite(entry.get("price"))
    cached_at = finite(entry.get("observed_at"))
    if cached_price > 0 and cached_at > 0 and now - cached_at <= ORTHOGONAL_PRICE_ANCHOR_MAX_AGE_SECONDS:
        return cached_price, "price_anchor_cache", None

    try:
        q = urllib.parse.urlencode({"category": "spot", "symbol": bybit_symbol(symbol)})
        payload = request_json(f"{BYBIT}/v5/market/tickers?{q}")
        if int(payload.get("retCode", -1)) != 0:
            return 0.0, "public_ticker", f"retCode={payload.get('retCode')}"
        rows = (payload.get("result") or {}).get("list") or []
        row = rows[0] if rows and isinstance(rows[0], dict) else {}
        bid = finite(row.get("bid1Price"))
        ask = finite(row.get("ask1Price"))
        last = finite(row.get("lastPrice"))
        price = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
        if price <= 0:
            return 0.0, "public_ticker", "non_positive_price"
        cache = dict(cache)
        cache[symbol] = {"price": price, "observed_at": now, "source": "bybit_public_ticker"}
        if len(cache) > 64:
            ranked = sorted(cache.items(), key=lambda kv: finite((kv[1] or {}).get("observed_at")), reverse=True)[:64]
            cache = dict(ranked)
        state["orthogonal_price_anchor_cache"] = cache
        return price, "public_ticker", None
    except Exception as exc:
        return 0.0, "public_ticker", f"{type(exc).__name__}: {exc}"


def orthogonal_flow_positioning_observations(now, contexts=None, raw_by_symbol=None, state=None):
    """Emit pre-registered, shadow-only cross-sensor hypotheses with bounded temporal joins.

    v1.20 adds flow-lineage quorum after v1.19 showed that most otherwise fresh flow blocks
    contained exactly one core component. Single-component flow is never treated as internally
    diversified. It can participate only when strong, cross-sectionally non-common, and paired
    with an independent derivatives/liquidation family under stricter gates and a lower
    confidence cap. Repeated one-component fingerprints remain context-only common factors.
    """
    latest, sensor_symbols = _orthogonal_sensor_snapshot()
    state = state if isinstance(state, dict) else {}
    raw_by_symbol = raw_by_symbol if isinstance(raw_by_symbol, dict) else {}
    snapshot_ts = finite(latest.get("timestamp") or latest.get("updated_at"), now)
    cache = _update_orthogonal_sensor_cache(state, sensor_symbols, snapshot_ts, now)

    # v1.20: orthogonal research is decoupled from the strategy/world-model selection list.
    # The core sensor fabric may rotate or cover symbols that the directional strategy selector
    # did not choose this cycle. Filtering to that list discarded valid independent evidence.
    selected = {normalize_symbol(x) for x in select_symbols(limit=20)}
    selected.discard("")
    all_universe = {normalize_symbol(k) for k in set(sensor_symbols.keys()) | set(cache.keys())}
    all_universe.discard("")

    def universe_rank(symbol):
        bundle = sensor_symbols.get(symbol) if isinstance(sensor_symbols.get(symbol), dict) else next((v for k, v in sensor_symbols.items() if normalize_symbol(k) == symbol and isinstance(v, dict)), {})
        crow = cache.get(symbol) if isinstance(cache.get(symbol), dict) else {}
        current_available = sum(1 for fam in ("flow_intelligence", "derivatives", "liquidations") if str((bundle.get(fam) or {}).get("status")) == "available")
        cached_available = sum(1 for fam in ("flow_intelligence", "derivatives", "liquidations") if isinstance((crow.get(fam) or {}).get("block"), dict))
        flow_first = 1 if (str((bundle.get("flow_intelligence") or {}).get("status")) == "available" or isinstance((crow.get("flow_intelligence") or {}).get("block"), dict)) else 0
        freshness = max([finite((e or {}).get("seen_at")) for e in crow.values() if isinstance(e, dict)] or [0.0])
        return (flow_first, current_available, cached_available, freshness, 1 if symbol in selected else 0, symbol)

    ranked_universe = sorted(all_universe, key=universe_rank, reverse=True)
    universe = set(ranked_universe[:ORTHOGONAL_RESEARCH_UNIVERSE_LIMIT])
    flow_profiles, flow_lineage_summary, common_flow_fingerprints = _orthogonal_flow_lineage_profiles(
        universe, sensor_symbols, cache, snapshot_ts, now
    )
    residual_rows, flow_residual_summary = _flow_residual_cross_section(flow_profiles)
    flow_residual_lab = _update_flow_residual_lab(state, residual_rows, flow_residual_summary, now)
    flow_residual_summary = dict(flow_residual_summary)
    flow_residual_summary.update({
        "samples_added_this_cycle": int(flow_residual_lab.get("samples_added_this_cycle") or 0),
        "symbols_tracked": int(flow_residual_lab.get("symbols_tracked") or 0),
        "persistent_candidate_count": int(flow_residual_lab.get("persistent_candidate_count") or 0),
        "persistent_diagnostic_candidates": list(flow_residual_lab.get("persistent_diagnostic_candidates") or [])[:8],
    })

    rows = []
    status = {
        "sensor_symbols": len(sensor_symbols),
        "strategy_selected_symbols": len(selected),
        "selected_sensor_symbols": len(all_universe & selected),
        "sensor_universe_candidates": len(all_universe),
        "sensor_universe_considered": len(universe),
        "sensor_universe_excluded_by_cap": max(0, len(all_universe) - len(universe)),
        "research_universe_is_decoupled_from_strategy_selection": True,
        "derivatives_available": 0,
        "flow_intelligence_available": 0,
        "liquidation_available": 0,
        "derivatives_available_after_alignment": 0,
        "flow_intelligence_available_after_alignment": 0,
        "liquidation_available_after_alignment": 0,
        "liquidation_event_window_seconds": LIQUIDATION_EVENT_WINDOW_SECONDS,
        "liquidation_event_window_fresh_snapshots_added": 0,
        "liquidation_event_window_payload_replays_suppressed": 0,
        "liquidation_event_window_source_timestamp_replays_suppressed": 0,
        "liquidation_event_window_reused_pairs": 0,
        "cache_reused_derivatives": 0,
        "cache_reused_flow_intelligence": 0,
        "cache_reused_liquidations": 0,
        "eligible_two_family_symbols_current": 0,
        "eligible_two_family_symbols": 0,
        "eligible_flow_derivatives_pairs": 0,
        "eligible_flow_liquidation_pairs": 0,
        "source_skew_rejections": 0,
        "quality_rejections": 0,
        "flow_lineage": flow_lineage_summary,
        "flow_residual_diagnostics": flow_residual_summary,
        "quality_rejections_by_family": {
            "flow_intelligence": {
                "rejected_blocks": 0,
                "components_lt_2": 0,
                "confidence_lt_045": 0,
                "single_component_common_factor": 0,
                "single_component_below_strict_magnitude": 0,
                "single_component_below_confidence": 0,
            },
            "derivatives": {"rejected_blocks": 0, "confidence_lt_055": 0},
            "liquidations": {"rejected_blocks": 0, "confidence_lt_045": 0},
        },
        "quality_rejection_symbols": [],
        "missing_price_anchor": 0,
        "price_anchor_from_raw": 0,
        "price_anchor_from_cache": 0,
        "price_anchor_public_fetch": 0,
        "price_anchor_fetch_errors": 0,
        "price_anchor_fetch_budget": ORTHOGONAL_MAX_PRICE_ANCHOR_FETCHES_PER_CYCLE,
        "coverage_gain_symbols": 0,
        "hypothesis_counts": {},
        "hypothesis_gate_rejections": {
            "flow_crowding_unwind": {},
            "flow_liquidation_alignment": {},
            "flow_leverage_confirmation": {},
        },
        "cache_policy": {
            "derivatives_max_age_seconds": ORTHOGONAL_CACHE_DERIVATIVES_MAX_AGE,
            "flow_max_age_seconds": ORTHOGONAL_CACHE_FLOW_MAX_AGE,
            "liquidations_max_age_seconds": ORTHOGONAL_CACHE_LIQUIDATIONS_MAX_AGE,
            "max_pair_source_skew_seconds": ORTHOGONAL_MAX_SOURCE_SKEW_SECONDS,
            "cache_can_increase_confidence": False,
        },
        "execution_authority": False,
    }

    for symbol in sorted(universe):
        bundle = sensor_symbols.get(symbol)
        if not isinstance(bundle, dict):
            # tolerate un-normalized keys in the core snapshot
            bundle = next((v for k, v in sensor_symbols.items() if normalize_symbol(k) == symbol and isinstance(v, dict)), {})
        cache_row = cache.get(symbol) if isinstance(cache.get(symbol), dict) else {}

        # Current availability is diagnostic only; eligibility below can use a bounded fresh cache.
        current = {}
        for family, key in (("derivatives", "derivatives_available"), ("flow_intelligence", "flow_intelligence_available"), ("liquidations", "liquidation_available")):
            b = bundle.get(family) if isinstance(bundle, dict) and isinstance(bundle.get(family), dict) else {}
            st, cf, vv = _sensor_block_values(b)
            current[family] = (st, cf, vv, _sensor_block_timestamp(b, snapshot_ts) if st == "available" else 0.0)
            if st == "available":
                status[key] += 1

        db, d_status, d_conf, d, d_ts, d_cached, _d_age = _resolve_orthogonal_family(bundle, cache_row, "derivatives", snapshot_ts, now)
        fb, f_status, f_conf, f, f_ts, f_cached, _f_age = _resolve_orthogonal_family(bundle, cache_row, "flow_intelligence", snapshot_ts, now)
        lb, l_status, l_conf, l, l_ts, l_cached, _l_age = _resolve_orthogonal_family(bundle, cache_row, "liquidations", snapshot_ts, now)
        current_liq_events = max(0, int(finite(l.get("events"))))
        current_liq_notional = max(0.0, finite(l.get("liquidation_notional")))
        current_liq_imbalance = clamp(finite(l.get("liquidation_imbalance")))
        liq_window = _liquidation_event_window(
            state, symbol, l, l_ts, now, cached=bool(l_cached), confidence=l_conf
        ) if l_status == "available" else {"events": 0, "liquidation_notional": 0.0, "liquidation_imbalance": 0.0, "snapshot_count": 0, "latest_source_timestamp": 0.0, "confidence": 0.0}
        if liq_window.get("fresh_event_snapshot_added"):
            status["liquidation_event_window_fresh_snapshots_added"] += 1
        if liq_window.get("payload_replay_suppressed"):
            status["liquidation_event_window_payload_replays_suppressed"] += 1
        if liq_window.get("source_timestamp_replay_suppressed"):
            status["liquidation_event_window_source_timestamp_replays_suppressed"] += 1
        liq_window_used = bool(
            l_status == "available"
            and int(liq_window.get("events") or 0) > 0
            and (int(liq_window.get("snapshot_count") or 0) > 1 or current_liq_events <= 0)
        )
        if l_status == "available" and int(liq_window.get("events") or 0) > 0:
            l = dict(l)
            l["events"] = int(liq_window.get("events") or 0)
            l["liquidation_notional"] = max(0.0, finite(liq_window.get("liquidation_notional")))
            l["liquidation_imbalance"] = clamp(finite(liq_window.get("liquidation_imbalance")))
            l_ts = finite(liq_window.get("latest_source_timestamp")) or l_ts
            wc = clamp(finite(liq_window.get("confidence")), 0.0, 1.0)
            if wc > 0:
                l_conf = min(l_conf, wc) if l_conf > 0 else wc
            if liq_window_used:
                status["liquidation_event_window_reused_pairs"] += 1
        if d_status == "available": status["derivatives_available_after_alignment"] += 1
        if f_status == "available": status["flow_intelligence_available_after_alignment"] += 1
        if l_status == "available": status["liquidation_available_after_alignment"] += 1
        if d_cached: status["cache_reused_derivatives"] += 1
        if f_cached: status["cache_reused_flow_intelligence"] += 1
        if l_cached: status["cache_reused_liquidations"] += 1

        components_available = int(finite(f.get("components_available")))
        flow_score_for_quality = clamp(finite(f.get("flow_score")))
        flow_profile = flow_profiles.get(symbol) if isinstance(flow_profiles.get(symbol), dict) else {}
        flow_multi_component_quality = f_status == "available" and components_available >= 2 and f_conf >= 0.45
        flow_single_component_quality = (
            f_status == "available"
            and components_available == 1
            and f_conf >= ORTHOGONAL_SINGLE_COMPONENT_MIN_CONFIDENCE
            and abs(flow_score_for_quality) >= ORTHOGONAL_SINGLE_COMPONENT_MIN_FLOW_SCORE
            and not bool(flow_profile.get("common_factor_like"))
        )
        flow_quality = flow_multi_component_quality or flow_single_component_quality
        flow_quality_tier = "multi_component" if flow_multi_component_quality else "single_component_cross_family" if flow_single_component_quality else "rejected"
        derivatives_quality = d_status == "available" and d_conf >= 0.55
        liquidation_quality = l_status == "available" and l_conf >= 0.45
        if f_status == "available" and not flow_quality:
            status["quality_rejections"] += 1
            q = status["quality_rejections_by_family"]["flow_intelligence"]
            q["rejected_blocks"] += 1
            if components_available < 2:
                q["components_lt_2"] += 1
            if f_conf < 0.45:
                q["confidence_lt_045"] += 1
            if components_available == 1 and bool(flow_profile.get("common_factor_like")):
                q["single_component_common_factor"] += 1
            if components_available == 1 and not bool(flow_profile.get("common_factor_like")) and abs(flow_score_for_quality) < ORTHOGONAL_SINGLE_COMPONENT_MIN_FLOW_SCORE:
                q["single_component_below_strict_magnitude"] += 1
            if components_available == 1 and not bool(flow_profile.get("common_factor_like")) and f_conf < ORTHOGONAL_SINGLE_COMPONENT_MIN_CONFIDENCE:
                q["single_component_below_confidence"] += 1
            if symbol not in status["quality_rejection_symbols"] and len(status["quality_rejection_symbols"]) < 24:
                status["quality_rejection_symbols"].append(symbol)
        if d_status == "available" and not derivatives_quality:
            status["quality_rejections"] += 1
            q = status["quality_rejections_by_family"]["derivatives"]
            q["rejected_blocks"] += 1
            if d_conf < 0.55:
                q["confidence_lt_055"] += 1
            if symbol not in status["quality_rejection_symbols"] and len(status["quality_rejection_symbols"]) < 24:
                status["quality_rejection_symbols"].append(symbol)
        if l_status == "available" and not liquidation_quality:
            status["quality_rejections"] += 1
            q = status["quality_rejections_by_family"]["liquidations"]
            q["rejected_blocks"] += 1
            if l_conf < 0.45:
                q["confidence_lt_045"] += 1
            if symbol not in status["quality_rejection_symbols"] and len(status["quality_rejection_symbols"]) < 24:
                status["quality_rejection_symbols"].append(symbol)

        fd_aligned = flow_quality and derivatives_quality and _orthogonal_pair_aligned(f_ts, d_ts)
        fl_aligned = flow_quality and liquidation_quality and _orthogonal_pair_aligned(f_ts, l_ts)
        if flow_quality and derivatives_quality and not fd_aligned:
            status["source_skew_rejections"] += 1
        if flow_quality and liquidation_quality and not fl_aligned:
            status["source_skew_rejections"] += 1
        if fd_aligned:
            status["eligible_flow_derivatives_pairs"] += 1
        if fl_aligned:
            status["eligible_flow_liquidation_pairs"] += 1
        if fd_aligned or fl_aligned:
            status["eligible_two_family_symbols"] += 1

        c_f = current["flow_intelligence"]
        c_d = current["derivatives"]
        c_l = current["liquidations"]
        c_flow_components = int(finite(c_f[2].get("components_available")))
        c_flow_score = clamp(finite(c_f[2].get("flow_score")))
        c_flow_fp = _orthogonal_flow_fingerprint(c_flow_score)
        c_flow_quality = (
            c_f[0] == "available"
            and c_f[3] > 0
            and now - c_f[3] <= ORTHOGONAL_CACHE_FLOW_MAX_AGE
            and (
                (c_flow_components >= 2 and c_f[1] >= 0.45)
                or (
                    c_flow_components == 1
                    and c_f[1] >= ORTHOGONAL_SINGLE_COMPONENT_MIN_CONFIDENCE
                    and abs(c_flow_score) >= ORTHOGONAL_SINGLE_COMPONENT_MIN_FLOW_SCORE
                    and c_flow_fp not in common_flow_fingerprints
                )
            )
        )
        c_deriv_quality = c_d[0] == "available" and c_d[1] >= 0.55 and c_d[3] > 0 and now - c_d[3] <= ORTHOGONAL_CACHE_DERIVATIVES_MAX_AGE
        c_liq_quality = c_l[0] == "available" and c_l[1] >= 0.45 and c_l[3] > 0 and now - c_l[3] <= ORTHOGONAL_CACHE_LIQUIDATIONS_MAX_AGE
        current_eligible = (c_flow_quality and c_deriv_quality and _orthogonal_pair_aligned(c_f[3], c_d[3])) or (c_flow_quality and c_liq_quality and _orthogonal_pair_aligned(c_f[3], c_l[3]))
        if current_eligible:
            status["eligible_two_family_symbols_current"] += 1

        if not flow_quality or not (fd_aligned or fl_aligned):
            continue

        flow = clamp(finite(f.get("flow_score")))
        oi = finite(d.get("open_interest_change_15m_window")) if fd_aligned else 0.0
        skew = clamp(finite(d.get("positioning_skew"))) if fd_aligned else 0.0
        funding = finite(d.get("funding_rate")) if fd_aligned else 0.0
        basis = finite(d.get("perpetual_basis")) if fd_aligned else 0.0
        liq_imb = clamp(finite(l.get("liquidation_imbalance"))) if fl_aligned else 0.0
        liq_events = int(finite(l.get("events"))) if fl_aligned else 0
        liq_notional = max(0.0, finite(l.get("liquidation_notional"))) if fl_aligned else 0.0
        context = market_context(symbol, raw_by_symbol.get(symbol) or {}, contexts or {})
        single_component_flow = flow_quality_tier == "single_component_cross_family"
        flow_gate = max(ORTHOGONAL_MIN_FLOW_SCORE, ORTHOGONAL_SINGLE_COMPONENT_MIN_FLOW_SCORE) if single_component_flow else ORTHOGONAL_MIN_FLOW_SCORE
        oi_gate = max(ORTHOGONAL_MIN_OI_BUILD, ORTHOGONAL_SINGLE_COMPONENT_MIN_OI_BUILD) if single_component_flow else ORTHOGONAL_MIN_OI_BUILD
        skew_gate = max(ORTHOGONAL_MIN_POSITIONING_SKEW, ORTHOGONAL_SINGLE_COMPONENT_MIN_POSITIONING_SKEW) if single_component_flow else ORTHOGONAL_MIN_POSITIONING_SKEW
        liq_imbalance_gate = max(ORTHOGONAL_MIN_LIQUIDATION_IMBALANCE, ORTHOGONAL_SINGLE_COMPONENT_MIN_LIQUIDATION_IMBALANCE) if single_component_flow else ORTHOGONAL_MIN_LIQUIDATION_IMBALANCE
        liq_event_gate = 3 if single_component_flow else 2

        candidates = []

        def reject(hypothesis, reason):
            bucket = status["hypothesis_gate_rejections"].setdefault(hypothesis, {})
            bucket[reason] = int(bucket.get(reason, 0)) + 1

        if fd_aligned:
            if abs(flow) < flow_gate: reject("flow_crowding_unwind", "flow_below_threshold")
            if oi < oi_gate: reject("flow_crowding_unwind", "oi_not_building")
            if abs(skew) < skew_gate: reject("flow_crowding_unwind", "positioning_skew_too_small")
            if flow * skew >= 0.0: reject("flow_crowding_unwind", "crowding_not_opposed_to_flow")
        if fl_aligned:
            if abs(flow) < flow_gate: reject("flow_liquidation_alignment", "flow_below_threshold")
            if liq_events < liq_event_gate: reject("flow_liquidation_alignment", "liquidation_events_lt_required")
            if liq_notional <= 0.0: reject("flow_liquidation_alignment", "liquidation_notional_nonpositive")
            if abs(liq_imb) < liq_imbalance_gate: reject("flow_liquidation_alignment", "liquidation_imbalance_below_threshold")
            if flow * liq_imb <= 0.0: reject("flow_liquidation_alignment", "liquidations_not_aligned_with_flow")
        if fd_aligned:
            if single_component_flow:
                reject("flow_leverage_confirmation", "single_component_flow_not_allowed")
            else:
                if abs(flow) < max(0.40, ORTHOGONAL_MIN_FLOW_SCORE): reject("flow_leverage_confirmation", "flow_below_strong_threshold")
                if oi < max(0.03, ORTHOGONAL_MIN_OI_BUILD): reject("flow_leverage_confirmation", "oi_below_strong_threshold")
                if abs(skew) < 0.05: reject("flow_leverage_confirmation", "positioning_skew_too_small")
                if abs(skew) > 0.30: reject("flow_leverage_confirmation", "positioning_skew_too_extreme")
                if flow * skew <= 0.0: reject("flow_leverage_confirmation", "positioning_not_aligned_with_flow")
                if abs(funding) > 0.0010: reject("flow_leverage_confirmation", "funding_too_crowded")

        if (
            fd_aligned
            and abs(flow) >= flow_gate
            and oi >= oi_gate
            and abs(skew) >= skew_gate
            and flow * skew < 0.0
        ):
            candidates.append(("flow_crowding_unwind", 1.0 if flow > 0 else -1.0, ["flow_intelligence", "derivatives_positioning"], [f_ts, d_ts]))

        if (
            fl_aligned
            and abs(flow) >= flow_gate
            and liq_events >= liq_event_gate
            and liq_notional > 0.0
            and abs(liq_imb) >= liq_imbalance_gate
            and flow * liq_imb > 0.0
        ):
            candidates.append(("flow_liquidation_alignment", 1.0 if flow > 0 else -1.0, ["flow_intelligence", "liquidation_tape"], [f_ts, l_ts]))

        if (
            fd_aligned
            and not single_component_flow
            and abs(flow) >= max(0.40, ORTHOGONAL_MIN_FLOW_SCORE)
            and oi >= max(0.03, ORTHOGONAL_MIN_OI_BUILD)
            and abs(skew) >= 0.05
            and abs(skew) <= 0.30
            and flow * skew > 0.0
            and abs(funding) <= 0.0010
        ):
            candidates.append(("flow_leverage_confirmation", 1.0 if flow > 0 else -1.0, ["flow_intelligence", "derivatives_positioning"], [f_ts, d_ts]))

        if not candidates:
            continue
        raw_anchor = finite((raw_by_symbol.get(symbol) or {}).get("mid_price"))
        if raw_anchor > 0:
            price_anchor, anchor_source, anchor_error = raw_anchor, "raw_microstructure", None
        elif status["price_anchor_public_fetch"] + status["price_anchor_fetch_errors"] < ORTHOGONAL_MAX_PRICE_ANCHOR_FETCHES_PER_CYCLE:
            price_anchor, anchor_source, anchor_error = _orthogonal_price_anchor(symbol, raw_by_symbol, state, now)
        else:
            price_anchor, anchor_source, anchor_error = 0.0, "budget_exhausted", "price_anchor_fetch_budget_exhausted"
        if anchor_source == "raw_microstructure":
            status["price_anchor_from_raw"] += 1
        elif anchor_source == "price_anchor_cache":
            status["price_anchor_from_cache"] += 1
        elif anchor_source == "public_ticker" and price_anchor > 0:
            status["price_anchor_public_fetch"] += 1
        if anchor_error:
            status["price_anchor_fetch_errors"] += 1
        if price_anchor <= 0:
            status["missing_price_anchor"] += 1
            continue

        for hypothesis_id, direction, families, family_ts in candidates:
            flow_strength = clamp(abs(flow), 0.0, 1.0)
            if "liquidation_tape" in families:
                liq_strength = clamp(abs(liq_imb), 0.0, 1.0)
                raw_strength = clamp(0.65 * flow_strength + 0.35 * liq_strength, 0.0, 1.0)
                source_quality = min(f_conf, l_conf)
            else:
                oi_strength = clamp(max(0.0, oi) / 0.08, 0.0, 1.0)
                skew_strength = clamp(abs(skew) / 0.30, 0.0, 1.0)
                raw_strength = clamp(0.58 * flow_strength + 0.24 * oi_strength + 0.18 * skew_strength, 0.0, 1.0)
                source_quality = min(d_conf, f_conf)
            confidence = min(ORTHOGONAL_MAX_SIGNAL_CONFIDENCE, 0.18 + 0.32 * raw_strength) * clamp(source_quality, 0.35, 1.0)
            if single_component_flow:
                confidence = min(confidence, ORTHOGONAL_SINGLE_COMPONENT_MAX_SIGNAL_CONFIDENCE)
            score = direction * max(0.05, raw_strength)
            status["hypothesis_counts"][hypothesis_id] = int(status["hypothesis_counts"].get(hypothesis_id, 0)) + 1
            rows.append({
                "symbol": symbol,
                "kind": "signal",
                "score": score,
                "confidence": clamp(confidence, 0.0, ORTHOGONAL_MAX_SIGNAL_CONFIDENCE),
                "source": "LeanTrader temporally aligned native public flow + derivatives/liquidation sensor fabric",
                "provenance": f"native-sensor:orthogonal-flow-v1.25:{hypothesis_id}:{symbol}:{int(max(family_ts))}",
                "observed_at": now,
                "horizon_seconds": ORTHOGONAL_FLOW_HORIZON_SECONDS,
                "metadata": {
                    "model": "orthogonal_flow_positioning_discovery_v1.25",
                    "hypothesis_id": hypothesis_id,
                    "source_families": families,
                    "source_family_count": len(set(families)),
                    "sensor_snapshot_timestamp": snapshot_ts,
                    "source_family_timestamps": family_ts,
                    "source_pair_skew_seconds": abs(family_ts[0] - family_ts[1]),
                    "temporal_alignment_max_skew_seconds": ORTHOGONAL_MAX_SOURCE_SKEW_SECONDS,
                    "used_cached_derivatives": bool(d_cached and "derivatives_positioning" in families),
                    "used_cached_flow_intelligence": bool(f_cached),
                    "used_cached_liquidations": bool(l_cached and "liquidation_tape" in families),
                    "flow_score": flow,
                    "open_interest_change_15m_window": oi,
                    "positioning_skew": skew,
                    "funding_rate": funding,
                    "perpetual_basis": basis,
                    "liquidation_imbalance": liq_imb,
                    "liquidation_events": liq_events,
                    "liquidation_notional": liq_notional,
                    "current_liquidation_events": current_liq_events,
                    "current_liquidation_notional": current_liq_notional,
                    "current_liquidation_imbalance": current_liq_imbalance,
                    "liquidation_evidence_window_used": bool(liq_window_used),
                    "liquidation_evidence_window_snapshots": int(liq_window.get("snapshot_count") or 0),
                    "liquidation_evidence_window_seconds": LIQUIDATION_EVENT_WINDOW_SECONDS,
                    "liquidation_payload_replay_suppressed": bool(liq_window.get("payload_replay_suppressed")),
                    "components_available": components_available,
                    "flow_quality_tier": flow_quality_tier,
                    "flow_component_fingerprint": _orthogonal_flow_fingerprint(flow),
                    "flow_component_common_factor_like": bool(flow_profile.get("common_factor_like")),
                    "single_component_cross_family_quorum": bool(single_component_flow),
                    "single_component_internal_diversity_claim": False,
                    "single_component_stricter_gate_applied": bool(single_component_flow),
                    "market_context": context,
                    "mid_price": price_anchor,
                    "price_anchor_source": anchor_source,
                    "price_anchor_used_only_for_shadow_outcome_accounting": True,
                    "direction_not_derived_from_price_or_orderbook": True,
                    "pre_registered_hypothesis": True,
                    "council_eligible": False,
                    "diversity_eligible": False,
                    "recovery_probe": False,
                    "shadow_only": True,
                    "execution_authority": False,
                    "risk_increase_allowed": False,
                },
            })

    status["coverage_gain_symbols"] = max(0, int(status["eligible_two_family_symbols"]) - int(status["eligible_two_family_symbols_current"]))
    return rows, status



def derivatives_liquidation_observations(now, raw_by_symbol=None, state=None):
    """Research a mechanism-distinct derivatives + liquidation squeeze without flow direction.

    v1.20 proved that most one-component flow is a single common factor (+0.031 across the
    cross-section), so v1.21 does not relax that lineage guard. Instead it opens a separate,
    pre-registered shadow hypothesis using derivatives positioning and forced-liquidation
    evidence directly. These families can share venue lineage, therefore confidence is penalized
    and the challenger never participates in council/diversity voting or execution.
    """
    raw_by_symbol = raw_by_symbol if isinstance(raw_by_symbol, dict) else {}
    state = state if isinstance(state, dict) else {}
    latest, sensor_symbols = _orthogonal_sensor_snapshot()
    snapshot_ts = _sensor_block_timestamp(latest, now) if isinstance(latest, dict) else finite(now)
    if snapshot_ts <= 0:
        snapshot_ts = finite(now)
    cache = state.get("orthogonal_sensor_cache") if isinstance(state.get("orthogonal_sensor_cache"), dict) else {}
    if not cache:
        cache = _update_orthogonal_sensor_cache(state, sensor_symbols, snapshot_ts, now)
    universe = sorted({normalize_symbol(k) for k in list((sensor_symbols or {}).keys()) + list((cache or {}).keys()) if normalize_symbol(k)})[:ORTHOGONAL_RESEARCH_UNIVERSE_LIMIT]
    status = {
        "sensor_universe_considered": len(universe),
        "derivatives_available_after_alignment": 0,
        "liquidations_available_after_alignment": 0,
        "aligned_mechanism_pairs": 0,
        "liquidation_transport_available_pairs": 0,
        "liquidation_event_evidence_pairs": 0,
        "liquidation_current_event_evidence_pairs": 0,
        "liquidation_quiet_zero_event_pairs": 0,
        "liquidation_event_window_seconds": LIQUIDATION_EVENT_WINDOW_SECONDS,
        "liquidation_event_window_fresh_snapshots_added": 0,
        "liquidation_event_window_payload_replays_suppressed": 0,
        "liquidation_event_window_source_timestamp_replays_suppressed": 0,
        "liquidation_event_window_reused_pairs": 0,
        "quality_rejections": {"derivatives": 0, "liquidations": 0},
        "source_skew_rejections": 0,
        "hypothesis_gate_rejections": {
            "positioning_liquidation_squeeze": {
                "positioning_skew_below_threshold": 0,
                "liquidation_events_lt_required": 0,
                "liquidation_notional_nonpositive": 0,
                "liquidation_imbalance_below_threshold": 0,
                "positioning_not_opposed_to_liquidation_direction": 0,
                "funding_contradicts_positioning": 0,
                "oi_collapse_extreme": 0,
            }
        },
        "hypothesis_counts": {},
        "price_anchor_sources": {"raw": 0, "cache": 0, "public_fetch": 0, "fetch_errors": 0, "missing": 0},
        "same_venue_lineage_penalty": DERIV_LIQ_SAME_VENUE_CONFIDENCE_MULTIPLIER,
        "direction_not_derived_from_price_orderbook_or_flow": True,
        "flow_common_factor_is_not_used_for_direction": True,
        "execution_authority": False,
    }
    rows = []
    for symbol in universe:
        bundle = sensor_symbols.get(symbol) if isinstance(sensor_symbols.get(symbol), dict) else next((v for k,v in (sensor_symbols or {}).items() if normalize_symbol(k)==symbol and isinstance(v,dict)), {})
        cache_row = cache.get(symbol) if isinstance(cache.get(symbol), dict) else {}
        _db, d_status, d_conf, d, d_ts, d_cached, _ = _resolve_orthogonal_family(bundle, cache_row, "derivatives", snapshot_ts, now)
        _lb, l_status, l_conf, l, l_ts, l_cached, _ = _resolve_orthogonal_family(bundle, cache_row, "liquidations", snapshot_ts, now)
        current_liq_events = max(0, int(finite(l.get("events"))))
        current_liq_notional = max(0.0, finite(l.get("liquidation_notional")))
        current_liq_imbalance = clamp(finite(l.get("liquidation_imbalance")))
        if current_liq_events > 0 and current_liq_notional > 0.0:
            status["liquidation_current_event_evidence_pairs"] += 1
        liq_window = _liquidation_event_window(
            state, symbol, l, l_ts, now, cached=bool(l_cached), confidence=l_conf
        ) if l_status == "available" else {"events": 0, "liquidation_notional": 0.0, "liquidation_imbalance": 0.0, "snapshot_count": 0, "latest_source_timestamp": 0.0, "confidence": 0.0}
        if liq_window.get("fresh_event_snapshot_added"):
            status["liquidation_event_window_fresh_snapshots_added"] += 1
        if liq_window.get("payload_replay_suppressed"):
            status["liquidation_event_window_payload_replays_suppressed"] += 1
        if liq_window.get("source_timestamp_replay_suppressed"):
            status["liquidation_event_window_source_timestamp_replays_suppressed"] += 1
        liq_window_used = bool(
            l_status == "available"
            and int(liq_window.get("events") or 0) > 0
            and (int(liq_window.get("snapshot_count") or 0) > 1 or current_liq_events <= 0)
        )
        if l_status == "available" and int(liq_window.get("events") or 0) > 0:
            l = dict(l)
            l["events"] = int(liq_window.get("events") or 0)
            l["liquidation_notional"] = max(0.0, finite(liq_window.get("liquidation_notional")))
            l["liquidation_imbalance"] = clamp(finite(liq_window.get("liquidation_imbalance")))
            l_ts = finite(liq_window.get("latest_source_timestamp")) or l_ts
            wc = clamp(finite(liq_window.get("confidence")), 0.0, 1.0)
            if wc > 0:
                l_conf = min(l_conf, wc) if l_conf > 0 else wc
            if liq_window_used:
                status["liquidation_event_window_reused_pairs"] += 1
        if d_status == "available": status["derivatives_available_after_alignment"] += 1
        if l_status == "available": status["liquidations_available_after_alignment"] += 1
        d_quality = d_status == "available" and d_conf >= DERIV_LIQ_MIN_DERIV_CONFIDENCE
        l_quality = l_status == "available" and l_conf >= DERIV_LIQ_MIN_LIQ_CONFIDENCE
        if d_status == "available" and not d_quality: status["quality_rejections"]["derivatives"] += 1
        if l_status == "available" and not l_quality: status["quality_rejections"]["liquidations"] += 1
        if not (d_quality and l_quality):
            continue
        if not _orthogonal_pair_aligned(d_ts, l_ts):
            status["source_skew_rejections"] += 1
            continue
        status["aligned_mechanism_pairs"] += 1
        skew = clamp(finite(d.get("positioning_skew")))
        oi = finite(d.get("open_interest_change_15m_window"))
        funding = finite(d.get("funding_rate"))
        basis = finite(d.get("perpetual_basis"))
        liq_imb = clamp(finite(l.get("liquidation_imbalance")))
        liq_events = int(finite(l.get("events")))
        liq_notional = max(0.0, finite(l.get("liquidation_notional")))
        status["liquidation_transport_available_pairs"] += 1
        if liq_events > 0 and liq_notional > 0.0:
            status["liquidation_event_evidence_pairs"] += 1
        if current_liq_events <= 0 or current_liq_notional <= 0.0:
            status["liquidation_quiet_zero_event_pairs"] += 1
        rej = status["hypothesis_gate_rejections"]["positioning_liquidation_squeeze"]
        failed = False
        def fail(reason):
            nonlocal failed
            rej[reason] = int(rej.get(reason,0)) + 1
            failed = True
        if abs(skew) < DERIV_LIQ_MIN_POSITIONING_SKEW: fail("positioning_skew_below_threshold")
        if liq_events < DERIV_LIQ_MIN_LIQUIDATION_EVENTS: fail("liquidation_events_lt_required")
        if liq_notional <= 0.0: fail("liquidation_notional_nonpositive")
        if abs(liq_imb) < DERIV_LIQ_MIN_LIQUIDATION_IMBALANCE: fail("liquidation_imbalance_below_threshold")
        if skew * liq_imb >= 0.0: fail("positioning_not_opposed_to_liquidation_direction")
        # Funding is only a contradiction guard when materially non-neutral; it never sets direction.
        if abs(funding) >= 0.0002 and funding * skew < 0.0: fail("funding_contradicts_positioning")
        # Extremely fast OI collapse is treated as post-event exhaustion rather than continuation.
        if oi <= -0.08: fail("oi_collapse_extreme")
        if failed:
            continue
        price_anchor = finite((raw_by_symbol.get(symbol) or {}).get("mid_price"))
        anchor_source = "raw_microstructure" if price_anchor > 0 else ""
        anchor_error = None
        if price_anchor <= 0:
            price_anchor, anchor_source, anchor_error = _orthogonal_price_anchor(symbol, raw_by_symbol, state, now)
        if anchor_source == "raw_microstructure": status["price_anchor_sources"]["raw"] += 1
        elif anchor_source == "price_anchor_cache": status["price_anchor_sources"]["cache"] += 1
        elif anchor_source == "public_ticker" and price_anchor > 0: status["price_anchor_sources"]["public_fetch"] += 1
        if anchor_error: status["price_anchor_sources"]["fetch_errors"] += 1
        if price_anchor <= 0:
            status["price_anchor_sources"]["missing"] += 1
            continue
        direction = 1.0 if liq_imb > 0 else -1.0
        liq_strength = clamp(abs(liq_imb),0.0,1.0)
        skew_strength = clamp(abs(skew)/0.30,0.0,1.0)
        oi_support = clamp(max(0.0,oi)/0.06,0.0,1.0)
        raw_strength = clamp(0.58*liq_strength + 0.32*skew_strength + 0.10*oi_support,0.0,1.0)
        source_quality = min(d_conf,l_conf)
        confidence = min(DERIV_LIQ_MAX_SIGNAL_CONFIDENCE, (0.16 + 0.24*raw_strength) * source_quality * DERIV_LIQ_SAME_VENUE_CONFIDENCE_MULTIPLIER)
        status["hypothesis_counts"]["positioning_liquidation_squeeze"] = int(status["hypothesis_counts"].get("positioning_liquidation_squeeze",0)) + 1
        rows.append({
            "symbol": symbol,
            "kind": "signal",
            "score": direction * max(0.05,raw_strength),
            "confidence": clamp(confidence,0.0,DERIV_LIQ_MAX_SIGNAL_CONFIDENCE),
            "source": "LeanTrader native derivatives positioning + forced-liquidation mechanism research",
            "provenance": f"native-sensor:deriv-liq-v1.25:positioning_liquidation_squeeze:{symbol}:{int(max(d_ts,l_ts))}",
            "observed_at": now,
            "horizon_seconds": DERIV_LIQ_HORIZON_SECONDS,
            "metadata": {
                "model": "derivatives_liquidation_event_window_v1.25",
                "hypothesis_id": "positioning_liquidation_squeeze",
                "source_families": ["derivatives_positioning","liquidation_tape"],
                "sensor_snapshot_timestamp": snapshot_ts,
                "source_family_count": 2,
                "same_venue_lineage_possible": True,
                "mechanism_distinct_not_venue_independent": True,
                "same_venue_confidence_multiplier": DERIV_LIQ_SAME_VENUE_CONFIDENCE_MULTIPLIER,
                "source_family_timestamps": [d_ts,l_ts],
                "source_pair_skew_seconds": abs(d_ts-l_ts),
                "used_cached_derivatives": bool(d_cached),
                "used_cached_liquidations": bool(l_cached),
                "positioning_skew": skew,
                "open_interest_change_15m_window": oi,
                "funding_rate": funding,
                "perpetual_basis": basis,
                "liquidation_imbalance": liq_imb,
                "liquidation_events": liq_events,
                "liquidation_notional": liq_notional,
                "current_liquidation_events": current_liq_events,
                "current_liquidation_notional": current_liq_notional,
                "current_liquidation_imbalance": current_liq_imbalance,
                "liquidation_evidence_window_used": bool(liq_window_used),
                "liquidation_evidence_window_snapshots": int(liq_window.get("snapshot_count") or 0),
                "liquidation_evidence_window_seconds": LIQUIDATION_EVENT_WINDOW_SECONDS,
                "liquidation_payload_replay_suppressed": bool(liq_window.get("payload_replay_suppressed")),
                "mid_price": price_anchor,
                "price_anchor_source": anchor_source,
                "price_anchor_used_only_for_shadow_outcome_accounting": True,
                "direction_not_derived_from_price_or_orderbook": True,
                "direction_not_derived_from_flow_intelligence": True,
                "flow_common_factor_is_context_only": True,
                "pre_registered_hypothesis": True,
                "council_eligible": False,
                "diversity_eligible": False,
                "shadow_only": True,
                "execution_authority": False,
                "risk_increase_allowed": False,
            }
        })
    return rows,status


def _deriv_liq_prequential_stats(rows):
    rows = sorted([r for r in (rows or []) if isinstance(r,dict)], key=lambda r:(finite(r.get("opened_at")),str(r.get("episode_id") or "")))
    n=len(rows)
    holdout_n=max(MIN_DERIV_LIQ_HOLDOUT_SAMPLES,int(math.ceil(n*0.30))) if n else 0
    if n >= MIN_DERIV_LIQ_POLICY_SAMPLES:
        holdout_n=min(holdout_n,max(MIN_DERIV_LIQ_HOLDOUT_SAMPLES,n-5))
    else:
        holdout_n=min(holdout_n,n)
    split=max(0,n-holdout_n)
    tr,ho=rows[:split],rows[split:]
    return {"samples":n,"train_samples":len(tr),"holdout_samples":len(ho),"train":_horizon_metric_stats(tr),"holdout":_horizon_metric_stats(ho),"all":_horizon_metric_stats(rows)}


def derivatives_liquidation_policy(state, source_status=None, now=None):
    _=finite(now,time.time())
    resolved=state.get("specialist_resolved") if isinstance(state.get("specialist_resolved"),list) else []
    raw_rows=[r for r in resolved if isinstance(r,dict) and r.get("pack_id")==PACK_DERIV_LIQ]
    rows=nonoverlapping_resolved_rows(raw_rows)
    hypotheses=sorted({str(r.get("hypothesis_id") or "unknown") for r in rows})
    by_hypothesis={}; positive=[]; watches=[]; negative=[]
    for hid in hypotheses:
        st=_deriv_liq_prequential_stats([r for r in rows if str(r.get("hypothesis_id") or "unknown")==hid])
        n=int(st.get("samples") or 0); hn=int(st.get("holdout_samples") or 0); tn=int(st.get("train_samples") or 0)
        tr=st.get("train") or {}; ho=st.get("holdout") or {}
        if n < MIN_DERIV_LIQ_POLICY_SAMPLES or hn < MIN_DERIV_LIQ_HOLDOUT_SAMPLES or tn < 5:
            lifecycle="insufficient_prequential_evidence"
        elif (finite(tr.get("average_net_return"))>0 and finite(tr.get("robust_net_edge_score"))>0 and finite(ho.get("average_net_return"))>MIN_DERIV_LIQ_HOLDOUT_NET and finite(ho.get("ewma_net_return"))>0 and finite(ho.get("robust_net_edge_score"))>0 and finite(ho.get("win_rate"))>=0.45 and finite(ho.get("cost_coverage_ratio"))>=1.10):
            if n>=40:
                lifecycle="research_positive"; positive.append(hid)
            else:
                lifecycle="prequential_positive_watch"; watches.append(hid)
        elif finite(ho.get("average_net_return"))<0 and finite(ho.get("robust_net_edge_score"))<0:
            lifecycle="prequential_negative"; negative.append(hid)
        else:
            lifecycle="watch"
        by_hypothesis[hid]={**st,"lifecycle":lifecycle,"execution_authority":False,"can_increase_risk":False,"auto_promotion_allowed":False}
    return {
        "model":"derivatives_liquidation_prequential_policy_v1.24",
        "source_status":dict(source_status or {}),
        "by_hypothesis":by_hypothesis,
        "research_positive_hypotheses":sorted(positive),
        "prequential_positive_watches":sorted(watches),
        "prequential_negative_hypotheses":sorted(negative),
        "global":_deriv_liq_prequential_stats(rows),
        "raw_resolved_samples":len(raw_rows),
        "effective_nonoverlap_samples":len(rows),
        "overlap_discounted_samples":max(0,len(raw_rows)-len(rows)),
        "prequential_policy_uses_nonoverlap_effective_samples":True,
        "minimum_samples":MIN_DERIV_LIQ_POLICY_SAMPLES,
        "minimum_holdout_samples":MIN_DERIV_LIQ_HOLDOUT_SAMPLES,
        "minimum_holdout_net_return":MIN_DERIV_LIQ_HOLDOUT_NET,
        "hypotheses_are_pre_registered":True,
        "same_venue_lineage_is_penalized":True,
        "price_direction_is_not_an_input":True,
        "flow_direction_is_not_an_input":True,
        "execution_authority":False,"can_enable_live":False,"can_increase_risk":False,"auto_promotion_allowed":False,
        "promotion_requires_verified_core_release":True,
    }


def apply_derivatives_liquidation_policy(rows, policy):
    negative=set(policy.get("prequential_negative_hypotheses") or []) if isinstance(policy,dict) else set()
    decisions=[]
    for row in rows or []:
        if not isinstance(row,dict): continue
        meta=row.get("metadata") if isinstance(row.get("metadata"),dict) else {}
        hid=str(meta.get("hypothesis_id") or "unknown")
        before=clamp(finite(row.get("confidence")),0.0,1.0)
        after=before
        if hid in negative:
            after=min(before,before*DERIV_LIQ_NEGATIVE_CONFIDENCE_MULTIPLIER)
            meta["deriv_liq_policy_reason"]="prequential_negative_recovery_probe"
            meta["recovery_probe"]=True
        else:
            meta["deriv_liq_policy_reason"]="unproven_or_positive_shadow_only"
        row["confidence"]=after
        meta["deriv_liq_policy_can_only_preserve_or_reduce_confidence"]=True
        meta["execution_authority"]=False; meta["risk_increase_allowed"]=False; meta["council_eligible"]=False; meta["diversity_eligible"]=False
        row["metadata"]=meta
        decisions.append({"symbol":row.get("symbol"),"hypothesis_id":hid,"confidence_before":before,"confidence_after":after,"decision":meta["deriv_liq_policy_reason"]})
    return {"model":"derivatives_liquidation_shadow_gate_v1.21","counts":{"considered":len(decisions),"prequential_negative_recovery_probe":sum(1 for d in decisions if d["decision"]=="prequential_negative_recovery_probe")},"decisions":decisions,"gate_can_only_preserve_or_reduce_confidence":True,"execution_authority":False,"can_increase_risk":False}


def derivatives_liquidation_policy_observation(policy, gate, now):
    return {"symbol":"GLOBAL","kind":"context","score":0.0,"confidence":0.98,"source":"LeanTrader derivatives-liquidation prequential governance","provenance":f"leantrader:deriv-liq-policy:v1.21:{int(now)}","observed_at":now,"horizon_seconds":max(900,INTERVAL*4),"metadata":{"model":"derivatives_liquidation_prequential_policy_v1.24","policy":policy,"gate":gate,"directional_claim":False,"execution_authority":False,"can_increase_risk":False,"can_enable_live":False}}


def _binance_derivatives_carry_snapshot(symbol):
    """Public Binance USD-M carry/OI snapshot used only as external research evidence."""
    sym = bybit_symbol(symbol)
    try:
        premium = request_json_any(
            f"{BINANCE_FUTURES_PUBLIC}/fapi/v1/premiumIndex?{urllib.parse.urlencode({'symbol': sym})}",
            timeout=8,
        )
        if not isinstance(premium, dict) or premium.get("code") not in (None, 0, "0"):
            return None, "symbol_unavailable"
        funding = finite(premium.get("lastFundingRate"))
        mark = finite(premium.get("markPrice"))
        index = finite(premium.get("indexPrice"))
        basis = (mark / index - 1.0) if mark > 0 and index > 0 else 0.0
        hist = request_json_any(
            f"{BINANCE_FUTURES_PUBLIC}/futures/data/openInterestHist?{urllib.parse.urlencode({'symbol': sym, 'period': '5m', 'limit': 3})}",
            timeout=8,
        )
        rows = [r for r in (hist if isinstance(hist, list) else []) if isinstance(r, dict)]
        rows.sort(key=lambda r: finite(r.get("timestamp")))
        latest_oi = finite((rows[-1] if rows else {}).get("sumOpenInterestValue") or (rows[-1] if rows else {}).get("sumOpenInterest"))
        oldest_oi = finite((rows[0] if rows else {}).get("sumOpenInterestValue") or (rows[0] if rows else {}).get("sumOpenInterest"))
        oi_change = (latest_oi / oldest_oi - 1.0) if latest_oi > 0 and oldest_oi > 0 and len(rows) >= 2 else 0.0
        ts_ms = max([finite(r.get("timestamp")) for r in rows] + [finite(premium.get("time"))])
        source_ts = ts_ms / 1000.0 if ts_ms > 10_000_000_000 else finite(time.time())
        return {
            "venue": "binance",
            "symbol": normalize_symbol(symbol),
            "funding_rate": funding,
            "perpetual_basis": basis,
            "open_interest_change_window": oi_change,
            "mark_price": mark,
            "index_price": index,
            "source_timestamp": source_ts,
            "confidence": 0.90 if mark > 0 and index > 0 and rows else 0.72,
        }, "available"
    except urllib.error.HTTPError as exc:
        code = int(getattr(exc, "code", 0) or 0)
        if code in {400, 404}:
            return None, "symbol_unavailable"
        if code in {403, 451}:
            return None, f"regional_or_access_restriction_{code}"
        return None, f"http_{code or 'error'}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"




def _okx_derivatives_carry_snapshot(symbol, state=None, now=None):
    """Public OKX USDT-SWAP carry/OI snapshot for third-venue research triangulation.

    Funding, mark/index basis and open interest are research-only. The current OI is compared
    with the previous sidecar observation when available; it never acquires execution authority.
    """
    state = state if isinstance(state, dict) else {}
    now = finite(now, time.time())
    norm = normalize_symbol(symbol)
    if not norm or '/' not in norm:
        return None, 'symbol_unavailable'
    base, quote = norm.split('/', 1)
    if quote != 'USDT':
        return None, 'symbol_unavailable'
    inst = f"{base}-USDT-SWAP"
    index_inst = f"{base}-USDT"
    try:
        funding_payload = request_json_any(
            f"{OKX_PUBLIC}/api/v5/public/funding-rate?{urllib.parse.urlencode({'instId': inst})}", timeout=8
        )
        frows = funding_payload.get('data') if isinstance(funding_payload, dict) else None
        frow = next((r for r in (frows or []) if isinstance(r, dict)), None)
        if not frow:
            return None, 'symbol_unavailable'
        funding = finite(frow.get('fundingRate'))

        mark_payload = request_json_any(
            f"{OKX_PUBLIC}/api/v5/public/mark-price?{urllib.parse.urlencode({'instType': 'SWAP', 'instId': inst})}", timeout=8
        )
        mrows = mark_payload.get('data') if isinstance(mark_payload, dict) else None
        mrow = next((r for r in (mrows or []) if isinstance(r, dict)), {})
        mark = finite(mrow.get('markPx'))

        index_payload = request_json_any(
            f"{OKX_PUBLIC}/api/v5/market/index-tickers?{urllib.parse.urlencode({'instId': index_inst})}", timeout=8
        )
        irows = index_payload.get('data') if isinstance(index_payload, dict) else None
        irow = next((r for r in (irows or []) if isinstance(r, dict)), {})
        index = finite(irow.get('idxPx'))
        basis = (mark / index - 1.0) if mark > 0 and index > 0 else 0.0

        oi_payload = request_json_any(
            f"{OKX_PUBLIC}/api/v5/public/open-interest?{urllib.parse.urlencode({'instType': 'SWAP', 'instId': inst})}", timeout=8
        )
        orows = oi_payload.get('data') if isinstance(oi_payload, dict) else None
        orow = next((r for r in (orows or []) if isinstance(r, dict)), {})
        current_oi = finite(orow.get('oiUsd') or orow.get('oi'))
        oi_change = 0.0
        hist = state.get('cross_venue_derivatives_okx_oi_history') if isinstance(state.get('cross_venue_derivatives_okx_oi_history'), dict) else {}
        prev = hist.get(norm) if isinstance(hist.get(norm), dict) else {}
        prev_oi = finite(prev.get('open_interest'))
        prev_ts = finite(prev.get('observed_at'))
        if current_oi > 0 and prev_oi > 0 and 60 <= now - prev_ts <= CROSS_DERIV_OKX_OI_CACHE_MAX_AGE:
            oi_change = current_oi / prev_oi - 1.0
        if current_oi > 0:
            hist[norm] = {'open_interest': current_oi, 'observed_at': now}
            # bounded persistent research cache
            if len(hist) > CROSS_DERIV_QUERY_HISTORY_MAX:
                keep = sorted(hist.items(), key=lambda kv: finite((kv[1] or {}).get('observed_at')), reverse=True)[:CROSS_DERIV_QUERY_HISTORY_MAX]
                hist = dict(keep)
            state['cross_venue_derivatives_okx_oi_history'] = hist

        ts_ms = max(finite(frow.get('ts')), finite(mrow.get('ts')), finite(irow.get('ts')), finite(orow.get('ts')))
        source_ts = ts_ms / 1000.0 if ts_ms > 10_000_000_000 else now
        confidence = 0.90 if mark > 0 and index > 0 and current_oi > 0 else 0.78 if funding != 0 else 0.68
        return {
            'venue': 'okx', 'symbol': norm, 'funding_rate': funding,
            'perpetual_basis': basis, 'open_interest_change_window': oi_change,
            'open_interest': current_oi, 'mark_price': mark, 'index_price': index,
            'source_timestamp': source_ts, 'confidence': confidence,
        }, 'available'
    except urllib.error.HTTPError as exc:
        code = int(getattr(exc, 'code', 0) or 0)
        if code in {400, 404}:
            return None, 'symbol_unavailable'
        if code in {403, 451}:
            return None, f'regional_or_access_restriction_{code}'
        return None, f'http_{code or "error"}'
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'


def _select_cross_deriv_candidates(candidates, state, now, budget=None):
    """Use fixed exploitation + persistent least-recently-queried exploration slots.

    This expands evidence coverage without lowering any alpha gate. Query history is research-only.
    """
    state = state if isinstance(state, dict) else {}
    now = finite(now, time.time())
    budget = max(1, min(int(budget or CROSS_DERIV_MAX_SYMBOLS_PER_CYCLE), len(candidates) or 1))
    hist = state.get('cross_venue_derivatives_query_history') if isinstance(state.get('cross_venue_derivatives_query_history'), dict) else {}
    ranked = list(candidates)
    exploit_n = min(CROSS_DERIV_EXPLOIT_SLOTS, budget, len(ranked))
    exploit = ranked[:exploit_n]
    used = {row[1] for row in exploit}
    remaining = [row for row in ranked if row[1] not in used]
    explore_n = min(max(0, budget - len(exploit)), CROSS_DERIV_EXPLORATION_SLOTS, len(remaining))
    remaining.sort(key=lambda row: (
        finite((hist.get(row[1]) or {}).get('last_queried_at')) if isinstance(hist.get(row[1]), dict) else 0.0,
        int((hist.get(row[1]) or {}).get('queries') or 0) if isinstance(hist.get(row[1]), dict) else 0,
        -finite(row[0]),
        row[1],
    ))
    explore = remaining[:explore_n]
    selected = exploit + explore
    if len(selected) < min(budget, len(ranked)):
        selected_symbols = {row[1] for row in selected}
        selected.extend([row for row in ranked if row[1] not in selected_symbols][:min(budget, len(ranked))-len(selected)])
    never_before = 0
    for row in selected:
        symbol = row[1]
        old = hist.get(symbol) if isinstance(hist.get(symbol), dict) else {}
        if not old:
            never_before += 1
        hist[symbol] = {'last_queried_at': now, 'queries': int(old.get('queries') or 0) + 1}
    if len(hist) > CROSS_DERIV_QUERY_HISTORY_MAX:
        hist = dict(sorted(hist.items(), key=lambda kv: finite((kv[1] or {}).get('last_queried_at')), reverse=True)[:CROSS_DERIV_QUERY_HISTORY_MAX])
    state['cross_venue_derivatives_query_history'] = hist
    return selected, {
        'candidate_count': len(ranked),
        'budget': min(budget, len(ranked)),
        'exploitation_selected': len(exploit),
        'exploration_selected': len(explore),
        'never_queried_selected': never_before,
        'unique_symbols_queried_lifetime': len(hist),
        'selected_symbols': [row[1] for row in selected],
        'scheduler_does_not_relax_alpha_thresholds': True,
        'execution_authority': False,
    }


def _carry_vote(snapshot):
    """Return +1/-1/0 for meaningful carry direction from funding/basis."""
    if not isinstance(snapshot, dict):
        return 0.0
    votes=[]
    funding=finite(snapshot.get('funding_rate'))
    basis=finite(snapshot.get('perpetual_basis'))
    if abs(funding) >= CROSS_DERIV_MIN_FUNDING_ABS:
        votes.append(1.0 if funding > 0 else -1.0)
    if abs(basis) >= CROSS_DERIV_MIN_BASIS_ABS:
        votes.append(1.0 if basis > 0 else -1.0)
    if not votes:
        return 0.0
    sm=sum(votes)
    return 1.0 if sm > 0 else -1.0 if sm < 0 else 0.0

def _cross_deriv_prequential_stats(rows):
    rows = sorted([r for r in rows if isinstance(r, dict)], key=lambda r: finite(r.get("closed_at") or r.get("opened_at")))
    n = len(rows)
    holdout_n = max(MIN_CROSS_DERIV_HOLDOUT_SAMPLES, int(math.ceil(n * 0.30))) if n else 0
    if n >= MIN_CROSS_DERIV_POLICY_SAMPLES:
        holdout_n = min(holdout_n, max(MIN_CROSS_DERIV_HOLDOUT_SAMPLES, n - 5))
    else:
        holdout_n = min(holdout_n, n)
    split = max(0, n - holdout_n)
    tr, ho = rows[:split], rows[split:]
    return {
        "samples": n,
        "train_samples": len(tr),
        "holdout_samples": len(ho),
        "train": _horizon_metric_stats(tr),
        "holdout": _horizon_metric_stats(ho),
        "all": _horizon_metric_stats(rows),
    }


def cross_venue_derivatives_observations(now, raw_by_symbol=None, state=None):
    """Shadow-test pre-registered cross-venue carry crowding reversion with venue triangulation.

    Direction is derived from native Bybit positioning/carry plus independent Binance/OKX carry.
    Price/order-book direction is never an input. A persistent exploitation/exploration scheduler
    broadens candidate coverage while keeping every alpha threshold unchanged.
    """
    raw_by_symbol = raw_by_symbol if isinstance(raw_by_symbol, dict) else {}
    state = state if isinstance(state, dict) else {}
    latest, sensor_symbols = _orthogonal_sensor_snapshot()
    snapshot_ts = _sensor_block_timestamp(latest, now) if isinstance(latest, dict) else finite(now)
    if snapshot_ts <= 0:
        snapshot_ts = finite(now)
    cache = state.get("orthogonal_sensor_cache") if isinstance(state.get("orthogonal_sensor_cache"), dict) else {}
    if not cache:
        cache = _update_orthogonal_sensor_cache(state, sensor_symbols, snapshot_ts, now)
    universe = sorted({normalize_symbol(k) for k in list((sensor_symbols or {}).keys()) + list((cache or {}).keys()) if normalize_symbol(k)})[:ORTHOGONAL_RESEARCH_UNIVERSE_LIMIT]
    status = {
        "sensor_universe_considered": len(universe),
        "native_derivatives_quality_candidates": 0,
        "external_queries_budget": CROSS_DERIV_MAX_SYMBOLS_PER_CYCLE,
        "external_queries_attempted": 0,
        "external_available": 0,
        "external_unavailable": 0,
        "external_status_counts": {},
        "external_venue_queries": {
            "binance": {"attempted": 0, "available": 0, "unavailable": 0, "status_counts": {}},
            "okx": {"attempted": 0, "available": 0, "unavailable": 0, "status_counts": {}},
        },
        "candidate_selection": {},
        "triangulation": {"three_venue_consensus": 0, "two_venue_confirmation": 0, "external_venue_conflict": 0, "external_neutral_or_missing": 0},
        "hypothesis_gate_rejections": {
            "carry_crowding_reversion_triangulated": {
                "positioning_skew_below_threshold": 0,
                "bybit_carry_not_aligned_with_positioning": 0,
                "external_consensus_not_aligned": 0,
                "external_venue_conflict": 0,
                "bybit_oi_collapse": 0,
                "binance_oi_collapse": 0,
                "okx_oi_collapse": 0,
                "external_source_unavailable": 0,
            }
        },
        "hypothesis_counts": {},
        "price_anchor_sources": {"raw": 0, "cache": 0, "public_fetch": 0, "fetch_errors": 0, "missing": 0},
        "price_direction_is_not_an_input": True,
        "orderbook_direction_is_not_an_input": True,
        "liquidation_evidence_is_not_required": True,
        "venue_independent_external_confirmation_required": True,
        "third_venue_triangulation_enabled": True,
        "candidate_scheduler_preserves_thresholds": True,
        "execution_authority": False,
    }
    candidates = []
    for symbol in universe:
        bundle = sensor_symbols.get(symbol) if isinstance(sensor_symbols.get(symbol), dict) else next((v for k, v in (sensor_symbols or {}).items() if normalize_symbol(k) == symbol and isinstance(v, dict)), {})
        cache_row = cache.get(symbol) if isinstance(cache.get(symbol), dict) else {}
        _db, d_status, d_conf, d, d_ts, d_cached, _ = _resolve_orthogonal_family(bundle, cache_row, "derivatives", snapshot_ts, now)
        if d_status != "available" or d_conf < CROSS_DERIV_MIN_DERIV_CONFIDENCE:
            continue
        status["native_derivatives_quality_candidates"] += 1
        skew = clamp(finite(d.get("positioning_skew")))
        funding = finite(d.get("funding_rate"))
        basis = finite(d.get("perpetual_basis"))
        oi = finite(d.get("open_interest_change_15m_window"))
        priority = abs(skew) + min(1.0, abs(funding) / max(CROSS_DERIV_MIN_FUNDING_ABS, 1e-9)) * 0.10 + min(1.0, abs(basis) / max(CROSS_DERIV_MIN_BASIS_ABS, 1e-9)) * 0.10
        candidates.append((priority, symbol, d_conf, d, d_ts, d_cached, skew, funding, basis, oi))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    selected, selection_status = _select_cross_deriv_candidates(candidates, state, now, CROSS_DERIV_MAX_SYMBOLS_PER_CYCLE)
    status["candidate_selection"] = selection_status
    rows = []
    rej = status["hypothesis_gate_rejections"]["carry_crowding_reversion_triangulated"]
    for _priority, symbol, d_conf, d, d_ts, d_cached, skew, funding, basis, oi in selected:
        status["external_queries_attempted"] += 1
        ext_rows = {}
        for venue, fn in (("binance", _binance_derivatives_carry_snapshot), ("okx", lambda sym: _okx_derivatives_carry_snapshot(sym, state=state, now=now))):
            vstat = status["external_venue_queries"][venue]
            vstat["attempted"] += 1
            ext, ext_status = fn(symbol)
            vstat["status_counts"][ext_status] = int(vstat["status_counts"].get(ext_status, 0)) + 1
            status["external_status_counts"][f"{venue}:{ext_status}"] = int(status["external_status_counts"].get(f"{venue}:{ext_status}", 0)) + 1
            if ext:
                vstat["available"] += 1
                ext_rows[venue] = ext
            else:
                vstat["unavailable"] += 1
        if ext_rows:
            status["external_available"] += 1
        else:
            status["external_unavailable"] += 1
            rej["external_source_unavailable"] += 1
            continue

        failed = False
        def fail(reason):
            nonlocal failed
            rej[reason] = int(rej.get(reason, 0)) + 1
            failed = True

        if abs(skew) < CROSS_DERIV_MIN_POSITIONING_SKEW:
            fail("positioning_skew_below_threshold")
        crowd_sign = 1.0 if skew > 0 else -1.0 if skew < 0 else 0.0
        bybit_vote = _carry_vote({"funding_rate": funding, "perpetual_basis": basis})
        if crowd_sign == 0.0 or bybit_vote != crowd_sign:
            fail("bybit_carry_not_aligned_with_positioning")

        ext_votes = {venue: _carry_vote(row) for venue, row in ext_rows.items()}
        aligned = [venue for venue, vote in ext_votes.items() if vote == crowd_sign and crowd_sign != 0.0]
        opposed = [venue for venue, vote in ext_votes.items() if vote == -crowd_sign and crowd_sign != 0.0]
        if aligned and opposed:
            fail("external_venue_conflict")
            status["triangulation"]["external_venue_conflict"] += 1
        elif not aligned:
            fail("external_consensus_not_aligned")
            status["triangulation"]["external_neutral_or_missing"] += 1

        if oi <= -CROSS_DERIV_MAX_OI_COLLAPSE:
            fail("bybit_oi_collapse")
        binance_oi = finite((ext_rows.get("binance") or {}).get("open_interest_change_window"))
        okx_oi = finite((ext_rows.get("okx") or {}).get("open_interest_change_window"))
        if "binance" in ext_rows and binance_oi <= -CROSS_DERIV_MAX_OI_COLLAPSE:
            fail("binance_oi_collapse")
        if "okx" in ext_rows and okx_oi <= -CROSS_DERIV_MAX_OI_COLLAPSE:
            fail("okx_oi_collapse")
        if failed:
            continue

        three_venue = set(aligned) >= {"binance", "okx"}
        hypothesis_id = "carry_crowding_reversion_3venue_consensus" if three_venue else "carry_crowding_reversion_2venue_confirmation"
        if three_venue:
            status["triangulation"]["three_venue_consensus"] += 1
            signal_cap = CROSS_DERIV_MAX_SIGNAL_CONFIDENCE
        else:
            status["triangulation"]["two_venue_confirmation"] += 1
            signal_cap = CROSS_DERIV_TWO_VENUE_CONFIDENCE_CAP

        price_anchor = finite((raw_by_symbol.get(symbol) or {}).get("mid_price"))
        anchor_source = "raw_microstructure" if price_anchor > 0 else ""
        anchor_error = None
        if price_anchor <= 0:
            price_anchor, anchor_source, anchor_error = _orthogonal_price_anchor(symbol, raw_by_symbol, state, now)
        if anchor_source == "raw_microstructure": status["price_anchor_sources"]["raw"] += 1
        elif anchor_source == "price_anchor_cache": status["price_anchor_sources"]["cache"] += 1
        elif anchor_source == "public_ticker" and price_anchor > 0: status["price_anchor_sources"]["public_fetch"] += 1
        if anchor_error: status["price_anchor_sources"]["fetch_errors"] += 1
        if price_anchor <= 0:
            status["price_anchor_sources"]["missing"] += 1
            continue

        skew_strength = clamp(abs(skew) / 0.30, 0.0, 1.0)
        bybit_carry_strength = clamp(max(abs(funding) / 0.0005, abs(basis) / 0.003), 0.0, 1.0)
        ext_strengths=[]; ext_conf=[]
        for venue in aligned:
            row=ext_rows[venue]
            ext_strengths.append(clamp(max(abs(finite(row.get("funding_rate"))) / 0.0005, abs(finite(row.get("perpetual_basis"))) / 0.003), 0.0, 1.0))
            ext_conf.append(finite(row.get("confidence"),0.70))
        external_carry_strength = sum(ext_strengths)/len(ext_strengths) if ext_strengths else 0.0
        oi_support_values=[clamp(max(0.0, oi) / 0.05, 0.0, 1.0)]
        if "binance" in ext_rows: oi_support_values.append(clamp(max(0.0, binance_oi)/0.05,0.0,1.0))
        if "okx" in ext_rows and okx_oi != 0.0: oi_support_values.append(clamp(max(0.0, okx_oi)/0.05,0.0,1.0))
        oi_support=sum(oi_support_values)/len(oi_support_values)
        raw_strength = clamp(0.40 * skew_strength + 0.24 * bybit_carry_strength + 0.26 * external_carry_strength + 0.10 * oi_support, 0.0, 1.0)
        source_conf = min([d_conf] + ext_conf) if ext_conf else d_conf
        confidence = min(signal_cap, (0.14 + 0.22 * raw_strength) * source_conf)
        direction = -crowd_sign
        status["hypothesis_counts"][hypothesis_id] = int(status["hypothesis_counts"].get(hypothesis_id, 0)) + 1
        metadata = {
            "model": "cross_venue_derivatives_carry_triangulation_v1.24",
            "hypothesis_id": hypothesis_id,
            "source_families": ["bybit_derivatives_positioning"] + [f"{v}_derivatives_carry" for v in sorted(aligned)],
            "venue_count": 3 if three_venue else 2,
            "venue_independent_external_confirmation": True,
            "third_venue_consensus": bool(three_venue),
            "external_votes": ext_votes,
            "positioning_skew": skew,
            "bybit_funding_rate": funding,
            "bybit_perpetual_basis": basis,
            "bybit_open_interest_change": oi,
            "used_cached_bybit_derivatives": bool(d_cached),
            "bybit_source_timestamp": d_ts,
            "mid_price": price_anchor,
            "price_anchor_source": anchor_source,
            "price_anchor_used_only_for_shadow_outcome_accounting": True,
            "direction_not_derived_from_price_or_orderbook": True,
            "direction_not_derived_from_liquidations": True,
            "pre_registered_hypothesis": True,
            "council_eligible": False,
            "diversity_eligible": False,
            "shadow_only": True,
            "execution_authority": False,
            "risk_increase_allowed": False,
            "can_enable_live": False,
        }
        for venue, ext in ext_rows.items():
            metadata[f"{venue}_funding_rate"] = finite(ext.get("funding_rate"))
            metadata[f"{venue}_perpetual_basis"] = finite(ext.get("perpetual_basis"))
            metadata[f"{venue}_open_interest_change"] = finite(ext.get("open_interest_change_window"))
            metadata[f"{venue}_source_timestamp"] = finite(ext.get("source_timestamp"))
        rows.append({
            "symbol": symbol, "kind": "signal", "score": direction * max(0.05, raw_strength),
            "confidence": clamp(confidence, 0.0, signal_cap),
            "source": "LeanTrader Bybit positioning + Binance/OKX public derivatives carry triangulation research",
            "provenance": f"cross-venue-derivatives:v1.24:{hypothesis_id}:{symbol}:{int(now)}",
            "observed_at": now, "horizon_seconds": CROSS_DERIV_HORIZON_SECONDS, "metadata": metadata,
        })
    return rows, status

def _cross_deriv_global_quarantine(global_stats):
    """Fail-closed global brake for a challenger with mature, independently negative evidence.

    This does not declare the inverse direction profitable and never changes alpha thresholds.
    It only reduces future shadow confidence when both chronological train and holdout evidence
    are negative after costs.  A hypothesis can escape only after independently satisfying the
    stronger research-positive lifecycle on its own evidence.
    """
    st = global_stats if isinstance(global_stats, dict) else {}
    tr = st.get("train") if isinstance(st.get("train"), dict) else {}
    ho = st.get("holdout") if isinstance(st.get("holdout"), dict) else {}
    allm = st.get("all") if isinstance(st.get("all"), dict) else {}
    n = int(st.get("samples") or 0)
    tn = int(st.get("train_samples") or 0)
    hn = int(st.get("holdout_samples") or 0)
    mature = n >= MIN_CROSS_DERIV_GLOBAL_QUARANTINE_SAMPLES and tn >= 5 and hn >= MIN_CROSS_DERIV_HOLDOUT_SAMPLES
    negative = bool(
        mature
        and finite(tr.get("average_net_return")) < 0
        and finite(tr.get("robust_net_edge_score")) < 0
        and finite(ho.get("average_net_return")) < 0
        and finite(ho.get("ewma_net_return")) < 0
        and finite(ho.get("robust_net_edge_score")) < 0
        and finite(allm.get("average_net_return")) < 0
        and finite(allm.get("robust_net_edge_score")) < 0
    )
    return {
        "active": negative,
        "lifecycle": "global_prequential_negative_quarantine" if negative else ("mature_not_globally_negative" if mature else "insufficient_global_evidence"),
        "samples": n,
        "train_samples": tn,
        "holdout_samples": hn,
        "minimum_samples": MIN_CROSS_DERIV_GLOBAL_QUARANTINE_SAMPLES,
        "confidence_multiplier": CROSS_DERIV_GLOBAL_NEGATIVE_CONFIDENCE_MULTIPLIER if negative else 1.0,
        "requires_negative_train_and_holdout": True,
        "does_not_validate_inverse_direction": True,
        "does_not_relax_alpha_thresholds": True,
        "execution_authority": False,
        "can_increase_risk": False,
    }



def _cross_deriv_quarantine_release_qualified(global_stats):
    st = global_stats if isinstance(global_stats, dict) else {}
    tr = st.get("train") if isinstance(st.get("train"), dict) else {}
    ho = st.get("holdout") if isinstance(st.get("holdout"), dict) else {}
    allm = st.get("all") if isinstance(st.get("all"), dict) else {}
    return bool(
        int(st.get("samples") or 0) >= 40
        and int(st.get("train_samples") or 0) >= 10
        and int(st.get("holdout_samples") or 0) >= MIN_CROSS_DERIV_HOLDOUT_SAMPLES
        and finite(tr.get("average_net_return")) > 0
        and finite(tr.get("robust_net_edge_score")) > 0
        and finite(ho.get("average_net_return")) > MIN_CROSS_DERIV_HOLDOUT_NET
        and finite(ho.get("ewma_net_return")) > 0
        and finite(ho.get("robust_net_edge_score")) > 0
        and finite(ho.get("win_rate")) >= 0.45
        and finite(ho.get("cost_coverage_ratio")) >= 1.10
        and finite(allm.get("average_net_return")) > 0
        and finite(allm.get("robust_net_edge_score")) > 0
    )


def _cross_deriv_quarantine_with_hysteresis(state, raw_quarantine, global_stats, now):
    """Latch a mature negative quarantine until strong recovery persists across cycles.

    v1.27 telemetry showed the raw prequential classification could briefly oscillate between
    globally negative and mature-not-negative as the chronological holdout moved. A safety brake
    should not disappear because of one boundary crossing. Once activated, the latch can release
    only after a strong globally positive recovery condition persists for several sidecar cycles.
    """
    state = state if isinstance(state, dict) else {}
    raw = dict(raw_quarantine or {})
    prev = state.get("cross_venue_derivatives_global_quarantine_latch") if isinstance(state.get("cross_venue_derivatives_global_quarantine_latch"), dict) else {}
    was_active = bool(prev.get("active"))
    raw_active = bool(raw.get("active"))
    qualified = _cross_deriv_quarantine_release_qualified(global_stats)
    streak = int(prev.get("recovery_streak") or 0)
    activated_at = finite(prev.get("activated_at"))
    released_at = finite(prev.get("released_at"))

    if raw_active:
        active = True
        streak = 0
        if activated_at <= 0:
            activated_at = finite(now)
        lifecycle = "global_prequential_negative_quarantine"
    elif was_active:
        streak = streak + 1 if qualified else 0
        if qualified and streak >= CROSS_DERIV_QUARANTINE_RELEASE_STREAK:
            active = False
            lifecycle = "global_quarantine_released_after_persistent_positive_recovery"
            released_at = finite(now)
        else:
            active = True
            lifecycle = "global_prequential_negative_quarantine_recovery_watch" if qualified else "global_prequential_negative_quarantine_latched"
    else:
        active = False
        streak = 0
        lifecycle = str(raw.get("lifecycle") or "insufficient_global_evidence")

    latch = {
        **raw,
        "active": active,
        "lifecycle": lifecycle,
        "raw_quarantine_active": raw_active,
        "latched_from_prior_negative_evidence": bool(active and not raw_active),
        "recovery_release_qualified_this_cycle": qualified,
        "recovery_streak": streak,
        "required_recovery_streak": CROSS_DERIV_QUARANTINE_RELEASE_STREAK,
        "activated_at": activated_at,
        "released_at": released_at,
        "release_requires_strong_global_positive_train_and_holdout": True,
        "does_not_validate_inverse_direction": True,
        "does_not_relax_alpha_thresholds": True,
        "execution_authority": False,
        "can_increase_risk": False,
    }
    state["cross_venue_derivatives_global_quarantine_latch"] = latch
    return latch

def cross_venue_derivatives_policy(state, source_status=None, now=None):
    _ = finite(now, time.time())
    resolved = state.get("specialist_resolved") if isinstance(state.get("specialist_resolved"), list) else []
    raw_rows = [r for r in resolved if isinstance(r, dict) and r.get("pack_id") == PACK_CROSS_DERIV]
    rows = nonoverlapping_resolved_rows(raw_rows)
    hypotheses = sorted({str(r.get("hypothesis_id") or "unknown") for r in rows})
    by_hypothesis = {}; positive = []; watches = []; negative = []
    for hid in hypotheses:
        st = _cross_deriv_prequential_stats([r for r in rows if str(r.get("hypothesis_id") or "unknown") == hid])
        n = int(st.get("samples") or 0); hn = int(st.get("holdout_samples") or 0); tn = int(st.get("train_samples") or 0)
        tr = st.get("train") or {}; ho = st.get("holdout") or {}
        if n < MIN_CROSS_DERIV_POLICY_SAMPLES or hn < MIN_CROSS_DERIV_HOLDOUT_SAMPLES or tn < 5:
            lifecycle = "insufficient_prequential_evidence"
        elif (finite(tr.get("average_net_return")) > 0 and finite(tr.get("robust_net_edge_score")) > 0 and finite(ho.get("average_net_return")) > MIN_CROSS_DERIV_HOLDOUT_NET and finite(ho.get("ewma_net_return")) > 0 and finite(ho.get("robust_net_edge_score")) > 0 and finite(ho.get("win_rate")) >= 0.45 and finite(ho.get("cost_coverage_ratio")) >= 1.10):
            if n >= 40:
                lifecycle = "research_positive"; positive.append(hid)
            else:
                lifecycle = "prequential_positive_watch"; watches.append(hid)
        elif finite(ho.get("average_net_return")) < 0 and finite(ho.get("robust_net_edge_score")) < 0:
            lifecycle = "prequential_negative"; negative.append(hid)
        else:
            lifecycle = "watch"
        by_hypothesis[hid] = {**st, "lifecycle": lifecycle, "execution_authority": False, "can_increase_risk": False, "auto_promotion_allowed": False}
    global_stats = _cross_deriv_prequential_stats(rows)
    raw_global_quarantine = _cross_deriv_global_quarantine(global_stats)
    global_quarantine = _cross_deriv_quarantine_with_hysteresis(state, raw_global_quarantine, global_stats, finite(now, time.time()))
    return {
        "model": "cross_venue_derivatives_carry_prequential_policy_v1.28",
        "source_status": dict(source_status or {}),
        "by_hypothesis": by_hypothesis,
        "research_positive_hypotheses": sorted(positive),
        "prequential_positive_watches": sorted(watches),
        "prequential_negative_hypotheses": sorted(negative),
        "global": global_stats,
        "raw_global_quarantine": raw_global_quarantine,
        "global_quarantine": global_quarantine,
        "global_negative_quarantine_active": bool(global_quarantine.get("active")),
        "raw_resolved_samples": len(raw_rows),
        "effective_nonoverlap_samples": len(rows),
        "overlap_discounted_samples": max(0, len(raw_rows) - len(rows)),
        "prequential_policy_uses_nonoverlap_effective_samples": True,
        "minimum_samples": MIN_CROSS_DERIV_POLICY_SAMPLES,
        "minimum_holdout_samples": MIN_CROSS_DERIV_HOLDOUT_SAMPLES,
        "minimum_holdout_net_return": MIN_CROSS_DERIV_HOLDOUT_NET,
        "hypotheses_are_pre_registered": True,
        "venue_independent_external_confirmation_required": True,
        "third_venue_triangulation_enabled": True,
        "persistent_candidate_exploration_scheduler": True,
        "global_negative_quarantine_is_confidence_reducing_only": True,
        "price_direction_is_not_an_input": True,
        "orderbook_direction_is_not_an_input": True,
        "execution_authority": False, "can_enable_live": False, "can_increase_risk": False, "auto_promotion_allowed": False,
        "promotion_requires_verified_core_release": True,
    }


def apply_cross_venue_derivatives_policy(rows, policy):
    negative = set(policy.get("prequential_negative_hypotheses") or []) if isinstance(policy, dict) else set()
    positive = set(policy.get("research_positive_hypotheses") or []) if isinstance(policy, dict) else set()
    global_quarantine = bool((policy.get("global_quarantine") or {}).get("active")) if isinstance(policy, dict) else False
    decisions = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        hid = str(meta.get("hypothesis_id") or "unknown")
        before = clamp(finite(row.get("confidence")), 0.0, 1.0)
        after = before
        if hid in negative:
            multiplier = CROSS_DERIV_NEGATIVE_CONFIDENCE_MULTIPLIER
            if global_quarantine:
                multiplier = min(multiplier, CROSS_DERIV_GLOBAL_NEGATIVE_CONFIDENCE_MULTIPLIER)
            after = min(before, before * multiplier)
            meta["cross_deriv_policy_reason"] = "prequential_negative_recovery_probe"
            meta["recovery_probe"] = True
        elif global_quarantine and hid not in positive:
            after = min(before, before * CROSS_DERIV_GLOBAL_NEGATIVE_CONFIDENCE_MULTIPLIER)
            meta["cross_deriv_policy_reason"] = "global_prequential_negative_recovery_probe"
            meta["recovery_probe"] = True
        else:
            meta["cross_deriv_policy_reason"] = "unproven_or_positive_shadow_only"
        row["confidence"] = after
        meta["cross_deriv_global_negative_quarantine_active"] = global_quarantine
        meta["cross_deriv_policy_can_only_preserve_or_reduce_confidence"] = True
        meta["execution_authority"] = False; meta["risk_increase_allowed"] = False; meta["council_eligible"] = False; meta["diversity_eligible"] = False
        row["metadata"] = meta
        decisions.append({"symbol": row.get("symbol"), "hypothesis_id": hid, "confidence_before": before, "confidence_after": after, "decision": meta["cross_deriv_policy_reason"]})
    return {
        "model": "cross_venue_derivatives_carry_shadow_gate_v1.26",
        "counts": {
            "considered": len(decisions),
            "prequential_negative_recovery_probe": sum(1 for d in decisions if d["decision"] == "prequential_negative_recovery_probe"),
            "global_prequential_negative_recovery_probe": sum(1 for d in decisions if d["decision"] == "global_prequential_negative_recovery_probe"),
        },
        "global_negative_quarantine_active": global_quarantine,
        "gate_can_only_preserve_or_reduce_confidence": True,
        "execution_authority": False,
        "can_increase_risk": False,
    }


def cross_venue_derivatives_policy_observation(policy, gate, now):
    return {"symbol": "GLOBAL", "kind": "context", "score": 0.0, "confidence": 0.98, "source": "LeanTrader tri-venue derivatives carry prequential governance", "provenance": f"leantrader:cross-deriv-policy:v1.26:{int(now)}", "observed_at": now, "horizon_seconds": max(900, INTERVAL * 4), "metadata": {"model": "cross_venue_derivatives_carry_prequential_policy_v1.26", "policy": policy, "gate": gate, "directional_claim": False, "execution_authority": False, "can_increase_risk": False, "can_enable_live": False}}


def orthogonal_sensor_coverage_observation(source_status, now):
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.98,
        "source": "LeanTrader orthogonal sensor temporal coverage reconciler",
        "provenance": f"leantrader:orthogonal-sensor-coverage:v1.20:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(900, INTERVAL * 3),
        "metadata": {
            "model": "orthogonal_sensor_universe_quality_reconciliation_v1.20",
            "source_status": source_status,
            "directional_claim": False,
            "cache_can_increase_confidence": False,
            "candidate_specific_family_pairs": True,
            "research_universe_decoupled_from_strategy_selection": True,
            "quality_rejection_attribution": True,
            "single_component_flow_requires_independent_cross_family_quorum": True,
            "single_component_common_factor_is_context_only": True,
            "single_component_cannot_claim_internal_diversity": True,
            "price_anchor_does_not_influence_direction": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "can_enable_live": False,
        },
    }


def _orthogonal_prequential_stats(rows):
    rows = sorted([r for r in (rows or []) if isinstance(r, dict)], key=lambda r: (finite(r.get("opened_at")), str(r.get("episode_id") or "")))
    n = len(rows)
    holdout_n = max(MIN_ORTHOGONAL_HOLDOUT_SAMPLES, int(math.ceil(n * 0.30))) if n else 0
    if n >= MIN_ORTHOGONAL_POLICY_SAMPLES:
        holdout_n = min(holdout_n, max(MIN_ORTHOGONAL_HOLDOUT_SAMPLES, n - 4))
    else:
        holdout_n = min(holdout_n, n)
    split = max(0, n - holdout_n)
    train = rows[:split]
    holdout = rows[split:]
    return {
        "samples": n,
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "train": _horizon_metric_stats(train),
        "holdout": _horizon_metric_stats(holdout),
        "all": _horizon_metric_stats(rows),
    }


def orthogonal_sensor_policy(state, source_status=None, now=None):
    """Prequentially score the new source family without granting authority."""
    _ = finite(now, time.time())
    resolved = state.get("specialist_resolved") if isinstance(state.get("specialist_resolved"), list) else []
    raw_rows = [r for r in resolved if isinstance(r, dict) and r.get("pack_id") == PACK_ORTHOGONAL_FLOW]
    rows = nonoverlapping_resolved_rows(raw_rows)
    hypotheses = sorted({str(r.get("hypothesis_id") or "unknown") for r in rows})
    by_hypothesis = {}
    research_positive = []
    positive_watches = []
    negative = []
    for hid in hypotheses:
        stats = _orthogonal_prequential_stats([r for r in rows if str(r.get("hypothesis_id") or "unknown") == hid])
        n = int(stats.get("samples") or 0)
        tr = stats.get("train") or {}
        ho = stats.get("holdout") or {}
        hn = int(stats.get("holdout_samples") or 0)
        if n < MIN_ORTHOGONAL_POLICY_SAMPLES or hn < MIN_ORTHOGONAL_HOLDOUT_SAMPLES or int(stats.get("train_samples") or 0) < 4:
            lifecycle = "insufficient_prequential_evidence"
        elif (
            finite(tr.get("average_net_return")) > 0.0
            and finite(tr.get("robust_net_edge_score")) > 0.0
            and finite(ho.get("average_net_return")) > MIN_ORTHOGONAL_HOLDOUT_NET
            and finite(ho.get("ewma_net_return")) > 0.0
            and finite(ho.get("robust_net_edge_score")) > 0.0
            and finite(ho.get("win_rate")) >= 0.40
            and finite(ho.get("cost_coverage_ratio")) >= 1.10
        ):
            if n >= 30:
                lifecycle = "research_positive"
                research_positive.append(hid)
            else:
                lifecycle = "prequential_positive_watch"
                positive_watches.append(hid)
        elif finite(ho.get("average_net_return")) < 0.0 and finite(ho.get("robust_net_edge_score")) < 0.0:
            lifecycle = "prequential_negative"
            negative.append(hid)
        else:
            lifecycle = "watch"
        by_hypothesis[hid] = {
            **stats,
            "lifecycle": lifecycle,
            "execution_authority": False,
            "can_increase_risk": False,
            "auto_promotion_allowed": False,
        }
    global_stats = _orthogonal_prequential_stats(rows)
    return {
        "model": "orthogonal_sensor_prequential_policy_v1.24",
        "source_status": dict(source_status or {}),
        "by_hypothesis": by_hypothesis,
        "research_positive_hypotheses": sorted(research_positive),
        "prequential_positive_watches": sorted(positive_watches),
        "prequential_negative_hypotheses": sorted(negative),
        "global": global_stats,
        "raw_resolved_samples": len(raw_rows),
        "effective_nonoverlap_samples": len(rows),
        "overlap_discounted_samples": max(0, len(raw_rows) - len(rows)),
        "prequential_policy_uses_nonoverlap_effective_samples": True,
        "minimum_samples": MIN_ORTHOGONAL_POLICY_SAMPLES,
        "minimum_holdout_samples": MIN_ORTHOGONAL_HOLDOUT_SAMPLES,
        "minimum_holdout_net_return": MIN_ORTHOGONAL_HOLDOUT_NET,
        "hypotheses_are_pre_registered": True,
        "price_direction_is_not_an_input": True,
        "execution_authority": False,
        "can_enable_live": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
        "promotion_requires_verified_core_release": True,
    }


def apply_orthogonal_sensor_policy(rows, policy):
    """A negative hypothesis remains a low-confidence recovery probe.

    The policy can never increase confidence and never changes execution authority.
    """
    negative = set(policy.get("prequential_negative_hypotheses") or []) if isinstance(policy, dict) else set()
    decisions = []
    for row in rows or []:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        hid = str(meta.get("hypothesis_id") or "unknown")
        before = finite(row.get("confidence"))
        decision = "preserve_shadow"
        if hid in negative:
            row["confidence"] = clamp(before * ORTHOGONAL_NEGATIVE_CONFIDENCE_MULTIPLIER, 0.0, before)
            meta["recovery_probe"] = True
            meta["orthogonal_policy_reason"] = "prequential_negative_recovery_probe"
            decision = "prequential_negative_recovery_probe"
        else:
            row["confidence"] = clamp(before, 0.0, before)
            meta["orthogonal_policy_reason"] = "unproven_or_positive_shadow_only"
        meta["council_eligible"] = False
        meta["diversity_eligible"] = False
        meta["execution_authority"] = False
        meta["risk_increase_allowed"] = False
        meta["orthogonal_policy_can_only_preserve_or_reduce_confidence"] = True
        row["metadata"] = meta
        decisions.append({
            "symbol": normalize_symbol(row.get("symbol") or ""),
            "hypothesis_id": hid,
            "decision": decision,
            "confidence_before": before,
            "confidence_after": finite(row.get("confidence")),
        })
    return {
        "model": "orthogonal_sensor_shadow_gate_v1.20",
        "decisions": decisions,
        "counts": {
            "considered": len(decisions),
            "prequential_negative_recovery_probe": sum(1 for d in decisions if d["decision"] == "prequential_negative_recovery_probe"),
        },
        "execution_authority": False,
        "can_increase_risk": False,
        "gate_can_only_preserve_or_reduce_confidence": True,
    }


def orthogonal_sensor_policy_observation(policy, gate, now):
    return {
        "symbol": "GLOBAL",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.98,
        "source": "LeanTrader native sensor fabric + sidecar-owned costed shadow ledger",
        "provenance": f"leantrader:orthogonal-sensor-policy:v1.24:{int(now)}",
        "observed_at": now,
        "horizon_seconds": max(1800, INTERVAL * 4),
        "metadata": {
            "model": "orthogonal_sensor_prequential_policy_v1.24",
            "directional_claim": False,
            "policy": policy,
            "gate": gate,
            "execution_authority": False,
            "can_increase_risk": False,
        },
    }

def _ema(values, period):
    vals = [finite(v) for v in values if math.isfinite(finite(v))]
    if not vals:
        return 0.0
    alpha = 2.0 / (max(1, int(period)) + 1.0)
    out = vals[0]
    for v in vals[1:]:
        out = alpha * v + (1.0 - alpha) * out
    return out


def _rsi(values, period=14):
    vals = [finite(v) for v in values]
    if len(vals) < period + 1:
        return 50.0
    gains, losses = [], []
    for a, b in zip(vals[-period-1:-1], vals[-period:]):
        d = b - a
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    ag = sum(gains) / max(1, len(gains))
    al = sum(losses) / max(1, len(losses))
    if al <= 1e-12:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(bars, period=14):
    if len(bars) < 2:
        return 0.0
    trs = []
    for prev, cur in zip(bars[-period-1:-1], bars[-period:]):
        h, l, pc = finite(cur.get("high")), finite(cur.get("low")), finite(prev.get("close"))
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / max(1, len(trs))


def _mtf_closed_klines_from_payload(payload, interval_seconds, now):
    result = payload.get("result") if isinstance(payload, dict) else {}
    rows = (result or {}).get("list") or []
    out = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 7:
            continue
        try:
            start = int(row[0]) / 1000.0
            close_deadline = start + float(interval_seconds)
            if close_deadline > now + 1e-6:
                continue
            op, hi, lo, cl, vol, turnover = map(float, row[1:7])
        except Exception:
            continue
        if min(op, hi, lo, cl) <= 0:
            continue
        out.append({
            "start": start,
            "open": op,
            "high": hi,
            "low": lo,
            "close": cl,
            "volume": max(0.0, vol),
            "turnover": max(0.0, turnover),
            "source_timestamp": close_deadline,
        })
    out.sort(key=lambda r: r["start"])
    return out


def _bybit_linear_closed_klines(symbol, interval_code, interval_seconds, now, limit=None):
    limit = int(limit or MTF_KLINE_LIMIT)
    q = urllib.parse.urlencode({
        "category": "linear",
        "symbol": bybit_symbol(symbol),
        "interval": interval_code,
        "limit": limit,
    })
    payload = request_json(f"{BYBIT}/v5/market/kline?{q}")
    if int(payload.get("retCode", -1)) != 0:
        return [], str(payload.get("retMsg") or "bybit_error")
    rows = _mtf_closed_klines_from_payload(payload, interval_seconds, now)
    if len(rows) < MTF_MIN_CLOSED_BARS:
        return rows, "insufficient_closed_bars"
    return rows, "available"


def _bybit_linear_closed_klines_window(symbol, interval_code, interval_seconds, start_ts, end_ts, now):
    """Fetch a bounded historical closed-candle window for overdue v1.32 bundles.

    This path is intentionally separate from the rotating current-symbol scan. It lets an
    already-open research bundle finish from the first historical closed candle at/after
    each ex-ante target even when its symbol is no longer among the four current symbols.
    """
    start_ts = max(0.0, finite(start_ts))
    end_ts = max(start_ts + float(interval_seconds), finite(end_ts))
    span = max(float(interval_seconds), end_ts - start_ts)
    limit = max(4, min(1000, int(math.ceil(span / max(float(interval_seconds), 1.0))) + 6))
    q = urllib.parse.urlencode({
        "category": "linear",
        "symbol": bybit_symbol(symbol),
        "interval": interval_code,
        "start": int(start_ts * 1000.0),
        "end": int(end_ts * 1000.0),
        "limit": limit,
    })
    payload = request_json(f"{BYBIT}/v5/market/kline?{q}")
    if int(payload.get("retCode", -1)) != 0:
        return [], str(payload.get("retMsg") or "bybit_error")
    rows = _mtf_closed_klines_from_payload(payload, interval_seconds, now)
    return rows, ("available" if rows else "empty_window")


def _bybit_linear_spread(symbol):
    q = urllib.parse.urlencode({"category": "linear", "symbol": bybit_symbol(symbol), "limit": 1})
    payload = request_json(f"{BYBIT}/v5/market/orderbook?{q}")
    if int(payload.get("retCode", -1)) != 0:
        return None
    book = payload.get("result") or {}
    bids, asks = book.get("b") or [], book.get("a") or []
    if not bids or not asks:
        return None
    try:
        bid, ask = float(bids[0][0]), float(asks[0][0])
    except Exception:
        return None
    if bid <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return {
        "spread_bps": (ask - bid) / max(mid, 1e-12) * 10_000.0,
        "mid_price": mid,
        "source": "bybit_linear_orderbook",
    }


def _mtf_timeframe_features(bars):
    if len(bars) < 24:
        return {"available": False, "score": 0.0, "confidence": 0.0}
    closes = [finite(r.get("close")) for r in bars]
    close = closes[-1]
    atr = _atr(bars, 14)
    atr_pct = atr / max(close, 1e-12)
    ema9 = _ema(closes[-40:], 9)
    ema21 = _ema(closes[-50:], 21)
    trend_atr = (ema9 - ema21) / max(atr, close * 1e-6)
    ret3 = close / max(closes[-4], 1e-12) - 1.0 if len(closes) >= 4 else 0.0
    momentum = ret3 / max(atr_pct * math.sqrt(3.0), 1e-6)
    rsi = _rsi(closes, 14)
    rsi_score = (rsi - 50.0) / 50.0
    look = closes[-20:]
    lo20, hi20 = min(look), max(look)
    breakout = ((close - (hi20 + lo20) / 2.0) / max((hi20 - lo20) / 2.0, close * 1e-6)) if hi20 > lo20 else 0.0
    score = clamp(
        0.42 * math.tanh(trend_atr / 1.5)
        + 0.28 * math.tanh(momentum / 2.0)
        + 0.16 * clamp(rsi_score)
        + 0.14 * clamp(breakout)
    )
    recent_ranges = [(finite(r.get("high")) - finite(r.get("low"))) / max(finite(r.get("close")), 1e-12) for r in bars[-12:]]
    older_ranges = [(finite(r.get("high")) - finite(r.get("low"))) / max(finite(r.get("close")), 1e-12) for r in bars[-36:-12]]
    recent_vol = sum(recent_ranges) / max(1, len(recent_ranges))
    older_vol = sum(older_ranges) / max(1, len(older_ranges))
    vol_ratio = recent_vol / max(older_vol, 1e-8)
    if vol_ratio >= 1.45:
        volatility_regime = "expansion"
    elif vol_ratio <= 0.72:
        volatility_regime = "compression"
    else:
        volatility_regime = "normal"
    trend_strength = min(1.0, abs(trend_atr) / 2.0)
    component_agreement = sum(1 for v in (trend_atr, momentum, rsi_score, breakout) if v * score > 0) / 4.0 if abs(score) > 1e-12 else 0.0
    confidence = clamp(0.35 + 0.30 * trend_strength + 0.25 * component_agreement + 0.10 * min(1.0, len(bars) / 64.0), 0.0, 1.0)
    return {
        "available": True,
        "score": score,
        "confidence": confidence,
        "close": close,
        "atr": atr,
        "atr_pct": atr_pct,
        "ema9": ema9,
        "ema21": ema21,
        "rsi14": rsi,
        "return_3bars": ret3,
        "volatility_ratio": vol_ratio,
        "volatility_regime": volatility_regime,
        "source_timestamp": finite(bars[-1].get("source_timestamp")),
        "closed_bars": len(bars),
        # Internal-only recent closed points let v1.32 resolve fixed horizons at the
        # first closed candle at/after each target even if a symbol rotates out for
        # several sidecar cycles. They are stripped from public per-timeframe status.
        "recent_closed_points": [
            [finite(r.get("source_timestamp")), finite(r.get("close"))]
            for r in bars[-64:]
            if finite(r.get("source_timestamp")) > 0 and finite(r.get("close")) > 0
        ],
    }


def _weighted_score(features, names, weights):
    total = 0.0
    den = 0.0
    conf = 0.0
    for name, w in zip(names, weights):
        row = features.get(name) or {}
        if not row.get("available"):
            continue
        ww = float(w)
        total += finite(row.get("score")) * ww
        conf += finite(row.get("confidence")) * ww
        den += ww
    if den <= 0:
        return 0.0, 0.0
    return clamp(total / den), clamp(conf / den, 0.0, 1.0)


def _mtf_aggregate_symbol(symbol, features, micro=None):
    micro = micro if isinstance(micro, dict) else {}
    entry_score, entry_conf = _weighted_score(features, ("1m", "3m", "5m"), (0.25, 0.35, 0.40))
    structure_score, structure_conf = _weighted_score(features, ("15m", "30m"), (0.58, 0.42))
    regime_score, regime_conf = _weighted_score(features, ("1h", "4h", "1d"), (0.50, 0.32, 0.18))
    complete = sum(1 for tf in MTF_TIMEFRAMES if (features.get(tf) or {}).get("available"))
    spread_bps = finite(micro.get("spread_bps"), 999.0)
    spread_gate = spread_bps <= MTF_MAX_SPREAD_BPS

    def direction(v, threshold):
        if v >= threshold:
            return 1
        if v <= -threshold:
            return -1
        return 0

    ed = direction(entry_score, MTF_ENTRY_MIN_SCORE)
    sd = direction(structure_score, MTF_STRUCTURE_MIN_SCORE)
    regime_opposes = ed != 0 and regime_score * ed < -MTF_REGIME_OPPOSITION_LIMIT
    aligned = ed != 0 and sd == ed and not regime_opposes
    full_coverage = complete == len(MTF_TIMEFRAMES)
    min_layer_conf = min(entry_conf, structure_conf, regime_conf) if full_coverage else 0.0
    readiness = clamp(
        (0.45 * abs(entry_score) + 0.35 * abs(structure_score) + 0.20 * max(0.0, regime_score * ed if ed else 0.0))
        * min_layer_conf
    )
    shadow_ready = bool(full_coverage and spread_gate and aligned and min_layer_conf >= 0.45)

    one = features.get("1m") or {}
    atr_pct_1m = finite(one.get("atr_pct"))
    if atr_pct_1m >= 0.0030:
        preferred_tf = "1m"
        holding = 300
    elif atr_pct_1m >= 0.0015:
        preferred_tf = "3m"
        holding = 600
    else:
        preferred_tf = "5m"
        holding = 900

    if not full_coverage:
        mode = "incomplete_no_trade"
    elif not spread_gate:
        mode = "spread_blocked"
    elif ed and sd and ed != sd:
        mode = "timeframe_conflict_no_trade"
    elif ed and regime_opposes:
        mode = "higher_timeframe_opposition_wait"
    elif shadow_ready:
        mode = "scalp_aligned_long" if ed > 0 else "scalp_aligned_short"
    elif sd and ed == -sd:
        mode = "pullback_wait"
    else:
        mode = "neutral_no_trade"

    return {
        "symbol": normalize_symbol(symbol),
        "mode": mode,
        "direction": "long" if shadow_ready and ed > 0 else ("short" if shadow_ready and ed < 0 else "none"),
        "shadow_ready": shadow_ready,
        "readiness": readiness,
        "entry_score": entry_score,
        "structure_score": structure_score,
        "regime_score": regime_score,
        "entry_confidence": entry_conf,
        "structure_confidence": structure_conf,
        "regime_confidence": regime_conf,
        "timeframes_available": complete,
        "timeframes_required": len(MTF_TIMEFRAMES),
        "preferred_execution_timeframe": preferred_tf,
        "preferred_holding_seconds": holding,
        "spread_bps": spread_bps,
        "spread_gate_pass": spread_gate,
        "entry_price": finite((features.get(preferred_tf) or {}).get("close")),
        "entry_source_timestamp": finite((features.get(preferred_tf) or {}).get("source_timestamp")),
        "closed_points_by_execution_timeframe": {
            tf: list((features.get(tf) or {}).get("recent_closed_points") or [])
            for tf in ("1m", "3m", "5m")
        },
        "risk_ceiling_multiplier": min(1.0, readiness),
        "execution_authority": False,
        "diagnostic_only": True,
        "does_not_create_alpha": True,
        "does_not_count_as_independent_council_vote": True,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }


def _mtf_outcome_validation(resolved):
    rows = sorted([dict(r) for r in (resolved or []) if isinstance(r, dict)], key=lambda r: (finite(r.get("opened_at")), str(r.get("episode_id") or "")))
    n = len(rows)
    holdout_n = max(MTF_OUTCOME_MIN_HOLDOUT_SAMPLES, int(math.ceil(n * 0.30))) if n else 0
    holdout_n = min(holdout_n, n)
    split = max(0, n - holdout_n)
    train = rows[:split]
    holdout = rows[split:]
    train_stats = _horizon_metric_stats(train)
    holdout_stats = _horizon_metric_stats(holdout)
    all_stats = _horizon_metric_stats(rows)
    mature = n >= MTF_OUTCOME_MIN_VALIDATION_SAMPLES and len(holdout) >= MTF_OUTCOME_MIN_HOLDOUT_SAMPLES
    positive = bool(
        mature
        and finite(train_stats.get("average_net_return")) > 0.0
        and finite(train_stats.get("robust_net_edge_score")) > 0.0
        and finite(holdout_stats.get("average_net_return")) > 0.0
        and finite(holdout_stats.get("robust_net_edge_score")) > 0.0
        and finite(holdout_stats.get("cost_coverage_ratio")) > 1.0
    )
    negative = bool(
        mature
        and finite(train_stats.get("average_net_return")) < 0.0
        and finite(holdout_stats.get("average_net_return")) < 0.0
    )
    lifecycle = "research_positive_holdout" if positive else ("prequential_negative" if negative else ("mature_mixed" if mature else "collecting"))
    return {
        "samples": n,
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "minimum_samples": MTF_OUTCOME_MIN_VALIDATION_SAMPLES,
        "minimum_holdout_samples": MTF_OUTCOME_MIN_HOLDOUT_SAMPLES,
        "mature": mature,
        "lifecycle": lifecycle,
        "research_positive": positive,
        "research_negative": negative,
        "train": train_stats,
        "holdout": holdout_stats,
        "all": all_stats,
        "testnet_research_gate_satisfied": positive,
        "testnet_auto_enable_allowed": False,
        "execution_authority": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }


def _update_mtf_scalping_outcome_lab(state, symbol_rows, now):
    """Prospectively score shadow-ready multi-timeframe scalp candidates.

    Each symbol can have at most one pending episode. Entry comes from the latest fully
    closed candle on the dynamically selected 1m/3m/5m execution timeframe. Resolution
    occurs only after that timeframe's closed source timestamp reaches the ex-ante holding
    horizon. Costs are fail-closed at the larger of the global round-trip assumption or the
    observed entry+exit spread. The lab is research-only and cannot trade or vote.
    """
    state = state if isinstance(state, dict) else {}
    pending0 = state.get("mtf_scalping_outcome_pending")
    pending = {str(k): dict(v) for k, v in pending0.items() if isinstance(v, dict)} if isinstance(pending0, dict) else {}
    resolved0 = state.get("mtf_scalping_outcome_resolved")
    resolved = [dict(r) for r in resolved0 if isinstance(r, dict)][-MTF_OUTCOME_MAX_RESOLVED:] if isinstance(resolved0, list) else []
    last0 = state.get("mtf_scalping_outcome_last_opened")
    last_opened = {str(k): finite(v) for k, v in last0.items()} if isinstance(last0, dict) else {}
    rows_by_symbol = {normalize_symbol(r.get("symbol") or ""): r for r in (symbol_rows or []) if isinstance(r, dict) and normalize_symbol(r.get("symbol") or "")}
    opened = resolved_now = overlap_suppressed = stale_suppressed = 0

    # Resolve before opening new episodes, preserving non-overlap per symbol.
    for symbol, episode in list(pending.items()):
        row = rows_by_symbol.get(symbol)
        if not isinstance(row, dict):
            continue
        tf = str(episode.get("preferred_execution_timeframe") or "5m")
        feat = (row.get("timeframes") or {}).get(tf) if isinstance(row.get("timeframes"), dict) else None
        if not isinstance(feat, dict) or not feat.get("available"):
            continue
        source_ts = finite(feat.get("source_timestamp"))
        if source_ts + 1e-9 < finite(episode.get("target_source_timestamp")):
            continue
        exit_price = finite(feat.get("close"))
        entry_price = finite(episode.get("entry_price"))
        if source_ts <= 0 or exit_price <= 0 or entry_price <= 0:
            continue
        direction = 1.0 if str(episode.get("direction")) == "long" else -1.0
        gross = direction * (exit_price / entry_price - 1.0)
        entry_spread = max(0.0, finite(episode.get("entry_spread_bps")))
        exit_spread = max(0.0, finite(row.get("spread_bps")))
        cost_bps = max(ROUND_TRIP_COST_BPS, entry_spread + exit_spread)
        net = gross - cost_bps / 10_000.0
        resolved.append({
            **episode,
            "resolved_at": finite(now),
            "exit_source_timestamp": source_ts,
            "exit_price": exit_price,
            "exit_spread_bps": exit_spread,
            "actual_horizon_seconds": max(0.0, source_ts - finite(episode.get("entry_source_timestamp"))),
            "gross_return": gross,
            "net_return": net,
            "inverse_net_return": -gross - cost_bps / 10_000.0,
            "edge_vs_inverse": 2.0 * gross,
            "round_trip_cost_bps": cost_bps,
            "execution_authority": False,
            "diagnostic_only": True,
        })
        del pending[symbol]
        resolved_now += 1

    resolved = resolved[-MTF_OUTCOME_MAX_RESOLVED:]

    # v1.32 fail-closed behavior: once the legacy dynamic-horizon mapping has mature
    # negative train+holdout evidence, resolve already-open episodes but do not keep
    # manufacturing more samples from the falsified configuration. The separate
    # multi-horizon lab below becomes the recovery/falsification path.
    validation_before_open = _mtf_outcome_validation(resolved)
    legacy_negative_quarantine = bool(validation_before_open.get("research_negative"))
    negative_quarantine_suppressed = 0

    for row in sorted((symbol_rows or []), key=lambda r: finite((r or {}).get("readiness")), reverse=True):
        if not isinstance(row, dict) or not row.get("shadow_ready"):
            continue
        if legacy_negative_quarantine:
            negative_quarantine_suppressed += 1
            continue
        symbol = normalize_symbol(row.get("symbol") or "")
        direction = str(row.get("direction") or "none")
        tf = str(row.get("preferred_execution_timeframe") or "5m")
        holding = int(row.get("preferred_holding_seconds") or 0)
        source_ts = finite(row.get("entry_source_timestamp"))
        entry_price = finite(row.get("entry_price"))
        if not symbol or direction not in ("long", "short") or tf not in ("1m", "3m", "5m") or holding <= 0 or source_ts <= 0 or entry_price <= 0:
            stale_suppressed += 1
            continue
        if symbol in pending:
            overlap_suppressed += 1
            continue
        if source_ts <= finite(last_opened.get(symbol)) + 1e-9:
            overlap_suppressed += 1
            continue
        episode_id = hashlib.sha256(f"mtf-scalp|{symbol}|{source_ts:.3f}|{direction}|{tf}|{holding}".encode("utf-8")).hexdigest()[:24]
        pending[symbol] = {
            "episode_id": episode_id,
            "symbol": symbol,
            "direction": direction,
            "opened_at": finite(now),
            "entry_source_timestamp": source_ts,
            "target_source_timestamp": source_ts + holding,
            "entry_price": entry_price,
            "entry_spread_bps": max(0.0, finite(row.get("spread_bps"))),
            "readiness": finite(row.get("readiness")),
            "entry_score": finite(row.get("entry_score")),
            "structure_score": finite(row.get("structure_score")),
            "regime_score": finite(row.get("regime_score")),
            "preferred_execution_timeframe": tf,
            "horizon_seconds": holding,
            "source_family": "bybit_linear_multi_timeframe_scalping_orchestrator",
            "source_method": "closed_candle_8tf_alignment",
            "diagnostic_only": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "auto_promotion_allowed": False,
        }
        last_opened[symbol] = source_ts
        opened += 1
        if len(pending) >= MTF_OUTCOME_MAX_PENDING:
            break

    validation = _mtf_outcome_validation(resolved)
    lab = {
        "model": "multi_timeframe_scalping_forward_outcome_lab_v1.32",
        "opened_this_cycle": opened,
        "resolved_this_cycle": resolved_now,
        "overlap_suppressed_this_cycle": overlap_suppressed,
        "stale_or_invalid_suppressed_this_cycle": stale_suppressed,
        "legacy_negative_quarantine_active": legacy_negative_quarantine,
        "legacy_negative_quarantine_suppressed_this_cycle": negative_quarantine_suppressed,
        "quarantine_resolves_existing_pending_but_blocks_new_legacy_episodes": True,
        "quarantine_does_not_validate_inverse_direction": True,
        "pending_count": len(pending),
        "pending_symbols": sorted(pending)[:16],
        "resolved_count": len(resolved),
        "metrics": _horizon_metric_stats(resolved),
        "validation": validation,
        "dynamic_horizons_from_execution_timeframe": True,
        "non_overlapping_per_symbol": True,
        "uses_closed_execution_timeframe_candles": True,
        "cost_floor_bps": ROUND_TRIP_COST_BPS,
        "does_not_create_alpha": True,
        "does_not_count_as_independent_council_vote": True,
        "execution_authority": False,
        "can_increase_risk": False,
        "can_enable_live": False,
        "can_enable_testnet": False,
        "auto_promotion_allowed": False,
    }
    state["mtf_scalping_outcome_pending"] = pending
    state["mtf_scalping_outcome_resolved"] = resolved
    state["mtf_scalping_outcome_last_opened"] = last_opened
    state["mtf_scalping_outcome_lab"] = lab
    return lab



def _mtf_horizon_validation(rows):
    """Chronological per-horizon diagnostics; never an execution/testnet gate."""
    rows = sorted(
        [dict(r) for r in (rows or []) if isinstance(r, dict)],
        key=lambda r: (finite(r.get("opened_at")), str(r.get("bundle_id") or "")),
    )
    n = len(rows)
    holdout_n = max(MTF_HORIZON_MIN_HOLDOUT_SAMPLES, int(math.ceil(n * 0.30))) if n else 0
    holdout_n = min(holdout_n, n)
    split = max(0, n - holdout_n)
    train = rows[:split]
    holdout = rows[split:]
    train_stats = _horizon_metric_stats(train)
    holdout_stats = _horizon_metric_stats(holdout)
    all_stats = _horizon_metric_stats(rows)
    mature = n >= MTF_HORIZON_MIN_VALIDATION_SAMPLES and len(holdout) >= MTF_HORIZON_MIN_HOLDOUT_SAMPLES
    positive_watch = bool(
        mature
        and finite(train_stats.get("average_net_return")) > 0.0
        and finite(train_stats.get("robust_net_edge_score")) > 0.0
        and finite(holdout_stats.get("average_net_return")) > 0.0
        and finite(holdout_stats.get("robust_net_edge_score")) > 0.0
        and finite(holdout_stats.get("cost_coverage_ratio")) > 1.0
    )
    negative = bool(
        mature
        and finite(train_stats.get("average_net_return")) < 0.0
        and finite(holdout_stats.get("average_net_return")) < 0.0
    )
    lifecycle = "prospective_positive_watch" if positive_watch else ("prequential_negative" if negative else ("mature_mixed" if mature else "collecting"))
    return {
        "samples": n,
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "minimum_samples": MTF_HORIZON_MIN_VALIDATION_SAMPLES,
        "minimum_holdout_samples": MTF_HORIZON_MIN_HOLDOUT_SAMPLES,
        "mature": mature,
        "lifecycle": lifecycle,
        "prospective_positive_watch": positive_watch,
        "research_negative": negative,
        "train": train_stats,
        "holdout": holdout_stats,
        "all": all_stats,
        # A horizon watch is diagnostic evidence only. It cannot bypass the manual
        # testnet decision or become an independent council vote.
        "testnet_research_gate_satisfied": False,
        "testnet_auto_enable_allowed": False,
        "execution_authority": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }


def _mtf_legacy_decomposition(resolved):
    """Describe v1.31 evidence without tuning thresholds from the same sample."""
    rows = [dict(r) for r in (resolved or []) if isinstance(r, dict)]
    cells = {}
    provisional = []
    for tf in ("1m", "3m", "5m"):
        for horizon in (300, 600, 900):
            subset = [
                r for r in rows
                if str(r.get("preferred_execution_timeframe") or "") == tf
                and int(finite(r.get("horizon_seconds"))) == horizon
            ]
            if not subset:
                continue
            stats = _horizon_metric_stats(subset)
            key = f"{tf}:{horizon}"
            cells[key] = {"samples": len(subset), **stats}
            # This is a descriptive watch only. Five observations are enough to report
            # the cell, never enough to grant authority or to change the live thresholds.
            if len(subset) >= 5 and finite(stats.get("average_net_return")) > 0.0:
                provisional.append({
                    "cell": key,
                    "samples": len(subset),
                    "average_net_return": finite(stats.get("average_net_return")),
                    "robust_net_edge_score": finite(stats.get("robust_net_edge_score")),
                    "diagnostic_only": True,
                })
    provisional.sort(key=lambda r: (finite(r.get("average_net_return")), int(r.get("samples") or 0)), reverse=True)
    return {
        "cells": cells,
        "provisional_positive_cells": provisional[:8],
        "provisional_cells_do_not_change_thresholds": True,
        "provisional_cells_do_not_enable_testnet": True,
        "execution_authority": False,
    }


def _update_mtf_scalping_horizon_lab(state, symbol_rows, now):
    """Prospective multi-horizon falsification lab for v1.32-r1.

    One bundle per symbol starts from a shadow-ready setup and observes the same ex-ante
    direction at 5/10/15/20/30/45/60 minutes. v1.32-r1 resolves overdue bundles through
    a bounded historical closed-candle backfill that is independent of the rotating
    four-symbol current scan. This prevents symbol rotation from leaving old bundles
    pending indefinitely or resolving old targets from a much later recent candle.
    """
    state = state if isinstance(state, dict) else {}
    pending0 = state.get("mtf_scalping_horizon_pending")
    pending = {str(k): dict(v) for k, v in pending0.items() if isinstance(v, dict)} if isinstance(pending0, dict) else {}
    resolved0 = state.get("mtf_scalping_horizon_resolved")
    resolved = [dict(r) for r in resolved0 if isinstance(r, dict)][-MTF_HORIZON_MAX_RESOLVED:] if isinstance(resolved0, list) else []
    last0 = state.get("mtf_scalping_horizon_last_opened")
    last_opened = {str(k): finite(v) for k, v in last0.items()} if isinstance(last0, dict) else {}
    legacy_pending = state.get("mtf_scalping_outcome_pending") or {}
    rows_by_symbol = {
        normalize_symbol(r.get("symbol") or ""): r
        for r in (symbol_rows or [])
        if isinstance(r, dict) and normalize_symbol(r.get("symbol") or "")
    }
    bundles_opened = horizons_resolved_now = overlap_suppressed = stale_suppressed = 0
    backfill_attempted = backfill_available = backfill_rows = overdue_bundle_count = 0
    backfill_status_counts = {}
    backfill_budget = MTF_HORIZON_BACKFILL_MAX_QUERIES

    # Oldest bundles are serviced first so a rotating current-symbol scheduler cannot
    # starve a mature pending episode forever.
    pending_order = sorted(
        list(pending.items()),
        key=lambda kv: (finite((kv[1] or {}).get("entry_source_timestamp")), str(kv[0])),
    )

    for symbol, bundle in pending_order:
        tf = str(bundle.get("preferred_execution_timeframe") or "5m")
        interval_info = MTF_TIMEFRAMES.get(tf)
        entry_price = finite(bundle.get("entry_price"))
        entry_ts = finite(bundle.get("entry_source_timestamp"))
        remaining = [int(h) for h in (bundle.get("remaining_horizons") or []) if int(h) > 0]
        if not interval_info or entry_price <= 0 or entry_ts <= 0 or not remaining:
            stale_suppressed += 1
            continue
        interval_code, interval_seconds = interval_info
        interval_seconds = float(interval_seconds)
        due = [h for h in remaining if finite(now) + 1e-9 >= entry_ts + h]
        if due:
            overdue_bundle_count += 1

        row = rows_by_symbol.get(symbol)
        feat = None
        source_ts = 0.0
        if isinstance(row, dict):
            feat = (row.get("timeframes") or {}).get(tf) if isinstance(row.get("timeframes"), dict) else None
            if isinstance(feat, dict) and feat.get("available"):
                source_ts = finite(feat.get("source_timestamp"))

        # point tuples are (closed source timestamp, close price, provenance)
        points = []
        if isinstance(row, dict):
            points0 = ((row.get("closed_points_by_execution_timeframe") or {}).get(tf) or [])
            for point in points0:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    pts, px = finite(point[0]), finite(point[1])
                    if pts > 0 and px > 0:
                        points.append((pts, px, "current_rotation_points"))
            if not points and isinstance(feat, dict):
                px = finite(feat.get("close"))
                if source_ts > 0 and px > 0:
                    points.append((source_ts, px, "current_rotation_point"))

        points.sort(key=lambda p: p[0])

        def first_eligible(target):
            eligible = [p for p in points if p[0] + 1e-9 >= target]
            return eligible[0] if eligible else None

        # A recent rolling window is unsafe for an old target if its first available
        # point is more than one execution interval after the target. Fetch the bounded
        # historical hour around the ORIGINAL entry instead of using that stale point.
        missing_precise_due = []
        for horizon in due:
            target = entry_ts + horizon
            hit = first_eligible(target)
            if hit is None or hit[0] - target > interval_seconds + 1e-6:
                missing_precise_due.append(horizon)

        backfill_used = False
        backfill_status = "not_needed"
        if missing_precise_due and backfill_attempted < backfill_budget:
            backfill_attempted += 1
            earliest = entry_ts - 2.0 * interval_seconds
            latest_target = entry_ts + max(missing_precise_due)
            latest = min(finite(now), latest_target + 2.0 * interval_seconds)
            try:
                bars, backfill_status = _bybit_linear_closed_klines_window(
                    symbol, interval_code, interval_seconds, earliest, latest, now
                )
            except Exception as exc:
                bars, backfill_status = [], f"error:{type(exc).__name__}:{exc}"
            backfill_status_counts[backfill_status] = backfill_status_counts.get(backfill_status, 0) + 1
            if bars:
                backfill_available += 1
                backfill_rows += len(bars)
                backfill_used = True
                for bar in bars:
                    pts, px = finite(bar.get("source_timestamp")), finite(bar.get("close"))
                    if pts > 0 and px > 0:
                        points.append((pts, px, "historical_pending_backfill"))
                # Deduplicate by closed timestamp while preferring historical backfill,
                # which is the precise source for old ex-ante targets.
                dedup = {}
                for pts, px, provenance in points:
                    old = dedup.get(pts)
                    if old is None or provenance == "historical_pending_backfill":
                        dedup[pts] = (pts, px, provenance)
                points = sorted(dedup.values(), key=lambda p: p[0])
        elif missing_precise_due:
            backfill_status = "budget_exhausted"
            backfill_status_counts[backfill_status] = backfill_status_counts.get(backfill_status, 0) + 1

        direction = 1.0 if str(bundle.get("direction")) == "long" else -1.0
        entry_spread = max(0.0, finite(bundle.get("entry_spread_bps")))
        current_exit_spread = max(0.0, finite((row or {}).get("spread_bps"))) if isinstance(row, dict) else 0.0
        # Historical order-book spread is unavailable from the kline endpoint. Fail closed
        # rather than silently assuming zero exit spread when the symbol rotated out.
        exit_spread_proxy = max(entry_spread, current_exit_spread)
        cost_bps = max(ROUND_TRIP_COST_BPS, entry_spread + exit_spread_proxy)

        still = []
        for horizon in sorted(set(remaining)):
            target_ts = entry_ts + horizon
            # Never resolve a horizon before it is actually due.
            if finite(now) + 1e-9 < target_ts:
                still.append(horizon)
                continue
            hit = first_eligible(target_ts)
            # The first closed candle after target must occur within one execution
            # interval. Otherwise the precise historical candle is still unavailable and
            # we keep the horizon pending rather than contaminate evidence with a late price.
            if hit is None or hit[0] - target_ts > interval_seconds + 1e-6:
                still.append(horizon)
                continue
            exit_source_ts, exit_price, resolution_source = hit
            gross = direction * (exit_price / entry_price - 1.0)
            resolved.append({
                **{k: v for k, v in bundle.items() if k != "remaining_horizons"},
                "horizon_seconds": horizon,
                "target_source_timestamp": target_ts,
                "resolved_at": finite(now),
                "exit_source_timestamp": exit_source_ts,
                "actual_horizon_seconds": max(0.0, exit_source_ts - entry_ts),
                "exit_price": exit_price,
                "exit_spread_bps": exit_spread_proxy,
                "gross_return": gross,
                "net_return": gross - cost_bps / 10_000.0,
                "inverse_net_return": -gross - cost_bps / 10_000.0,
                "edge_vs_inverse": 2.0 * gross,
                "round_trip_cost_bps": cost_bps,
                "resolution_source": resolution_source,
                "historical_backfill_used_this_cycle": bool(backfill_used),
                "historical_backfill_status": backfill_status,
                "correlated_with_same_bundle_horizons": True,
                "counts_as_independent_council_vote": False,
                "execution_authority": False,
                "diagnostic_only": True,
            })
            horizons_resolved_now += 1
        if still:
            bundle["remaining_horizons"] = still
            bundle["last_resolution_attempt_at"] = finite(now)
            bundle["last_backfill_status"] = backfill_status
            pending[symbol] = bundle
        else:
            del pending[symbol]

    resolved = resolved[-MTF_HORIZON_MAX_RESOLVED:]

    for row in sorted((symbol_rows or []), key=lambda r: finite((r or {}).get("readiness")), reverse=True):
        if not isinstance(row, dict) or not row.get("shadow_ready"):
            continue
        symbol = normalize_symbol(row.get("symbol") or "")
        direction = str(row.get("direction") or "none")
        tf = str(row.get("preferred_execution_timeframe") or "5m")
        source_ts = finite(row.get("entry_source_timestamp"))
        entry_price = finite(row.get("entry_price"))
        if not symbol or direction not in ("long", "short") or tf not in ("1m", "3m", "5m") or source_ts <= 0 or entry_price <= 0:
            stale_suppressed += 1
            continue
        if symbol in pending or symbol in legacy_pending:
            overlap_suppressed += 1
            continue
        if source_ts <= finite(last_opened.get(symbol)) + 1e-9:
            overlap_suppressed += 1
            continue
        if len(pending) >= MTF_HORIZON_MAX_PENDING:
            break
        feat = (row.get("timeframes") or {}).get(tf) if isinstance(row.get("timeframes"), dict) else {}
        bundle_id = hashlib.sha256(f"mtf-horizon|{symbol}|{source_ts:.3f}|{direction}|{tf}".encode("utf-8")).hexdigest()[:24]
        pending[symbol] = {
            "bundle_id": bundle_id,
            "symbol": symbol,
            "direction": direction,
            "opened_at": finite(now),
            "entry_source_timestamp": source_ts,
            "entry_price": entry_price,
            "entry_spread_bps": max(0.0, finite(row.get("spread_bps"))),
            "readiness": finite(row.get("readiness")),
            "entry_score": finite(row.get("entry_score")),
            "structure_score": finite(row.get("structure_score")),
            "regime_score": finite(row.get("regime_score")),
            "preferred_execution_timeframe": tf,
            "legacy_preferred_holding_seconds": int(row.get("preferred_holding_seconds") or 0),
            "execution_atr_pct": finite((feat or {}).get("atr_pct")),
            "execution_volatility_regime": str((feat or {}).get("volatility_regime") or "unknown"),
            "remaining_horizons": list(MTF_HORIZON_GRID),
            "source_family": "bybit_linear_multi_timeframe_scalping_horizon_falsification",
            "source_method": "closed_candle_same_signal_multi_horizon",
            "diagnostic_only": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "auto_promotion_allowed": False,
        }
        last_opened[symbol] = source_ts
        bundles_opened += 1

    by_horizon = {}
    positive_watches = []
    for horizon in MTF_HORIZON_GRID:
        hrows = [r for r in resolved if int(finite(r.get("horizon_seconds"))) == int(horizon)]
        val = _mtf_horizon_validation(hrows)
        by_horizon[str(int(horizon))] = val
        if val.get("prospective_positive_watch"):
            positive_watches.append({
                "horizon_seconds": int(horizon),
                "samples": int(val.get("samples") or 0),
                "average_net_return": finite((val.get("all") or {}).get("average_net_return")),
                "holdout_average_net_return": finite((val.get("holdout") or {}).get("average_net_return")),
                "diagnostic_only": True,
            })
    positive_watches.sort(key=lambda r: (finite(r.get("holdout_average_net_return")), int(r.get("samples") or 0)), reverse=True)
    legacy = _mtf_legacy_decomposition(state.get("mtf_scalping_outcome_resolved") or [])
    lab = {
        "model": "multi_timeframe_scalping_multi_horizon_falsification_lab_v1.32-r1",
        "horizon_grid_seconds": list(MTF_HORIZON_GRID),
        "bundles_opened_this_cycle": bundles_opened,
        "horizons_resolved_this_cycle": horizons_resolved_now,
        "overlap_suppressed_this_cycle": overlap_suppressed,
        "stale_or_invalid_suppressed_this_cycle": stale_suppressed,
        "pending_bundle_count": len(pending),
        "pending_symbols": sorted(pending)[:24],
        "resolved_horizon_rows": len(resolved),
        "by_horizon": by_horizon,
        "prospective_positive_watches": positive_watches[:8],
        "legacy_decomposition": legacy,
        "historical_pending_backfill_budget": backfill_budget,
        "historical_pending_backfill_queries_attempted": backfill_attempted,
        "historical_pending_backfill_queries_available": backfill_available,
        "historical_pending_backfill_rows_fetched": backfill_rows,
        "historical_pending_backfill_status_counts": backfill_status_counts,
        "overdue_bundle_count": overdue_bundle_count,
        "rotating_symbol_scan_cannot_starve_pending_bundles": True,
        "late_recent_candles_cannot_substitute_for_target_candles": True,
        "historical_exit_spread_is_fail_closed_proxy": True,
        "same_signal_horizons_are_correlated_not_independent_votes": True,
        "thresholds_unchanged_from_v1_31": True,
        "cost_floor_bps": ROUND_TRIP_COST_BPS,
        "does_not_create_alpha": True,
        "does_not_count_as_independent_council_vote": True,
        "execution_authority": False,
        "can_increase_risk": False,
        "can_enable_live": False,
        "can_enable_testnet": False,
        "auto_promotion_allowed": False,
    }
    state["mtf_scalping_horizon_pending"] = pending
    state["mtf_scalping_horizon_resolved"] = resolved
    state["mtf_scalping_horizon_last_opened"] = last_opened
    state["mtf_scalping_horizon_lab"] = lab
    return lab


def _mtf_discovery_readiness_bucket(value):
    v = finite(value)
    if v < 0.20:
        return "lt_0.20"
    if v < 0.25:
        return "0.20_0.25"
    if v < 0.30:
        return "0.25_0.30"
    if v < 0.35:
        return "0.30_0.35"
    if v < 0.40:
        return "0.35_0.40"
    if v < 0.45:
        return "0.40_0.45"
    return "ge_0.45"


def _mtf_discovery_spread_bucket(value):
    v = max(0.0, finite(value))
    if v <= 2.0:
        return "le_2bps"
    if v <= 4.0:
        return "2_4bps"
    if v <= 6.0:
        return "4_6bps"
    if v <= 10.0:
        return "6_10bps"
    return "gt_10bps"


def _mtf_discovery_regime_alignment(row):
    direction = 1.0 if str(row.get("direction") or "").lower() == "long" else -1.0
    signed = finite(row.get("regime_score")) * direction
    if signed >= 0.25:
        return "strong_aligned"
    if signed >= 0.05:
        return "supportive"
    if signed <= -0.10:
        return "opposed"
    return "neutral"


def _mtf_discovery_metrics(rows):
    rows = [dict(r) for r in (rows or []) if isinstance(r, dict)]
    rows.sort(key=lambda r: (finite(r.get("opened_at")), str(r.get("bundle_id") or "")))
    base = _horizon_metric_stats(rows)
    nets = [finite(r.get("net_return")) for r in rows]
    symbols = sorted({normalize_symbol(r.get("symbol") or "") for r in rows if normalize_symbol(r.get("symbol") or "")})
    if nets:
        ordered = sorted(nets)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        leave = list(nets)
        leave.remove(max(leave))
        leave_best = sum(leave) / len(leave) if leave else 0.0
        win_rate = sum(v > 0 for v in nets) / len(nets)
    else:
        median = leave_best = win_rate = 0.0
    return {
        **base,
        "median_net_return": median,
        "leave_best_out_average_net_return": leave_best,
        "win_rate": win_rate,
        "unique_symbols": len(symbols),
        "symbols": symbols[:16],
    }


def _mtf_discovery_validation(rows, level, cell_key):
    rows = [dict(r) for r in (rows or []) if isinstance(r, dict)]
    rows.sort(key=lambda r: (finite(r.get("opened_at")), str(r.get("bundle_id") or "")))
    n = len(rows)
    holdout_n = max(MTF_DISCOVERY_MIN_HOLDOUT_SAMPLES, int(math.ceil(n * 0.30))) if n else 0
    holdout_n = min(holdout_n, n)
    split = max(0, n - holdout_n)
    train = rows[:split]
    holdout = rows[split:]
    allm = _mtf_discovery_metrics(rows)
    trainm = _mtf_discovery_metrics(train)
    holdm = _mtf_discovery_metrics(holdout)
    mature = bool(
        n >= MTF_DISCOVERY_MIN_SAMPLES
        and len(holdout) >= MTF_DISCOVERY_MIN_HOLDOUT_SAMPLES
        and int(allm.get("unique_symbols") or 0) >= MTF_DISCOVERY_MIN_UNIQUE_SYMBOLS
    )
    outlier_driven = bool(
        finite(allm.get("average_net_return")) > 0.0
        and finite(allm.get("leave_best_out_average_net_return")) <= 0.0
    )
    early_falsified = bool(
        n >= MTF_DISCOVERY_EARLY_FALSIFICATION_SAMPLES
        and int(allm.get("unique_symbols") or 0) >= MTF_DISCOVERY_MIN_UNIQUE_SYMBOLS
        and finite(allm.get("median_net_return")) <= 0.0
        and finite(allm.get("leave_best_out_average_net_return")) < 0.0
        and finite(allm.get("win_rate")) <= 0.50
    )
    positive = bool(
        mature
        and not outlier_driven
        and finite(trainm.get("average_net_return")) > 0.0
        and finite(trainm.get("median_net_return")) > 0.0
        and finite(trainm.get("leave_best_out_average_net_return")) > 0.0
        and finite(holdm.get("average_net_return")) > 0.0
        and finite(holdm.get("median_net_return")) > 0.0
        and finite(holdm.get("leave_best_out_average_net_return")) > 0.0
        and finite(holdm.get("robust_net_edge_score")) > 0.0
        and finite(holdm.get("cost_coverage_ratio")) > 1.0
        and finite(allm.get("win_rate")) > 0.50
    )
    mature_negative = bool(
        mature
        and (
            (finite(trainm.get("average_net_return")) < 0.0 and finite(holdm.get("average_net_return")) < 0.0)
            or (
                finite(allm.get("median_net_return")) <= 0.0
                and finite(allm.get("leave_best_out_average_net_return")) < 0.0
                and finite(allm.get("win_rate")) <= 0.50
            )
        )
    )
    lifecycle = (
        "research_positive_watch" if positive
        else "mature_negative_quarantine" if mature_negative
        else "early_falsified_watch" if early_falsified
        else "outlier_driven_watch" if outlier_driven
        else "mature_mixed" if mature
        else "collecting"
    )
    return {
        "level": level,
        "cell": cell_key,
        "samples": n,
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "minimum_samples": MTF_DISCOVERY_MIN_SAMPLES,
        "minimum_holdout_samples": MTF_DISCOVERY_MIN_HOLDOUT_SAMPLES,
        "minimum_unique_symbols": MTF_DISCOVERY_MIN_UNIQUE_SYMBOLS,
        "mature": mature,
        "lifecycle": lifecycle,
        "research_positive_watch": positive,
        "research_negative": mature_negative,
        "early_falsified_watch": early_falsified,
        "outlier_driven": outlier_driven,
        "train": trainm,
        "holdout": holdm,
        "all": allm,
        "testnet_research_gate_satisfied": False,
        "testnet_auto_enable_allowed": False,
        "execution_authority": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }


def _mtf_5m_long_60m_specialist_validation(rows):
    """Fresh v1.34-only prospective validation for the discovered 5m-long/60m family.

    Discovery evidence is never imported into this ledger. A specialist can only become a
    replication-positive candidate after new chronological train and holdout observations
    satisfy median, leave-best-out, robust after-cost and multi-symbol requirements.
    Even then this function grants no execution or Testnet authority.
    """
    rows = [dict(r) for r in (rows or []) if isinstance(r, dict)]
    rows.sort(key=lambda r: (finite(r.get("opened_at")), str(r.get("specialist_episode_id") or "")))
    n = len(rows)
    holdout_n = max(MTF_5M_LONG_60M_SPECIALIST_MIN_HOLDOUT, int(math.ceil(n * 0.30))) if n else 0
    holdout_n = min(holdout_n, n)
    split = max(0, n - holdout_n)
    train = rows[:split]
    holdout = rows[split:]
    allm = _mtf_discovery_metrics(rows)
    trainm = _mtf_discovery_metrics(train)
    holdm = _mtf_discovery_metrics(holdout)
    mature = bool(
        n >= MTF_5M_LONG_60M_SPECIALIST_MIN_SAMPLES
        and len(holdout) >= MTF_5M_LONG_60M_SPECIALIST_MIN_HOLDOUT
        and int(allm.get("unique_symbols") or 0) >= MTF_5M_LONG_60M_SPECIALIST_MIN_SYMBOLS
    )
    replication_positive = bool(
        mature
        and finite(allm.get("win_rate")) > 0.50
        and finite(allm.get("median_net_return")) > 0.0
        and finite(allm.get("leave_best_out_average_net_return")) > 0.0
        and finite(allm.get("robust_net_edge_score")) > 0.0
        and finite(trainm.get("average_net_return")) > 0.0
        and finite(trainm.get("median_net_return")) > 0.0
        and finite(trainm.get("leave_best_out_average_net_return")) > 0.0
        and finite(trainm.get("robust_net_edge_score")) > 0.0
        and finite(holdm.get("average_net_return")) > 0.0
        and finite(holdm.get("median_net_return")) > 0.0
        and finite(holdm.get("leave_best_out_average_net_return")) > 0.0
        and finite(holdm.get("robust_net_edge_score")) > 0.0
        and finite(holdm.get("cost_coverage_ratio")) > 1.0
    )
    replication_negative = bool(
        mature and (
            (finite(trainm.get("average_net_return")) < 0.0 and finite(holdm.get("average_net_return")) < 0.0)
            or (
                finite(allm.get("median_net_return")) <= 0.0
                and finite(allm.get("leave_best_out_average_net_return")) < 0.0
                and finite(allm.get("win_rate")) <= 0.50
            )
        )
    )
    lifecycle = (
        "dedicated_replication_positive" if replication_positive
        else "dedicated_replication_negative" if replication_negative
        else "dedicated_mature_mixed" if mature
        else "dedicated_collecting"
    )
    return {
        "samples": n,
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "minimum_samples": MTF_5M_LONG_60M_SPECIALIST_MIN_SAMPLES,
        "minimum_holdout_samples": MTF_5M_LONG_60M_SPECIALIST_MIN_HOLDOUT,
        "minimum_unique_symbols": MTF_5M_LONG_60M_SPECIALIST_MIN_SYMBOLS,
        "mature": mature,
        "lifecycle": lifecycle,
        "replication_positive": replication_positive,
        "replication_negative": replication_negative,
        "candidate_for_controlled_testnet_review": replication_positive,
        "train": trainm,
        "holdout": holdm,
        "all": allm,
        "testnet_research_gate_satisfied": False,
        "testnet_auto_enable_allowed": False,
        "execution_authority": False,
        "can_increase_risk": False,
        "auto_promotion_allowed": False,
    }


def _update_mtf_5m_long_60m_specialist(state, symbol_rows, now):
    """Dedicated v1.34 prospective challenger for the v1.33-discovered 5m LONG / 60m family.

    Entries are fresh after v1.34 installation and must independently satisfy the unchanged
    8-timeframe shadow-ready rules with direction=LONG and preferred execution timeframe=5m.
    One open episode per symbol is allowed. Historical closed 5m candles resolve exact path
    checkpoints even after symbol rotation. MAE/MFE and giveback are descriptive risk-path
    diagnostics only and cannot change thresholds, risk, council votes or execution authority.
    """
    state = state if isinstance(state, dict) else {}
    pending0 = state.get("mtf_5m_long_60m_specialist_pending")
    pending = {str(k): dict(v) for k, v in pending0.items() if isinstance(v, dict)} if isinstance(pending0, dict) else {}
    resolved0 = state.get("mtf_5m_long_60m_specialist_resolved")
    resolved = [dict(r) for r in resolved0 if isinstance(r, dict)][-MTF_5M_LONG_60M_SPECIALIST_MAX_RESOLVED:] if isinstance(resolved0, list) else []
    last0 = state.get("mtf_5m_long_60m_specialist_last_opened")
    last_opened = {str(k): finite(v) for k, v in last0.items()} if isinstance(last0, dict) else {}
    rows_by_symbol = {
        normalize_symbol(r.get("symbol") or ""): r
        for r in (symbol_rows or [])
        if isinstance(r, dict) and normalize_symbol(r.get("symbol") or "")
    }
    opened = resolved_now = path_points_now = overlap_suppressed = invalid_suppressed = 0
    backfill_attempted = backfill_available = 0
    backfill_status_counts = {}

    # Resolve/checkpoint oldest pending episodes first, independently of current scanner rotation.
    for symbol, episode in sorted(list(pending.items()), key=lambda kv: (finite((kv[1] or {}).get("entry_source_timestamp")), str(kv[0]))):
        entry_ts = finite(episode.get("entry_source_timestamp"))
        entry_price = finite(episode.get("entry_price"))
        if entry_ts <= 0 or entry_price <= 0:
            invalid_suppressed += 1
            continue
        remaining = [int(x) for x in (episode.get("remaining_checkpoints") or []) if int(x) > 0]
        if not remaining:
            invalid_suppressed += 1
            continue
        due = [h for h in remaining if finite(now) + 1e-9 >= entry_ts + h]
        if not due:
            continue
        if backfill_attempted >= MTF_5M_LONG_60M_SPECIALIST_BACKFILL_BUDGET:
            backfill_status_counts["budget_exhausted"] = backfill_status_counts.get("budget_exhausted", 0) + 1
            continue
        backfill_attempted += 1
        interval_code, interval_seconds = MTF_TIMEFRAMES["5m"]
        latest_target = entry_ts + max(due)
        try:
            bars, fetch_status = _bybit_linear_closed_klines_window(
                symbol,
                interval_code,
                interval_seconds,
                entry_ts - 2.0 * interval_seconds,
                min(finite(now), latest_target + 2.0 * interval_seconds),
                now,
            )
        except Exception as exc:
            bars, fetch_status = [], f"error:{type(exc).__name__}:{exc}"
        backfill_status_counts[fetch_status] = backfill_status_counts.get(fetch_status, 0) + 1
        if not bars:
            continue
        backfill_available += 1
        points = sorted(
            [(finite(b.get("source_timestamp")), finite(b.get("close"))) for b in bars if finite(b.get("source_timestamp")) > 0 and finite(b.get("close")) > 0],
            key=lambda x: x[0],
        )
        path = dict(episode.get("path") or {})
        still = []
        for h in sorted(set(remaining)):
            target = entry_ts + h
            if finite(now) + 1e-9 < target:
                still.append(h)
                continue
            eligible = [p for p in points if p[0] + 1e-9 >= target]
            hit = eligible[0] if eligible else None
            if hit is None or hit[0] - target > float(interval_seconds) + 1e-6:
                still.append(h)
                continue
            source_ts, px = hit
            gross = px / entry_price - 1.0  # challenger is LONG-only by construction
            path[str(h)] = {
                "horizon_seconds": h,
                "target_source_timestamp": target,
                "exit_source_timestamp": source_ts,
                "actual_horizon_seconds": max(0.0, source_ts - entry_ts),
                "price": px,
                "gross_return": gross,
            }
            path_points_now += 1
        episode["path"] = path
        episode["remaining_checkpoints"] = still
        if MTF_5M_LONG_60M_SPECIALIST_HORIZON not in {int(k) for k in path.keys()}:
            pending[symbol] = episode
            continue

        final = path[str(MTF_5M_LONG_60M_SPECIALIST_HORIZON)]
        row = rows_by_symbol.get(symbol) or {}
        entry_spread = max(0.0, finite(episode.get("entry_spread_bps")))
        current_spread = max(0.0, finite(row.get("spread_bps"))) if isinstance(row, dict) else 0.0
        exit_spread_proxy = max(entry_spread, current_spread)
        cost_bps = max(ROUND_TRIP_COST_BPS, entry_spread + exit_spread_proxy)
        gross = finite(final.get("gross_return"))
        net = gross - cost_bps / 10000.0
        path_rows = [v for _, v in sorted(path.items(), key=lambda kv: int(kv[0])) if isinstance(v, dict)]
        path_gross = [finite(v.get("gross_return")) for v in path_rows]
        mfe = max(path_gross) if path_gross else gross
        mae = min(path_gross) if path_gross else gross
        mfe_row = max(path_rows, key=lambda v: finite(v.get("gross_return"))) if path_rows else final
        mae_row = min(path_rows, key=lambda v: finite(v.get("gross_return"))) if path_rows else final
        resolved.append({
            **{k: v for k, v in episode.items() if k != "remaining_checkpoints"},
            "resolved_at": finite(now),
            "exit_source_timestamp": finite(final.get("exit_source_timestamp")),
            "actual_horizon_seconds": finite(final.get("actual_horizon_seconds")),
            "exit_price": finite(final.get("price")),
            "exit_spread_bps_proxy": exit_spread_proxy,
            "gross_return": gross,
            "net_return": net,
            "inverse_net_return": -gross - cost_bps / 10000.0,
            "edge_vs_inverse": 2.0 * gross,
            "round_trip_cost_bps": cost_bps,
            "maximum_favorable_excursion": mfe,
            "maximum_adverse_excursion": mae,
            "time_to_mfe_seconds": int(finite(mfe_row.get("horizon_seconds"))),
            "time_to_mae_seconds": int(finite(mae_row.get("horizon_seconds"))),
            "giveback_from_mfe_to_exit": max(0.0, mfe - gross),
            "path_checkpoint_count": len(path_rows),
            "execution_authority": False,
            "diagnostic_only": True,
        })
        del pending[symbol]
        resolved_now += 1

    resolved = resolved[-MTF_5M_LONG_60M_SPECIALIST_MAX_RESOLVED:]

    # Open only FRESH 5m LONG setups after this challenger exists. No v1.33 discovery rows
    # are copied into the specialist ledger, preventing discovery/revalidation leakage.
    for row in sorted((symbol_rows or []), key=lambda r: finite((r or {}).get("readiness")), reverse=True):
        if not isinstance(row, dict) or not row.get("shadow_ready"):
            continue
        symbol = normalize_symbol(row.get("symbol") or "")
        if (
            not symbol
            or str(row.get("direction") or "") != "long"
            or str(row.get("preferred_execution_timeframe") or "") != "5m"
        ):
            continue
        source_ts = finite(row.get("entry_source_timestamp"))
        entry_price = finite(row.get("entry_price"))
        if source_ts <= 0 or entry_price <= 0:
            invalid_suppressed += 1
            continue
        if symbol in pending or source_ts <= finite(last_opened.get(symbol)) + 1e-9:
            overlap_suppressed += 1
            continue
        if len(pending) >= MTF_5M_LONG_60M_SPECIALIST_MAX_PENDING:
            break
        feat = (row.get("timeframes") or {}).get("5m") if isinstance(row.get("timeframes"), dict) else {}
        sid = hashlib.sha256(f"mtf-5m-long-60m-v134|{symbol}|{source_ts:.3f}".encode("utf-8")).hexdigest()[:24]
        pending[symbol] = {
            "specialist_episode_id": sid,
            "symbol": symbol,
            "direction": "long",
            "preferred_execution_timeframe": "5m",
            "horizon_seconds": MTF_5M_LONG_60M_SPECIALIST_HORIZON,
            "opened_at": finite(now),
            "entry_source_timestamp": source_ts,
            "entry_price": entry_price,
            "entry_spread_bps": max(0.0, finite(row.get("spread_bps"))),
            "readiness": finite(row.get("readiness")),
            "entry_score": finite(row.get("entry_score")),
            "structure_score": finite(row.get("structure_score")),
            "regime_score": finite(row.get("regime_score")),
            "execution_atr_pct": finite((feat or {}).get("atr_pct")),
            "execution_volatility_regime": str((feat or {}).get("volatility_regime") or "unknown"),
            "remaining_checkpoints": list(MTF_5M_LONG_60M_SPECIALIST_CHECKPOINTS),
            "path": {},
            "source_family": "v1.34_dedicated_5m_long_60m_prospective_challenger",
            "discovery_source_cell": "5m|long|3600s",
            "fresh_post_discovery_episode": True,
            "discovery_rows_are_not_validation_samples": True,
            "diagnostic_only": True,
            "execution_authority": False,
            "can_increase_risk": False,
            "auto_promotion_allowed": False,
        }
        last_opened[symbol] = source_ts
        opened += 1

    validation = _mtf_5m_long_60m_specialist_validation(resolved)
    risk_path = {
        "resolved_samples": len(resolved),
        "average_mfe": (sum(finite(r.get("maximum_favorable_excursion")) for r in resolved) / len(resolved)) if resolved else 0.0,
        "average_mae": (sum(finite(r.get("maximum_adverse_excursion")) for r in resolved) / len(resolved)) if resolved else 0.0,
        "average_giveback_from_mfe_to_exit": (sum(finite(r.get("giveback_from_mfe_to_exit")) for r in resolved) / len(resolved)) if resolved else 0.0,
        "full_path_samples": sum(int(r.get("path_checkpoint_count") or 0) >= len(MTF_5M_LONG_60M_SPECIALIST_CHECKPOINTS) for r in resolved),
    }
    lab = {
        "model": "dedicated_5m_long_60m_prospective_challenger_v1.34",
        "discovered_by": "automated_mtf_regime_horizon_discovery_v1.33",
        "source_cell": "5m|long|3600s",
        "fresh_post_discovery_only": True,
        "opened_this_cycle": opened,
        "resolved_this_cycle": resolved_now,
        "path_points_resolved_this_cycle": path_points_now,
        "pending_count": len(pending),
        "pending_symbols": sorted(pending)[:16],
        "resolved_count": len(resolved),
        "overlap_suppressed_this_cycle": overlap_suppressed,
        "invalid_suppressed_this_cycle": invalid_suppressed,
        "historical_backfill_queries_this_cycle": backfill_attempted,
        "historical_backfill_available_this_cycle": backfill_available,
        "historical_backfill_status_counts": backfill_status_counts,
        "checkpoint_grid_seconds": list(MTF_5M_LONG_60M_SPECIALIST_CHECKPOINTS),
        "cost_floor_bps": ROUND_TRIP_COST_BPS,
        "risk_path_metrics": risk_path,
        "validation": validation,
        "does_not_import_discovery_rows_as_validation": True,
        "does_not_relax_signal_thresholds": True,
        "does_not_validate_inverse_direction": True,
        "does_not_count_as_independent_council_vote": True,
        "execution_authority": False,
        "can_increase_risk": False,
        "can_enable_testnet": False,
        "can_enable_live": False,
        "auto_promotion_allowed": False,
    }
    state["mtf_5m_long_60m_specialist_pending"] = pending
    state["mtf_5m_long_60m_specialist_resolved"] = resolved
    state["mtf_5m_long_60m_specialist_last_opened"] = last_opened
    state["mtf_5m_long_60m_specialist_lab"] = lab
    return lab


def _mtf_automated_regime_discovery(state, now=None):
    """Discover MTF timeframe/direction/regime/horizon cells without tuning on winners.

    Every cell is defined only from information stored at bundle entry. A fixed horizon
    contributes at most one observation from each bundle to that cell. Positive watches
    require chronological train/holdout support, multiple symbols, positive median and
    leave-best-out edge, robust after-cost holdout performance, and >50% overall wins.
    Negative/outlier cells can be quarantined from nomination, never inverted into alpha.
    """
    _ = finite(now, time.time())
    rows = [dict(r) for r in (state.get("mtf_scalping_horizon_resolved") or []) if isinstance(r, dict)]
    rows.sort(key=lambda r: (finite(r.get("opened_at")), str(r.get("bundle_id") or ""), int(finite(r.get("horizon_seconds")))))
    levels = {"coarse": {}, "regime": {}, "fine": {}}
    for r in rows:
        tf = str(r.get("preferred_execution_timeframe") or "unknown")
        direction = str(r.get("direction") or "unknown").lower()
        horizon = int(finite(r.get("horizon_seconds")))
        vol = str(r.get("execution_volatility_regime") or "unknown")
        align = _mtf_discovery_regime_alignment(r)
        ready = _mtf_discovery_readiness_bucket(r.get("readiness"))
        spread = _mtf_discovery_spread_bucket(r.get("entry_spread_bps"))
        keys = {
            "coarse": f"{tf}|{direction}|{horizon}s",
            "regime": f"{tf}|{direction}|{horizon}s|vol:{vol}|regime:{align}",
            "fine": f"{tf}|{direction}|{horizon}s|vol:{vol}|regime:{align}|ready:{ready}|spread:{spread}",
        }
        for level, key in keys.items():
            levels[level].setdefault(key, []).append(r)

    reports = {}
    positive = []
    negative = []
    outliers = []
    for level, cells in levels.items():
        vals = []
        for key, rs in cells.items():
            v = _mtf_discovery_validation(rs, level, key)
            vals.append(v)
            if v.get("research_positive_watch"):
                positive.append(v)
            if v.get("research_negative") or v.get("early_falsified_watch"):
                negative.append(v)
            if v.get("outlier_driven"):
                outliers.append(v)
        vals.sort(key=lambda v: (int(v.get("samples") or 0), finite((v.get("all") or {}).get("robust_net_edge_score"))), reverse=True)
        reports[level] = vals[:MTF_DISCOVERY_MAX_REPORTED_CELLS]

    positive.sort(key=lambda v: (finite((v.get("holdout") or {}).get("robust_net_edge_score")), int(v.get("samples") or 0)), reverse=True)
    negative.sort(key=lambda v: (int(v.get("samples") or 0), -finite((v.get("all") or {}).get("median_net_return"))), reverse=True)
    outliers.sort(key=lambda v: (finite((v.get("all") or {}).get("average_net_return")), int(v.get("samples") or 0)), reverse=True)

    # Explicitly surface the two cells falsified by the real v1.32 replication run.
    branch_finalization = []
    for h in (1800, 2700):
        key = f"1m|short|{h}s"
        rs = levels["coarse"].get(key) or []
        if rs:
            v = _mtf_discovery_validation(rs, "coarse", key)
            branch_finalization.append({
                "branch": f"1m_short_{h//60}m",
                "samples": v.get("samples"),
                "unique_symbols": (v.get("all") or {}).get("unique_symbols"),
                "win_rate": (v.get("all") or {}).get("win_rate"),
                "median_net_return": (v.get("all") or {}).get("median_net_return"),
                "leave_best_out_average_net_return": (v.get("all") or {}).get("leave_best_out_average_net_return"),
                "falsified": bool(v.get("early_falsified_watch") or v.get("research_negative")),
                "execution_authority": False,
            })

    result = {
        "model": "automated_mtf_regime_horizon_discovery_v1.33",
        "rows_considered": len(rows),
        "independent_bundles_considered": len({str(r.get("bundle_id") or "") for r in rows if str(r.get("bundle_id") or "")}),
        "cell_counts": {level: len(cells) for level, cells in levels.items()},
        "reported_cells": reports,
        "research_positive_watches": positive[:MTF_DISCOVERY_MAX_REPORTED_CELLS],
        "negative_or_falsified_cells": negative[:MTF_DISCOVERY_MAX_REPORTED_CELLS],
        "outlier_driven_cells": outliers[:MTF_DISCOVERY_MAX_REPORTED_CELLS],
        "branch_finalization": branch_finalization,
        "cell_dimensions": {
            "coarse": ["execution_timeframe", "direction", "horizon"],
            "regime": ["execution_timeframe", "direction", "horizon", "execution_volatility_regime", "higher_timeframe_regime_alignment"],
            "fine": ["execution_timeframe", "direction", "horizon", "execution_volatility_regime", "higher_timeframe_regime_alignment", "readiness_bucket", "entry_spread_bucket"],
        },
        "same_bundle_different_horizons_are_never_combined_as_independent_samples": True,
        "ex_ante_context_only": True,
        "chronological_train_holdout_required": True,
        "positive_median_required": True,
        "leave_best_out_positive_required": True,
        "multiple_symbols_required": True,
        "cost_floor_bps": ROUND_TRIP_COST_BPS,
        "does_not_relax_signal_thresholds": True,
        "does_not_validate_inverse_direction": True,
        "does_not_create_alpha": True,
        "does_not_count_as_independent_council_vote": True,
        "execution_authority": False,
        "can_increase_risk": False,
        "can_enable_testnet": False,
        "can_enable_live": False,
        "auto_promotion_allowed": False,
    }
    state["mtf_scalping_regime_discovery"] = result
    return result


def multi_timeframe_scalping_research(state, raw_by_symbol, now):
    selected = select_symbols(limit=MTF_MAX_SYMBOLS_PER_CYCLE)
    symbol_rows = []
    query_attempts = 0
    query_available = 0
    spread_queries = 0
    spread_available = 0
    query_errors = {}
    for symbol in selected:
        features = {}
        for tf, (interval_code, interval_seconds) in MTF_TIMEFRAMES.items():
            query_attempts += 1
            try:
                bars, status = _bybit_linear_closed_klines(symbol, interval_code, interval_seconds, now)
                if status == "available":
                    query_available += 1
                feat = _mtf_timeframe_features(bars)
                feat["source_status"] = status
                features[tf] = feat
            except Exception as exc:
                query_errors[f"{symbol}:{tf}"] = f"{type(exc).__name__}: {exc}"
                features[tf] = {"available": False, "score": 0.0, "confidence": 0.0, "source_status": "error"}
        spread_queries += 1
        try:
            linear_micro = _bybit_linear_spread(symbol)
        except Exception as exc:
            linear_micro = None
            query_errors[f"{symbol}:linear_spread"] = f"{type(exc).__name__}: {exc}"
        if linear_micro:
            spread_available += 1
        spot_reference = raw_by_symbol.get(normalize_symbol(symbol)) if isinstance(raw_by_symbol, dict) else None
        micro = linear_micro or {"spread_bps": 999.0}
        agg = _mtf_aggregate_symbol(symbol, features, micro=micro)
        agg["linear_spread_available"] = bool(linear_micro)
        agg["spot_reference_spread_bps"] = finite((spot_reference or {}).get("spread_bps"), 0.0)
        agg["timeframes"] = {
            tf: {k: v for k, v in feat.items() if k != "recent_closed_points"}
            for tf, feat in features.items()
        }
        symbol_rows.append(agg)

    ready = [r for r in symbol_rows if r.get("shadow_ready")]
    mode_counts = {}
    for row in symbol_rows:
        mode_counts[row.get("mode") or "unknown"] = mode_counts.get(row.get("mode") or "unknown", 0) + 1
    ready.sort(key=lambda r: finite(r.get("readiness")), reverse=True)
    outcome_lab = _update_mtf_scalping_outcome_lab(state, symbol_rows, now)
    horizon_lab = _update_mtf_scalping_horizon_lab(state, symbol_rows, now)
    regime_discovery = _mtf_automated_regime_discovery(state, now)
    dedicated_specialist = _update_mtf_5m_long_60m_specialist(state, symbol_rows, now)
    status = {
        "model": "multi_timeframe_scalping_dedicated_challenger_v1.34",
        "symbols_attempted": len(selected),
        "symbols": symbol_rows,
        "query_attempts": query_attempts,
        "query_available": query_available,
        "spread_queries": spread_queries,
        "spread_available": spread_available,
        "timeframes": list(MTF_TIMEFRAMES),
        "full_timeframe_symbols": sum(int(r.get("timeframes_available") or 0) == len(MTF_TIMEFRAMES) for r in symbol_rows),
        "shadow_ready_count": len(ready),
        "shadow_long_count": sum(r.get("direction") == "long" for r in ready),
        "shadow_short_count": sum(r.get("direction") == "short" for r in ready),
        "mode_counts": mode_counts,
        "top_shadow_candidates": [
            {k: r.get(k) for k in ("symbol", "direction", "readiness", "preferred_execution_timeframe", "preferred_holding_seconds", "spread_bps")}
            for r in ready[:5]
        ],
        "forward_outcome_lab": outcome_lab,
        "multi_horizon_falsification_lab": horizon_lab,
        "automated_regime_horizon_discovery": regime_discovery,
        "dedicated_5m_long_60m_challenger": dedicated_specialist,
        "legacy_negative_quarantine_active": bool(outcome_lab.get("legacy_negative_quarantine_active")),
        "errors": query_errors,
        "all_timeframes_required_for_shadow_ready": True,
        "higher_timeframe_opposition_can_block": True,
        "spread_gate_enabled": True,
        "execution_authority": False,
        "diagnostic_only": True,
        "does_not_create_alpha": True,
        "does_not_count_as_independent_council_vote": True,
        "can_increase_risk": False,
        "can_enable_live": False,
        "can_enable_testnet": False,
        "auto_promotion_allowed": False,
    }
    state["multi_timeframe_scalping_research"] = status
    return status


def multi_timeframe_scalping_context_observation(status, now):
    return {
        "symbol": "MARKET",
        "kind": "context",
        "score": 0.0,
        "confidence": 0.99,
        "source": "LeanTrader Bybit linear multi-timeframe scalping orchestration lab",
        "provenance": f"leantrader:mtf-dedicated-challenger:v1.34:{int(now)}",
        "observed_at": now,
        "horizon_seconds": 0,
        "metadata": {
            "model": "multi_timeframe_scalping_dedicated_challenger_v1.34",
            "status": status,
            "directional_claim": False,
            "execution_authority": False,
            "risk_increase_allowed": False,
            "auto_promotion_allowed": False,
        },
    }


def build_packs():
    now = time.time()
    state = load_json(STATE)
    resolved_this_cycle = resolve_specialist_episodes(state, now)
    council_resolved_this_cycle = resolve_council_episodes(state, now)
    council_counterfactual_resolved_this_cycle = resolve_council_counterfactuals(state, now)
    contexts = world_context_map()
    hist = state.get("micro_history") if isinstance(state.get("micro_history"), dict) else {}
    cross_hist = state.get("cross_venue_history") if isinstance(state.get("cross_venue_history"), dict) else {}
    errors = {}
    try:
        horizon_backfill_status = backfill_horizon_lab(state, now)
    except Exception as exc:
        horizon_backfill_status = {"queries_this_cycle": 0, "rows_added_this_cycle": 0, "resolved_rows": len(state.get("horizon_lab_resolved") or []), "errors": [f"{type(exc).__name__}:{exc}"], "execution_authority": False}
        errors["horizon_lab_backfill"] = f"{type(exc).__name__}: {exc}"
    raw = []
    for symbol in select_symbols():
        try:
            row = raw_microstructure(symbol, now)
            if row:
                raw.append(row)
        except Exception as exc:
            errors[f"microstructure:{symbol}"] = f"{type(exc).__name__}: {exc}"

    cal5, cal15 = calibration(PACK_MICRO_5M), calibration(PACK_MICRO_15M)
    calcross = calibration(PACK_CROSS_VENUE)
    m5, m15, crossvenue, venue_quality_rows, whales = [], [], [], [], []
    e5 = e15 = ecross = 0
    newhist = dict(hist)
    new_cross_hist = dict(cross_hist)
    cross_source_status = {"binance_available": 0, "okx_available": 0, "symbols_with_external_venue": 0, "symbols_with_two_external_venues": 0, "symbols_with_trusted_external_venue": 0, "symbols_with_two_trusted_external_venues": 0, "quality_rejections": 0}
    reputation_decisions = {PACK_MICRO_5M: [], PACK_MICRO_15M: [], PACK_CROSS_VENUE: []}
    for row in raw:
        prev = hist.get(row["symbol"]) if isinstance(hist, dict) else None
        prev = prev if isinstance(prev, dict) else None
        context = market_context(row["symbol"], row, contexts)
        rep5 = specialist_reputation(state, PACK_MICRO_5M, context, 300)
        rep15 = specialist_reputation(state, PACK_MICRO_15M, context, 900)
        repcross = specialist_reputation(state, PACK_CROSS_VENUE, context, 900)
        o5 = micro_observation(row, prev, cal5, 300, "microstructure_persistence_5m_v1.14", reputation=rep5, context=context)
        o15 = micro_observation(row, prev, cal15, 900, "microstructure_persistence_15m_v1.14", reputation=rep15, context=context)
        external_rows = []
        for venue_name, venue_fn in (("binance", _binance_orderbook), ("okx", _okx_orderbook)):
            try:
                venue_row, venue_status = venue_fn(row["symbol"])
            except Exception as exc:
                venue_row, venue_status = None, f"{type(exc).__name__}: {exc}"
            if venue_row:
                external_rows.append(venue_row)
                cross_source_status[f"{venue_name}_available"] += 1
        if external_rows:
            cross_source_status["symbols_with_external_venue"] += 1
        if len(external_rows) >= 2:
            cross_source_status["symbols_with_two_external_venues"] += 1
        qobs = venue_quality_observation(row, external_rows, now=now)
        venue_quality_rows.append(qobs)
        qmeta = qobs.get("metadata") or {}
        trusted_count = int(qmeta.get("trusted_count") or 0)
        rejected_count = int(qmeta.get("rejected_count") or 0)
        if trusted_count >= 1:
            cross_source_status["symbols_with_trusted_external_venue"] += 1
        if trusted_count >= 2:
            cross_source_status["symbols_with_two_trusted_external_venues"] += 1
        cross_source_status["quality_rejections"] += rejected_count
        cprev = cross_hist.get(row["symbol"]) if isinstance(cross_hist, dict) else None
        cprev = cprev if isinstance(cprev, dict) else None
        ocross = cross_venue_observation(row, cprev, calcross, reputation=repcross, context=context, now=now, external_rows=external_rows)
        m5.append(o5)
        m15.append(o15)
        crossvenue.append(ocross)
        whales.append(whale_context(row))
        e5 += o5["kind"] == "signal"
        e15 += o15["kind"] == "signal"
        ecross += ocross["kind"] == "signal"
        reputation_decisions[PACK_MICRO_5M].append({"symbol": row["symbol"], "regime": context["regime"], "multiplier": rep5["confidence_multiplier"], "hard_negative": rep5["hard_negative"]})
        reputation_decisions[PACK_MICRO_15M].append({"symbol": row["symbol"], "regime": context["regime"], "multiplier": rep15["confidence_multiplier"], "hard_negative": rep15["hard_negative"]})
        reputation_decisions[PACK_CROSS_VENUE].append({"symbol": row["symbol"], "regime": context["regime"], "multiplier": repcross["confidence_multiplier"], "hard_negative": repcross["hard_negative"]})
        newhist[row["symbol"]] = {"score": row["score"], "observed_at": now}
        cscore = finite((ocross.get("metadata") or {}).get("consensus_strength"))
        if (ocross.get("metadata") or {}).get("independent_external_venues", 0) > 0:
            new_cross_hist[row["symbol"]] = {"score": cscore, "observed_at": now}

    fx = []
    try:
        v = ecb_fx_context(now)
        if v:
            fx.append(v)
    except Exception as exc:
        errors["ecb_fx"] = f"{type(exc).__name__}: {exc}"

    try:
        lead = leadlag_observations(now, contexts=contexts, state=state)
    except Exception as exc:
        errors["leadlag"] = f"{type(exc).__name__}: {exc}"
        lead = []

    try:
        bridge_obs, bridge_caps = core_sensor_bridge(now)
    except Exception as exc:
        errors["core_sensor_bridge"] = f"{type(exc).__name__}: {exc}"
        bridge_obs, bridge_caps = None, []

    # v1.25: preserve v1.20 flow lineage quorum and reconcile sparse liquidation events through
    # a bounded replay-safe temporal window; correlated/common flow still cannot create direction.
    raw_by_symbol = {normalize_symbol(r.get("symbol") or ""): r for r in raw if isinstance(r, dict)}
    try:
        orthogonal_rows, orthogonal_source_status = orthogonal_flow_positioning_observations(
            now, contexts=contexts, raw_by_symbol=raw_by_symbol, state=state
        )
    except Exception as exc:
        errors["orthogonal_sensor_discovery"] = f"{type(exc).__name__}: {exc}"
        orthogonal_rows, orthogonal_source_status = [], {"error": f"{type(exc).__name__}: {exc}", "execution_authority": False}
    orthogonal_policy = orthogonal_sensor_policy(state, orthogonal_source_status, now=now)
    orthogonal_gate = apply_orthogonal_sensor_policy(orthogonal_rows, orthogonal_policy)

    # v1.29: retain bounded independent aggressive-flow acquisition and add non-overlapping
    # forward outcome tracking for already-persistent diagnostics. Neither layer can create alpha,
    # vote in council, increase risk, or alter execution authority.
    try:
        external_flow_source_status = external_aggressive_flow_research(state, now)
    except Exception as exc:
        errors["external_aggressive_flow_research"] = f"{type(exc).__name__}: {exc}"
        external_flow_source_status = {"error": f"{type(exc).__name__}: {exc}", "execution_authority": False, "diagnostic_only": True}

    # v1.32: preserve the eight-timeframe candidate generator and v1.31 evidence, quarantine
    # the mature-negative legacy horizon mapping, and add a prospective fixed multi-horizon
    # falsification lab. Thresholds are unchanged and every new layer remains research-only.
    try:
        mtf_scalping_status = multi_timeframe_scalping_research(state, raw_by_symbol, now)
    except Exception as exc:
        errors["multi_timeframe_scalping_research"] = f"{type(exc).__name__}: {exc}"
        mtf_scalping_status = {"error": f"{type(exc).__name__}: {exc}", "execution_authority": False, "diagnostic_only": True}

    # v1.25: preserve the mechanism-distinct derivatives/liquidation test while allowing only
    # unique eventful liquidation snapshots to persist briefly across quiet transport cycles.
    try:
        deriv_liq_rows, deriv_liq_source_status = derivatives_liquidation_observations(now, raw_by_symbol=raw_by_symbol, state=state)
    except Exception as exc:
        errors["derivatives_liquidation_discovery"] = f"{type(exc).__name__}: {exc}"
        deriv_liq_rows, deriv_liq_source_status = [], {"error": f"{type(exc).__name__}: {exc}", "execution_authority": False}
    deriv_liq_policy = derivatives_liquidation_policy(state, deriv_liq_source_status, now=now)
    deriv_liq_gate = apply_derivatives_liquidation_policy(deriv_liq_rows, deriv_liq_policy)

    # v1.22: liquidation tape can legitimately be quiet for many symbols. Rather than lowering
    # event thresholds, test a separate venue-independent derivatives carry/crowding hypothesis
    # using Bybit positioning and Binance public USD-M carry/OI data.
    try:
        cross_deriv_rows, cross_deriv_source_status = cross_venue_derivatives_observations(now, raw_by_symbol=raw_by_symbol, state=state)
    except Exception as exc:
        errors["cross_venue_derivatives_discovery"] = f"{type(exc).__name__}: {exc}"
        cross_deriv_rows, cross_deriv_source_status = [], {"error": f"{type(exc).__name__}: {exc}", "execution_authority": False}
    cross_deriv_policy = cross_venue_derivatives_policy(state, cross_deriv_source_status, now=now)
    cross_deriv_gate = apply_cross_venue_derivatives_policy(cross_deriv_rows, cross_deriv_policy)

    try:
        macro_obs, macro_errors, macro_source_status = macro_calendar_context(now)
        macro_warning = "; ".join(macro_errors) if macro_errors else None
    except Exception as exc:
        errors["macro_calendar"] = f"{type(exc).__name__}: {exc}"
        macro_obs = None
        macro_warning = None
        macro_source_status = {"bls_live_ics": "error", "bls_verified_snapshot": "unknown", "federal_reserve_snapshot": "unknown"}

    rep_obs = reputation_pack_observation(state, now)
    evidence_obs = evidence_governance_observation(state, now)
    packs = [
        make_pack(PACK_MICRO_5M, ["microstructure_pressure"], m5, now),
        make_pack(PACK_MICRO_15M, ["microstructure_pressure"], m15, now),
        make_pack(PACK_CROSS_VENUE, ["cross_venue_orderbook_consensus"], crossvenue, now),
        make_pack(PACK_VENUE_QUALITY, ["cross_venue_data_quality", "venue_price_basis_integrity", "venue_spread_integrity"], venue_quality_rows, now),
        make_pack(PACK_WHALE, ["whale_concentration"], whales, now),
        make_pack(PACK_FX, ["rates_fx_cross_asset"], fx, now),
        make_pack(PACK_LEADLAG, ["causal_lead_lag_discovery"], lead, now),
        make_pack(PACK_CORE_BRIDGE, bridge_caps, bridge_obs or [], now),
        make_pack(PACK_ORTHOGONAL_FLOW, ["orthogonal_flow_positioning", "cross_sensor_positioning_divergence", "deduplicated_liquidation_event_window"], orthogonal_rows, now),
        make_pack(PACK_FLOW_RESIDUAL_CONTEXT, ["common_factor_residualization", "robust_cross_sectional_residual_diagnostics", "residual_persistence_lab"], [flow_residual_context_observation(orthogonal_source_status, state, now)], now),
        make_pack(PACK_EXTERNAL_AGGRESSIVE_FLOW, ["external_aggressive_trade_flow", "binance_usdm_taker_quote_volume", "independent_flow_lineage", "cross_sectional_residual_persistence", "persistent_candidate_forward_outcome_tracking"], [external_aggressive_flow_context_observation(external_flow_source_status, now)], now),
        make_pack(PACK_MTF_SCALPING_CONTEXT, ["multi_timeframe_crypto_scalping_orchestration", "1m_3m_5m_entry_layer", "15m_30m_structure_layer", "1h_4h_1d_regime_layer", "dynamic_execution_timeframe_selection", "spread_gate", "higher_timeframe_conflict_suppression", "legacy_negative_quarantine", "fixed_multi_horizon_falsification_5m_to_60m", "cost_aware_after_cost_diagnostics"], [multi_timeframe_scalping_context_observation(mtf_scalping_status, now)], now),
        make_pack(PACK_DERIV_LIQ, ["derivatives_liquidation_squeeze", "mechanism_distinct_positioning_liquidation_research", "sparse_liquidation_event_window", "event_payload_replay_suppression"], deriv_liq_rows, now),
        make_pack(PACK_DERIV_LIQ_POLICY, ["derivatives_liquidation_prequential_validation", "same_venue_lineage_penalty", "rolling_event_window_governance"], [derivatives_liquidation_policy_observation(deriv_liq_policy, deriv_liq_gate, now)], now),
        make_pack(PACK_CROSS_DERIV, ["cross_venue_derivatives_carry", "venue_independent_crowding_reversion", "tri_venue_carry_triangulation", "persistent_candidate_exploration"], cross_deriv_rows, now),
        make_pack(PACK_CROSS_DERIV_POLICY, ["cross_venue_derivatives_prequential_validation", "venue_independent_external_confirmation", "tri_venue_consensus_governance", "candidate_coverage_scheduler", "global_negative_edge_quarantine"], [cross_venue_derivatives_policy_observation(cross_deriv_policy, cross_deriv_gate, now)], now),
        make_pack(PACK_ORTHOGONAL_COVERAGE, ["orthogonal_sensor_temporal_alignment", "bounded_last_good_sensor_cache", "candidate_specific_sensor_pairing", "orthogonal_sensor_universe_decoupling", "sensor_quality_rejection_attribution", "shadow_price_anchor_reconciliation", "sparse_event_temporal_reconciliation"], [orthogonal_sensor_coverage_observation(orthogonal_source_status, now)], now),
        make_pack(PACK_ORTHOGONAL_SENSOR_POLICY, ["orthogonal_sensor_prequential_validation", "independent_information_research"], [orthogonal_sensor_policy_observation(orthogonal_policy, orthogonal_gate, now)], now),
        make_pack(PACK_MACRO, ["macro_calendar"], [macro_obs] if macro_obs else [], now),
        make_pack(PACK_REPUTATION, ["specialist_reputation_calibration"], [rep_obs], now),
        make_pack(PACK_EVIDENCE_GOVERNANCE, ["specialist_evidence_governance"], [evidence_obs], now),
    ]
    packs = [p for p in packs if p]

    # Contextual routing removes globally/context-suppressed specialists from
    # council/diversity voting while retaining low-confidence recovery probes.
    evidence_gov = (evidence_obs.get("metadata") or {}).get("governance") or {}
    routing_gov = apply_specialist_routing(packs, evidence_gov, now=now)
    routing_obs = routing_observation(routing_gov, now)
    routing_pack = make_pack(PACK_ROUTING, ["specialist_contextual_routing"], [routing_obs], now)
    if routing_pack:
        packs.append(routing_pack)

    # v1.14: test whether the signal's holding horizon actually clears modeled costs.
    # Known sub-cost horizons are removed from council/diversity voting, while a
    # separate shadow-only challenger can test an evidence-positive alternate horizon.
    horizon_policy = net_edge_horizon_policy(state, now=now)
    net_edge_gate = apply_net_edge_horizon_gate(packs, horizon_policy)
    horizon_policy_obs = net_edge_horizon_policy_observation(horizon_policy, horizon_backfill_status, now)
    horizon_policy_pack = make_pack(
        PACK_HORIZON_POLICY,
        ["cost_aware_multi_horizon_net_edge", "economic_tradeability_hurdle", "holding_horizon_research"],
        [horizon_policy_obs],
        now,
    )
    if horizon_policy_pack:
        packs.append(horizon_policy_pack)
    horizon_challenger_rows = horizon_optimized_observations(packs, horizon_policy, now)
    horizon_challenger_pack = make_pack(
        PACK_HORIZON_CHALLENGER,
        ["cost_aware_horizon_optimized_shadow_challenger"],
        horizon_challenger_rows,
        now,
    )
    if horizon_challenger_pack:
        packs.append(horizon_challenger_pack)

    # v1.14: chronological falsification. A direction that is negative on newer
    # holdout data while its inverse is positive can be blocked from voting, but the
    # inverse is emitted only as a separately named shadow challenger.
    falsification_policy = prequential_falsification_policy(state, now=now)
    falsification_gate = apply_prequential_falsification_gate(packs, falsification_policy)
    falsification_obs = prequential_falsification_observation(falsification_policy, falsification_gate, now)
    falsification_pack = make_pack(
        PACK_FALSIFICATION_POLICY,
        ["prequential_directional_falsification", "chronological_holdout_validation", "inverse_hypothesis_research"],
        [falsification_obs],
        now,
    )
    if falsification_pack:
        packs.append(falsification_pack)
    inverse_rows = falsification_inverse_observations(packs, falsification_policy, now)
    inverse_pack = make_pack(
        PACK_INVERSE_CHALLENGER,
        ["prequential_inverse_shadow_challenger"],
        inverse_rows,
        now,
    )
    if inverse_pack:
        packs.append(inverse_pack)

    # v1.20: look for narrow, disjoint market contexts where an otherwise weak
    # specialist actually clears modeled costs on a newer chronological holdout.
    # Mature negative contexts are blocked from voting; positive contexts create a
    # separate shadow-only challenger and never raise original confidence.
    conditional_policy = conditional_edge_policy(state, now=now)
    conditional_gate = apply_conditional_edge_gate(packs, conditional_policy)
    conditional_obs = conditional_edge_policy_observation(conditional_policy, conditional_gate, now)
    conditional_pack = make_pack(
        PACK_CONDITIONAL_EDGE_POLICY,
        ["context_selective_alpha_discovery", "chronological_context_holdout", "multiple_comparison_guard"],
        [conditional_obs],
        now,
    )
    if conditional_pack:
        packs.append(conditional_pack)
    conditional_rows = conditional_challenger_observations(packs, conditional_policy, now)
    conditional_challenger_pack = make_pack(
        PACK_CONDITIONAL_CHALLENGER,
        ["context_selective_alpha_shadow_challenger"],
        conditional_rows,
        now,
    )
    if conditional_challenger_pack:
        packs.append(conditional_challenger_pack)

    # v1.20: sparse exact contexts are generalized through a fixed ex-ante hierarchy.
    # Parent-child chronological agreement can identify a shadow-only positive niche,
    # while broad negative consensus can only reduce/block the original signal.
    hierarchical_policy = hierarchical_context_policy(state, now=now)
    hierarchical_gate = apply_hierarchical_context_gate(packs, hierarchical_policy)
    hierarchical_obs = hierarchical_context_policy_observation(hierarchical_policy, hierarchical_gate, now)
    hierarchical_pack = make_pack(
        PACK_HIERARCHICAL_CONTEXT_POLICY,
        ["hierarchical_context_generalization", "sparse_evidence_shrinkage", "parent_child_holdout_agreement"],
        [hierarchical_obs],
        now,
    )
    if hierarchical_pack:
        packs.append(hierarchical_pack)
    hierarchical_rows = hierarchical_challenger_observations(packs, hierarchical_policy, now)
    hierarchical_challenger_pack = make_pack(
        PACK_HIERARCHICAL_CHALLENGER,
        ["hierarchical_context_alpha_shadow_challenger"],
        hierarchical_rows,
        now,
    )
    if hierarchical_challenger_pack:
        packs.append(hierarchical_challenger_pack)

    # Diversity governance detects duplicated/correlated votes before shadow episodes
    # are opened. Its caps are asymmetric: they can only preserve/reduce confidence.
    diversity_obs = diversity_observation(state, packs, now)
    diversity_gov = (diversity_obs.get("metadata") or {}).get("governance") or {}
    apply_diversity_caps(packs, diversity_gov)
    diversity_pack = make_pack(PACK_DIVERSITY, ["specialist_diversity_governance"], [diversity_obs], now)
    if diversity_pack:
        packs.append(diversity_pack)

    # Separate council directional-alpha evidence from risk-overlay evidence.
    # This prevents useful reduce/veto decisions from masking negative directional expectancy.
    council_role_obs = council_role_policy_observation(state, now)
    council_role_gov = (council_role_obs.get("metadata") or {}).get("policy") or {}
    council_role_pack = make_pack(
        PACK_COUNCIL_ROLE_POLICY,
        ["council_role_separation", "council_directional_alpha_evidence", "council_risk_overlay_evidence"],
        [council_role_obs],
        now,
    )
    if council_role_pack:
        packs.append(council_role_pack)

    # Learn how strongly successful reduce decisions should reduce shadow exposure.
    # The policy is derived only from resolved counterfactuals and is monotone: it
    # may tighten the council cap but can never raise it.
    risk_fraction_obs = council_risk_fraction_policy_observation(state, now)
    risk_fraction_policy = (risk_fraction_obs.get("metadata") or {}).get("policy") or {}
    risk_fraction_pack = make_pack(
        PACK_RISK_FRACTION_POLICY,
        ["council_counterfactual_risk_fraction_calibration", "adaptive_exposure_ceiling"],
        [risk_fraction_obs],
        now,
    )
    if risk_fraction_pack:
        packs.append(risk_fraction_pack)

    no_trade_frontier_policy = council_no_trade_frontier_policy(risk_fraction_policy)

    council_gov = specialist_council_advisory(packs, diversity_gov, evidence_gov, macro_obs=macro_obs, now=now, role_policy=council_role_gov)
    risk_fraction_application = apply_council_risk_fraction_policy(council_gov, risk_fraction_policy)
    no_trade_frontier_application = apply_no_trade_frontier_policy(council_gov, no_trade_frontier_policy)
    no_trade_frontier_obs = council_no_trade_frontier_observation(no_trade_frontier_policy, no_trade_frontier_application, now)
    no_trade_frontier_pack = make_pack(
        PACK_NO_TRADE_FRONTIER,
        ["counterfactual_zero_exposure_frontier", "no_trade_shadow_policy", "capital_preservation_frontier"],
        [no_trade_frontier_obs],
        now,
    )
    if no_trade_frontier_pack:
        packs.append(no_trade_frontier_pack)
    apply_council_caps(packs, council_gov)
    council_obs = council_observation(council_gov, state, now)
    council_pack = make_pack(PACK_COUNCIL, ["specialist_council_advisory"], [council_obs], now)
    if council_pack:
        packs.append(council_pack)

    opened_this_cycle = open_specialist_episodes(state, packs, now)
    episode_independence_status = state.get("specialist_episode_independence") or {}
    evidence_independence_status = shadow_evidence_independence(state)
    independence_obs = episode_independence_observation(episode_independence_status, evidence_independence_status, now)
    independence_pack = make_pack(
        PACK_EPISODE_INDEPENDENCE,
        ["nonoverlapping_shadow_episodes", "effective_sample_size_governance", "persistence_without_sample_inflation"],
        [independence_obs],
        now,
    )
    if independence_pack:
        packs.append(independence_pack)
    council_opened_this_cycle = open_council_episodes(state, council_gov, now)
    council_counterfactual_opened_this_cycle = open_council_counterfactuals(state, council_gov, now)
    council_cf_obs = council_counterfactual_observation(state, now)
    council_cf_pack = make_pack(PACK_COUNCIL_COUNTERFACTUAL, ["council_decision_counterfactual_evidence"], [council_cf_obs], now)
    if council_cf_pack:
        packs.append(council_cf_pack)

    # Rebuild summaries after newly opened episodes so the operator sees the current ledger.
    rep_obs = reputation_pack_observation(state, now)
    evidence_obs = evidence_governance_observation(state, now)
    for pack in packs:
        if pack.get("pack_id") == PACK_REPUTATION:
            pack["observations"] = [rep_obs]
        elif pack.get("pack_id") == PACK_EVIDENCE_GOVERNANCE:
            pack["observations"] = [evidence_obs]
        elif pack.get("pack_id") == PACK_COUNCIL:
            pack["observations"] = [council_observation(council_gov, state, now)]
        elif pack.get("pack_id") == PACK_COUNCIL_COUNTERFACTUAL:
            pack["observations"] = [council_counterfactual_observation(state, now)]
        elif pack.get("pack_id") == PACK_COUNCIL_ROLE_POLICY:
            pack["observations"] = [council_role_policy_observation(state, now)]
        elif pack.get("pack_id") == PACK_RISK_FRACTION_POLICY:
            pack["observations"] = [council_risk_fraction_policy_observation(state, now)]
        elif pack.get("pack_id") == PACK_NO_TRADE_FRONTIER:
            pack["observations"] = [council_no_trade_frontier_observation(no_trade_frontier_policy, no_trade_frontier_application, now)]
        elif pack.get("pack_id") == PACK_HORIZON_POLICY:
            pack["observations"] = [net_edge_horizon_policy_observation(horizon_policy, horizon_backfill_status, now)]
        elif pack.get("pack_id") == PACK_FALSIFICATION_POLICY:
            pack["observations"] = [prequential_falsification_observation(falsification_policy, falsification_gate, now)]
        elif pack.get("pack_id") == PACK_CONDITIONAL_EDGE_POLICY:
            pack["observations"] = [conditional_edge_policy_observation(conditional_policy, conditional_gate, now)]
        elif pack.get("pack_id") == PACK_HIERARCHICAL_CONTEXT_POLICY:
            pack["observations"] = [hierarchical_context_policy_observation(hierarchical_policy, hierarchical_gate, now)]
        elif pack.get("pack_id") == PACK_ORTHOGONAL_SENSOR_POLICY:
            pack["observations"] = [orthogonal_sensor_policy_observation(orthogonal_policy, orthogonal_gate, now)]
        elif pack.get("pack_id") == PACK_DERIV_LIQ_POLICY:
            pack["observations"] = [derivatives_liquidation_policy_observation(deriv_liq_policy, deriv_liq_gate, now)]
        elif pack.get("pack_id") == PACK_CROSS_DERIV_POLICY:
            pack["observations"] = [cross_venue_derivatives_policy_observation(cross_deriv_policy, cross_deriv_gate, now)]

    # Refresh role policy after all due outcomes were resolved; newly opened episodes
    # do not count as evidence until they close.
    council_role_gov = council_role_policy(state, now=now)

    status = {
        "schema_version": 24,
        "version": VERSION,
        "patch_release": PATCH_RELEASE,
        "updated_at": now,
        "errors": errors,
        "execution_authority": False,
        "symbols": sorted({r["symbol"] for r in raw}),
        "raw_microstructure_observations": len(raw),
        "microstructure_5m_signals": int(e5),
        "microstructure_15m_signals": int(e15),
        "cross_venue_signals": int(ecross),
        "cross_venue_observations": len(crossvenue),
        "venue_quality_observations": len(venue_quality_rows),
        "cross_venue_source_status": cross_source_status,
        "leadlag_rows": len(lead),
        "native_capabilities_bridged": bridge_caps,
        "macro_calendar_available": bool(macro_obs),
        "macro_calendar_warning": macro_warning,
        "macro_calendar_source_status": macro_source_status,
        "packs_published": [p["pack_id"] for p in packs],
        "calibration": {
            PACK_MICRO_5M: cal5,
            PACK_MICRO_15M: cal15,
            PACK_CROSS_VENUE: calcross,
            PACK_LEADLAG: calibration(PACK_LEADLAG),
            OLD_PACK: calibration(OLD_PACK),
        },
        "specialist_outcomes_resolved_this_cycle": resolved_this_cycle,
        "specialist_episodes_opened_this_cycle": opened_this_cycle,
        "specialist_episode_independence": episode_independence_status,
        "shadow_evidence_independence": evidence_independence_status,
        "specialist_pending": state.get("specialist_pending") or {},
        "specialist_resolved": state.get("specialist_resolved") or [],
        "specialist_reputation_summary": (rep_obs.get("metadata") or {}).get("summaries") or {},
        "specialist_evidence_governance": (evidence_obs.get("metadata") or {}).get("governance") or {},
        "specialist_routing_governance": routing_gov,
        "specialist_diversity_governance": diversity_gov,
        "specialist_council_advisory": council_gov,
        "council_role_policy": council_role_gov,
        "council_risk_fraction_policy": risk_fraction_policy,
        "council_risk_fraction_application": risk_fraction_application,
        "council_no_trade_frontier_policy": no_trade_frontier_policy,
        "council_no_trade_frontier_application": no_trade_frontier_application,
        "net_edge_horizon_policy": horizon_policy,
        "net_edge_horizon_gate": net_edge_gate,
        "horizon_lab_backfill_status": horizon_backfill_status,
        "horizon_lab_resolved": state.get("horizon_lab_resolved") or [],
        "horizon_optimized_signals": len(horizon_challenger_rows),
        "prequential_falsification_policy": falsification_policy,
        "prequential_falsification_gate": falsification_gate,
        "conditional_edge_policy": conditional_policy,
        "conditional_edge_gate": conditional_gate,
        "conditional_edge_shadow_signals": len(conditional_rows),
        "conditional_edge_shadow_metrics": metric_stats([r for r in (state.get("specialist_resolved") or []) if isinstance(r, dict) and r.get("pack_id") == PACK_CONDITIONAL_CHALLENGER]),
        "hierarchical_context_policy": hierarchical_policy,
        "hierarchical_context_gate": hierarchical_gate,
        "hierarchical_context_shadow_signals": len(hierarchical_rows),
        "hierarchical_context_shadow_metrics": metric_stats([r for r in (state.get("specialist_resolved") or []) if isinstance(r, dict) and r.get("pack_id") == PACK_HIERARCHICAL_CHALLENGER]),
        "orthogonal_sensor_source_status": orthogonal_source_status,
        "flow_residual_lab": state.get("flow_residual_lab") or {},
        "flow_residual_persistent_candidates": int(((state.get("flow_residual_lab") or {}).get("persistent_candidate_count") or 0)),
        "external_aggressive_flow_source_status": external_flow_source_status,
        "external_aggressive_flow_lab": state.get("external_aggressive_flow_lab") or {},
        "external_aggressive_flow_query_history": state.get("external_aggressive_flow_query_history") or {},
        "external_aggressive_flow_persistent_candidates": int(((state.get("external_aggressive_flow_lab") or {}).get("persistent_candidate_count") or 0)),
        "external_aggressive_flow_outcome_lab": state.get("external_aggressive_flow_outcome_lab") or {},
        "external_aggressive_flow_outcome_pending": state.get("external_aggressive_flow_outcome_pending") or {},
        "external_aggressive_flow_outcome_resolved": state.get("external_aggressive_flow_outcome_resolved") or [],
        "multi_timeframe_scalping_research": state.get("multi_timeframe_scalping_research") or mtf_scalping_status,
        "mtf_scalping_outcome_lab": state.get("mtf_scalping_outcome_lab") or {},
        "mtf_scalping_outcome_pending": state.get("mtf_scalping_outcome_pending") or {},
        "mtf_scalping_outcome_resolved": state.get("mtf_scalping_outcome_resolved") or [],
        "mtf_scalping_horizon_lab": state.get("mtf_scalping_horizon_lab") or {},
        "mtf_scalping_horizon_pending": state.get("mtf_scalping_horizon_pending") or {},
        "mtf_scalping_horizon_resolved": state.get("mtf_scalping_horizon_resolved") or [],
        "mtf_scalping_regime_discovery": state.get("mtf_scalping_regime_discovery") or {},
        "mtf_5m_long_60m_specialist_lab": state.get("mtf_5m_long_60m_specialist_lab") or {},
        "mtf_5m_long_60m_specialist_pending": state.get("mtf_5m_long_60m_specialist_pending") or {},
        "mtf_5m_long_60m_specialist_resolved": state.get("mtf_5m_long_60m_specialist_resolved") or [],
        "orthogonal_sensor_cache": state.get("orthogonal_sensor_cache") or {},
        "orthogonal_price_anchor_cache": state.get("orthogonal_price_anchor_cache") or {},
        "liquidation_event_window": state.get("liquidation_event_window") or {},
        "orthogonal_sensor_policy": orthogonal_policy,
        "orthogonal_sensor_gate": orthogonal_gate,
        "orthogonal_flow_signals": len(orthogonal_rows),
        "orthogonal_flow_shadow_metrics": _horizon_metric_stats([r for r in (state.get("specialist_resolved") or []) if isinstance(r, dict) and r.get("pack_id") == PACK_ORTHOGONAL_FLOW]),
        "derivatives_liquidation_source_status": deriv_liq_source_status,
        "derivatives_liquidation_policy": deriv_liq_policy,
        "derivatives_liquidation_gate": deriv_liq_gate,
        "derivatives_liquidation_signals": len(deriv_liq_rows),
        "derivatives_liquidation_shadow_metrics": _horizon_metric_stats([r for r in (state.get("specialist_resolved") or []) if isinstance(r, dict) and r.get("pack_id") == PACK_DERIV_LIQ]),
        "cross_venue_derivatives_source_status": cross_deriv_source_status,
        "cross_venue_derivatives_query_history": state.get("cross_venue_derivatives_query_history") or {},
        "cross_venue_derivatives_okx_oi_history": state.get("cross_venue_derivatives_okx_oi_history") or {},
        "cross_venue_derivatives_global_quarantine_latch": state.get("cross_venue_derivatives_global_quarantine_latch") or {},
        "cross_venue_derivatives_policy": cross_deriv_policy,
        "cross_venue_derivatives_gate": cross_deriv_gate,
        "cross_venue_derivatives_signals": len(cross_deriv_rows),
        "cross_venue_derivatives_shadow_metrics": _horizon_metric_stats([r for r in (state.get("specialist_resolved") or []) if isinstance(r, dict) and r.get("pack_id") == PACK_CROSS_DERIV]),
        "inverse_shadow_signals": len(inverse_rows),
        "inverse_shadow_metrics": metric_stats([r for r in (state.get("specialist_resolved") or []) if isinstance(r, dict) and r.get("pack_id") == PACK_INVERSE_CHALLENGER]),
        "council_shadow_metrics": metric_stats(state.get("council_resolved") or []),
        "council_counterfactual_metrics": council_counterfactual_metrics(state.get("council_counterfactual_resolved") or []),
        "council_counterfactual_pending": state.get("council_counterfactual_pending") or {},
        "council_counterfactual_resolved": state.get("council_counterfactual_resolved") or [],
        "council_counterfactual_opened_this_cycle": council_counterfactual_opened_this_cycle,
        "council_counterfactual_resolved_this_cycle": council_counterfactual_resolved_this_cycle,
        "council_pending": state.get("council_pending") or {},
        "council_resolved": state.get("council_resolved") or [],
        "council_episodes_opened_this_cycle": council_opened_this_cycle,
        "council_outcomes_resolved_this_cycle": council_resolved_this_cycle,
        "effective_independent_votes": diversity_gov.get("effective_independent_votes_total", 0.0),
        "redundancy_clusters": diversity_gov.get("redundancy_clusters") or [],
        "specialist_conflicts": diversity_gov.get("directional_conflicts") or [],
        "research_validation_nominations": (evidence_obs.get("metadata") or {}).get("research_validation_nominations") or [],
        "suppressed_specialists": (evidence_obs.get("metadata") or {}).get("suppressed_specialists") or [],
        "reputation_decisions": reputation_decisions,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "leadlag_history": state.get("leadlag_history") or {},
        "micro_history": newhist,
        "cross_venue_history": new_cross_hist,
    }
    return packs, status

def cycle():
    packs, status = build_packs()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    for pack in packs:
        atomic_write(INBOX / f"{pack['pack_id']}.json", pack)
    atomic_write(STATE, status)
    print(json.dumps({
        "event": "evolution_sidecar_cycle_v1_34",
        "version": VERSION,
        "patch_release": PATCH_RELEASE,
        "errors": status["errors"],
        "execution_authority": False,
        "raw_microstructure_observations": status["raw_microstructure_observations"],
        "microstructure_5m_signals": status["microstructure_5m_signals"],
        "microstructure_15m_signals": status["microstructure_15m_signals"],
        "cross_venue_signals": status.get("cross_venue_signals"),
        "venue_quality_observations": status.get("venue_quality_observations"),
        "cross_venue_source_status": status.get("cross_venue_source_status"),
        "leadlag_rows": status["leadlag_rows"],
        "native_capabilities_bridged": status["native_capabilities_bridged"],
        "macro_calendar_available": status["macro_calendar_available"],
        "macro_calendar_source_status": status.get("macro_calendar_source_status"),
        "specialist_outcomes_resolved_this_cycle": status.get("specialist_outcomes_resolved_this_cycle"),
        "specialist_episodes_opened_this_cycle": status.get("specialist_episodes_opened_this_cycle"),
        "episode_overlap_merged": (status.get("specialist_episode_independence") or {}).get("overlapping_confirmations_merged", 0),
        "episode_direction_conflicts_suppressed": (status.get("specialist_episode_independence") or {}).get("opposite_direction_conflicts_suppressed", 0),
        "effective_nonoverlap_shadow_samples": (status.get("shadow_evidence_independence") or {}).get("effective_nonoverlap_samples_total", 0),
        "overlapping_shadow_samples_discounted": (status.get("shadow_evidence_independence") or {}).get("overlapping_samples_discounted_total", 0),
        "specialist_pending": len(status.get("specialist_pending") or {}),
        "specialist_resolved": len(status.get("specialist_resolved") or []),
        "research_validation_nominations": status.get("research_validation_nominations") or [],
        "suppressed_specialists": status.get("suppressed_specialists") or [],
        "effective_independent_votes": status.get("effective_independent_votes"),
        "redundancy_clusters": len(status.get("redundancy_clusters") or []),
        "specialist_conflicts": len(status.get("specialist_conflicts") or []),
        "council_actions": (status.get("specialist_council_advisory") or {}).get("action_counts") or {},
        "council_directional_alpha_lifecycle": ((status.get("council_role_policy") or {}).get("directional_alpha") or {}).get("lifecycle"),
        "council_risk_overlay_lifecycle": ((status.get("council_role_policy") or {}).get("risk_overlay") or {}).get("lifecycle"),
        "council_risk_fraction_lifecycle": ((status.get("council_risk_fraction_policy") or {}).get("global") or {}).get("lifecycle"),
        "council_risk_fraction_recommended_ceiling": ((status.get("council_risk_fraction_policy") or {}).get("global") or {}).get("recommended_exposure_ceiling"),
        "council_risk_fraction_caps_tightened": (status.get("council_risk_fraction_application") or {}).get("caps_tightened"),
        "council_no_trade_frontier": (status.get("council_no_trade_frontier_policy") or {}).get("global_no_trade_shadow_candidate"),
        "council_reduce_to_veto_shadow": (status.get("council_no_trade_frontier_application") or {}).get("reduce_to_veto_shadow"),
        "horizon_lab_rows_added_this_cycle": (status.get("horizon_lab_backfill_status") or {}).get("rows_added_this_cycle"),
        "horizon_lab_resolved": len(status.get("horizon_lab_resolved") or []),
        "horizon_positive_packs": (status.get("net_edge_horizon_policy") or {}).get("research_positive_horizon_packs") or [],
        "net_edge_sub_cost_blocked": ((status.get("net_edge_horizon_gate") or {}).get("counts") or {}).get("sub_cost_blocked", 0),
        "horizon_optimized_signals": status.get("horizon_optimized_signals"),
        "inverse_shadow_candidates": len((status.get("prequential_falsification_policy") or {}).get("inverse_shadow_candidates") or []),
        "inverse_watches": len((status.get("prequential_falsification_policy") or {}).get("inverse_watches") or []),
        "prequential_original_falsified_blocked": ((status.get("prequential_falsification_gate") or {}).get("counts") or {}).get("original_falsified_blocked", 0),
        "conditional_validated_contexts": len((status.get("conditional_edge_policy") or {}).get("validated_contexts") or []),
        "conditional_negative_contexts": len((status.get("conditional_edge_policy") or {}).get("mature_negative_contexts") or []),
        "conditional_negative_blocked": ((status.get("conditional_edge_gate") or {}).get("counts") or {}).get("mature_negative_blocked", 0),
        "conditional_edge_shadow_signals": int(status.get("conditional_edge_shadow_signals") or 0),
        "conditional_edge_shadow_resolved_samples": (status.get("conditional_edge_shadow_metrics") or {}).get("samples", 0),
        "conditional_edge_shadow_average_net_return": (status.get("conditional_edge_shadow_metrics") or {}).get("average_net_return", 0.0),
        "hierarchical_validated_contexts": len((status.get("hierarchical_context_policy") or {}).get("validated_contexts") or []),
        "hierarchical_negative_contexts": len((status.get("hierarchical_context_policy") or {}).get("negative_consensus_contexts") or []),
        "hierarchical_positive_watches": len((status.get("hierarchical_context_policy") or {}).get("positive_watches") or []),
        "hierarchical_parent_negative_blocked": ((status.get("hierarchical_context_gate") or {}).get("counts") or {}).get("parent_negative_blocked", 0),
        "hierarchical_context_shadow_signals": int(status.get("hierarchical_context_shadow_signals") or 0),
        "orthogonal_flow_signals": int(status.get("orthogonal_flow_signals") or 0),
        "orthogonal_sensor_hypothesis_counts": (status.get("orthogonal_sensor_source_status") or {}).get("hypothesis_counts") or {},
        "orthogonal_research_positive_hypotheses": (status.get("orthogonal_sensor_policy") or {}).get("research_positive_hypotheses") or [],
        "orthogonal_prequential_positive_watches": (status.get("orthogonal_sensor_policy") or {}).get("prequential_positive_watches") or [],
        "orthogonal_prequential_negative_hypotheses": (status.get("orthogonal_sensor_policy") or {}).get("prequential_negative_hypotheses") or [],
        "orthogonal_flow_shadow_resolved_samples": (status.get("orthogonal_flow_shadow_metrics") or {}).get("samples", 0),
        "orthogonal_flow_shadow_average_net_return": (status.get("orthogonal_flow_shadow_metrics") or {}).get("average_net_return", 0.0),
        "flow_residual_cross_section_active": bool((((status.get("orthogonal_sensor_source_status") or {}).get("flow_residual_diagnostics") or {}).get("active"))),
        "flow_residual_current_single_rows": int((((status.get("orthogonal_sensor_source_status") or {}).get("flow_residual_diagnostics") or {}).get("current_single_component_rows") or 0)),
        "flow_residual_common_factor": finite((((status.get("orthogonal_sensor_source_status") or {}).get("flow_residual_diagnostics") or {}).get("common_factor_median"))),
        "flow_residual_diagnostic_outliers": int((((status.get("orthogonal_sensor_source_status") or {}).get("flow_residual_diagnostics") or {}).get("diagnostic_outliers") or 0)),
        "flow_residual_max_abs_robust_z": finite((((status.get("orthogonal_sensor_source_status") or {}).get("flow_residual_diagnostics") or {}).get("max_abs_robust_z"))),
        "flow_residual_persistent_candidates": int(status.get("flow_residual_persistent_candidates") or 0),
        "external_flow_queries_attempted": int((status.get("external_aggressive_flow_source_status") or {}).get("queries_attempted") or 0),
        "external_flow_available": int((status.get("external_aggressive_flow_source_status") or {}).get("available") or 0),
        "external_flow_cross_section_active": bool(((status.get("external_aggressive_flow_source_status") or {}).get("cross_section") or {}).get("active")),
        "external_flow_common_factor": finite(((status.get("external_aggressive_flow_source_status") or {}).get("cross_section") or {}).get("common_factor_median")),
        "external_flow_diagnostic_outliers": int(((status.get("external_aggressive_flow_source_status") or {}).get("cross_section") or {}).get("diagnostic_outliers") or 0),
        "external_flow_max_abs_robust_z": finite(((status.get("external_aggressive_flow_source_status") or {}).get("cross_section") or {}).get("max_abs_robust_z")),
        "external_flow_persistent_candidates": int(status.get("external_aggressive_flow_persistent_candidates") or 0),
        "external_flow_persistent_symbols": list(((status.get("external_aggressive_flow_outcome_lab") or {}).get("persistent_candidate_symbols") or []))[:8],
        "external_flow_outcome_pending": int(((status.get("external_aggressive_flow_outcome_lab") or {}).get("pending_count") or 0)),
        "external_flow_outcome_resolved": int(((status.get("external_aggressive_flow_outcome_lab") or {}).get("resolved_count") or 0)),
        "external_flow_outcome_opened_this_cycle": int(((status.get("external_aggressive_flow_outcome_lab") or {}).get("opened_this_cycle") or 0)),
        "external_flow_outcome_resolved_this_cycle": int(((status.get("external_aggressive_flow_outcome_lab") or {}).get("resolved_this_cycle") or 0)),
        "external_flow_outcome_average_net_return": finite((((status.get("external_aggressive_flow_outcome_lab") or {}).get("metrics") or {}).get("average_net_return"))),
        "external_flow_outcome_win_rate": finite((((status.get("external_aggressive_flow_outcome_lab") or {}).get("metrics") or {}).get("win_rate"))),
        "external_flow_outcome_cost_coverage_ratio": finite((((status.get("external_aggressive_flow_outcome_lab") or {}).get("metrics") or {}).get("cost_coverage_ratio"))),
        "mtf_scalping_symbols_attempted": int(((status.get("multi_timeframe_scalping_research") or {}).get("symbols_attempted") or 0)),
        "mtf_scalping_query_attempts": int(((status.get("multi_timeframe_scalping_research") or {}).get("query_attempts") or 0)),
        "mtf_scalping_query_available": int(((status.get("multi_timeframe_scalping_research") or {}).get("query_available") or 0)),
        "mtf_scalping_spread_available": int(((status.get("multi_timeframe_scalping_research") or {}).get("spread_available") or 0)),
        "mtf_scalping_full_timeframe_symbols": int(((status.get("multi_timeframe_scalping_research") or {}).get("full_timeframe_symbols") or 0)),
        "mtf_scalping_shadow_ready": int(((status.get("multi_timeframe_scalping_research") or {}).get("shadow_ready_count") or 0)),
        "mtf_scalping_shadow_long": int(((status.get("multi_timeframe_scalping_research") or {}).get("shadow_long_count") or 0)),
        "mtf_scalping_shadow_short": int(((status.get("multi_timeframe_scalping_research") or {}).get("shadow_short_count") or 0)),
        "mtf_scalping_mode_counts": (status.get("multi_timeframe_scalping_research") or {}).get("mode_counts") or {},
        "mtf_scalping_top_shadow_candidates": (status.get("multi_timeframe_scalping_research") or {}).get("top_shadow_candidates") or [],
        "mtf_scalping_outcome_pending": int(((status.get("mtf_scalping_outcome_lab") or {}).get("pending_count") or 0)),
        "mtf_scalping_outcome_pending_symbols": (status.get("mtf_scalping_outcome_lab") or {}).get("pending_symbols") or [],
        "mtf_scalping_outcome_opened_this_cycle": int(((status.get("mtf_scalping_outcome_lab") or {}).get("opened_this_cycle") or 0)),
        "mtf_scalping_outcome_resolved_this_cycle": int(((status.get("mtf_scalping_outcome_lab") or {}).get("resolved_this_cycle") or 0)),
        "mtf_scalping_outcome_resolved": int(((status.get("mtf_scalping_outcome_lab") or {}).get("resolved_count") or 0)),
        "mtf_scalping_outcome_average_net_return": finite((((status.get("mtf_scalping_outcome_lab") or {}).get("metrics") or {}).get("average_net_return"))),
        "mtf_scalping_outcome_win_rate": finite((((status.get("mtf_scalping_outcome_lab") or {}).get("metrics") or {}).get("win_rate"))),
        "mtf_scalping_outcome_cost_coverage_ratio": finite((((status.get("mtf_scalping_outcome_lab") or {}).get("metrics") or {}).get("cost_coverage_ratio"))),
        "mtf_scalping_validation_lifecycle": (((status.get("mtf_scalping_outcome_lab") or {}).get("validation") or {}).get("lifecycle") or "collecting"),
        "mtf_scalping_validation_samples": int((((status.get("mtf_scalping_outcome_lab") or {}).get("validation") or {}).get("samples") or 0)),
        "mtf_scalping_testnet_research_gate_satisfied": bool((((status.get("mtf_scalping_outcome_lab") or {}).get("validation") or {}).get("testnet_research_gate_satisfied"))),
        "mtf_scalping_legacy_negative_quarantine": bool((status.get("mtf_scalping_outcome_lab") or {}).get("legacy_negative_quarantine_active")),
        "mtf_scalping_legacy_suppressed_this_cycle": int((status.get("mtf_scalping_outcome_lab") or {}).get("legacy_negative_quarantine_suppressed_this_cycle") or 0),
        "mtf_scalping_horizon_pending_bundles": int((status.get("mtf_scalping_horizon_lab") or {}).get("pending_bundle_count") or 0),
        "mtf_scalping_horizon_pending_symbols": (status.get("mtf_scalping_horizon_lab") or {}).get("pending_symbols") or [],
        "mtf_scalping_horizon_bundles_opened_this_cycle": int((status.get("mtf_scalping_horizon_lab") or {}).get("bundles_opened_this_cycle") or 0),
        "mtf_scalping_horizon_rows_resolved_this_cycle": int((status.get("mtf_scalping_horizon_lab") or {}).get("horizons_resolved_this_cycle") or 0),
        "mtf_scalping_horizon_rows_resolved": int((status.get("mtf_scalping_horizon_lab") or {}).get("resolved_horizon_rows") or 0),
        "mtf_scalping_horizon_positive_watches": (status.get("mtf_scalping_horizon_lab") or {}).get("prospective_positive_watches") or [],
        "mtf_scalping_legacy_provisional_cells": (((status.get("mtf_scalping_horizon_lab") or {}).get("legacy_decomposition") or {}).get("provisional_positive_cells") or []),
        "mtf_regime_discovery_rows": int((status.get("mtf_scalping_regime_discovery") or {}).get("rows_considered") or 0),
        "mtf_regime_discovery_independent_bundles": int((status.get("mtf_scalping_regime_discovery") or {}).get("independent_bundles_considered") or 0),
        "mtf_regime_discovery_cell_counts": (status.get("mtf_scalping_regime_discovery") or {}).get("cell_counts") or {},
        "mtf_regime_discovery_positive_watches": (status.get("mtf_scalping_regime_discovery") or {}).get("research_positive_watches") or [],
        "mtf_regime_discovery_negative_cells": (status.get("mtf_scalping_regime_discovery") or {}).get("negative_or_falsified_cells") or [],
        "mtf_regime_discovery_outlier_cells": (status.get("mtf_scalping_regime_discovery") or {}).get("outlier_driven_cells") or [],
        "mtf_regime_discovery_branch_finalization": (status.get("mtf_scalping_regime_discovery") or {}).get("branch_finalization") or [],
        "mtf_5m_long_60m_specialist_opened_this_cycle": int((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("opened_this_cycle") or 0),
        "mtf_5m_long_60m_specialist_resolved_this_cycle": int((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("resolved_this_cycle") or 0),
        "mtf_5m_long_60m_specialist_pending": int((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("pending_count") or 0),
        "mtf_5m_long_60m_specialist_pending_symbols": (status.get("mtf_5m_long_60m_specialist_lab") or {}).get("pending_symbols") or [],
        "mtf_5m_long_60m_specialist_resolved": int((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("resolved_count") or 0),
        "mtf_5m_long_60m_specialist_lifecycle": (((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("validation") or {}).get("lifecycle") or "dedicated_collecting"),
        "mtf_5m_long_60m_specialist_replication_positive": bool(((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("validation") or {}).get("replication_positive")),
        "mtf_5m_long_60m_specialist_candidate_for_testnet_review": bool(((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("validation") or {}).get("candidate_for_controlled_testnet_review")),
        "mtf_5m_long_60m_specialist_average_net_return": finite(((((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("validation") or {}).get("all") or {}).get("average_net_return"))),
        "mtf_5m_long_60m_specialist_holdout_average_net_return": finite(((((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("validation") or {}).get("holdout") or {}).get("average_net_return"))),
        "mtf_5m_long_60m_specialist_average_mfe": finite(((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("risk_path_metrics") or {}).get("average_mfe")),
        "mtf_5m_long_60m_specialist_average_mae": finite(((status.get("mtf_5m_long_60m_specialist_lab") or {}).get("risk_path_metrics") or {}).get("average_mae")),
        "derivatives_liquidation_signals": int(status.get("derivatives_liquidation_signals") or 0),
        "derivatives_liquidation_hypothesis_counts": (status.get("derivatives_liquidation_source_status") or {}).get("hypothesis_counts") or {},
        "derivatives_liquidation_prequential_positive_watches": (status.get("derivatives_liquidation_policy") or {}).get("prequential_positive_watches") or [],
        "derivatives_liquidation_research_positive_hypotheses": (status.get("derivatives_liquidation_policy") or {}).get("research_positive_hypotheses") or [],
        "derivatives_liquidation_shadow_resolved_samples": (status.get("derivatives_liquidation_shadow_metrics") or {}).get("samples", 0),
        "derivatives_liquidation_shadow_average_net_return": (status.get("derivatives_liquidation_shadow_metrics") or {}).get("average_net_return", 0.0),
        "liquidation_event_window_reused_pairs": (status.get("derivatives_liquidation_source_status") or {}).get("liquidation_event_window_reused_pairs", 0),
        "liquidation_event_window_fresh_snapshots_added": (status.get("derivatives_liquidation_source_status") or {}).get("liquidation_event_window_fresh_snapshots_added", 0),
        "liquidation_event_window_payload_replays_suppressed": (status.get("derivatives_liquidation_source_status") or {}).get("liquidation_event_window_payload_replays_suppressed", 0),
        "liquidation_event_window_source_timestamp_replays_suppressed": (status.get("derivatives_liquidation_source_status") or {}).get("liquidation_event_window_source_timestamp_replays_suppressed", 0),
        "cross_venue_derivatives_signals": int(status.get("cross_venue_derivatives_signals") or 0),
        "cross_venue_derivatives_hypothesis_counts": (status.get("cross_venue_derivatives_source_status") or {}).get("hypothesis_counts") or {},
        "cross_venue_derivatives_positive_watches": (status.get("cross_venue_derivatives_policy") or {}).get("prequential_positive_watches") or [],
        "cross_venue_derivatives_research_positive": (status.get("cross_venue_derivatives_policy") or {}).get("research_positive_hypotheses") or [],
        "cross_venue_derivatives_negative": (status.get("cross_venue_derivatives_policy") or {}).get("prequential_negative_hypotheses") or [],
        "cross_venue_derivatives_global_quarantine": bool((status.get("cross_venue_derivatives_policy") or {}).get("global_negative_quarantine_active")),
        "cross_venue_derivatives_global_lifecycle": ((status.get("cross_venue_derivatives_policy") or {}).get("global_quarantine") or {}).get("lifecycle", ""),
        "cross_venue_derivatives_global_recovery_probes": (((status.get("cross_venue_derivatives_gate") or {}).get("counts") or {}).get("global_prequential_negative_recovery_probe", 0)),
        "cross_venue_derivatives_shadow_resolved_samples": (status.get("cross_venue_derivatives_shadow_metrics") or {}).get("samples", 0),
        "cross_venue_derivatives_shadow_average_net_return": (status.get("cross_venue_derivatives_shadow_metrics") or {}).get("average_net_return", 0.0),
        "inverse_shadow_signals": status.get("inverse_shadow_signals"),
        "inverse_shadow_resolved_samples": (status.get("inverse_shadow_metrics") or {}).get("samples", 0),
        "inverse_shadow_average_net_return": (status.get("inverse_shadow_metrics") or {}).get("average_net_return", 0.0),
        "council_episodes_opened_this_cycle": status.get("council_episodes_opened_this_cycle"),
        "council_outcomes_resolved_this_cycle": status.get("council_outcomes_resolved_this_cycle"),
        "council_pending": len(status.get("council_pending") or {}),
        "council_resolved": len(status.get("council_resolved") or []),
        "council_counterfactual_opened_this_cycle": status.get("council_counterfactual_opened_this_cycle"),
        "council_counterfactual_resolved_this_cycle": status.get("council_counterfactual_resolved_this_cycle"),
        "council_counterfactual_pending": len(status.get("council_counterfactual_pending") or {}),
        "council_counterfactual_resolved": len(status.get("council_counterfactual_resolved") or []),
        "routing_counts": (status.get("specialist_routing_governance") or {}).get("counts") or {},
        "packs_published": status["packs_published"],
    }, sort_keys=True), flush=True)


def handle_stop(_s, _f):
    global STOP
    STOP = True


def main():
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    once = "--once" in sys.argv
    while not STOP:
        try:
            cycle()
        except Exception as exc:
            status = {
                "schema_version": 24,
                "version": VERSION,
                "patch_release": PATCH_RELEASE,
                "updated_at": time.time(),
                "errors": {"cycle": f"{type(exc).__name__}: {exc}"},
                "execution_authority": False,
            }
            atomic_write(STATE, status)
            print(json.dumps({"event": "evolution_sidecar_error_v1_34", **status}, sort_keys=True), flush=True)
        if once:
            return 0
        for _ in range(INTERVAL):
            if STOP:
                break
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
