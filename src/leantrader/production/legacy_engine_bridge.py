from __future__ import annotations

import asyncio
import ast
import copy
import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

from ..agents.swarm_service import ReadOnlySwarmService

LOGGER = logging.getLogger("leantrader.production.legacy_restoration")


def _n(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _direction(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"buy", "long", "bull", "bullish"}:
        return "long"
    if raw in {"sell", "short", "bear", "bearish"}:
        return "short"
    return "flat"


def merge_candidate_symbols(
    base: list[str],
    legacy: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    bounded = max(1, int(limit))
    ranked_legacy = sorted(
        [
            row
            for row in legacy
            if isinstance(row, dict)
            and str(row.get("symbol") or "").strip()
        ],
        key=lambda row: (
            _n(row.get("score")),
            _n(row.get("confidence")),
            _n(row.get("absolute_percentage_24h")),
            _n(row.get("quote_volume_usd")),
        ),
        reverse=True,
    )
    output: list[str] = []

    def add(symbol: Any) -> None:
        normalized = str(symbol or "").upper().strip()
        if normalized and normalized not in output and len(output) < bounded:
            output.append(normalized)

    for row in ranked_legacy:
        add(row.get("symbol"))
    for symbol in base:
        add(symbol)
    return output[:bounded]


def merge_restored_signal(
    base: dict[str, Any],
    legacy: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(legacy, dict) or not legacy:
        return base

    result = copy.deepcopy(base)
    result["legacy_restoration"] = copy.deepcopy(legacy)

    ranked = dict(result.get("ranked_opportunity") or {})
    legacy_score = max(-1.0, min(1.0, _n(legacy.get("score"))))
    legacy_confidence = max(0.0, min(1.0, _n(legacy.get("confidence"))))

    existing_quality = max(
        0.0,
        min(1.0, _n(ranked.get("quality_multiplier"))),
    )
    restored_quality = min(
        1.0,
        max(
            existing_quality,
            abs(legacy_score) * 0.65 + legacy_confidence * 0.35,
        ),
    )
    ranked["quality_multiplier"] = restored_quality
    ranked["legacy_restoration_score"] = legacy_score
    ranked["legacy_restoration_confidence"] = legacy_confidence
    ranked["legacy_restoration_contributors"] = list(
        legacy.get("contributors") or []
    )[:32]
    result["ranked_opportunity"] = ranked

    assessments = copy.deepcopy(
        result.get("timeframe_assessments") or {}
    )
    for index, contribution in enumerate(
        legacy.get("contributions") or []
    ):
        if not isinstance(contribution, dict):
            continue
        direction = _direction(contribution.get("direction"))
        confidence = max(
            0.0,
            min(1.0, _n(contribution.get("confidence"))),
        )
        edge = max(0.0, _n(contribution.get("expected_edge_bps")))
        if direction == "flat" or confidence <= 0.0 or edge <= 0.0:
            continue
        metadata = (
            contribution.get("metadata")
            if isinstance(contribution.get("metadata"), dict)
            else {}
        )
        source = str(
            contribution.get("source") or f"legacy_{index}"
        ).replace(" ", "_")
        timeframe = str(contribution.get("timeframe") or "1m")
        key = f"legacy:{source}:{timeframe}:{index}"
        assessments[key] = {
            "direction": direction,
            "confidence": confidence,
            "expected_edge_bps": edge,
            "independently_qualified": False,
            "legacy_restoration": True,
            "source": source,
            "timeframe": timeframe,
            "legacy_gross_edge_bps": _n(
                metadata.get("gross_edge_bps")
            ),
            "legacy_modeled_round_trip_cost_bps": _n(
                metadata.get("modeled_round_trip_cost_bps")
            ),
            "legacy_conservative_net_edge_bps": _n(
                metadata.get("conservative_net_edge_bps")
            ),
            "legacy_economically_positive": (
                metadata.get("economically_positive") is True
            ),
            "execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    result["timeframe_assessments"] = assessments
    result["legacy_restoration_active"] = True
    result["legacy_direct_execution_authority"] = False
    result["live_authority"] = False
    return result


class LegacyEngineBridge:
    """Reconnect legacy LeanTrader intelligence to the modern signal bus.

    Legacy engines never submit orders directly here. They consume real public
    market data and contribute ranked evidence to the modern authenticated
    Testnet executor.
    """

    VERSION = "1.61.4"

    ENGINE_CLASS_TOKENS = (
        "engine",
        "trader",
        "scanner",
        "agent",
        "strategy",
        "brain",
        "intelligence",
        "swarm",
        "sniper",
        "arbitrage",
        "forex",
        "risk",
        "momentum",
        "predictor",
        "detector",
        "orchestrator",
    )

    def __init__(
        self,
        *,
        feed: Any,
        market_quote: str,
        min_quote_volume_usd: float,
        max_spread_bps: float,
        scan_limit: int = 64,
        analysis_batch_size: int = 12,
        cadence_seconds: float = 3.0,
        manifest_path: Path | None = None,
        arbitrage_venues: tuple[str, ...] = ("bybit", "okx"),
        minimum_round_trip_cost_bps: float = 30.0,
        minimum_positive_net_edge_bps: float = 5.0,
    ) -> None:
        self.feed = feed
        self.market_quote = str(market_quote).upper()
        self.min_quote_volume_usd = float(min_quote_volume_usd)
        self.max_spread_bps = float(max_spread_bps)
        self.scan_limit = max(8, min(256, int(scan_limit)))
        self.analysis_batch_size = max(
            2, min(self.scan_limit, int(analysis_batch_size))
        )
        self.cadence_seconds = max(1.0, float(cadence_seconds))
        self.manifest_path = manifest_path
        self.arbitrage_venues = tuple(
            dict.fromkeys(str(v).lower() for v in arbitrage_venues if v)
        )
        self.minimum_round_trip_cost_bps = max(
            0.0,
            float(minimum_round_trip_cost_bps),
        )
        self.minimum_positive_net_edge_bps = max(
            0.0,
            float(minimum_positive_net_edge_bps),
        )

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._cursor = 0
        self._discovery: list[dict[str, Any]] = []
        self._signals: dict[str, dict[str, Any]] = {}
        self._last_discovery_at = 0.0
        self._last_success_at = 0.0
        self._last_error: str | None = None
        self.cycles = 0
        self.symbols_analyzed = 0
        self.engine_calls = 0
        self.engine_failures = 0
        self.family_calls: dict[str, int] = {}
        self.family_failures: dict[str, int] = {}
        self.family_last_outputs: dict[str, Any] = {}
        self.loaded_families: dict[str, str] = {}
        self._components: dict[str, Any] = {}
        self._manifest_summary: dict[str, Any] = {}
        self._load_legacy_components()
        self._inventory_repository_engines()

    def _load_legacy_components(self) -> None:
        try:
            from ultra_scalping_engine import UltraScalpingEngine

            scalp = object.__new__(UltraScalpingEngine)
            scalp.MICRO_PIP_TARGET = 0.5
            scalp.MAX_PIP_TARGET = 2.0
            scalp.TIGHT_STOP_LOSS = 1.0
            scalp.MIN_CONFIDENCE = 0.75
            scalp.MAX_POSITIONS = 10
            scalp.logger = logging.getLogger(
                "leantrader.legacy.ultra_scalping"
            )
            self._components["ultra_scalping"] = scalp
            self.loaded_families["ultra_scalping"] = "active_real_data"
        except Exception as exc:
            self.loaded_families["ultra_scalping"] = (
                f"unavailable:{type(exc).__name__}"
            )

        try:
            from ultra_continuous_trading import (
                UltraContinuousTradingOrchestrator,
            )

            continuous = object.__new__(
                UltraContinuousTradingOrchestrator
            )
            continuous.logger = logging.getLogger(
                "leantrader.legacy.ultra_continuous"
            )
            self._components["ultra_continuous"] = continuous
            self.loaded_families["ultra_continuous"] = (
                "active_strategy_logic_real_data"
            )
        except Exception as exc:
            self.loaded_families["ultra_continuous"] = (
                f"unavailable:{type(exc).__name__}"
            )

        try:
            from ultra_swarm_consciousness import SwarmAgent

            agents = {}
            for specialization in (
                "scalping",
                "arbitrage",
                "trend",
                "black_swan",
                "general",
            ):
                agents[specialization] = SwarmAgent(
                    f"restored_{specialization}",
                    specialization,
                    None,
                )
            self._components["legacy_swarm"] = agents
            self.loaded_families["legacy_swarm"] = (
                "active_specializations_real_data"
            )
        except Exception as exc:
            self.loaded_families["legacy_swarm"] = (
                f"unavailable:{type(exc).__name__}"
            )

        try:
            from ultra_quantum_intelligence import (
                AdaptiveMarketRegimeDetector,
                FractalResonanceDetector,
                MicrostructureDecoder,
                QuantumMomentumOscillator,
            )

            self._components["quantum_microstructure"] = (
                MicrostructureDecoder()
            )
            self._components["quantum_momentum"] = (
                QuantumMomentumOscillator()
            )
            self._components["quantum_regime"] = (
                AdaptiveMarketRegimeDetector()
            )
            self._components["quantum_fractal"] = (
                FractalResonanceDetector()
            )
            self.loaded_families["ultra_quantum"] = (
                "active_microstructure_momentum_regime_fractal_real_data"
            )
        except Exception as exc:
            self.loaded_families["ultra_quantum"] = (
                f"unavailable:{type(exc).__name__}"
            )

        try:
            from ultra_god_mode import (
                FractalMarketAnalyzer,
                QuantumPricePredictor,
            )

            self._components["god_quantum_price"] = (
                QuantumPricePredictor()
            )
            self._components["god_fractal"] = (
                FractalMarketAnalyzer()
            )
            self.loaded_families["ultra_god_mode"] = (
                "active_quantum_price_and_fractal_real_data;"
                "simulated_moneyflow_and_random_swarm_excluded"
            )
        except Exception as exc:
            self.loaded_families["ultra_god_mode"] = (
                f"unavailable:{type(exc).__name__}"
            )

        try:
            from ultra_fluid_mechanics import FluidMechanicsEngine

            self._components["fluid_mechanics"] = (
                FluidMechanicsEngine(None, None)
            )
            self.loaded_families["ultra_fluid_mechanics"] = (
                "active_real_ohlcv_intelligence"
            )
        except Exception as exc:
            self.loaded_families["ultra_fluid_mechanics"] = (
                f"unavailable:{type(exc).__name__}"
            )

        try:
            from ultra_forex_master import (
                ForexPatternRecognizer,
                MultiTimeframeAnalyzer,
            )

            self._components["legacy_pattern_recognizer"] = (
                ForexPatternRecognizer()
            )
            self._components["legacy_mtf_analyzer"] = (
                MultiTimeframeAnalyzer()
            )
            self.loaded_families["legacy_pattern_mtf"] = (
                "active_real_ohlcv_pattern_and_multitimeframe"
            )
        except Exception as exc:
            self.loaded_families["legacy_pattern_mtf"] = (
                f"unavailable:{type(exc).__name__}"
            )

        try:
            from scanners.arbitrage import cross_exchange_spreads

            self._components["cross_exchange_spreads"] = (
                cross_exchange_spreads
            )
            self.loaded_families["legacy_arbitrage"] = (
                "available_real_public_scanner;"
                "modern_cross_venue_arbitrage_runtime_preferred"
            )
        except Exception as exc:
            self.loaded_families["legacy_arbitrage"] = (
                f"unavailable:{type(exc).__name__}"
            )

        self.loaded_families["legacy_testnet_trader"] = (
            "preserved_but_simulated_market_excluded;"
            "authenticated_bybit_testnet_executor_active"
        )
        self.loaded_families["legacy_backtest"] = (
            "preserved_but_synthetic_history_loader_excluded;"
            "modern_prospective_validation_and_research_active"
        )
        self.loaded_families["legacy_multi_platform"] = (
            "preserved_but_random_market_sources_excluded;"
            "real_cross_venue_engine_preferred"
        )
        self.loaded_families["november_growth"] = (
            "preserved_goal_manager_not_execution_authority;"
            "modern_capital_growth_engine_active"
        )
        self.loaded_families["ultra_moon_spotter"] = (
            "preserved_external_research_engine;"
            "not_allowed_to_inject_unverified_or_placeholder_market_data"
        )

    def _inventory_repository_engines(self) -> None:
        root = Path("/app")
        if not root.exists():
            return
        classes: list[dict[str, str]] = []
        files_scanned = 0
        for path in root.rglob("*.py"):
            if any(
                part in {".git", ".venv", "venv", "__pycache__"}
                for part in path.parts
            ):
                continue
            try:
                text = path.read_text(
                    encoding="utf-8", errors="ignore"
                )
                tree = ast.parse(text)
            except Exception:
                continue
            files_scanned += 1
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                lower = node.name.lower()
                if any(token in lower for token in self.ENGINE_CLASS_TOKENS):
                    classes.append(
                        {
                            "class": node.name,
                            "path": str(path.relative_to(root)),
                        }
                    )
        classes.sort(key=lambda row: (row["path"], row["class"]))
        summary = {
            "version": self.VERSION,
            "files_scanned": files_scanned,
            "engine_like_classes": len(classes),
            "classes": classes,
            "loaded_families": dict(self.loaded_families),
            "generated_at": time.time(),
            "direct_execution_authority": False,
            "live_authority": False,
        }
        self._manifest_summary = {
            key: value
            for key, value in summary.items()
            if key != "classes"
        }
        if self.manifest_path is not None:
            try:
                self.manifest_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                temporary = self.manifest_path.with_suffix(
                    self.manifest_path.suffix + ".tmp"
                )
                temporary.write_text(
                    json.dumps(summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                temporary.replace(self.manifest_path)
            except Exception:
                pass

    @staticmethod
    def _frame_payload(
        frame: Any,
        *,
        bid: float,
        ask: float,
    ) -> dict[str, Any]:
        closes = [
            _n(value)
            for value in list(frame["close"])
            if _n(value) > 0.0
        ]
        if not closes:
            return {}
        return {
            "close": closes[-1],
            "prices": closes,
            "bid": bid,
            "ask": ask,
        }

    @staticmethod
    def _run_async(awaitable: Any) -> Any:
        return asyncio.run(awaitable)

    def _append(
        self,
        output: list[dict[str, Any]],
        *,
        source: str,
        timeframe: str,
        direction: Any,
        confidence: Any,
        expected_edge_bps: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        normalized_direction = _direction(direction)
        conf = max(0.0, min(1.0, _n(confidence)))
        gross_edge = max(0.0, _n(expected_edge_bps))

        if normalized_direction == "flat" or conf <= 0.0:
            return

        conservative_net_edge = max(
            0.0,
            gross_edge - self.minimum_round_trip_cost_bps,
        )
        economically_positive = bool(
            conservative_net_edge
            >= self.minimum_positive_net_edge_bps
        )

        handoff_edge = (
            conservative_net_edge
            if economically_positive
            else 0.0
        )

        meta = copy.deepcopy(metadata or {})
        meta.update(
            {
                "gross_edge_bps": gross_edge,
                "modeled_round_trip_cost_bps": (
                    self.minimum_round_trip_cost_bps
                ),
                "minimum_positive_net_edge_bps": (
                    self.minimum_positive_net_edge_bps
                ),
                "conservative_net_edge_bps": (
                    conservative_net_edge
                ),
                "economically_positive": (
                    economically_positive
                ),
            }
        )

        output.append(
            {
                "source": source,
                "timeframe": timeframe,
                "direction": normalized_direction,
                "confidence": conf,
                "expected_edge_bps": handoff_edge,
                "metadata": meta,
            }
        )

    def _record_family_call(
        self,
        family: str,
        output: dict[str, Any] | None = None,
    ) -> None:
        self.family_calls[family] = (
            self.family_calls.get(family, 0) + 1
        )
        if output is not None:
            self.family_last_outputs[family] = copy.deepcopy(
                output
            )

    def _record_family_failure(
        self,
        family: str,
    ) -> None:
        self.family_failures[family] = (
            self.family_failures.get(family, 0) + 1
        )

    def _analyze_symbol(
        self,
        row: dict[str, Any],
    ) -> dict[str, Any]:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            return {}

        order_book = self.feed.order_book(symbol, limit=10)
        bids = order_book.get("bids") or []
        asks = order_book.get("asks") or []
        bid = _n(bids[0][0]) if bids else 0.0
        ask = _n(asks[0][0]) if asks else 0.0

        frame_1m = self.feed.candles(symbol, "1m", 100)
        frame_5m = self.feed.candles(symbol, "5m", 100)
        frame_15m = self.feed.candles(symbol, "15m", 100)

        payload_1m = self._frame_payload(
            frame_1m, bid=bid, ask=ask
        )
        payload_5m = self._frame_payload(
            frame_5m, bid=bid, ask=ask
        )
        payload_15m = self._frame_payload(
            frame_15m, bid=bid, ask=ask
        )
        contributions: list[dict[str, Any]] = []

        scalp = self._components.get("ultra_scalping")
        if scalp is not None:
            try:
                signals = self._run_async(
                    scalp._analyze_scalp_opportunities(
                        symbol, "M1", payload_1m
                    )
                )
                self.engine_calls += 1
                signal_rows = list(signals or [])
                self._record_family_call(
                    "ultra_scalping",
                    {"signals": len(signal_rows)},
                )

                prices = payload_1m.get("prices") or []
                for signal in signal_rows:
                    strategy = str(
                        getattr(signal, "strategy", "") or ""
                    )

                    if strategy == "spread_capture":
                        continue

                    direction = (
                        "long"
                        if _n(signal.target_price)
                        > _n(signal.entry_price)
                        else "short"
                    )

                    lookback = (
                        10
                        if strategy == "micro_momentum"
                        else 5
                    )
                    gross_edge = 0.0
                    if (
                        len(prices) >= lookback
                        and _n(prices[-lookback]) > 0.0
                    ):
                        gross_edge = abs(
                            _n(prices[-1])
                            / _n(prices[-lookback])
                            - 1.0
                        ) * 10_000.0

                    self._append(
                        contributions,
                        source=f"ultra_scalping.{strategy}",
                        timeframe="1m",
                        direction=direction,
                        confidence=signal.confidence,
                        expected_edge_bps=gross_edge,
                        metadata={
                            "legacy_nominal_target_bps": _n(
                                signal.profit_pips
                            ),
                            "observed_move_bps": gross_edge,
                        },
                    )
            except Exception:
                self.engine_failures += 1

        continuous = self._components.get("ultra_continuous")
        if continuous is not None:
            calls = (
                (
                    "continuous_scalping",
                    "1m",
                    continuous._detect_scalping_opportunities,
                    payload_1m,
                    0.40,
                ),
                (
                    "continuous_momentum",
                    "5m",
                    continuous._detect_momentum_opportunities,
                    payload_5m,
                    0.30,
                ),
                (
                    "continuous_mean_reversion",
                    "15m",
                    continuous._detect_mean_reversion_opportunities,
                    payload_15m,
                    0.20,
                ),
                (
                    "continuous_breakout",
                    "5m",
                    continuous._detect_breakout_opportunities,
                    payload_5m,
                    0.30,
                ),
            )
            for name, timeframe, method, payload, weight in calls:
                try:
                    opportunities = self._run_async(
                        method(
                            symbol,
                            (
                                "M1"
                                if timeframe == "1m"
                                else "M5"
                                if timeframe == "5m"
                                else "M15"
                            ),
                            payload,
                            weight,
                        )
                    )
                    self.engine_calls += 1
                    opportunity_rows = list(
                        opportunities or []
                    )
                    self._record_family_call(
                        name,
                        {"signals": len(opportunity_rows)},
                    )

                    for opportunity in opportunity_rows:
                        prices = [
                            _n(value)
                            for value in (
                                payload.get("prices") or []
                            )
                            if _n(value) > 0.0
                        ]
                        direction = _direction(
                            opportunity.get("direction")
                        )
                        gross_edge = _n(
                            opportunity.get(
                                "expected_edge_bps"
                            )
                        )

                        if (
                            name == "continuous_scalping"
                            and len(prices) >= 5
                        ):
                            move = (
                                prices[-1] / prices[-5] - 1.0
                            )
                            direction = (
                                "long"
                                if move > 0.0
                                else "short"
                                if move < 0.0
                                else "flat"
                            )
                            gross_edge = abs(move) * 10_000.0

                        elif (
                            name == "continuous_momentum"
                            and len(prices) >= 20
                        ):
                            short_mean = (
                                sum(prices[-5:]) / 5.0
                            )
                            long_mean = (
                                sum(prices[-20:]) / 20.0
                            )
                            move = (
                                short_mean / long_mean - 1.0
                                if long_mean > 0.0
                                else 0.0
                            )
                            direction = (
                                "long"
                                if move > 0.0
                                else "short"
                                if move < 0.0
                                else "flat"
                            )
                            gross_edge = abs(move) * 10_000.0

                        elif (
                            name
                            == "continuous_mean_reversion"
                            and len(prices) >= 50
                        ):
                            mean_price = (
                                sum(prices[-50:]) / 50.0
                            )
                            deviation = (
                                prices[-1] / mean_price - 1.0
                                if mean_price > 0.0
                                else 0.0
                            )
                            direction = (
                                "short"
                                if deviation > 0.0
                                else "long"
                                if deviation < 0.0
                                else "flat"
                            )
                            gross_edge = (
                                abs(deviation) * 10_000.0
                            )

                        elif (
                            name == "continuous_breakout"
                            and len(prices) >= 20
                        ):
                            prior = prices[-20:-1]
                            prior_high = max(prior)
                            prior_low = min(prior)
                            current = prices[-1]

                            if current > prior_high:
                                direction = "long"
                                gross_edge = (
                                    current / prior_high - 1.0
                                ) * 10_000.0
                            elif current < prior_low:
                                direction = "short"
                                gross_edge = (
                                    prior_low / current - 1.0
                                ) * 10_000.0

                        self._append(
                            contributions,
                            source=name,
                            timeframe=timeframe,
                            direction=direction,
                            confidence=opportunity.get(
                                "confidence"
                            ),
                            expected_edge_bps=gross_edge,
                            metadata={
                                "strategy": opportunity.get(
                                    "strategy"
                                ),
                                "legacy_nominal_target_bps": (
                                    _n(
                                        opportunity.get(
                                            "profit_target"
                                        )
                                    )
                                ),
                                "observed_move_bps": gross_edge,
                            },
                        )
                except Exception:
                    self.engine_failures += 1

        agents = self._components.get("legacy_swarm") or {}
        swarm_calls = (
            ("scalping", "1m", "_analyze_scalping", frame_1m),
            ("arbitrage", "5m", "_analyze_arbitrage", frame_5m),
            ("trend", "15m", "_analyze_trend", frame_15m),
            ("black_swan", "1m", "_analyze_black_swan", frame_1m),
            ("general", "5m", "_analyze_general", frame_5m),
        )
        for specialization, timeframe, method_name, frame in swarm_calls:
            agent = agents.get(specialization)
            if agent is None:
                continue
            try:
                signal = self._run_async(
                    getattr(agent, method_name)(
                        frame, symbol, timeframe
                    )
                )
                self.engine_calls += 1
                if signal is not None:
                    if signal.signal_type == "black_swan":
                        # Black-swan evidence is defensive context, not
                        # a direct long/short entry signal.
                        continue
                    self._append(
                        contributions,
                        source=f"legacy_swarm.{specialization}",
                        timeframe=timeframe,
                        direction=signal.signal_type,
                        confidence=signal.confidence,
                        expected_edge_bps=max(
                            1.0,
                            abs(
                                _n(
                                    (
                                        signal.metadata
                                        or {}
                                    ).get("momentum")
                                )
                            )
                            * 10_000.0,
                        ),
                    )
            except Exception:
                self.engine_failures += 1

        decoder = self._components.get(
            "quantum_microstructure"
        )
        if decoder is not None and bids and asks:
            try:
                decoded = decoder.decode_hidden_liquidity(
                    order_book
                )
                self.engine_calls += 1
                intent = str(
                    (decoded or {}).get("mm_intent") or ""
                ).lower()
                if intent in {"accumulation", "distribution"}:
                    self._append(
                        contributions,
                        source="ultra_quantum.microstructure_decoder",
                        timeframe="micro",
                        direction=(
                            "long"
                            if intent == "accumulation"
                            else "short"
                        ),
                        confidence=0.60,
                        expected_edge_bps=max(
                            1.0,
                            abs(
                                _n(
                                    (decoded or {}).get(
                                        "imbalance"
                                    )
                                )
                            )
                            * 10.0,
                        ),
                    )
            except Exception:
                self.engine_failures += 1

        momentum = self._components.get("quantum_momentum")
        if momentum is not None:
            try:
                result = momentum.calculate(
                    frame_1m["close"].to_numpy()
                )
                self.engine_calls += 1
                probability = _n(
                    (result or {}).get("shift_probability")
                )
                direction = str(
                    (result or {}).get("shift_direction") or ""
                ).lower()
                if probability > 0.0 and direction in {
                    "bullish",
                    "bearish",
                }:
                    self._append(
                        contributions,
                        source="ultra_quantum.momentum",
                        timeframe="1m",
                        direction=direction,
                        confidence=probability,
                        expected_edge_bps=max(
                            1.0,
                            abs(
                                _n(
                                    (result or {}).get(
                                        "shift_magnitude"
                                    )
                                )
                            ),
                        ),
                    )
            except Exception:
                self.engine_failures += 1

        # --------------------------------------------------------
        # Adaptive regime intelligence
        # --------------------------------------------------------
        regime_detector = self._components.get(
            "quantum_regime"
        )
        if regime_detector is not None:
            try:
                regime_result = regime_detector.detect_regime(
                    frame_15m
                )
                self.engine_calls += 1
                indicators = (
                    regime_result.get("indicators") or {}
                )
                momentum_value = _n(
                    indicators.get("momentum")
                )
                regime_name = str(
                    regime_result.get("current_regime") or ""
                ).lower()

                if regime_name == "crash":
                    regime_direction = "short"
                elif momentum_value > 0.0:
                    regime_direction = "long"
                elif momentum_value < 0.0:
                    regime_direction = "short"
                else:
                    regime_direction = "flat"

                regime_conf = max(
                    0.0,
                    min(
                        1.0,
                        _n(regime_result.get("confidence")),
                    ),
                )

                self._record_family_call(
                    "ultra_quantum.regime",
                    {
                        "regime": regime_name,
                        "confidence": regime_conf,
                        "momentum": momentum_value,
                    },
                )

                self._append(
                    contributions,
                    source="ultra_quantum.regime",
                    timeframe="15m",
                    direction=regime_direction,
                    confidence=regime_conf,
                    expected_edge_bps=min(
                        100.0,
                        max(
                            1.0,
                            abs(momentum_value) * 10_000.0,
                        ),
                    ),
                    metadata={
                        "regime": regime_name,
                        "regime_stability": _n(
                            regime_result.get(
                                "regime_stability"
                            )
                        ),
                    },
                )
            except Exception:
                self.engine_failures += 1
                self._record_family_failure(
                    "ultra_quantum.regime"
                )

        # --------------------------------------------------------
        # Quantum multi-timeframe fractal resonance
        # --------------------------------------------------------
        quantum_fractal = self._components.get(
            "quantum_fractal"
        )
        if quantum_fractal is not None:
            try:
                resonance = quantum_fractal.detect_resonance(
                    {
                        1: frame_1m,
                        5: frame_5m,
                        15: frame_15m,
                    }
                )
                self.engine_calls += 1

                raw_direction = str(
                    resonance.get("resonance_type")
                    or resonance.get("cascade_direction")
                    or ""
                ).lower()

                if raw_direction in {
                    "bullish",
                    "up",
                    "long",
                }:
                    resonance_direction = "long"
                elif raw_direction in {
                    "bearish",
                    "down",
                    "short",
                }:
                    resonance_direction = "short"
                else:
                    resonance_direction = "flat"

                resonance_conf = max(
                    _n(resonance.get("resonance_score")),
                    _n(
                        resonance.get(
                            "cascade_probability"
                        )
                    ),
                )
                resonance_conf = max(
                    0.0,
                    min(1.0, resonance_conf),
                )
                cascade_magnitude = abs(
                    _n(
                        resonance.get(
                            "cascade_magnitude"
                        )
                    )
                )

                self._record_family_call(
                    "ultra_quantum.fractal_resonance",
                    {
                        "direction": resonance_direction,
                        "confidence": resonance_conf,
                        "cascade_magnitude": (
                            cascade_magnitude
                        ),
                    },
                )

                self._append(
                    contributions,
                    source=(
                        "ultra_quantum."
                        "fractal_resonance"
                    ),
                    timeframe="1m+5m+15m",
                    direction=resonance_direction,
                    confidence=resonance_conf,
                    expected_edge_bps=min(
                        100.0,
                        max(
                            1.0,
                            cascade_magnitude
                            * 10_000.0,
                            resonance_conf * 10.0,
                        ),
                    ),
                )
            except Exception:
                self.engine_failures += 1
                self._record_family_failure(
                    "ultra_quantum.fractal_resonance"
                )

        # --------------------------------------------------------
        # Existing God Mode components that are based on real price
        # data only. Simulated smart-money/options pieces remain
        # excluded.
        # --------------------------------------------------------
        god_fractal = self._components.get(
            "god_fractal"
        )
        if god_fractal is not None:
            try:
                closes = frame_1m[
                    "close"
                ].to_numpy()
                fractal_result = (
                    god_fractal.analyze_fractals(
                        closes
                    )
                )
                self.engine_calls += 1

                state = str(
                    fractal_result.get(
                        "market_state"
                    )
                    or ""
                ).lower()
                predictability = max(
                    0.0,
                    min(
                        1.0,
                        _n(
                            fractal_result.get(
                                "predictability"
                            )
                        ),
                    ),
                )

                recent_return = 0.0
                if len(closes) >= 20 and closes[-20] > 0:
                    recent_return = (
                        closes[-1] / closes[-20] - 1.0
                    )

                if "mean_revert" in state:
                    fractal_direction = (
                        "short"
                        if recent_return > 0.0
                        else "long"
                        if recent_return < 0.0
                        else "flat"
                    )
                elif (
                    "trend" in state
                    or "persistent" in state
                ):
                    fractal_direction = (
                        "long"
                        if recent_return > 0.0
                        else "short"
                        if recent_return < 0.0
                        else "flat"
                    )
                else:
                    fractal_direction = "flat"

                self._record_family_call(
                    "ultra_god_mode.fractal",
                    {
                        "state": state,
                        "predictability": (
                            predictability
                        ),
                        "recent_return": recent_return,
                    },
                )

                self._append(
                    contributions,
                    source="ultra_god_mode.fractal",
                    timeframe="1m",
                    direction=fractal_direction,
                    confidence=predictability,
                    expected_edge_bps=min(
                        100.0,
                        max(
                            1.0,
                            abs(recent_return)
                            * 10_000.0,
                        ),
                    ),
                    metadata={
                        "market_state": state,
                        "chaos_level": (
                            fractal_result.get(
                                "chaos_level"
                            )
                        ),
                    },
                )
            except Exception:
                self.engine_failures += 1
                self._record_family_failure(
                    "ultra_god_mode.fractal"
                )

        # Quantum price forecast is computationally heavier.
        # Keep it active, but do not repeat it on every 3-second
        # pass for every symbol.
        god_quantum = self._components.get(
            "god_quantum_price"
        )
        if (
            god_quantum is not None
            and self.cycles % 3 == 0
        ):
            try:
                closes = frame_1m[
                    "close"
                ].to_numpy()
                forecast = (
                    god_quantum.quantum_forecast(
                        closes
                    )
                )
                self.engine_calls += 1

                current_price = _n(closes[-1])
                expected_price = _n(
                    forecast.get(
                        "most_likely_price"
                    )
                )
                if expected_price > current_price:
                    forecast_direction = "long"
                elif (
                    expected_price > 0.0
                    and expected_price < current_price
                ):
                    forecast_direction = "short"
                else:
                    forecast_direction = "flat"

                forecast_conf = max(
                    0.0,
                    min(
                        1.0,
                        _n(
                            forecast.get(
                                "quantum_probability"
                            )
                        ),
                    ),
                )
                forecast_edge = 0.0
                if current_price > 0.0:
                    forecast_edge = abs(
                        expected_price
                        / current_price
                        - 1.0
                    ) * 10_000.0

                self._record_family_call(
                    "ultra_god_mode.quantum_price",
                    {
                        "direction": (
                            forecast_direction
                        ),
                        "confidence": forecast_conf,
                        "edge_bps": forecast_edge,
                    },
                )

                self._append(
                    contributions,
                    source=(
                        "ultra_god_mode."
                        "quantum_price"
                    ),
                    timeframe="1m",
                    direction=forecast_direction,
                    confidence=forecast_conf,
                    expected_edge_bps=min(
                        200.0,
                        max(1.0, forecast_edge),
                    ),
                )
            except Exception:
                self.engine_failures += 1
                self._record_family_failure(
                    "ultra_god_mode.quantum_price"
                )

        # --------------------------------------------------------
        # Fluid-mechanics / sentinel price-flow intelligence
        # --------------------------------------------------------
        fluid_engine = self._components.get(
            "fluid_mechanics"
        )
        if fluid_engine is not None:
            try:
                fluid_state = self._run_async(
                    fluid_engine.analyze_fluid_dynamics(
                        frame_1m
                    )
                )
                self.engine_calls += 1

                flow = str(
                    getattr(
                        fluid_state,
                        "flow_direction",
                        "",
                    )
                ).lower()
                velocity = _n(
                    getattr(
                        fluid_state,
                        "velocity",
                        0.0,
                    )
                )
                turbulence = _n(
                    getattr(
                        fluid_state,
                        "turbulence_level",
                        0.0,
                    )
                )

                if flow == "up":
                    fluid_direction = "long"
                elif flow == "down":
                    fluid_direction = "short"
                else:
                    fluid_direction = "flat"

                fluid_conf = min(
                    0.95,
                    max(
                        0.50,
                        0.50
                        + min(
                            0.45,
                            abs(velocity) * 50.0,
                        ),
                    ),
                )

                self._record_family_call(
                    "ultra_fluid_mechanics",
                    {
                        "flow": flow,
                        "velocity": velocity,
                        "turbulence": turbulence,
                    },
                )

                self._append(
                    contributions,
                    source="ultra_fluid_mechanics",
                    timeframe="1m",
                    direction=fluid_direction,
                    confidence=fluid_conf,
                    expected_edge_bps=min(
                        100.0,
                        max(
                            1.0,
                            abs(velocity) * 10_000.0,
                        ),
                    ),
                    metadata={
                        "turbulence": turbulence,
                        "pressure": _n(
                            getattr(
                                fluid_state,
                                "pressure",
                                0.0,
                            )
                        ),
                    },
                )
            except Exception:
                self.engine_failures += 1
                self._record_family_failure(
                    "ultra_fluid_mechanics"
                )

        # --------------------------------------------------------
        # Existing price-pattern engine
        # --------------------------------------------------------
        pattern_engine = self._components.get(
            "legacy_pattern_recognizer"
        )
        if pattern_engine is not None:
            try:
                patterns = (
                    pattern_engine.detect_patterns(
                        frame_15m
                    )
                    or []
                )
                self.engine_calls += 1

                best_direction = "flat"
                best_confidence = 0.0
                best_pattern = ""

                bullish_names = {
                    "double_bottom",
                    "hammer",
                    "morning_star",
                    "three_white_soldiers",
                }
                bearish_names = {
                    "double_top",
                    "head_shoulders",
                    "shooting_star",
                    "evening_star",
                    "three_black_crows",
                }

                for pattern in patterns:
                    name = str(
                        pattern.get("name") or ""
                    ).lower()
                    signal_name = str(
                        pattern.get("signal") or ""
                    ).lower()
                    reliability = max(
                        0.0,
                        min(
                            1.0,
                            _n(
                                pattern.get(
                                    "reliability"
                                )
                            ),
                        ),
                    )

                    direction = "flat"
                    if (
                        "bullish" in signal_name
                        or name in bullish_names
                    ):
                        direction = "long"
                    elif (
                        "bearish" in signal_name
                        or name in bearish_names
                    ):
                        direction = "short"
                    elif name == "engulfing":
                        direction = (
                            "long"
                            if _n(
                                frame_15m[
                                    "close"
                                ].iloc[-1]
                            )
                            >= _n(
                                frame_15m[
                                    "open"
                                ].iloc[-1]
                            )
                            else "short"
                        )

                    if (
                        direction != "flat"
                        and reliability
                        > best_confidence
                    ):
                        best_direction = direction
                        best_confidence = reliability
                        best_pattern = name

                self._record_family_call(
                    "legacy_pattern_recognizer",
                    {
                        "patterns": len(patterns),
                        "best_pattern": best_pattern,
                        "confidence": best_confidence,
                    },
                )

                recent_close = frame_15m[
                    "close"
                ].to_numpy()
                pattern_edge = 1.0
                if (
                    len(recent_close) >= 20
                    and recent_close[-20] > 0
                ):
                    pattern_edge = abs(
                        recent_close[-1]
                        / recent_close[-20]
                        - 1.0
                    ) * 10_000.0

                self._append(
                    contributions,
                    source=(
                        "legacy_pattern_recognizer."
                        + (
                            best_pattern
                            or "pattern"
                        )
                    ),
                    timeframe="15m",
                    direction=best_direction,
                    confidence=best_confidence,
                    expected_edge_bps=min(
                        100.0,
                        max(1.0, pattern_edge),
                    ),
                )
            except Exception:
                self.engine_failures += 1
                self._record_family_failure(
                    "legacy_pattern_recognizer"
                )

        # --------------------------------------------------------
        # Existing multi-timeframe confluence engine
        # --------------------------------------------------------
        mtf_engine = self._components.get(
            "legacy_mtf_analyzer"
        )
        if mtf_engine is not None:
            try:
                mtf_result = (
                    mtf_engine.analyze_all_timeframes(
                        symbol,
                        {
                            "M1": frame_1m,
                            "M5": frame_5m,
                            "M15": frame_15m,
                        },
                    )
                )
                self.engine_calls += 1

                dominant = str(
                    mtf_result.get(
                        "dominant_trend"
                    )
                    or ""
                ).lower()
                if dominant in {
                    "bullish",
                    "up",
                    "long",
                }:
                    mtf_direction = "long"
                elif dominant in {
                    "bearish",
                    "down",
                    "short",
                }:
                    mtf_direction = "short"
                else:
                    mtf_direction = "flat"

                raw_confluence = abs(
                    _n(
                        mtf_result.get(
                            "confluence_score"
                        )
                    )
                )
                mtf_conf = min(
                    1.0,
                    (
                        raw_confluence
                        if raw_confluence <= 1.0
                        else raw_confluence / 100.0
                    ),
                )

                self._record_family_call(
                    "legacy_multitimeframe",
                    {
                        "dominant_trend": dominant,
                        "confluence": raw_confluence,
                    },
                )

                self._append(
                    contributions,
                    source="legacy_multitimeframe",
                    timeframe="1m+5m+15m",
                    direction=mtf_direction,
                    confidence=mtf_conf,
                    expected_edge_bps=min(
                        100.0,
                        max(
                            1.0,
                            mtf_conf * 30.0,
                        ),
                    ),
                )
            except Exception:
                self.engine_failures += 1
                self._record_family_failure(
                    "legacy_multitimeframe"
                )

        long_support = sum(
            _n(item.get("confidence"))
            * _n(item.get("expected_edge_bps"))
            for item in contributions
            if item.get("direction") == "long"
        )
        short_support = sum(
            _n(item.get("confidence"))
            * _n(item.get("expected_edge_bps"))
            for item in contributions
            if item.get("direction") == "short"
        )
        denominator = max(
            1.0,
            long_support + short_support,
        )
        signed = (long_support - short_support) / denominator
        confidence = min(
            1.0,
            max(
                [
                    _n(item.get("confidence"))
                    for item in contributions
                ]
                or [0.0]
            ),
        )
        return {
            "symbol": symbol,
            "timestamp": time.time(),
            "score": max(-1.0, min(1.0, signed)),
            "confidence": confidence,
            "contributors": sorted(
                {
                    str(item.get("source"))
                    for item in contributions
                    if item.get("source")
                }
            ),
            "contributions": contributions,
            "quote_volume_usd": _n(
                row.get("quote_volume_usd")
            ),
            "absolute_percentage_24h": _n(
                row.get("absolute_percentage_24h")
            ),
            "spread_bps": _n(row.get("spread_bps")),
            "real_public_market_data": True,
            "legacy_simulated_execution_used": False,
            "legacy_mock_market_data_used": False,
            "direct_execution_authority": False,
            "testnet_authority": False,
            "live_authority": False,
        }

    def _refresh_discovery(self, now: float) -> None:
        if (
            self._discovery
            and now - self._last_discovery_at < 20.0
        ):
            return
        payload = self.feed.discover_markets(
            quote=self.market_quote,
            min_quote_volume_usd=self.min_quote_volume_usd,
            max_spread_bps=self.max_spread_bps,
        )
        rows = [
            dict(row)
            for row in payload.get("candidates") or []
            if isinstance(row, dict)
        ]
        rows.sort(
            key=lambda row: (
                _n(row.get("absolute_percentage_24h")),
                _n(row.get("quote_volume_usd")),
                -_n(row.get("spread_bps"), 1_000_000.0),
            ),
            reverse=True,
        )
        self._discovery = rows[: self.scan_limit]
        self._last_discovery_at = now

    def _step(self) -> None:
        now = time.time()
        self._refresh_discovery(now)
        if not self._discovery:
            return

        batch: list[dict[str, Any]] = []
        total = len(self._discovery)
        for offset in range(
            min(self.analysis_batch_size, total)
        ):
            batch.append(
                self._discovery[
                    (self._cursor + offset) % total
                ]
            )
        self._cursor = (
            self._cursor + len(batch)
        ) % max(1, total)

        updates: dict[str, dict[str, Any]] = {}
        for row in batch:
            symbol = str(row.get("symbol") or "").upper()
            try:
                signal = self._analyze_symbol(row)
            except Exception as exc:
                self.engine_failures += 1
                self._last_error = (
                    f"{symbol}:{type(exc).__name__}"
                )
                continue
            if signal:
                updates[symbol] = signal
                self.symbols_analyzed += 1

        with self._lock:
            self._signals.update(updates)
            stale_before = now - 180.0
            self._signals = {
                symbol: signal
                for symbol, signal in self._signals.items()
                if _n(signal.get("timestamp")) >= stale_before
            }
            self.cycles += 1
            self._last_success_at = now
            self._last_error = None

            if self.cycles == 1 or self.cycles % 10 == 0:
                LOGGER.info(
                    "legacy-restoration cycle=%s symbols=%s "
                    "cached=%s engine_calls=%s failures=%s "
                    "family_calls=%s family_failures=%s",
                    self.cycles,
                    self.symbols_analyzed,
                    len(self._signals),
                    self.engine_calls,
                    self.engine_failures,
                    dict(self.family_calls),
                    dict(self.family_failures),
                )

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._step()
            except Exception as exc:
                with self._lock:
                    self._last_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    self.engine_failures += 1
            elapsed = time.monotonic() - started
            self._stop.wait(
                max(0.0, self.cadence_seconds - elapsed)
            )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="leantrader-legacy-engine-restoration",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)

    def candidates(
        self,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                copy.deepcopy(row)
                for row in self._signals.values()
            ]
        rows.sort(
            key=lambda row: (
                _n(row.get("score")),
                _n(row.get("confidence")),
                _n(row.get("absolute_percentage_24h")),
                _n(row.get("quote_volume_usd")),
            ),
            reverse=True,
        )
        return rows[: max(1, int(limit))]

    def signal(self, symbol: str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(
                self._signals.get(
                    str(symbol or "").upper(),
                    {},
                )
            )

    def health(self) -> dict[str, Any]:
        with self._lock:
            active_contributors = sorted(
                {
                    str(item.get("source"))
                    for signal in self._signals.values()
                    for item in (
                        signal.get("contributions") or []
                    )
                    if isinstance(item, dict)
                    and item.get("source")
                }
            )
            return {
                "version": self.VERSION,
                "running": bool(
                    self._thread is not None
                    and self._thread.is_alive()
                    and not self._stop.is_set()
                ),
                "cycles": self.cycles,
                "symbols_analyzed": self.symbols_analyzed,
                "signals_cached": len(self._signals),
                "engine_calls": self.engine_calls,
                "engine_failures": self.engine_failures,
                "last_success_at": self._last_success_at,
                "last_error": self._last_error,
                "loaded_families": dict(
                    self.loaded_families
                ),
                "family_calls": dict(
                    self.family_calls
                ),
                "family_failures": dict(
                    self.family_failures
                ),
                "family_last_outputs": copy.deepcopy(
                    self.family_last_outputs
                ),
                "active_contributors": (
                    active_contributors
                ),
                "manifest": dict(
                    self._manifest_summary
                ),
                "real_public_market_data": True,
                "legacy_simulated_execution_used": False,
                "legacy_mock_market_data_used": False,
                "direct_execution_authority": False,
                "testnet_authority": False,
                "live_authority": False,
            }


class RestoredSwarmService(ReadOnlySwarmService):
    """Current fast swarm plus restored legacy-intelligence contributors."""

    VERSION = "1.61.4"

    def __init__(
        self,
        *args: Any,
        legacy_bridge: LegacyEngineBridge,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.legacy_bridge = legacy_bridge

    def start(self) -> None:
        super().start()
        self.legacy_bridge.start()

    def stop(self) -> None:
        self.legacy_bridge.stop()
        super().stop()

    def collective_candidates(
        self,
        limit: int = 8,
    ) -> list[str]:
        base = super().collective_candidates(limit=limit)
        legacy = self.legacy_bridge.candidates(
            limit=max(limit, 32)
        )
        return merge_candidate_symbols(
            base,
            legacy,
            limit=limit,
        )

    def collective_signal(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        base = super().collective_signal(symbol)
        legacy = self.legacy_bridge.signal(symbol)
        return merge_restored_signal(base, legacy)

    def health(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = super().health(*args, **kwargs)
        payload["legacy_engine_restoration"] = (
            self.legacy_bridge.health()
        )
        payload["legacy_restoration_active"] = True
        payload["version"] = self.VERSION
        return payload
