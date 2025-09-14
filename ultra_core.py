"""
ultra_core.py
Ultra Reasoning, Scanning, Planning, and Learning Engine for God Mode Trading Bot
"""

import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401  # intentionally kept

import numpy as np
import pandas as pd

# local helpers
from ledger import log_entry

# New imports for enhancements
try:
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential

    lstm_model = Sequential([LSTM(50, input_shape=(10, 1)), Dense(1)])
    lstm_model.compile(loss="mean_squared_error", optimizer="adam")
except ImportError:
    lstm_model = None

# Quantum-inspired imports
try:
    from qiskit import Aer, QuantumCircuit, execute
except ImportError:
    Aer, execute, QuantumCircuit = None, None, None


class UltraCore:
    def ultra_advanced_cycle(self):
        """Run advanced ultra features (non-blocking where possible).

        This is intentionally conservative: side-effecting actions (orders) are guarded by
        the router.live flag. The routines below populate the knowledge base for
        later planning and offline analysis.
        """
        try:
            # On-chain analytics (best-effort)
            try:
                self.knowledge_base["onchain"] = self.scout.fetch_onchain_analytics(
                    "0x0000000000000000000000000000000000000000"
                )
            except Exception:
                self.knowledge_base["onchain"] = {}

            # Backtesting (quick sample run)
            try:
                self.knowledge_base["backtest"] = self.scout.run_backtest("trend", {"ema_fast": 20, "ema_slow": 50})
            except Exception:
                self.knowledge_base["backtest"] = None

            # Swarm intelligence and risk alerts
            signals = self.knowledge_base.get("last_signals", []) or []
            self.knowledge_base["swarm"] = self.scout.swarm_collaboration(signals) if signals else {}
            self.knowledge_base["risk_alerts"] = self.scout.detect_risk_alerts(signals) if signals else []

            # Broker API / RL / Dashboard / Voice are non-critical stubs
            for key, fn, arg in (
                ("broker_api", self.scout.broker_api_integration, "Binance"),
                ("rl", self.scout.reinforcement_learning_update, {"state": "init"}),
                ("voice_chat", self.scout.voice_chat_interface, "Hello bot, status?"),
            ):
                try:
                    self.knowledge_base[key] = fn(arg)
                except Exception:
                    self.knowledge_base[key] = None

            try:
                self.scout.update_dashboard(
                    {
                        "trades": signals,
                        "alerts": self.knowledge_base.get("risk_alerts", []),
                    }
                )
            except Exception:
                pass
        except Exception:
            # swallow to avoid killing the main loop
            if self.logger:
                self.logger.exception("ultra_advanced_cycle failed")

    def __init__(self, router, universe, logger=None):
        # restore previous behavior: use router directly
        self.router = router
        self.universe = universe
        self.logger = logger
        self.knowledge_base = {}
        self.performance_log = []
        self.last_update = time.time()
        from ultra_scout import UltraScout

        self.scout = UltraScout()
        # internal logger
        if self.logger is None:
            self.logger = logging.getLogger("UltraCore")
            self.logger.addHandler(logging.NullHandler())
        # small thread pool size for parallel fetches
        self._thread_pool_size = 6
        self.lstm_model = lstm_model
        try:
            import torch  # noqa: F401  # intentionally kept
            import torch_geometric as pyg

            self.gnn_model = pyg.nn.GCNConv(10, 1)  # Placeholder GNN
        except ImportError:
            self.gnn_model = None
        self.anomaly_detector = None  # Will be set in scout_all if available
        try:
            from qiskit import Aer, QuantumCircuit, execute  # noqa: F401  # intentionally kept

            self.quantum_backend = Aer.get_backend("qasm_simulator")
        except ImportError:
            self.quantum_backend = None
        self.retrade_count = 0
        self.max_retrades = 3
        # Force demo mode
        if os.getenv("TRADING_MODE", "demo").lower() == "demo":
            setattr(self.router, "live", False)

    def scan_markets(self):
        """Scan all supported markets for opportunities."""
        results: Dict[str, Any] = {}

        # Determine market list
        try:
            if self.universe and hasattr(self.universe, "get_all_markets"):
                markets = list(self.universe.get_all_markets())
            else:
                # fallback: sample from router markets
                markets = list(self.router.markets.keys())[:200]
        except Exception:
            markets = list(self.router.markets.keys())[:200]

        # threaded fetch to improve throughput
        lock = threading.Lock()

        def worker(market: str):
            try:
                data = self.router.safe_fetch_ohlcv(market)
                if data:
                    ana = self.analyze_market(data)
                    with lock:
                        results[market] = ana
            except Exception:
                with lock:
                    results[market] = None

        threads: List[threading.Thread] = []
        for i, m in enumerate(markets):
            t = threading.Thread(target=worker, args=(m,), daemon=True)
            threads.append(t)
            t.start()
            # throttle thread creation
            if len(threads) >= self._thread_pool_size:
                for tt in threads:
                    tt.join(timeout=5)
                threads = []

        # join remaining
        for tt in threads:
            tt.join(timeout=2)

        return results

    def analyze_market(self, ohlcv):
        """Advanced reasoning: pattern recognition, anomaly detection, regime analysis."""
        # Enhanced: LSTM-based regime prediction
        try:
            prices = np.array([x[4] for x in ohlcv if isinstance(x, (list, tuple)) and len(x) > 4])
            highs = np.array([x[2] for x in ohlcv if isinstance(x, (list, tuple)) and len(x) > 2])
            lows = np.array([x[3] for x in ohlcv if isinstance(x, (list, tuple)) and len(x) > 3])
            if len(prices) < 14:
                return None
            if self.lstm_model:
                X = prices[-10:].reshape(1, 10, 1)
                pred = self.lstm_model.predict(X)[0][0]
                regime = "bull" if pred > np.mean(prices) else "bear" if pred < np.mean(prices) else "neutral"
            else:
                # Fallback to existing
                mean = float(np.mean(prices))
                std = float(np.std(prices))
                regime = "bull" if prices[-1] > mean + std else "bear" if prices[-1] < mean - std else "neutral"
            atr = float(np.mean(np.abs(highs - lows))) if len(highs) and len(lows) else float(std)
        except Exception:
            regime, atr = "neutral", 0.0

        # Enhanced: Add Ichimoku Cloud and adaptive RSI
        try:
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            # Ichimoku
            high_9 = df["high"].rolling(9).max()
            low_9 = df["low"].rolling(9).min()
            df["tenkan"] = (high_9 + low_9) / 2
            # ... (add full Ichimoku calc)
            # Adaptive RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
            # Fuse with sentiment/on-chain
            sentiment_score = self.scout.analyze_sentiment([f"Market {df['close'].iloc[-1]}"])  # Placeholder
            regime = "bull" if df["tenkan"].iloc[-1] > df["close"].iloc[-1] and sentiment_score > 0 else "bear"
        except Exception:
            regime = "neutral"

        return {
            "regime": regime,
            "mean": mean,
            "std": std,
            "atr": atr,
            "last": float(prices[-1]),
        }

    def scout_opportunities(self, scan_results):
        """Scout for best trade setups across all assets."""
        opportunities = []
        for market, analysis in scan_results.items():
            if analysis and analysis["regime"] == "bull":
                opportunities.append({"market": market, "action": "buy"})
            elif analysis and analysis["regime"] == "bear":
                opportunities.append({"market": market, "action": "sell"})
        return opportunities

    def plan_trades(self, opportunities):
        """Plan entries/exits with adaptive risk management."""
        plans = []
        for opp in opportunities:
            try:
                market = opp.get("market")
                action = opp.get("action")
                # Example: Dynamic position sizing
                size = self.dynamic_position_size(market)
                # clamp USD-denominated stake to sane bounds and convert to units
                try:
                    min_usd = float(os.getenv("ULTRA_MIN_STAKE_USD", "5.0"))
                    max_usd = float(os.getenv("ULTRA_MAX_STAKE_USD", "200.0"))
                except Exception:
                    min_usd, max_usd = 5.0, 200.0
                size_usd = max(min_usd, min(max_usd, float(size or 0.0)))

                # Best-effort: determine an entry price using router ticker or recent OHLCV
                entry = 0.0
                sl = 0.0
                tp1 = None
                try:
                    # prefer ticker last
                    tk = {}
                    try:
                        tk = self.router.safe_fetch_ticker(market) or {}
                    except Exception:
                        tk = {}
                    last = float(tk.get("last") or tk.get("price") or tk.get("close") or 0.0)
                    if last and last > 0.0:
                        entry = last
                    else:
                        # fallback to ohlcv last close
                        try:
                            ohlcv = self.router.safe_fetch_ohlcv(market, timeframe="5m", limit=20) or []
                            if ohlcv and isinstance(ohlcv, list) and len(ohlcv[-1]) > 4:
                                entry = float(ohlcv[-1][4])
                        except Exception:
                            entry = 0.0
                except Exception:
                    entry = 0.0
                # convert USD stake to instrument units if we have a price
                size_units = 0.0
                try:
                    if entry and entry > 0:
                        size_units = float(size_usd) / float(entry)
                except Exception:
                    size_units = 0.0

                # compute a conservative SL using ATR from analysis or small fraction of price
                try:
                    ana = None
                    try:
                        ohlcv = self.router.safe_fetch_ohlcv(market, timeframe="5m", limit=50) or []
                        ana = self.analyze_market(ohlcv) if ohlcv else None
                    except Exception:
                        ana = None
                    atr = float(ana.get("atr")) if ana and ana.get("atr") else 0.0
                    if entry and atr and atr > 0:
                        # conservative SL 1.5x ATR away
                        # For BUY: SL below entry; for SELL: SL above entry.
                        if action == "buy":
                            sl = float(max(0.0, entry - 1.5 * atr))
                        else:
                            sl = float(entry + 1.5 * atr)
                    else:
                        # fallback small percent distance (0.3%)
                        if entry and entry > 0:
                            diff = max(0.001, entry * 0.003)
                            if action == "buy":
                                sl = float(max(0.0, entry - diff))
                            else:
                                sl = float(entry + diff)
                        else:
                            sl = 0.0
                except Exception:
                    sl = 0.0

                # ensure stop-loss isn't accidentally identical to entry (tiny numerical guard)
                eps = max(1e-8, abs(entry) * 1e-6)
                if entry and sl and abs(entry - sl) < eps:
                    # nudge SL by a tiny fraction so it's distinct
                    if action == "buy":
                        sl = float(max(0.0, entry - max(eps, entry * 1e-4)))
                    else:
                        sl = float(entry + max(eps, entry * 1e-4))

                # simple tp1 placeholder (1x R)
                try:
                    if entry and sl:
                        R = abs(entry - sl)
                        if R > 1e-8:
                            if action == "buy":
                                tp1 = entry + 1.0 * R
                            else:
                                tp1 = entry - 1.0 * R
                except Exception:
                    tp1 = None

                # produce plan with both USD stake and units (units are used for execution)
                plan = {
                    "market": market,
                    "action": action,
                    "size": size_units,
                    "size_usd": size_usd,
                    "entry": entry,
                    "sl": sl,
                }
                if tp1 is not None:
                    plan["tp1"] = tp1
                plans.append(plan)
            except Exception:
                # on any plan-level failure, include minimal plan so caller can continue
                try:
                    plans.append(
                        {
                            "market": opp.get("market"),
                            "action": opp.get("action"),
                            "size": 0.0,
                        }
                    )
                except Exception:
                    continue
        # persist plans for debugging/inspection
        try:
            runtime = Path("runtime")
            runtime.mkdir(parents=True, exist_ok=True)
            with open(runtime / "ultra_plans.json", "w", encoding="utf-8") as f:
                json.dump(plans, f, indent=2)
        except Exception:
            pass
        return plans

    def dynamic_position_size(self, market):
        """Adaptive sizing using ATR and recent win-rate.

        Returns a USD-denominated stake (or abstract units) suitable for `safe_place_order`.
        """
        # Enhanced: Kelly criterion for sizing
        try:
            win_rate = float(self.knowledge_base.get("win_rate", 0.5))
            avg_win = float(self.knowledge_base.get("avg_win", 1.0))
            avg_loss = float(self.knowledge_base.get("avg_loss", 1.0))
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win if avg_win > 0 else 0.1
            base_usd = 50.0
            stake = base_usd * max(0.01, min(0.5, kelly))
            return max(1.0, stake)
        except Exception:
            return 10.0

    def enter_trades(self, plans):
        """Execute planned trades using router wrappers."""
        results = []
        for plan in plans:
            try:
                # respect live flag
                if not getattr(self.router, "live", False):
                    # simulated log for dry-run
                    px = 0.0
                    try:
                        px = float(self.router.safe_fetch_ticker(plan["market"]).get("last") or 0.0)
                    except Exception:
                        px = 0.0
                    qty = float(plan.get("size") or 0.0)
                    trade_id = None
                    try:
                        trade_id = log_entry(
                            venue=getattr(self.router, "id", "exchange"),
                            market="auto",
                            symbol=plan["market"],
                            tf="auto",
                            side=plan["action"],
                            entry_px=px,
                            qty=qty,
                            sl=None,
                            tp=None,
                            meta={"sim": True},
                        )
                    except Exception:
                        trade_id = None
                    results.append(
                        {
                            "ok": False,
                            "dry_run": True,
                            "market": plan["market"],
                            "side": plan["action"],
                            "qty": qty,
                            "px": px,
                            "trade_id": trade_id,
                        }
                    )
                    # append debug line to price_fill_debug.ndjson for post-mortem
                    try:
                        runtime = Path("runtime")
                        runtime.mkdir(parents=True, exist_ok=True)
                        dbg = {
                            "market": plan.get("market"),
                            "side": plan.get("action"),
                            "qty": qty,
                            "entry": px,
                            "sl": plan.get("sl"),
                            "size_usd": plan.get("size_usd"),
                            "context": "UltraCore dry-run",
                        }
                        with open(runtime / "price_fill_debug.ndjson", "a", encoding="utf-8") as f:
                            f.write(json.dumps(dbg) + "\n")
                    except Exception:
                        pass
                else:
                    result = self.router.safe_place_order(plan["market"], plan["action"], plan["size"])
                    results.append(result)
            except Exception as e:
                results.append({"ok": False, "error": str(e), "market": plan.get("market")})
        return results

    def close_trades(self):
        """Smart exit logic: trailing stops, profit targets, regime change detection."""
        # Enhanced: Anomaly-based closing
        try:
            open_positions = []
            if self.universe and hasattr(self.universe, "get_open_positions"):
                open_positions = list(self.universe.get_open_positions())
            else:
                # if universe doesn't provide, ask ledger/open positions via knowledge base
                open_positions = [
                    o.get("symbol") for o in (self.knowledge_base.get("open_positions") or []) if o.get("symbol")
                ]

            if self.anomaly_detector:
                # Use anomaly detector for early closure
                for market in open_positions:
                    try:
                        ohlcv = self.router.safe_fetch_ohlcv(market)
                        features = [float(x[4]) for x in ohlcv[-10:]]
                        if self.anomaly_detector.predict([features])[0] == -1:
                            if getattr(self.router, "live", False):
                                self.router.safe_close_position(market)
                            else:
                                # simulate close in dry-run
                                try:
                                    log_entry(
                                        venue=getattr(self.router, "id", "exchange"),
                                        market="close-sim",
                                        symbol=market,
                                        tf="auto",
                                        side="close",
                                        entry_px=0.0,
                                        qty=0.0,
                                        sl=None,
                                        tp=None,
                                        meta={"sim_close": True},
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        continue
            else:
                # Example: Close trades if regime changes
                for market in open_positions:
                    try:
                        ohlcv = self.router.safe_fetch_ohlcv(market)
                        analysis = self.analyze_market(ohlcv)
                        if analysis and analysis.get("regime") == "neutral":
                            # close conservatively
                            if getattr(self.router, "live", False):
                                self.router.safe_close_position(market)
                            else:
                                # simulate close in dry-run
                                try:
                                    log_entry(
                                        venue=getattr(self.router, "id", "exchange"),
                                        market="close-sim",
                                        symbol=market,
                                        tf="auto",
                                        side="close",
                                        entry_px=0.0,
                                        qty=0.0,
                                        sl=None,
                                        tp=None,
                                        meta={"sim_close": True},
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        continue
        except Exception:
            if self.logger:
                self.logger.exception("close_trades failed")

    def learn(self):
        """Online learning: update knowledge base from recent trades and market data."""
        # Enhanced: Track avg_win/avg_loss for Kelly
        # Example: Track win rate, update strategies
        self.performance_log.append({"timestamp": time.time(), "result": "placeholder"})
        if len(self.performance_log) > 100:
            self.performance_log = self.performance_log[-100:]
        # Placeholder for more advanced learning
        # compute crude win_rate from recent performance_log (placeholder semantics)
        try:
            wins = sum(1 for p in self.performance_log[-50:] if p.get("result") == "win")
            total = max(1, len(self.performance_log[-50:]))
            self.knowledge_base["win_rate"] = wins / total
        except Exception:
            self.knowledge_base["win_rate"] = random.uniform(0.4, 0.7)
        try:
            wins = [p.get("result") for p in self.performance_log[-50:] if p.get("result") == "win"]
            losses = [p.get("result") for p in self.performance_log[-50:] if p.get("result") == "loss"]
            self.knowledge_base["avg_win"] = np.mean([1.0] * len(wins)) if wins else 1.0
            self.knowledge_base["avg_loss"] = np.mean([1.0] * len(losses)) if losses else 1.0
        except Exception:
            pass

    def sharpen(self):
        """Self-improvement: periodically optimize parameters and strategies."""
        # Example: Randomly adjust parameters
        self.knowledge_base["param"] = random.uniform(0, 1)
        self.last_update = time.time()

    def quantum_optimize_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Ultra-rare: Quantum-inspired optimization for strategy params."""
        if not self.quantum_backend:
            return params
        try:
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            job = execute(qc, self.quantum_backend, shots=1000)
            result = job.result()
            counts = result.get_counts()
            # Simulate optimization: adjust ATR mult based on quantum randomness
            rand_factor = float(counts.get("00", 0)) / 1000
            params["atr_mult"] = params.get("atr_mult", 1.5) * (0.9 + rand_factor)
            return params
        except Exception:
            return params

    def self_adaptive_retrade(self, opportunities):
        """Autonomous retrading: Retrade up to max_retrades if analysis improves."""
        for opp in opportunities:
            if self.retrade_count < self.max_retrades:
                # Simulate improved analysis (e.g., via RL)
                improved_conf = float(opp.get("confidence", 0)) * 1.1
                if improved_conf > 0.8:
                    opp["confidence"] = improved_conf
                    self.retrade_count += 1
                    # Retrade logic
                    self.plan_trades([opp])
                    self.enter_trades([opp])

    def gnn_correlation_prediction(self, market_data: Dict[str, Any]) -> float:
        """Ultra-rare: GNN for multi-asset correlations."""
        if not self.gnn_model:
            return 0.0
        try:
            # Local import to avoid top-level dependency and static undefined-name
            try:
                import torch
            except Exception:
                return 0.0
            # Simulate graph data
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            x = torch.randn(2, 10)  # Node features
            pred = self.gnn_model(x, edge_index)
            return pred.mean().item()
        except Exception:
            return 0.0

    def auto_healing_anomalies(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect and heal anomalies in signals."""
        if not self.anomaly_detector:
            return signals
        try:
            features = [[s.get("confidence", 0), s.get("entry", 0)] for s in signals]
            preds = self.anomaly_detector.predict(features)
            healed = [s for s, pred in zip(signals, preds) if pred != -1]
            return healed
        except Exception:
            return signals

    def god_mode_cycle(self):
        """Full god mode trading cycle: scan, reason, scout, plan, enter, close, learn, sharpen, ultra scout, advanced features."""
        # UltraScout: fetch news, sentiment, patterns, trends
        scout_data = self.scout.scout_all()
        self.knowledge_base["scout"] = scout_data

        # Run advanced ultra features
        self.ultra_advanced_cycle()

        scan_results = self.scan_markets()
        opportunities = self.scout_opportunities(scan_results)
        self.gnn_correlation_prediction(scan_results)
        opportunities = self.auto_healing_anomalies(opportunities)
        plans = self.plan_trades(opportunities)
        self.enter_trades(plans)
        self.close_trades()
        self.learn()
        self.sharpen()
        # Enhanced: Quantum optimize and retrade
        self.quantum_optimize_params({"atr_mult": 1.5})
        self.self_adaptive_retrade(opportunities)
        if self.logger:
            self.logger.info(f"God mode cycle complete. Knowledge base: {self.knowledge_base}")


# Ultra feature suggestions for future upgrades:
# - Deep reinforcement learning for trade decision optimization
# - NLP-based news sentiment analysis for market impact
# - On-chain analytics for web3/token/meme/altcoin trading
# - Social media trend detection
# - Automated strategy backtesting and self-tuning
# - Multi-agent collaboration (swarm intelligence)
# - Real-time anomaly detection and risk alerts
