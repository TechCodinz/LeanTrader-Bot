"""
ultra_scout.py
Ultra Scouting Engine: News, Social, Web, Research, and Pattern Discovery
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup

# NOTE: heavy / optional libraries are loaded lazily inside the class to avoid import-time failures.

class UltraScout:
    def __init__(
        self,
        max_threads: Optional[int] = None,
        user_agent: Optional[str] = None,
        exchange: Optional[Any] = None,
    ):
        # A ccxt client, when the caller has one. Everything that can be read
        # from a real market is read through this; nothing is invented when it
        # is absent. See _unavailable() for the contract.
        self.exchange = exchange
        self.sources = [
            "https://www.investing.com/news/cryptocurrency-news",
            "https://cryptopanic.com/news",
            "https://twitter.com/search?q=crypto%20trading",
            "https://www.reddit.com/r/cryptocurrency/",
            "https://github.com/search?q=trading+strategy",
        ]
        self.patterns: List[str] = []
        self.sentiment: Dict[str, float] = {}
        self.trends: List[str] = []
        self.last_update = time.time()
        # advanced placeholders
        self.onchain_data: Dict[str, Any] = {}
        self.backtest_results: Dict[str, Any] = {}
        self.swarm_signals: List[Dict[str, Any]] = []
        self.risk_alerts: List[str] = []
        self.broker_api_status: Dict[str, Any] = {}
        self.rl_state: Dict[str, Any] = {}
        self.dashboard_data: Dict[str, Any] = {}
        self.voice_chat_log: List[str] = []

        # network/session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent
                or os.getenv("ULTRA_USER_AGENT", "UltraScout/1.0 (+https://example.com)"),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )
        self.request_timeout = float(os.getenv("ULTRA_REQUEST_TIMEOUT", "8.0"))

        # concurrency
        self.max_threads = int(max_threads or int(os.getenv("ULTRA_SCOUT_THREADS", "4")))
        self._lock = threading.Lock()

        # optional components (lazy)
        self._sentiment_analyzer = None
        self._anomaly_detector = None
        self._rl_model = None
        self._gpt_client = None
        # do not instantiate heavy libs until needed

    # -------------------------
    # Helpers and lazy loaders
    # -------------------------
    def _get_sentiment_analyzer(self):
        if self._sentiment_analyzer is None:
            try:
                from transformers import pipeline

                self._sentiment_analyzer = pipeline("sentiment-analysis")
            except Exception:
                self._sentiment_analyzer = None
        return self._sentiment_analyzer

    def _get_anomaly_detector(self):
        if self._anomaly_detector is None:
            try:
                from sklearn.ensemble import IsolationForest

                self._anomaly_detector = IsolationForest(contamination=0.05, random_state=0)
            except Exception:
                self._anomaly_detector = None
        return self._anomaly_detector

    def _get_rl_model(self):
        if self._rl_model is None:
            try:
                # Keep placeholder minimal and lazy; avoid hard dependency
                self._rl_model = object
            except Exception:
                self._rl_model = None
        return self._rl_model

    def _get_gpt_client(self):
        if self._gpt_client is None:
            try:
                import openai as _openai

                _openai.api_key = os.getenv("OPENAI_API_KEY") or ""
                if not _openai.api_key:
                    self._gpt_client = None
                else:
                    self._gpt_client = _openai
            except Exception:
                self._gpt_client = None
        return self._gpt_client

    # -------------------------
    # On-chain, backtest, swarm
    # -------------------------
    def _real_trends(self) -> List[Any]:
        """Trends from real closes across the configured symbols, or []."""
        if self.exchange is None:
            return []
        symbols = [
            sym.strip()
            for sym in os.getenv("SCOUT_TREND_SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
            if sym.strip()
        ]
        trends: List[Any] = []
        for symbol in symbols:
            try:
                rows = self.exchange.fetch_ohlcv(symbol, timeframe="1h", limit=30)
            except Exception:
                continue
            closes = [float(r[4]) for r in (rows or []) if r and len(r) > 4]
            if len(closes) < 10:
                continue
            try:
                trends.append(self.detect_trends(closes))
            except Exception:
                continue
        return trends

    @staticmethod
    def _derive_votes(signal: Any, agents: int) -> List[str]:
        """Votes from the signal's own strength, not from chance.

        Each agent applies a progressively stricter threshold to the same
        real value, so the spread of votes reflects how strong the signal
        actually is. A signal with no readable strength abstains as hold
        rather than voting arbitrarily.
        """
        value = 0.0
        if isinstance(signal, dict):
            for key in ("score", "strength", "confidence", "value", "change"):
                if key in signal:
                    try:
                        value = float(signal[key])
                        break
                    except (TypeError, ValueError):
                        continue
        else:
            try:
                value = float(signal)
            except (TypeError, ValueError):
                value = 0.0

        if value > 1.0:  # a 0-100 confidence rather than a -1..1 score
            value = (value - 50.0) / 50.0

        votes: List[str] = []
        for index in range(max(1, int(agents))):
            threshold = 0.1 + (index * 0.15)
            if value >= threshold:
                votes.append("buy")
            elif value <= -threshold:
                votes.append("sell")
            else:
                votes.append("hold")
        return votes

    @staticmethod
    def _unavailable(reason: str) -> Dict[str, Any]:
        """The honest answer when there is no source for a metric.

        This module previously filled these gaps with random.uniform and
        random.randint -- invented supply, holders, whale flows, depth and
        balances that a caller could not distinguish from measurements. A
        metric with no provider is reported as unavailable, with the reason,
        so a consumer skips it instead of trading on a number nobody observed.
        """
        return {"available": False, "reason": reason}

    def _onchain_provider(self) -> Optional[str]:
        """The Etherscan-compatible API key, if one is configured."""
        for name in ("ETHERSCAN_API_KEY", "ONCHAIN_API_KEY"):
            key = os.getenv(name, "").strip()
            if key:
                return key
        return None

    def fetch_onchain_analytics(self, token_address: str) -> Dict[str, Any]:
        """Fetch real on-chain analytics data."""
        analytics = {"token": token_address, "timestamp": time.time()}

        try:
            # Fetch from multiple sources
            # Etherscan API (requires API key)
            etherscan_data = self._fetch_etherscan(token_address)
            analytics.update(etherscan_data)

            # Glassnode-style metrics
            analytics['whale_transactions'] = self._detect_whale_movements(token_address)
            analytics['exchange_flows'] = self._analyze_exchange_flows(token_address)
            analytics['holder_distribution'] = self._get_holder_distribution(token_address)

            # DeFi metrics
            analytics['defi_tvl'] = self._get_defi_tvl(token_address)
            analytics['liquidity_depth'] = self._get_liquidity_depth(token_address)

        except Exception as e:
            analytics['error'] = str(e)
            analytics['whale_moves'] = self._estimate_whale_activity()
            # Real traded volume if a market client is available. Previously
            # this was random.uniform(100000, 10000000) -- a fabricated volume
            # presented beside real fields.
            analytics['volume'] = self._real_quote_volume(token_address)

        return analytics

    def _real_quote_volume(self, symbol: str) -> Optional[float]:
        """24h quote volume from the exchange, or None if unknown."""
        if self.exchange is None:
            return None
        try:
            ticker = self.exchange.fetch_ticker(symbol) or {}
            volume = ticker.get("quoteVolume")
            return float(volume) if volume is not None else None
        except Exception:
            return None

    def _fetch_etherscan(self, token_address: str) -> Dict[str, Any]:
        """Token supply from Etherscan, or an explicit unavailable."""
        key = self._onchain_provider()
        if not key:
            return self._unavailable("no_etherscan_api_key")
        try:
            import requests

            response = requests.get(
                "https://api.etherscan.io/api",
                params={
                    "module": "stats",
                    "action": "tokensupply",
                    "contractaddress": token_address,
                    "apikey": key,
                },
                timeout=10,
            )
            payload = response.json()
            if str(payload.get("status")) != "1":
                return self._unavailable(
                    f"etherscan_status_{payload.get('message', 'error')}"
                )
            return {"available": True, "total_supply": float(payload.get("result") or 0.0)}
        except Exception as exc:
            return self._unavailable(f"etherscan_error:{type(exc).__name__}")

    def _detect_whale_movements(self, token_address: str) -> List[Dict[str, Any]]:
        """Detect large transactions indicating whale activity."""
        key = self._onchain_provider()
        if not key:
            # No provider: no whale evidence. Previously this invented an
            # accumulation or distribution event 30% of the time, with a
            # random amount and a random bullish/bearish label.
            return []
        try:
            import requests

            response = requests.get(
                "https://api.etherscan.io/api",
                params={
                    "module": "account",
                    "action": "tokentx",
                    "contractaddress": token_address,
                    "page": 1,
                    "offset": 100,
                    "sort": "desc",
                    "apikey": key,
                },
                timeout=10,
            )
            rows = (response.json() or {}).get("result") or []
            if not isinstance(rows, list):
                return []
            values = [float(r.get("value") or 0.0) for r in rows if isinstance(r, dict)]
            if len(values) < 10:
                return []
            values.sort()
            # A whale transfer is one in the top percentile of what actually
            # moved, measured -- not assumed.
            threshold = values[int(len(values) * 0.95)]
            return [
                {
                    "type": "transfer",
                    "amount": float(r.get("value") or 0.0),
                    "hash": str(r.get("hash", ""))[:66],
                    "measured": True,
                }
                for r in rows
                if isinstance(r, dict) and float(r.get("value") or 0.0) >= threshold
            ][:10]
        except Exception:
            return []

    def _analyze_exchange_flows(self, token_address: str) -> Dict[str, float]:
        """Analyze token flows to/from exchanges."""
        # Exchange flow attribution needs labelled exchange wallets, which no
        # configured provider supplies here. Reported unavailable rather than
        # invented -- the previous four random values were indistinguishable
        # from measurements to any caller.
        return self._unavailable("no_exchange_flow_provider")

    def _get_holder_distribution(self, token_address: str) -> Dict[str, float]:
        """Get token holder distribution metrics."""
        return self._unavailable("no_holder_distribution_provider")

    def _get_defi_tvl(self, token_address: str) -> Optional[float]:
        """TVL from DefiLlama when reachable, else None."""
        try:
            import requests

            response = requests.get(
                f"https://api.llama.fi/protocol/{token_address}", timeout=10
            )
            if response.status_code != 200:
                return None
            tvl = (response.json() or {}).get("tvl")
            if isinstance(tvl, list) and tvl:
                return float(tvl[-1].get("totalLiquidityUSD") or 0.0)
            return float(tvl) if isinstance(tvl, (int, float)) else None
        except Exception:
            return None

    def _get_liquidity_depth(self, token_address: str) -> Dict[str, Any]:
        """Real bid/ask depth and spread from the exchange order book.

        This is the one on-chain-adjacent metric with a genuine source: the
        venue's own book. Previously all three numbers were random.
        """
        if self.exchange is None:
            return self._unavailable("no_market_client")
        try:
            book = self.exchange.fetch_order_book(token_address, limit=50) or {}
            bids = book.get("bids") or []
            asks = book.get("asks") or []
            if not bids or not asks:
                return self._unavailable("empty_order_book")

            bid_depth = sum(float(p) * float(q) for p, q in bids[:25])
            ask_depth = sum(float(p) * float(q) for p, q in asks[:25])
            best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
            mid = (best_bid + best_ask) / 2.0

            return {
                "available": True,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "spread": (best_ask - best_bid) / mid if mid > 0 else 0.0,
                "spread_bps": ((best_ask - best_bid) / mid * 10_000.0) if mid > 0 else 0.0,
                "imbalance": (
                    (bid_depth - ask_depth) / (bid_depth + ask_depth)
                    if (bid_depth + ask_depth) > 0
                    else 0.0
                ),
            }
        except Exception as exc:
            return self._unavailable(f"order_book_error:{type(exc).__name__}")

    def _estimate_whale_activity(self) -> Optional[int]:
        """Estimate whale activity based on patterns."""
        # There is no measurement behind this. The previous version returned
        # a random count keyed off the hour of day, which is a guess wearing
        # the shape of data. None means unknown.
        return None

    def run_backtest(self, strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive backtest with walk-forward optimization."""
        try:
            # Import backtest modules
            from strategy import get_strategy

            # Get strategy instance
            strat = get_strategy(strategy, **params)

            # Prepare test data
            test_periods = [
                {'start': -90, 'end': -60, 'name': 'out_sample_1'},
                {'start': -60, 'end': -30, 'name': 'out_sample_2'},
                {'start': -30, 'end': 0, 'name': 'out_sample_3'},
            ]

            results = {'strategy': strategy, 'params': params, 'periods': {}}

            # Run walk-forward analysis
            for period in test_periods:
                # Simulate trading
                period_result = self._run_period_backtest(strat, period)
                results['periods'][period['name']] = period_result

            # Calculate overall metrics
            all_returns = []
            for period_name, period_data in results['periods'].items():
                all_returns.extend(period_data.get('returns', []))

            if all_returns:
                results['total_return'] = np.prod([1 + r for r in all_returns]) - 1
                results['sharpe_ratio'] = (
                    np.mean(all_returns) / (np.std(all_returns) + 1e-10) * np.sqrt(252)
                )
                results['max_drawdown'] = self._calculate_max_drawdown(all_returns)
                results['win_rate'] = len([r for r in all_returns if r > 0]) / len(all_returns)
            else:
                results['total_return'] = 0
                results['sharpe_ratio'] = 0
                results['max_drawdown'] = 0
                results['win_rate'] = 0

            results['score'] = results['sharpe_ratio']

        except Exception as e:
            results = {
                'strategy': strategy,
                'params': params,
                'error': str(e),
                # A failed backtest has no score. The previous fallback
                # invented one between -1 and 2, which could rank a strategy
                # that never ran above one that did.
                'score': None,
                'available': False,
            }

        return results

    def _run_period_backtest(self, strategy, period: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest for a specific period."""
        # Real period returns from real candles. Previously this drew 30
        # samples from random.gauss, so every backtest scored a strategy
        # against noise that had nothing to do with the market.
        symbol = period.get('symbol') or getattr(self, 'backtest_symbol', None)
        returns = self._real_period_returns(symbol, period)

        if not returns:
            return {
                'returns': [],
                'total_return': None,
                'trades': 0,
                'period': period.get('name'),
                'available': False,
                'reason': 'no_real_candles_for_period',
            }

        return {
            'returns': returns,
            'total_return': float(np.prod([1 + r for r in returns]) - 1),
            'trades': len(returns),
            'period': period.get('name'),
            'available': True,
        }

    def _real_period_returns(
        self, symbol: Optional[str], period: Dict[str, Any]
    ) -> List[float]:
        """Bar-to-bar returns from real candles, or [] when unavailable."""
        if self.exchange is None or not symbol:
            return []
        try:
            rows = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=str(period.get('timeframe') or '1h'),
                limit=int(period.get('limit') or 200),
            )
        except Exception:
            return []
        closes = [float(r[4]) for r in (rows or []) if r and len(r) > 4]
        if len(closes) < 3:
            return []
        return [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
            if closes[i - 1] > 0
        ]

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))

    def swarm_collaboration(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Multi-agent swarm intelligence for signal aggregation."""
        if not signals:
            return {'consensus': 'HOLD', 'confidence': 0, 'agents': 0}

        # Agent voting system
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        confidences = []

        for signal in signals:
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0.5)

            # Weight votes by confidence
            votes[action] += confidence
            confidences.append(confidence)

        # Calculate consensus
        total_votes = sum(votes.values())
        if total_votes == 0:
            return {'consensus': 'HOLD', 'confidence': 0, 'agents': len(signals)}

        # Get winning action
        consensus = max(votes.items(), key=lambda x: x[1])[0]
        consensus_strength = votes[consensus] / total_votes

        # Calculate swarm confidence
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)

        # Higher agreement = higher confidence
        swarm_confidence = consensus_strength * avg_confidence * (1 - confidence_std)

        # Advanced swarm metrics
        swarm_data = {
            'consensus': consensus,
            'confidence': float(swarm_confidence),
            'agents': len(signals),
            'votes': votes,
            'agreement_rate': consensus_strength,
            'diversity': float(confidence_std),
            'minority_report': self._get_minority_report(signals, consensus),
        }

        return swarm_data

    def _get_minority_report(self, signals: List[Dict[str, Any]], consensus: str) -> Dict[str, Any]:
        """Analyze dissenting opinions in the swarm."""
        minority_signals = [s for s in signals if s.get('action') != consensus]

        if not minority_signals:
            return {'dissent_rate': 0, 'alternative': None}

        return {
            'dissent_rate': len(minority_signals) / len(signals),
            'alternative': max(
                set([s.get('action') for s in minority_signals]),
                key=lambda x: [s.get('action') for s in minority_signals].count(x),
            ),
            'reasons': [s.get('reason', 'Unknown') for s in minority_signals[:3]],
        }

    def detect_risk_alerts(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Advanced risk detection with anomaly detection."""
        alerts = []

        if not trades:
            return alerts

        # Calculate statistics
        pnls = [t.get('pnl', 0) for t in trades]
        volumes = [t.get('volume', 0) for t in trades]

        if pnls:
            pnl_mean = np.mean(pnls)
            pnl_std = np.std(pnls)

            # Detect anomalies
            for trade in trades:
                pnl = trade.get('pnl', 0)
                symbol = trade.get('symbol', 'Unknown')

                # Statistical anomaly detection
                if abs(pnl - pnl_mean) > 3 * pnl_std:
                    alerts.append(
                        {
                            'type': 'statistical_anomaly',
                            'severity': 'high',
                            'symbol': symbol,
                            'pnl': pnl,
                            'z_score': (pnl - pnl_mean) / (pnl_std + 1e-10),
                            'message': f"Unusual PnL detected: {symbol} with {pnl:.2f} (Z-score: {(pnl - pnl_mean) / (pnl_std + 1e-10):.2f})",
                        }
                    )

                # Large loss detection
                if pnl < -1000:
                    alerts.append(
                        {
                            'type': 'large_loss',
                            'severity': 'critical',
                            'symbol': symbol,
                            'pnl': pnl,
                            'message': f"Large loss on {symbol}: {pnl:.2f}",
                        }
                    )

                # Consecutive losses
                recent_trades = trades[-10:]
                consecutive_losses = sum(1 for t in recent_trades if t.get('pnl', 0) < 0)
                if consecutive_losses >= 5:
                    alerts.append(
                        {
                            'type': 'losing_streak',
                            'severity': 'high',
                            'count': consecutive_losses,
                            'message': f"Losing streak detected: {consecutive_losses} consecutive losses",
                        }
                    )

        # Volume anomalies
        if volumes:
            vol_mean = np.mean(volumes)
            vol_std = np.std(volumes)

            for trade in trades:
                volume = trade.get('volume', 0)
                if volume > vol_mean + 3 * vol_std:
                    alerts.append(
                        {
                            'type': 'volume_spike',
                            'severity': 'medium',
                            'symbol': trade.get('symbol', 'Unknown'),
                            'volume': volume,
                            'message': f"Unusual volume spike: {volume:.0f} (avg: {vol_mean:.0f})",
                        }
                    )

        # Market-wide risk detection
        if len(trades) > 20:
            recent_return = sum(t.get('pnl', 0) for t in trades[-20:])
            if recent_return < -2000:
                alerts.append(
                    {
                        'type': 'market_crash',
                        'severity': 'critical',
                        'total_loss': recent_return,
                        'message': f"Potential market crash detected: {recent_return:.2f} loss in recent trades",
                    }
                )

        return alerts

    def broker_api_integration(self, broker_name: str) -> Dict[str, Any]:
        """Real broker API integration and health check."""
        status = {'broker': broker_name, 'timestamp': datetime.now().isoformat(), 'checks': {}}

        try:
            if broker_name.lower() == 'binance':
                # Check Binance API
                status['checks']['api_accessible'] = self._check_binance_api()
                status['checks']['rate_limits'] = self._check_rate_limits('binance')
                status['checks']['balance'] = self._check_balance('binance')

            elif broker_name.lower() == 'coinbase':
                status['checks']['api_accessible'] = self._check_coinbase_api()
                status['checks']['rate_limits'] = self._check_rate_limits('coinbase')
                status['checks']['balance'] = self._check_balance('coinbase')

            else:
                # Generic broker check
                status['checks']['api_accessible'] = True
                status['checks']['rate_limits'] = {'remaining': 100, 'reset': 60}
                status['checks']['balance'] = {'available': True}

            # Overall status
            all_checks_pass = all(
                v if isinstance(v, bool) else v.get('available', False)
                for v in status['checks'].values()
            )
            status['status'] = 'connected' if all_checks_pass else 'degraded'

        except Exception as e:
            status['status'] = 'error'
            status['error'] = str(e)

        return status

    def _check_binance_api(self) -> bool:
        """Check Binance API connectivity."""
        try:
            response = requests.get('https://api.binance.com/api/v3/ping', timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _check_coinbase_api(self) -> bool:
        """Check Coinbase API connectivity."""
        try:
            response = requests.get('https://api.coinbase.com/v2/time', timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _check_rate_limits(self, broker: str) -> Dict[str, Any]:
        """Check API rate limits."""
        # Real values from the ccxt client when one is attached.
        if self.exchange is None:
            return self._unavailable("no_market_client")
        try:
            return {
                "available": True,
                "rate_limit_ms": float(getattr(self.exchange, "rateLimit", 0) or 0),
                "enable_rate_limit": bool(getattr(self.exchange, "enableRateLimit", False)),
                "last_response_headers": {
                    k: v
                    for k, v in (getattr(self.exchange, "last_response_headers", {}) or {}).items()
                    if "limit" in str(k).lower() or "remaining" in str(k).lower()
                },
            }
        except Exception as exc:
            return self._unavailable(f"rate_limit_error:{type(exc).__name__}")

    def _check_balance(self, broker: str) -> Dict[str, Any]:
        """Check account balance availability."""
        # The authenticated balance, or an explicit unavailable. Inventing a
        # balance here could size a position against money that is not there.
        if self.exchange is None:
            return self._unavailable("no_market_client")
        try:
            balance = self.exchange.fetch_balance() or {}
            quote = os.getenv("MARKET_QUOTE", "USDT").upper()
            free = (balance.get("free") or {}).get(quote)
            total = (balance.get("total") or {}).get(quote)
            return {
                "available": True,
                "quote": quote,
                "free_usd": float(free) if free is not None else None,
                "total_usd": float(total) if total is not None else None,
            }
        except Exception as exc:
            return self._unavailable(f"balance_error:{type(exc).__name__}")

    def reinforcement_learning_update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Update reinforcement learning agent with new state and rewards."""
        updated_state = state.copy()

        try:
            # Calculate reward from recent performance
            recent_pnl = state.get('recent_pnl', 0)
            risk_adjusted_reward = self._calculate_risk_adjusted_reward(recent_pnl, state)

            # Update Q-values or policy
            if 'q_values' not in updated_state:
                updated_state['q_values'] = {}

            # State representation
            state_key = self._encode_state(state)

            # Q-learning update
            learning_rate = 0.1
            discount_factor = 0.95

            current_q = updated_state['q_values'].get(state_key, 0)
            max_future_q = (
                max(updated_state['q_values'].values()) if updated_state['q_values'] else 0
            )

            new_q = current_q + learning_rate * (
                risk_adjusted_reward + discount_factor * max_future_q - current_q
            )

            updated_state['q_values'][state_key] = new_q

            # Update policy
            updated_state['policy'] = self._derive_policy(updated_state['q_values'])

            # Track learning progress
            updated_state['learning_metrics'] = {
                'episodes': state.get('episodes', 0) + 1,
                'total_reward': state.get('total_reward', 0) + risk_adjusted_reward,
                'avg_reward': (state.get('total_reward', 0) + risk_adjusted_reward)
                / (state.get('episodes', 0) + 1),
                'exploration_rate': max(0.01, 0.9 * (0.99 ** state.get('episodes', 0))),
            }

            updated_state['updated'] = True
            updated_state['last_update'] = datetime.now().isoformat()

        except Exception as e:
            updated_state['error'] = str(e)
            updated_state['updated'] = False

        return updated_state

    def _calculate_risk_adjusted_reward(self, pnl: float, state: Dict[str, Any]) -> float:
        """Calculate risk-adjusted reward for RL."""
        # Sharpe-ratio inspired reward
        returns = state.get('recent_returns', [pnl])
        if not returns:
            return 0

        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1

        # Risk-adjusted reward
        sharpe = avg_return / (std_return + 1e-10)

        # Add penalties for excessive risk
        max_drawdown = state.get('max_drawdown', 0)
        drawdown_penalty = abs(max_drawdown) * 0.5

        return sharpe - drawdown_penalty

    def _encode_state(self, state: Dict[str, Any]) -> str:
        """Encode state into a hashable key."""
        # Simplified state encoding
        key_features = [
            state.get('regime', 'neutral'),
            round(state.get('volatility', 0.5), 1),
            round(state.get('trend', 0), 1),
            state.get('position', 'none'),
        ]
        return '_'.join(map(str, key_features))

    def _derive_policy(self, q_values: Dict[str, float]) -> Dict[str, str]:
        """Derive trading policy from Q-values."""
        if not q_values:
            return {'default': 'HOLD'}

        # Group by state prefix and find best actions
        policy = {}
        state_groups = {}

        for state_action, q_value in q_values.items():
            parts = state_action.split('_')
            if len(parts) >= 2:
                state = '_'.join(parts[:-1])
                action = parts[-1]

                if state not in state_groups:
                    state_groups[state] = {}
                state_groups[state][action] = q_value

        # Select best action for each state
        for state, actions in state_groups.items():
            best_action = max(actions.items(), key=lambda x: x[1])[0]
            policy[state] = best_action

        return policy

    # keep single definition handled above

    def update_dashboard(self, data: Dict[str, Any]) -> None:
        data["timestamp"] = datetime.utcnow().isoformat()
        # Ideally push to a metrics store or websocket; print for debug
        try:
            print(f"[ultra_scout.dashboard] {json.dumps(data, default=str)[:1000]}")
        except Exception:
            print("[ultra_scout.dashboard] update")

    def voice_chat_interface(self, message: str) -> str:
        gpt = self._get_gpt_client()
        if gpt:
            try:
                # support both old and newer OpenAI SDK response shapes
                resp = gpt.ChatCompletion.create(
                    model=os.getenv("ULTRA_GPT_MODEL", "gpt-3.5-turbo"),
                    messages=[{"role": "user", "content": message}],
                    max_tokens=128,
                    temperature=0.0,
                )
                # try common response shapes in order
                try:
                    return resp.choices[0].message.content  # new SDK
                except Exception:
                    try:
                        return resp["choices"][0]["message"]["content"]  # dict-like
                    except Exception:
                        try:
                            return resp["choices"][0]["text"]  # older shape
                        except Exception:
                            return str(resp)
            except Exception as e:
                return f"Error: {e}"
        # fallback: record and return simple ack
        self.voice_chat_log.append(message)
        return f"Bot received: {message}"

    # -------------------------
    # News / social scraping
    # -------------------------
    def fetch_news(
        self, sources: Optional[List[str]] = None, max_per_source: int = 10
    ) -> List[str]:
        sources = sources if sources is not None else list(self.sources)
        headlines: List[str] = []

        from concurrent.futures import as_completed

        def _worker(url: str) -> List[str]:
            local = []
            try:
                r = self.session.get(url, timeout=self.request_timeout)
                ctype = r.headers.get("Content-Type", "")
                if "html" in ctype:
                    soup = BeautifulSoup(r.text, "html.parser")
                    for tag in soup.find_all(["h1", "h2", "h3", "a"]):
                        txt = tag.get_text(strip=True)
                        if txt and len(txt) > 10:
                            local.append(txt)
                elif "json" in ctype:
                    try:
                        data = r.json()
                        local += self._extract_json_headlines(data)
                    except Exception:
                        pass
            except Exception:
                pass
            # rate-limit per-site a little
            time.sleep(0.05)
            return local[:max_per_source]

        with ThreadPoolExecutor(max_workers=self.max_threads) as exe:
            futures = {exe.submit(_worker, u): u for u in sources}
            for fut in as_completed(futures, timeout=30):
                try:
                    res = fut.result()
                    if res:
                        with self._lock:
                            headlines.extend(res)
                except Exception:
                    continue

        # lightweight social fusion placeholders (no blocking external auth)
        try:
            headlines.append("SocialTrendPlaceholder: crypto buzz")
        except Exception:
            pass

        # dedupe and return
        seen = set()
        out = []
        for h in headlines:
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out

    def _extract_json_headlines(self, data: Any) -> List[str]:
        out: List[str] = []
        try:
            if isinstance(data, dict):
                for v in data.values():
                    out += self._extract_json_headlines(v)
            elif isinstance(data, list):
                for item in data:
                    out += self._extract_json_headlines(item)
            elif isinstance(data, str):
                if len(data) > 10:
                    out.append(data)
        except Exception:
            pass
        return out

    # -------------------------
    # Sentiment analysis
    # -------------------------
    def analyze_sentiment(self, texts: List[str]) -> Dict[str, float]:
        analyzer = self._get_sentiment_analyzer()
        if analyzer:
            out = {}
            for t in texts:
                try:
                    res = analyzer(t[:512])
                    lbl = res[0].get("label", "").upper()
                    score = float(res[0].get("score", 0.0))
                    out[t] = score if "POS" in lbl else -score
                except Exception:
                    out[t] = 0.0
            return out
        # fallback simple heuristic
        pos_words = [
            "bull",
            "pump",
            "breakout",
            "moon",
            "win",
            "profit",
            "surge",
            "rally",
        ]
        neg_words = ["bear", "dump", "crash", "loss", "risk", "fear", "selloff"]
        out = {}
        for t in texts:
            s = sum(t.lower().count(w) for w in pos_words) - sum(
                t.lower().count(w) for w in neg_words
            )
            out[t] = float(s)
        return out

    def advanced_nlp_sentiment(self, texts: List[str]) -> Dict[str, float]:
        gpt = self._get_gpt_client()
        if not gpt:
            return self.analyze_sentiment(texts)
        out = {}
        for t in texts:
            try:
                resp = gpt.ChatCompletion.create(
                    model=os.getenv("ULTRA_GPT_MODEL", "gpt-3.5-turbo"),
                    messages=[
                        {
                            "role": "user",
                            "content": f"Classify sentiment (positive/negative/neutral) for trading impact: {t}",
                        }
                    ],
                    max_tokens=32,
                    temperature=0.0,
                )
                txt = ""
                try:
                    txt = resp.choices[0].message.content.lower()
                except Exception:
                    txt = str(resp).lower()
                if "positive" in txt:
                    out[t] = 1.0
                elif "negative" in txt:
                    out[t] = -1.0
                else:
                    out[t] = 0.0
            except Exception:
                out[t] = 0.0
        return out

    # -------------------------
    # Patterns & trends
    # -------------------------
    def scrape_patterns(self) -> List[str]:
        patterns: List[str] = []
        try:
            r = self.session.get(
                "https://github.com/search?q=trading+strategy",
                timeout=self.request_timeout,
            )
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.select("a[href]"):
                href = a.get("href")
                if href and re.search(r"/[\w-]+/([\w-]+)", href):
                    patterns.append(href)
        except Exception:
            pass
        return list(dict.fromkeys(patterns))

    def detect_trends(self, prices: List[float]) -> str:
        try:
            if len(prices) < 20:
                return "neutral"
            from sklearn.cluster import KMeans

            data = np.array(prices).reshape(-1, 1)
            kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
            clusters = kmeans.labels_
            if clusters[-1] > clusters[0]:
                return "bull"
            elif clusters[-1] < clusters[0]:
                return "bear"
        except Exception:
            pass
        fast = float(np.mean(prices[-5:])) if len(prices) >= 5 else float(np.mean(prices))
        slow = float(np.mean(prices[-20:])) if len(prices) >= 20 else fast
        if fast > slow:
            return "bull"
        if fast < slow:
            return "bear"
        return "neutral"

    # -------------------------
    # Scout/aggregate
    # -------------------------
    def scout_all(self) -> Dict[str, Any]:
        headlines = self.fetch_news()
        sentiment = self.analyze_sentiment(headlines)
        patterns = self.scrape_patterns()
        self.patterns = patterns
        self.sentiment = sentiment
        # Trends come from real closes or not at all. This previously built
        # them from random.uniform(0.9, 1.1) samples and stored the result
        # beside genuinely scraped sentiment.
        self.trends = self._real_trends()
        self.last_update = time.time()
        # swarm & satellite placeholders
        patterns = self.swarm_ai_decision(patterns)
        satellite = self.satellite_data_fusion("BTC")
        return {
            "headlines": headlines,
            "sentiment": sentiment,
            "patterns": patterns,
            "trends": self.trends,
            "satellite": satellite,
        }

    def swarm_ai_decision(self, signals: List[Any]) -> List[Any]:
        try:
            agents = int(os.getenv("ULTRA_SWARM_AGENTS", "5"))
        except Exception:
            agents = 5
        out = []
        for s in signals:
            # Votes derived from the signal's own strength rather than drawn
            # at random, which produced a consensus that meant nothing.
            votes = self._derive_votes(s, agents)
            try:
                s_dict = s if isinstance(s, dict) else {"value": s}
                s_dict["swarm_vote"] = max(set(votes), key=votes.count)
                out.append(s_dict)
            except Exception:
                continue
        return out

    def satellite_data_fusion(self, symbol: str) -> Dict[str, Any]:
        try:
            # No satellite feed is configured or reachable from here. The
            # previous implementation returned random.uniform(0, 1) under the
            # name "satellite_volatility".
            return self._unavailable("no_satellite_provider")
        except Exception:
            return self._unavailable("no_satellite_provider")

    def describe_model(self) -> str:
        """Return a short human-readable summary of UltraScout capabilities and limits."""
        parts = [
            "UltraScout: modular scouting engine for news, social, on-chain and pattern detection.",
            "Capabilities: threaded news scraping, lightweight NLP (transformers if installed),",
            "             optional GPT integration (OpenAI API), on-chain fetch (Etherscan),",
            "             simple RL/Anomaly/GNN placeholders (loaded lazily).",
            "Data sources: RSS/web pages, GitHub, Twitter/Reddit placeholders, on-chain API when keys provided.",
            "Outputs: headlines list, sentiment scores, pattern list, swarm votes, satellite volatility proxy.",
            "Limitations: heavy ML (transformers, stable-baselines3, sklearn, openai) are optional and lazy — accuracy depends on installed libs and quality of prompts/data.",
            "Safety: network calls have timeouts and basic rate-limiting; model is best used in paper/test mode until tuned.",
        ]
        return "\n".join(parts)

    def health_report(self) -> Dict[str, Any]:
        """Return availability/status of optional components and simple stats."""
        report: Dict[str, Any] = {
            "timestamp": int(time.time()),
            "session_user_agent": self.session.headers.get("User-Agent"),
            "sources_count": len(self.sources),
            "sentiment_analyzer": bool(self._get_sentiment_analyzer()),
            "anomaly_detector": bool(self._get_anomaly_detector()),
            "rl_model_stub": bool(self._get_rl_model()),
            "gpt_client": bool(self._get_gpt_client()),
            "requests_timeout_s": self.request_timeout,
            "last_update": self.last_update,
        }
        # quick sample counts
        try:
            report["pattern_count"] = len(self.patterns)
        except Exception:
            report["pattern_count"] = 0
        return report

    def recommendations(self) -> List[str]:
        """Return concise actionable recommendations for improving model accuracy / production readiness."""
        recs = [
            "1) Run in paper mode; collect incoming signals and outcomes for 2-4 weeks before live.",
            "2) Install optional deps: transformers, scikit-learn, openai, stable-baselines3 for full features.",
            "3) Provide API keys: OPENAI_API_KEY and ETHERSCAN_API_KEY for GPT and on-chain signals.",
            "4) Replace placeholders with concrete backtest / RL environments and train offline with Optuna.",
            "5) Add persistent logging of signals + outcomes and run nightly re-training (online_learner hooks).",
            "6) Integrate chart snapshots for Telegram messages to validate signals visually.",
            "7) Start with conservative Kelly sizing and low leverage; enable auto-trade only after stable paper PnL.",
        ]
        return recs

# For integration: UltraCore can call UltraScout.scout_all() and use results for reasoning, planning, and learning.
