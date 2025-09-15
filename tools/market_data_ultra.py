"""Ultra Advanced Market Data Module - Extended Features.

This module extends the basic market_data.py with ultra-advanced features.
Import this alongside market_data.py for full functionality.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import random
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)


class UltraMarketData:
    """Ultra Advanced Market Data Handler with Professional Trading Features."""

    def __init__(
        self,
        exchange_id: str = "binance",
        cache_dir: Optional[Path] = None,
        enable_ml: bool = True,
        enable_news: bool = True,
        enable_onchain: bool = True,
        max_threads: int = 8,
        api_keys: Optional[Dict[str, str]] = None,
    ):
        """Initialize Ultra Market Data handler."""
        self.exchange_id = exchange_id
        self.cache_dir = cache_dir or Path("runtime/ultra_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.enable_ml = enable_ml
        self.enable_news = enable_news
        self.enable_onchain = enable_onchain
        self.max_threads = max_threads
        self.api_keys = api_keys or {}

        # Initialize components
        self._exchange = None
        self._ml_models = {}
        self._sentiment_analyzer = None
        self._anomaly_detector = None
        self._scaler = None

        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "UltraMarketData/2.0",
            "Accept": "application/json",
        })

        # Thread safety
        self._lock = threading.Lock()

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all components."""
        if self.enable_ml:
            self._init_ml_components()
        if self.enable_news:
            self._init_news_components()

    def _init_ml_components(self):
        """Initialize machine learning components."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            self._anomaly_detector = IsolationForest(
                contamination=0.05,
                random_state=42,
                n_estimators=100
            )
            self._scaler = StandardScaler()
            logger.info("Initialized ML components")
        except ImportError:
            logger.warning("scikit-learn not installed, ML features disabled")

    def _init_news_components(self):
        """Initialize news and sentiment analysis components."""
        try:
            from transformers import pipeline
            self._sentiment_analyzer = pipeline("sentiment-analysis")
            logger.info("Initialized sentiment analyzer")
        except ImportError:
            logger.warning("transformers not installed, using basic sentiment")
            self._sentiment_analyzer = None

    def fetch_ohlcv_ultra(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch OHLCV data with ultra features."""
        # Import the basic fetch function
        try:
            from .market_data import fetch_ohlcv
        except ImportError:
            from market_data import fetch_ohlcv
        
        # Get basic data
        data = fetch_ohlcv(self.exchange_id, symbol, timeframe, since, limit)

        if not data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Add technical indicators
        df = self._add_technical_indicators(df)

        # Add market microstructure features
        df = self._add_microstructure_features(df)

        # Add ML predictions if enabled
        if self.enable_ml and self._anomaly_detector:
            df = self._add_ml_predictions(df)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators."""
        try:
            # Moving Averages
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()
            df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
            df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

            # RSI
            df["rsi_14"] = self._calculate_rsi(df["close"], 14)

            # MACD
            exp1 = df["close"].ewm(span=12, adjust=False).mean()
            exp2 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp1 - exp2
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_histogram"] = df["macd"] - df["macd_signal"]

            # Bollinger Bands
            df["bb_middle"] = df["close"].rolling(window=20).mean()
            bb_std = df["close"].rolling(window=20).std()
            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
            df["bb_width"] = df["bb_upper"] - df["bb_lower"]
            df["bb_percent"] = (df["close"] - df["bb_lower"]) / df["bb_width"]

            # ATR
            df["atr_14"] = self._calculate_atr(df, 14)

            # ADX
            df["adx_14"] = self._calculate_adx(df, 14)

            # Stochastic
            df["stoch_k"], df["stoch_d"] = self._calculate_stochastic(df, 14, 3)

            # Volume indicators
            df["volume_sma"] = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]
            df["obv"] = self._calculate_obv(df)

            # Volatility
            df["volatility"] = df["close"].pct_change().rolling(window=20).std()

            # Support and Resistance
            df["pivot"] = (df["high"] + df["low"] + df["close"]) / 3
            df["r1"] = 2 * df["pivot"] - df["low"]
            df["s1"] = 2 * df["pivot"] - df["high"]

        except Exception as e:
            logger.warning(f"Error adding technical indicators: {e}")

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        try:
            # Price levels
            df["hh"] = df["high"].rolling(window=20).max()
            df["ll"] = df["low"].rolling(window=20).min()

            # Market structure
            df["trend_structure"] = np.where(
                (df["close"] > df["hh"].shift(1)), 1,
                np.where((df["close"] < df["ll"].shift(1)), -1, 0)
            )

            # Fair Value Gaps
            df["fvg_bull"] = np.where(
                (df["low"] > df["high"].shift(2)),
                df["low"] - df["high"].shift(2),
                0
            )
            df["fvg_bear"] = np.where(
                (df["high"] < df["low"].shift(2)),
                df["low"].shift(2) - df["high"],
                0
            )

            # Liquidity Sweeps
            df["liquidity_sweep_high"] = np.where(
                (df["high"] > df["hh"].shift(1)) & (df["close"] < df["hh"].shift(1)),
                1, 0
            )
            df["liquidity_sweep_low"] = np.where(
                (df["low"] < df["ll"].shift(1)) & (df["close"] > df["ll"].shift(1)),
                1, 0
            )

            # Order Flow
            df["delta"] = df["close"] - df["open"]
            df["buying_pressure"] = np.where(df["delta"] > 0, df["volume"], 0)
            df["selling_pressure"] = np.where(df["delta"] < 0, df["volume"], 0)

            # Market Regime
            df["regime"] = self._detect_market_regime(df)

            # Session identification
            df["session"] = df.index.map(self._identify_trading_session)

        except Exception as e:
            logger.warning(f"Error adding microstructure features: {e}")

        return df

    def _add_ml_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML-based predictions."""
        try:
            # Prepare features
            feature_cols = ["close", "volume", "rsi_14", "macd", "atr_14"]
            available_features = [col for col in feature_cols if col in df.columns]

            if len(available_features) >= 3:
                X = df[available_features].dropna()

                if len(X) >= 20:
                    # Anomaly detection
                    X_scaled = self._scaler.fit_transform(X)
                    anomalies = self._anomaly_detector.fit_predict(X_scaled)
                    df.loc[X.index, "anomaly"] = anomalies

                    # ML signals
                    df["ml_signal"] = self._generate_ml_signals(df)
                    df["ml_confidence"] = self._calculate_ml_confidence(df)

        except Exception as e:
            logger.warning(f"ML prediction error: {e}")

        return df

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX."""
        plus_dm = df["high"].diff()
        minus_dm = -df["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = self._calculate_atr(df, 1)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx

    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14, smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df["low"].rolling(window=period).min()
        high_max = df["high"].rolling(window=period).max()
        k_percent = 100 * ((df["close"] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=smooth).mean()
        return k_percent, d_percent

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate OBV."""
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        return obv

    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime."""
        adx = df.get("adx_14", pd.Series(index=df.index, dtype=float))
        volatility = df.get("volatility", pd.Series(index=df.index, dtype=float))

        regime = pd.Series(index=df.index, dtype=str)
        regime[adx > 25] = "trending"
        regime[adx <= 25] = "ranging"
        regime[volatility > volatility.quantile(0.8)] = "volatile"

        return regime

    def _identify_trading_session(self, timestamp: pd.Timestamp) -> str:
        """Identify trading session."""
        hour = timestamp.hour
        if 0 <= hour < 8:
            return "asia"
        elif 8 <= hour < 16:
            return "london"
        elif 16 <= hour < 24:
            return "newyork"
        return "unknown"

    def _generate_ml_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate ML-based trading signals."""
        signals = pd.Series(index=df.index, dtype=float)

        if all(col in df.columns for col in ["rsi_14", "macd", "bb_percent"]):
            # Bullish signal
            bullish = (
                (df["rsi_14"] < 30) |
                ((df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))) |
                (df["bb_percent"] < 0.2)
            )

            # Bearish signal
            bearish = (
                (df["rsi_14"] > 70) |
                ((df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))) |
                (df["bb_percent"] > 0.8)
            )

            signals[bullish] = 1
            signals[bearish] = -1
            signals.fillna(0, inplace=True)

        return signals

    def _calculate_ml_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate confidence score for ML predictions."""
        confidence = pd.Series(index=df.index, dtype=float, data=0.5)

        if "ml_signal" in df.columns:
            # Multiple confirmations increase confidence
            confirmations = 0

            if "rsi_14" in df.columns:
                rsi_confirms = (
                    ((df["ml_signal"] == 1) & (df["rsi_14"] < 40)) |
                    ((df["ml_signal"] == -1) & (df["rsi_14"] > 60))
                )
                confidence += rsi_confirms.astype(float) * 0.15
                confirmations += 1

            if "macd" in df.columns and "macd_signal" in df.columns:
                macd_confirms = (
                    ((df["ml_signal"] == 1) & (df["macd"] > df["macd_signal"])) |
                    ((df["ml_signal"] == -1) & (df["macd"] < df["macd_signal"]))
                )
                confidence += macd_confirms.astype(float) * 0.15
                confirmations += 1

            if "trend_structure" in df.columns:
                trend_confirms = (
                    ((df["ml_signal"] == 1) & (df["trend_structure"] == 1)) |
                    ((df["ml_signal"] == -1) & (df["trend_structure"] == -1))
                )
                confidence += trend_confirms.astype(float) * 0.2
                confirmations += 1

        return confidence.clip(0, 1)

    def fetch_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch news and analyze sentiment."""
        if not self.enable_news:
            return {"enabled": False}

        news_data = []

        # Simplified news fetching
        try:
            # Placeholder for news sources
            sources = [
                f"https://cryptopanic.com/api/v1/posts/?auth_token={
                    self.api_keys.get(
                        'cryptopanic', '')}&currencies={symbol}",
            ]

            for source in sources:
                if self.api_keys.get('cryptopanic'):
                    try:
                        response = self.session.get(source, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            news_data.extend(data.get("results", [])[:5])
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"News fetch error: {e}")

        # Analyze sentiment
        sentiment_scores = []
        for article in news_data:
            score = self._analyze_sentiment(article.get("title", ""))
            sentiment_scores.append(score)
            article["sentiment"] = score

        return {
            "symbol": symbol,
            "articles": news_data,
            "aggregate_sentiment": np.mean(sentiment_scores) if sentiment_scores else 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text."""
        if not text:
            return 0.0

        if self._sentiment_analyzer:
            try:
                result = self._sentiment_analyzer(text[:512])
                label = result[0]["label"].upper()
                score = result[0]["score"]

                if "POS" in label:
                    return score
                elif "NEG" in label:
                    return -score
                else:
                    return 0.0
            except Exception:
                pass

        # Fallback to keyword-based sentiment
        positive_words = ["bull", "bullish", "pump", "moon", "breakout", "surge", "rally"]
        negative_words = ["bear", "bearish", "dump", "crash", "plunge", "loss", "risk"]

        text_lower = text.lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)

        if pos_count + neg_count == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count)

    def fetch_onchain_data(self, token_address: str, chain: str = "ethereum") -> Dict[str, Any]:
        """Fetch on-chain analytics."""
        if not self.enable_onchain:
            return {"enabled": False}

        onchain_data = {
            "token_address": token_address,
            "chain": chain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Etherscan API integration
        api_key = self.api_keys.get("etherscan")
        if api_key:
            try:
                base_url = {
                    "ethereum": "https://api.etherscan.io/api",
                    "bsc": "https://api.bscscan.com/api",
                    "polygon": "https://api.polygonscan.com/api",
                }.get(chain, "https://api.etherscan.io/api")

                url = f"{base_url}?module=account&action=txlist&address={token_address}&apikey={api_key}"
                response = self.session.get(url, timeout=10)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "1":
                        txs = result.get("result", [])
                        onchain_data["tx_count"] = len(txs)
                        onchain_data["whale_txs"] = sum(1 for tx in txs if float(tx.get("value", 0)) > 10**18)
            except Exception as e:
                logger.warning(f"Etherscan API error: {e}")

        return onchain_data

    async def stream_realtime_data(self, symbols: List[str], callback: Callable):
        """Stream real-time market data via WebSocket."""
        try:
            import ccxt.pro as ccxtpro

            exchange = getattr(ccxtpro, self.exchange_id)()

            while True:
                try:
                    for symbol in symbols:
                        ohlcv = await exchange.watch_ohlcv(symbol, "1m")

                        # Convert to DataFrame
                        df = pd.DataFrame(
                            ohlcv,
                            columns=["timestamp", "open", "high", "low", "close", "volume"]
                        )
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df.set_index("timestamp", inplace=True)

                        # Add indicators
                        df = self._add_technical_indicators(df)

                        # Callback with enhanced data
                        await callback(symbol, df)

                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    await asyncio.sleep(5)

        except ImportError:
            logger.error("ccxt.pro not installed, WebSocket streaming unavailable")

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality."""
        validation_report = {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "quality_score": 100.0,
        }

        # Calculate quality score
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        dup_pct = df.duplicated().sum() / len(df)

        validation_report["quality_score"] -= (missing_pct * 20 + dup_pct * 10)
        validation_report["quality_score"] = max(0, validation_report["quality_score"])

        return validation_report


# Example usage
if __name__ == "__main__":
    print("Ultra Market Data Module - Extended Features")
    print("=" * 60)

    # Initialize with ultra features
    ultra_data = UltraMarketData(
        exchange_id="binance",
        enable_ml=True,
        enable_news=False,
        enable_onchain=False
    )

    # Fetch enhanced market data
    print("\nFetching BTC/USDT with ultra features...")
    df = ultra_data.fetch_ohlcv_ultra("BTC/USDT", timeframe="1h", limit=100)

    if not df.empty:
        print(f"\nData shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nLatest values:")
        print(df.iloc[-1])

        # Validate data
        validation = ultra_data.validate_data(df)
        print(f"\nData Quality Score: {validation['quality_score']:.2f}/100")
    else:
        print("No data fetched - check connection")

    print("\n" + "=" * 60)
    print("Ultra features loaded successfully!")
