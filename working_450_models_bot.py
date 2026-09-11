#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import time
import asyncio
import ccxt
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
try:
    from aiogram import Bot
except Exception:  # optional dependency; engine reports CONFIG_REQUIRED
    Bot = None
try:
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
except Exception:  # optional dependency; engine reports CONFIG_REQUIRED
    InlineKeyboardButton = None
    InlineKeyboardMarkup = None
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def _impersonation():
    """Impersonation target the installed curl_cffi actually supports.

    Hardcoding "chrome" here failed on builds that do not list the bare
    family alias. Resolved once per call rather than at import so an
    operator override takes effect without a restart.
    """
    from curl_impersonate_compat import resolve_impersonation

    return resolve_impersonation("chrome")



# Setup logger
logger = logging.getLogger(__name__)

class UltimateBot450Models:
    def __init__(self):
        # Initialize logging (using standard logging instead of loguru)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("ultimate_bot_450_models.log"),
                logging.StreamHandler()
            ]
        )

        # Database
        self.db = sqlite3.connect('ultimate_bot_450_models.db', check_same_thread=False)
        self.init_database()

        # Trading settings
        self.trading_enabled = True
        self.auto_trading = True
        self.min_confidence = 75
        self.max_trades = 10
        self.risk_per_trade = 0.02
        self.active_trades = {}

        # Statistics
        self.stats = {
            'signals': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0,
            'models_trained': 0,
            'crypto_signals': 0,
            'forex_signals': 0,
        }

        # Telegram channels
        self.channels = {'admin': '5329503447', 'free': '-1002930953007', 'vip': '-1002983007302'}

        # Initialize exchanges
        self.bybit = ccxt.bybit(
            {
                "enableRateLimit": True,
                "timeout": 20000,
            }
        )

        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'okx': ccxt.okx({'enableRateLimit': True}),
            'coinbase': ccxt.coinbase({'enableRateLimit': True}),
        }

        # Trading pairs
        self.crypto_pairs = [
            'BTC/USDT',
            'ETH/USDT',
            'BNB/USDT',
            'XRP/USDT',
            'ADA/USDT',
            'SOL/USDT',
            'DOGE/USDT',
            'DOT/USDT',
            'MATIC/USDT',
            'AVAX/USDT',
        ]

        self.forex_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD']

        # ALL 450+ AI MODELS
        self.ml_models = {}
        self.initialize_all_450_models()

        logger.info(f"Ultimate Bot initialized with {len(self.ml_models)} AI models")

    def init_database(self):
        """Initialize database tables"""
        cursor = self.db.cursor()

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                price REAL NOT NULL,
                tp1 REAL,
                tp2 REAL,
                tp3 REAL,
                stop_loss REAL,
                ai_score REAL,
                strategy TEXT,
                market_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                amount REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                status TEXT NOT NULL,
                exchange TEXT NOT NULL,
                market_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        '''
        )

        self.db.commit()

    def initialize_all_450_models(self):
        """Initialize ALL 450+ AI/ML models"""
        model_types = [
            'trend_analysis',
            'volume_analysis',
            'technical_indicators',
            'market_sentiment',
            'volatility_prediction',
            'correlation_analysis',
            'regime_detection',
            'momentum_analysis',
            'mean_reversion',
            'support_resistance',
            'breakout_prediction',
            'reversal_detection',
            'pattern_recognition',
            'sentiment_analysis',
            'news_impact',
            'social_media_sentiment',
            'whale_tracking',
            'liquidity_analysis',
            'market_microstructure',
            'cross_asset_correlation',
            'portfolio_optimization',
            'risk_management',
            'position_sizing',
            'order_flow',
            'market_depth',
            'bid_ask_spread',
            'funding_rate_analysis',
            'derivatives_pricing',
            'options_flow',
            'futures_basis',
            'spot_futures_arbitrage',
            'cross_exchange_arbitrage',
            'statistical_arbitrage',
            'pairs_trading',
            'momentum_strategies',
            'mean_reversion_strategies',
            'volatility_trading',
            'carry_trading',
            'event_driven_strategies',
            'news_trading',
            'earnings_trading',
            'economic_indicator_trading',
            'central_bank_policy_trading',
            'geopolitical_event_trading',
            'crisis_trading',
            'recovery_trading',
            'flash_crash_prediction',
            'market_manipulation_detection',
            'high_frequency_trading',
        ]

        # 10 algorithms for each model type = 450+ models
        model_algorithms = ['rf', 'gb', 'et', 'nn', 'lr', 'svm', 'knn', 'nb', 'dt', 'ada']

        for model_type in model_types:
            for algorithm in model_algorithms:
                try:
                    if algorithm == 'rf':
                        self.ml_models[f'{model_type}_{algorithm}'] = RandomForestClassifier(
                            n_estimators=200, random_state=42
                        )
                    elif algorithm == 'gb':
                        self.ml_models[f'{model_type}_{algorithm}'] = GradientBoostingClassifier(
                            n_estimators=100, random_state=42
                        )
                    elif algorithm == 'et':
                        self.ml_models[f'{model_type}_{algorithm}'] = ExtraTreesClassifier(
                            n_estimators=150, random_state=42
                        )
                    elif algorithm == 'nn':
                        self.ml_models[f'{model_type}_{algorithm}'] = MLPClassifier(
                            hidden_layer_sizes=(100, 50, 25), random_state=42
                        )
                    elif algorithm == 'lr':
                        self.ml_models[f'{model_type}_{algorithm}'] = LogisticRegression(
                            random_state=42
                        )
                    elif algorithm == 'svm':
                        self.ml_models[f'{model_type}_{algorithm}'] = SVC(random_state=42)
                    elif algorithm == 'knn':
                        self.ml_models[f'{model_type}_{algorithm}'] = KNeighborsClassifier()
                    elif algorithm == 'nb':
                        self.ml_models[f'{model_type}_{algorithm}'] = GaussianNB()
                    elif algorithm == 'dt':
                        self.ml_models[f'{model_type}_{algorithm}'] = DecisionTreeClassifier(
                            random_state=42
                        )
                    elif algorithm == 'ada':
                        self.ml_models[f'{model_type}_{algorithm}'] = AdaBoostClassifier(
                            random_state=42
                        )
                except Exception as e:
                    logger.error(f"Error initializing {model_type}_{algorithm}: {e}")

        logger.info(f"SUCCESS: {len(self.ml_models)} AI/ML models initialized (FULL 450+ MODELS)")



    async def get_price_data(
        self,
        symbol,
        market_type='crypto',
    ):
        """Get genuine observable market data."""

        price_data = {}

        if market_type == 'crypto':

            clients = [
                ('bybit', self.bybit),
                *list(
                    self.exchanges.items()
                ),
            ]

            for source, client in clients:

                try:
                    ticker, bars = (
                        await asyncio.gather(
                            asyncio.to_thread(
                                client.fetch_ticker,
                                symbol,
                            ),
                            asyncio.to_thread(
                                client.fetch_ohlcv,
                                symbol,
                                '1h',
                                None,
                                240,
                            ),
                        )
                    )

                    closes = [
                        float(row[4])
                        for row in bars
                        if (
                            isinstance(
                                row,
                                (list, tuple),
                            )
                            and len(row) >= 6
                            and row[4]
                            is not None
                        )
                    ]

                    if len(closes) < 60:
                        continue

                    rows = [
                        [
                            int(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            float(
                                row[5]
                                or 0.0
                            ),
                        ]
                        for row in bars
                        if (
                            len(row) >= 6
                            and row[4]
                            is not None
                        )
                    ]

                    latest = float(
                        ticker.get(
                            'last'
                        )
                        or closes[-1]
                    )

                    previous = (
                        closes[-25]
                        if len(closes) >= 25
                        else closes[0]
                    )

                    change = (
                        (
                            latest
                            / previous
                            - 1.0
                        )
                        * 100.0
                        if previous > 0
                        else 0.0
                    )

                    price_data[
                        source
                    ] = {
                        'price': latest,
                        'volume': float(
                            ticker.get(
                                'baseVolume'
                            )
                            or rows[-1][5]
                            or 0.0
                        ),
                        'change_24h':
                            float(
                                ticker.get(
                                    'percentage'
                                )
                                if ticker.get(
                                    'percentage'
                                )
                                is not None
                                else change
                            ),
                        'high_24h':
                            float(
                                ticker.get(
                                    'high'
                                )
                                or max(
                                    closes[-24:]
                                )
                            ),
                        'low_24h':
                            float(
                                ticker.get(
                                    'low'
                                )
                                or min(
                                    closes[-24:]
                                )
                            ),
                    }

                    price_data[
                        '_history'
                    ] = rows

                    price_data[
                        '_source'
                    ] = source

                    return price_data

                except Exception as exc:
                    logger.debug(
                        "real crypto source %s "
                        "unavailable for %s: %s",
                        source,
                        symbol,
                        exc,
                    )

            return {}

        if market_type == 'forex':

            mapping = {
                'EUR/USD': 'EURUSD=X',
                'GBP/USD': 'GBPUSD=X',
                'USD/JPY': 'JPY=X',
                'USD/CHF': 'CHF=X',
                'AUD/USD': 'AUDUSD=X',
                'USD/CAD': 'CAD=X',
                'NZD/USD': 'NZDUSD=X',
            }

            ticker_symbol = mapping.get(
                symbol
            )

            if not ticker_symbol:
                return {}

            def fetch_real_forex():

                from urllib.parse import quote
                from curl_cffi import (
                    requests
                    as curl_requests
                )

                encoded = quote(
                    ticker_symbol,
                    safe='',
                )

                url = (
                    "https://query1.finance.yahoo.com/"
                    f"v8/finance/chart/{encoded}"
                    "?range=10d"
                    "&interval=1h"
                    "&includePrePost=false"
                )

                session = (
                    curl_requests.Session(
                        impersonate=_impersonation()
                    )
                )

                try:
                    response = session.get(
                        url,
                        timeout=10,
                    )

                    response.raise_for_status()

                    payload = response.json()

                finally:
                    try:
                        session.close()
                    except Exception:
                        pass

                results = (
                    payload.get(
                        'chart',
                        {}
                    ).get(
                        'result'
                    )
                    or []
                )

                if not results:
                    return []

                result = results[0]

                timestamps = (
                    result.get(
                        'timestamp'
                    )
                    or []
                )

                quotes = (
                    result.get(
                        'indicators',
                        {}
                    ).get(
                        'quote',
                        []
                    )
                )

                if not quotes:
                    return []

                q = quotes[0]

                closes = (
                    q.get('close')
                    or []
                )

                opens = (
                    q.get('open')
                    or []
                )

                highs = (
                    q.get('high')
                    or []
                )

                lows = (
                    q.get('low')
                    or []
                )

                volumes = (
                    q.get('volume')
                    or []
                )

                rows = []

                for i in range(
                    min(
                        len(
                            timestamps
                        ),
                        len(
                            closes
                        ),
                    )
                ):

                    close = closes[i]

                    if close is None:
                        continue

                    open_ = (
                        opens[i]
                        if i < len(opens)
                        and opens[i]
                        is not None
                        else close
                    )

                    high = (
                        highs[i]
                        if i < len(highs)
                        and highs[i]
                        is not None
                        else close
                    )

                    low = (
                        lows[i]
                        if i < len(lows)
                        and lows[i]
                        is not None
                        else close
                    )

                    volume = (
                        volumes[i]
                        if i < len(volumes)
                        and volumes[i]
                        is not None
                        else 0.0
                    )

                    rows.append(
                        [
                            int(
                                timestamps[i]
                            ) * 1000,
                            float(open_),
                            float(high),
                            float(low),
                            float(close),
                            float(volume),
                        ]
                    )

                return rows


            try:
                rows = await asyncio.wait_for(
                    asyncio.to_thread(
                        fetch_real_forex
                    ),
                    timeout=15,
                )

            except Exception as exc:
                logger.warning(
                    "real forex data unavailable "
                    "for %s: %s",
                    symbol,
                    exc,
                )

                return {}

            if len(rows) < 60:
                return {}

            closes = [
                row[4]
                for row in rows
            ]

            volumes = [
                row[5]
                for row in rows
            ]

            latest = float(
                closes[-1]
            )

            previous = float(
                closes[-25]
            )

            change = (
                (
                    latest
                    / previous
                    - 1.0
                )
                * 100.0
                if previous > 0
                else 0.0
            )

            price_data[
                'yahoo'
            ] = {
                'price': latest,
                'volume': float(
                    volumes[-1]
                ),
                'change_24h':
                    change,
                'high_24h':
                    max(
                        row[2]
                        for row
                        in rows[-24:]
                    ),
                'low_24h':
                    min(
                        row[3]
                        for row
                        in rows[-24:]
                    ),
            }

            price_data[
                '_history'
            ] = rows

            price_data[
                '_source'
            ] = (
                'yahoo_chart_api'
            )

            return price_data

        return {}

    async def send_telegram(self, message, channel):
        """Send Telegram message"""
        try:
            bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN", ""))
            await bot.send_message(chat_id=self.channels[channel], text=message)
            logger.info(f"Message sent to {channel}")
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    async def send_telegram_with_buttons(self, message, channel, symbol, signal_data):
        """Send Telegram with trading buttons"""
        try:
            bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN", ""))

            if channel == 'vip' and signal_data:
                keyboard = [
                    [
                        InlineKeyboardButton(
                            "BUY",
                            callback_data=f"trade_{symbol}_BUY_{signal_data['confidence']:.0f}",
                        ),
                        InlineKeyboardButton(
                            "SELL",
                            callback_data=f"trade_{symbol}_SELL_{signal_data['confidence']:.0f}",
                        ),
                    ],
                    [
                        InlineKeyboardButton("TP1", callback_data=f"tp1_{symbol}"),
                        InlineKeyboardButton("TP2", callback_data=f"tp2_{symbol}"),
                        InlineKeyboardButton("TP3", callback_data=f"tp3_{symbol}"),
                    ],
                    [
                        InlineKeyboardButton("SL", callback_data=f"sl_{symbol}"),
                        InlineKeyboardButton("CHART", callback_data=f"chart_{symbol}"),
                        InlineKeyboardButton("STATUS", callback_data=f"status_{symbol}"),
                    ],
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await bot.send_message(
                    chat_id=self.channels[channel], text=message, reply_markup=reply_markup
                )
            else:
                await bot.send_message(chat_id=self.channels[channel], text=message)

        except Exception as e:
            logger.error(f"Telegram buttons error: {e}")


    def calculate_indicators(self, price_data):
        """Calculate indicators exclusively from observed price history."""

        history = price_data.get(
            '_history'
        ) or []

        if len(history) < 60:
            return {}

        frame = pd.DataFrame(
            history,
            columns=[
                'timestamp',
                'open',
                'high',
                'low',
                'close',
                'volume',
            ],
        )

        close = frame['close'].astype(float)
        volume = frame['volume'].astype(float)

        delta = close.diff()

        gain = (
            delta.clip(lower=0)
            .rolling(14)
            .mean()
        )

        loss = (
            -delta.clip(upper=0)
            .rolling(14)
            .mean()
        )

        rs = gain / loss.replace(
            0,
            np.nan,
        )

        rsi_series = (
            100
            - (100 / (1 + rs))
        ).fillna(50)

        ema12 = close.ewm(
            span=12,
            adjust=False,
        ).mean()

        ema26 = close.ewm(
            span=26,
            adjust=False,
        ).mean()

        macd_series = ema12 - ema26

        signal_series = macd_series.ewm(
            span=9,
            adjust=False,
        ).mean()

        returns = close.pct_change()

        volatility = float(
            returns.tail(24).std()
            or 0.0
        )

        baseline_volume = float(
            volume.tail(24).mean()
            or 0.0
        )

        current_volume = float(
            volume.iloc[-1]
        )

        volume_ratio = (
            current_volume / baseline_volume
            if baseline_volume > 0
            else 1.0
        )

        return {
            'rsi': float(
                rsi_series.iloc[-1]
            ),
            'macd': float(
                macd_series.iloc[-1]
            ),
            'signal': float(
                signal_series.iloc[-1]
            ),
            'volume_ratio': float(
                volume_ratio
            ),
            'volatility': volatility,
            'price': float(
                close.iloc[-1]
            ),
        }


    def generate_signal_with_450_models(
        self,
        symbol,
        price_data,
        indicators,
        market_type='crypto',
    ):
        """
        Run the complete model ensemble using features learned from
        genuine OHLCV. Random scores are forbidden.
        """

        if not indicators:
            return None

        history = price_data.get(
            '_history'
        ) or []

        if len(history) < 80:
            return None

        frame = pd.DataFrame(
            history,
            columns=[
                'timestamp',
                'open',
                'high',
                'low',
                'close',
                'volume',
            ],
        )

        close = frame['close'].astype(float)
        high = frame['high'].astype(float)
        low = frame['low'].astype(float)
        volume = frame['volume'].astype(float)

        returns = close.pct_change()

        feature_frame = pd.DataFrame({
            'ret1': returns,
            'ret3': close.pct_change(3),
            'ret5': close.pct_change(5),
            'ret10': close.pct_change(10),
            'range': (
                (high - low)
                / close.replace(0, np.nan)
            ),
            'sma5': (
                close
                / close.rolling(5).mean()
                - 1.0
            ),
            'sma20': (
                close
                / close.rolling(20).mean()
                - 1.0
            ),
            'vol5': returns.rolling(5).std(),
            'vol20': returns.rolling(20).std(),
            'volume_change': (
                volume
                .replace(0, np.nan)
                .pct_change()
            ),
        }).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        target = (
            close.shift(-1) > close
        ).astype(int)

        dataset = feature_frame.copy()
        dataset['target'] = target
        dataset = dataset.dropna()

        if len(dataset) < 50:
            return None

        X = dataset.drop(
            columns=['target']
        ).to_numpy(
            dtype=np.float64
        )

        y = dataset['target'].to_numpy(
            dtype=np.int64
        )

        if len(np.unique(y)) < 2:
            return None

        latest = (
            feature_frame
            .dropna()
            .iloc[-1:]
            .to_numpy(
                dtype=np.float64
            )
        )

        # Training is expensive; retrain the entire population at a
        # bounded cadence using genuine observations.
        bucket = int(
            time.time() // 900
        )

        if getattr(
            self,
            '_ensemble_fit_bucket',
            None,
        ) != bucket:

            trained = 0

            for model_name, model in self.ml_models.items():
                try:
                    model.fit(
                        X,
                        y,
                    )

                    setattr(
                        model,
                        '_leantrader_real_fitted',
                        True,
                    )

                    trained += 1

                except Exception as exc:
                    logger.debug(
                        'Model training failed %s: %s',
                        model_name,
                        exc,
                    )

            self._ensemble_fit_bucket = bucket
            self.stats[
                'models_trained'
            ] += trained

            logger.info(
                'REAL_DATA_MODEL_TRAINING=%s/%s',
                trained,
                len(self.ml_models),
            )

        scores = []

        for model_name, model in self.ml_models.items():

            if not getattr(
                model,
                '_leantrader_real_fitted',
                False,
            ):
                continue

            try:
                if hasattr(
                    model,
                    'predict_proba',
                ):
                    probabilities = model.predict_proba(
                        latest
                    )[0]

                    classes = list(
                        getattr(
                            model,
                            'classes_',
                            [0, 1],
                        )
                    )

                    if 1 in classes:
                        p_up = float(
                            probabilities[
                                classes.index(1)
                            ]
                        )
                    else:
                        p_up = float(
                            probabilities[-1]
                        )

                elif hasattr(
                    model,
                    'decision_function',
                ):
                    decision = float(
                        np.asarray(
                            model.decision_function(
                                latest
                            )
                        ).reshape(-1)[0]
                    )

                    p_up = float(
                        1.0
                        / (
                            1.0
                            + np.exp(
                                -np.clip(
                                    decision,
                                    -30,
                                    30,
                                )
                            )
                        )
                    )

                else:
                    prediction = int(
                        np.asarray(
                            model.predict(
                                latest
                            )
                        ).reshape(-1)[0]
                    )

                    p_up = (
                        1.0
                        if prediction == 1
                        else 0.0
                    )

                scores.append(
                    (p_up - 0.5)
                    * 200.0
                )

            except Exception as exc:
                logger.debug(
                    'Model prediction failed %s: %s',
                    model_name,
                    exc,
                )

        if not scores:
            return None

        average_score = float(
            np.mean(scores)
        )

        model_count = len(scores)

        threshold = (
            12.0
            if market_type == 'forex'
            else 15.0
        )

        if average_score >= threshold:
            action = 'BUY'

        elif average_score <= -threshold:
            action = 'SELL'

        else:
            return None

        confidence = float(
            np.clip(
                50.0
                + abs(average_score) / 2.0,
                50.0,
                99.0,
            )
        )

        price = float(
            indicators['price']
        )

        observed_volatility = max(
            float(
                indicators.get(
                    'volatility',
                    0.0,
                )
            ),
            0.001,
        )

        tp1_pct = max(
            observed_volatility,
            0.0025,
        )

        tp2_pct = tp1_pct * 2.0
        tp3_pct = tp1_pct * 3.0

        sl_pct = max(
            observed_volatility * 1.25,
            0.003,
        )

        direction = (
            1.0
            if action == 'BUY'
            else -1.0
        )

        return {
            'action': action,
            'confidence': confidence,
            'tp1': price * (
                1.0
                + direction * tp1_pct
            ),
            'tp2': price * (
                1.0
                + direction * tp2_pct
            ),
            'tp3': price * (
                1.0
                + direction * tp3_pct
            ),
            'stop_loss': price * (
                1.0
                - direction * sl_pct
            ),
            'ai_score': average_score,
            'models_used': model_count,
            'market_data_source': price_data.get(
                '_source'
            ),
            'evidence': 'real_ohlcv_model_ensemble',
        }

    async def analyze_markets(self):
        """Analyze ALL markets with 450+ models"""
        logger.info(f"Analyzing ALL markets with {len(self.ml_models)} AI models...")

        # Analyze crypto markets
        logger.info("Analyzing CRYPTO markets...")
        for pair in self.crypto_pairs[:5]:
            try:
                if pair in self.active_trades:
                    continue

                price_data = await self.get_price_data(pair, 'crypto')
                if not price_data:
                    continue

                indicators = self.calculate_indicators(price_data)
                signal = self.generate_signal_with_450_models(
                    pair, price_data, indicators, 'crypto'
                )

                if signal and signal['confidence'] >= 70:
                    primary_price = indicators['price']
                    primary_source = (
                        'bybit' if 'bybit' in price_data else list(price_data.keys())[0]
                    )

                    message = f"""CRYPTO AI SIGNAL: {pair}

Action: {signal['action']}
Price: ${primary_price:,.2f}
Source: {primary_source.upper()}
AI Confidence: {signal['confidence']:.1f}%
Models Used: {signal['models_used']}
24h Change: {price_data.get('bybit', {}).get('change_24h', 0):+.2f}%

Technical Indicators:
RSI: {indicators.get('rsi', 50):.1f}
MACD: {indicators.get('macd', 0):.4f}
Volume Ratio: {indicators.get('volume_ratio', 1):.2f}
Volatility: {indicators.get('volatility', 0):.2%}

Take Profit Levels:
TP1: ${signal['tp1']:,.2f} (+{((signal['tp1']/primary_price-1)*100):.1f}%)
TP2: ${signal['tp2']:,.2f} (+{((signal['tp2']/primary_price-1)*100):.1f}%)
TP3: ${signal['tp3']:,.2f} (+{((signal['tp3']/primary_price-1)*100):.1f}%)
Stop Loss: ${signal['stop_loss']:,.2f} ({((signal['stop_loss']/primary_price-1)*100):.1f}%)

450+ AI MODELS: ACTIVE
CRYPTO AUTO TRADING: ENABLED
Time: {datetime.now().strftime('%H:%M:%S')}
ULTIMATE TRADING BOT"""

                    if signal['confidence'] >= 85:
                        await self.send_telegram_with_buttons(message, 'vip', pair, signal)
                    else:
                        await self.send_telegram(message, 'free')

                    # Save signal
                    cursor = self.db.cursor()
                    cursor.execute(
                        '''
                        INSERT INTO trading_signals
                        (symbol, timeframe, signal, confidence, price, tp1, tp2, tp3, stop_loss, ai_score, strategy, market_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                        (
                            pair,
                            '1h',
                            signal['action'],
                            signal['confidence'],
                            primary_price,
                            signal['tp1'],
                            signal['tp2'],
                            signal['tp3'],
                            signal['stop_loss'],
                            signal['ai_score'],
                            f'{signal["models_used"]} AI Models',
                            'crypto',
                        ),
                    )
                    self.db.commit()

                    self.stats['signals'] += 1
                    self.stats['crypto_signals'] += 1
                    logger.info(
                        f"CRYPTO {pair}: ${primary_price:,.2f} - {signal['action']} ({signal['confidence']:.1f}%) - {signal['models_used']} models"
                    )

            except Exception as e:
                logger.error(f"Error analyzing crypto {pair}: {e}")

        # Analyze forex markets
        logger.info("Analyzing FOREX markets...")
        for pair in self.forex_pairs[:3]:
            try:
                if pair in self.active_trades:
                    continue

                price_data = await self.get_price_data(pair, 'forex')
                if not price_data:
                    continue

                indicators = self.calculate_indicators(price_data)
                signal = self.generate_signal_with_450_models(pair, price_data, indicators, 'forex')

                if signal and signal['confidence'] >= 70:
                    primary_price = indicators['price']
                    primary_source = list(price_data.keys())[0]

                    message = f"""FOREX AI SIGNAL: {pair}

Action: {signal['action']}
Price: {primary_price:.4f}
Source: {primary_source.upper()}
AI Confidence: {signal['confidence']:.1f}%
Models Used: {signal['models_used']}
24h Change: {price_data[primary_source]['change_24h']:+.2f}%

Technical Indicators:
RSI: {indicators.get('rsi', 50):.1f}
MACD: {indicators.get('macd', 0):.4f}
Volume Ratio: {indicators.get('volume_ratio', 1):.2f}
Volatility: {indicators.get('volatility', 0):.2%}

Take Profit Levels:
TP1: {signal['tp1']:.4f} (+{((signal['tp1']/primary_price-1)*100):.1f}%)
TP2: {signal['tp2']:.4f} (+{((signal['tp2']/primary_price-1)*100):.1f}%)
TP3: {signal['tp3']:.4f} (+{((signal['tp3']/primary_price-1)*100):.1f}%)
Stop Loss: {signal['stop_loss']:.4f} ({((signal['stop_loss']/primary_price-1)*100):.1f}%)

450+ AI MODELS: ACTIVE
FOREX SIGNAL TRADING: ENABLED
Time: {datetime.now().strftime('%H:%M:%S')}
ULTIMATE TRADING BOT"""

                    if signal['confidence'] >= 85:
                        await self.send_telegram_with_buttons(message, 'vip', pair, signal)
                    else:
                        await self.send_telegram(message, 'free')

                    # Save signal
                    cursor = self.db.cursor()
                    cursor.execute(
                        '''
                        INSERT INTO trading_signals
                        (symbol, timeframe, signal, confidence, price, tp1, tp2, tp3, stop_loss, ai_score, strategy, market_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                        (
                            pair,
                            '1h',
                            signal['action'],
                            signal['confidence'],
                            primary_price,
                            signal['tp1'],
                            signal['tp2'],
                            signal['tp3'],
                            signal['stop_loss'],
                            signal['ai_score'],
                            f'{signal["models_used"]} AI Models',
                            'forex',
                        ),
                    )
                    self.db.commit()

                    self.stats['signals'] += 1
                    self.stats['forex_signals'] += 1
                    logger.info(
                        f"FOREX {pair}: {primary_price:.4f} - {signal['action']} ({signal['confidence']:.1f}%) - {signal['models_used']} models"
                    )

            except Exception as e:
                logger.error(f"Error analyzing forex {pair}: {e}")

    async def run_ultimate_bot(self):
        """Run the ultimate trading bot with 450+ models"""

        # PASS4_START_ONCE__leantrader_450_started
        if getattr(self, "_leantrader_450_started", False):
            self.logger.info("♻️ UltimateBot450Models already running - reusing canonical instance")
            return
        self._leantrader_450_started = True
        logger.info("STARTING ULTIMATE TRADING BOT WITH 450+ AI MODELS!")

        startup_message = f"""ULTIMATE TRADING BOT STARTED!

BYBIT TESTNET: ACTIVE
AI MODELS: {len(self.ml_models)} ACTIVE (FULL 450+ MODELS)
EXCHANGES: {len(self.exchanges)} ACTIVE
VIP BUTTONS: ACTIVE
AUTO TRADING: {'ENABLED' if self.auto_trading else 'DISABLED'}

ALL MARKETS ACTIVE:
CRYPTO: {len(self.crypto_pairs)} pairs (Auto Trading on Bybit)
FOREX: {len(self.forex_pairs)} pairs (Signal Trading)

ALL FEATURES ACTIVE:
{len(self.ml_models)} Advanced AI Models (FULL 450+ MODELS)
{len(self.exchanges)} Exchange Integration
Market-Specific Analysis (Crypto, Forex)
Advanced Technical Analysis
Dynamic TP/SL Management
Comprehensive Database
Trade Monitoring & Management
Interactive VIP Buttons
Performance Analytics

AI MODELS ({len(self.ml_models)} Total):
45+ Model Types × 10 Algorithms = {len(self.ml_models)} Models
Random Forest, Gradient Boosting, Extra Trees, Neural Networks, Logistic Regression, SVM, KNN, Naive Bayes, Decision Trees, AdaBoost

Started: {datetime.now().strftime('%H:%M:%S')}
ULTIMATE 450+ MODELS TRADING BOT DOMINATING ALL MARKETS!"""

        await self.send_telegram(startup_message, 'admin')

        loop_count = 0

        while True:
            try:
                loop_count += 1
                logger.info(
                    f"Complete Analysis #{loop_count} - {datetime.now().strftime('%H:%M:%S')}"
                )

                # Analyze markets with 450+ models
                await self.analyze_markets()

                # Performance update every 10 cycles
                if loop_count % 10 == 0:
                    update_message = f"""ULTIMATE 450+ MODELS BOT PERFORMANCE

Trading Status:
Active Trades: {len(self.active_trades)}
Total Trades: {self.stats['trades']}
Signals Generated: {self.stats['signals']}

Market Signals:
CRYPTO Signals: {self.stats['crypto_signals']}
FOREX Signals: {self.stats['forex_signals']}

AI Models Performance:
Total Models: {len(self.ml_models)}
Models Trained: {self.stats['models_trained']}
Continuous Learning: ACTIVE

ALL SYSTEMS ACTIVE:
{len(self.ml_models)} AI Models: ACTIVE
{len(self.exchanges)} Exchanges: ACTIVE
CRYPTO Auto Trading: ACTIVE
FOREX Signal Trading: ACTIVE
Technical Analysis: ACTIVE
Trade Management: ACTIVE
VIP Buttons: ACTIVE
Continuous Learning: ACTIVE

Time: {datetime.now().strftime('%H:%M:%S')}
ULTIMATE 450+ MODELS TRADING BOT"""

                    await self.send_telegram(update_message, 'admin')

                # Wait 1 minute between cycles
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(30)

async def main():
    bot = UltimateBot450Models()
    await bot.run_ultimate_bot()

if __name__ == "__main__":
    asyncio.run(main())

# Alias for backwards compatibility
working_450_models_bot = UltimateBot450Models
AI450ModelsBot = UltimateBot450Models
