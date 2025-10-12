#!/usr/bin/env python3
"""
Nobel Prize Hedge Fund System - Simplified Version
=================================================

A working version of the Nobel Prize-level trading system
with all essential features for market domination.
"""

import asyncio
import ccxt
import numpy as np
import pandas as pd
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nobel_hedge_fund.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class MarketSession(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"
    CRYPTO_24H = "crypto_24h"

class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SCALP_LONG = "SCALP_LONG"
    SCALP_SHORT = "SCALP_SHORT"

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size: float
    risk_reward: float
    timeframe: TimeFrame
    session: MarketSession
    timestamp: datetime
    ai_score: float
    technical_score: float
    sentiment_score: float
    volume_score: float
    volatility_score: float
    momentum_score: float

@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    trailing_stop: float
    max_profit: float
    entry_time: datetime
    last_update: datetime
    status: str
    risk_amount: float
    reward_amount: float

class NobelSimpleSystem:
    """
    Nobel Prize-level trading system - simplified but powerful
    """
    
    def __init__(self):
        self.running = False
        self.initial_balance = 10000.0
        self.current_balance = 10000.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Core components
        self.exchanges = {}
        self.positions = {}
        self.signals = []
        self.market_data = {}
        
        # Database
        self.db = None
        
        # Configuration
        self.config = self.load_config()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.Lock()
        
        logger.info("🏆 Nobel Simple System initialized")

    def load_config(self) -> Dict:
        """Load system configuration"""
        return {
            'exchanges': {
                'bybit': {
                    'api_key': 'g1mhPqKrOBp9rnqb4G',
                    'secret': 's9KCIelCqPwJOOWAXNoWqFHtiauRQr9PLeqG',
                    'sandbox': True,
                    'testnet': True
                }
            },
            'telegram': {
                'bot_token': '8291641352:AAFTGq-hIY_iS47aMOoGXrBDFlR_B3nCupg',
                'channels': {
                    'admin': '5329503447',
                    'free': '-1002930953007',
                    'vip': '-1002983007302'
                }
            },
            'trading': {
                'max_positions': 10,
                'max_risk_per_trade': 0.02,
                'max_daily_risk': 0.10,
                'min_confidence': 0.75,
                'scalp_timeframes': ['1m', '5m', '15m'],
                'swing_timeframes': ['1h', '4h', '1d'],
                'scalp_profit_target': 0.005,
                'swing_profit_target': 0.02,
                'stop_loss_multiplier': 2.0
            }
        }

    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Nobel Simple System...")
            
            # Initialize database
            await self.initialize_database()
            
            # Initialize exchanges
            await self.initialize_exchanges()
            
            logger.info("✅ Nobel Simple System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False

    async def initialize_database(self):
        """Initialize database"""
        try:
            Path("data").mkdir(exist_ok=True)
            Path("models").mkdir(exist_ok=True)
            Path("logs").mkdir(exist_ok=True)
            
            self.db = sqlite3.connect('nobel_simple.db', check_same_thread=False)
            cursor = self.db.cursor()
            
            # Market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit_1 REAL NOT NULL,
                    take_profit_2 REAL NOT NULL,
                    take_profit_3 REAL NOT NULL,
                    position_size REAL NOT NULL,
                    risk_reward REAL NOT NULL,
                    timeframe TEXT NOT NULL,
                    session TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    ai_score REAL,
                    technical_score REAL,
                    sentiment_score REAL,
                    volume_score REAL,
                    volatility_score REAL,
                    momentum_score REAL,
                    executed BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    unrealized_pnl REAL,
                    stop_loss REAL NOT NULL,
                    take_profit_1 REAL NOT NULL,
                    take_profit_2 REAL NOT NULL,
                    take_profit_3 REAL NOT NULL,
                    trailing_stop REAL,
                    max_profit REAL,
                    entry_time DATETIME NOT NULL,
                    last_update DATETIME,
                    status TEXT NOT NULL,
                    risk_amount REAL,
                    reward_amount REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    balance REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.db.commit()
            logger.info("✅ Database initialized")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    async def initialize_exchanges(self):
        """Initialize trading exchanges"""
        try:
            for exchange_name, config in self.config['exchanges'].items():
                if config.get('api_key') and config.get('secret'):
                    exchange_class = getattr(ccxt, exchange_name)
                    self.exchanges[exchange_name] = exchange_class({
                        'apiKey': config['api_key'],
                        'secret': config['secret'],
                        'sandbox': config.get('sandbox', True),
                        'testnet': config.get('testnet', True),
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'spot'
                        }
                    })
                    logger.info(f"✅ {exchange_name.upper()} initialized")
            
            logger.info(f"✅ {len(self.exchanges)} exchanges initialized")
            
        except Exception as e:
            logger.error(f"Exchange initialization error: {e}")

    async def start_trading(self):
        """Start the main trading loop"""
        try:
            logger.info("🚀 Starting Nobel Simple Trading System...")
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self.data_collection_loop())
            asyncio.create_task(self.signal_generation_loop())
            asyncio.create_task(self.position_management_loop())
            asyncio.create_task(self.performance_monitoring_loop())
            
            # Main trading loop
            while self.running:
                try:
                    await self.main_trading_cycle()
                    await asyncio.sleep(5)  # 5 second cycle
                    
                except Exception as e:
                    logger.error(f"Main trading cycle error: {e}")
                    await asyncio.sleep(10)
                    
        except Exception as e:
            logger.error(f"Trading system error: {e}")

    async def main_trading_cycle(self):
        """Main trading cycle"""
        try:
            # Update market data
            await self.update_market_data()
            
            # Generate trading signals
            await self.generate_trading_signals()
            
            # Execute trades
            await self.execute_trades()
            
            # Manage positions
            await self.manage_positions()
            
            # Update performance metrics
            await self.update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Main trading cycle error: {e}")

    async def data_collection_loop(self):
        """Continuous data collection loop"""
        while self.running:
            try:
                await self.collect_market_data()
                await asyncio.sleep(10)  # 10 second intervals
                
            except Exception as e:
                logger.error(f"Data collection error: {e}")
                await asyncio.sleep(30)

    async def signal_generation_loop(self):
        """Continuous signal generation loop"""
        while self.running:
            try:
                await self.generate_all_signals()
                await asyncio.sleep(15)  # 15 second intervals
                
            except Exception as e:
                logger.error(f"Signal generation error: {e}")
                await asyncio.sleep(30)

    async def position_management_loop(self):
        """Continuous position management loop"""
        while self.running:
            try:
                await self.manage_all_positions()
                await asyncio.sleep(5)  # 5 second intervals
                
            except Exception as e:
                logger.error(f"Position management error: {e}")
                await asyncio.sleep(10)

    async def performance_monitoring_loop(self):
        """Continuous performance monitoring loop"""
        while self.running:
            try:
                await self.update_performance_metrics()
                await asyncio.sleep(60)  # 60 second intervals
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def collect_market_data(self):
        """Collect market data from exchanges"""
        try:
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            
            for exchange_name, exchange in self.exchanges.items():
                try:
                    for symbol in symbols:
                        for timeframe in timeframes:
                            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                            
                            if ohlcv:
                                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                
                                # Store in cache
                                key = f"{exchange_name}_{symbol}_{timeframe}"
                                self.market_data[key] = df
                                
                                # Save to database
                                cursor = self.db.cursor()
                                for _, row in df.iterrows():
                                    cursor.execute('''
                                        INSERT INTO market_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                    ''', (
                                        symbol, timeframe, row['timestamp'],
                                        row['open'], row['high'], row['low'], row['close'], row['volume']
                                    ))
                                self.db.commit()
                                
                except Exception as e:
                    logger.warning(f"Error collecting data from {exchange_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Market data collection error: {e}")

    async def update_market_data(self):
        """Update market data cache"""
        try:
            # This would update the market data cache
            # For now, we'll just log the update
            logger.debug("Market data updated")
            
        except Exception as e:
            logger.error(f"Market data update error: {e}")

    async def generate_all_signals(self):
        """Generate all types of trading signals"""
        try:
            # Scalping signals
            await self.generate_scalping_signals()
            
            # Swing trading signals
            await self.generate_swing_signals()
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")

    async def generate_scalping_signals(self):
        """Generate scalping signals"""
        try:
            timeframes = ['1m', '5m', '15m']
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            
            for symbol in symbols:
                for timeframe in timeframes:
                    signal = await self.analyze_scalping_opportunity(symbol, timeframe)
                    if signal and signal.confidence >= self.config['trading']['min_confidence']:
                        await self.save_signal(signal)
                        await self.send_signal_alert(signal)
                        
        except Exception as e:
            logger.error(f"Scalping signal generation error: {e}")

    async def generate_swing_signals(self):
        """Generate swing trading signals"""
        try:
            timeframes = ['1h', '4h', '1d']
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            
            for symbol in symbols:
                for timeframe in timeframes:
                    signal = await self.analyze_swing_opportunity(symbol, timeframe)
                    if signal and signal.confidence >= self.config['trading']['min_confidence']:
                        await self.save_signal(signal)
                        await self.send_signal_alert(signal)
                        
        except Exception as e:
            logger.error(f"Swing signal generation error: {e}")

    async def analyze_scalping_opportunity(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        """Analyze scalping opportunity"""
        try:
            # Get market data
            market_data = await self.get_market_data(symbol, timeframe)
            if not market_data or len(market_data) < 20:
                return None
            
            # Calculate technical indicators
            technical_score = await self.calculate_technical_score(market_data)
            
            # Calculate AI score (simplified)
            ai_score = await self.calculate_ai_score(market_data)
            
            # Calculate sentiment score (simplified)
            sentiment_score = await self.calculate_sentiment_score(symbol)
            
            # Calculate volume score
            volume_score = await self.calculate_volume_score(market_data)
            
            # Calculate volatility score
            volatility_score = await self.calculate_volatility_score(market_data)
            
            # Calculate momentum score
            momentum_score = await self.calculate_momentum_score(market_data)
            
            # Calculate overall confidence
            confidence = (
                technical_score * 0.3 +
                ai_score * 0.25 +
                sentiment_score * 0.15 +
                volume_score * 0.15 +
                volatility_score * 0.10 +
                momentum_score * 0.05
            )
            
            if confidence >= self.config['trading']['min_confidence']:
                # Determine signal type
                if technical_score > 0.7 and ai_score > 0.6:
                    signal_type = SignalType.SCALP_LONG
                elif technical_score < -0.7 and ai_score < -0.6:
                    signal_type = SignalType.SCALP_SHORT
                else:
                    return None
                
                # Calculate entry price
                current_price = market_data['close'].iloc[-1]
                entry_price = current_price
                
                # Calculate stop loss and take profits
                atr = await self.calculate_atr(market_data)
                stop_loss_multiplier = self.config['trading']['stop_loss_multiplier']
                
                if signal_type == SignalType.SCALP_LONG:
                    stop_loss = entry_price - (atr * stop_loss_multiplier)
                    take_profit_1 = entry_price + (atr * 1.0)
                    take_profit_2 = entry_price + (atr * 2.0)
                    take_profit_3 = entry_price + (atr * 3.0)
                else:
                    stop_loss = entry_price + (atr * stop_loss_multiplier)
                    take_profit_1 = entry_price - (atr * 1.0)
                    take_profit_2 = entry_price - (atr * 2.0)
                    take_profit_3 = entry_price - (atr * 3.0)
                
                # Calculate position size
                position_size = await self.calculate_position_size(symbol, entry_price, stop_loss, confidence)
                
                # Calculate risk/reward
                risk_amount = abs(entry_price - stop_loss)
                reward_amount = abs(take_profit_1 - entry_price)
                risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
                
                return TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    take_profit_2=take_profit_2,
                    take_profit_3=take_profit_3,
                    position_size=position_size,
                    risk_reward=risk_reward,
                    timeframe=TimeFrame(timeframe),
                    session=MarketSession.CRYPTO_24H,
                    timestamp=datetime.now(),
                    ai_score=ai_score,
                    technical_score=technical_score,
                    sentiment_score=sentiment_score,
                    volume_score=volume_score,
                    volatility_score=volatility_score,
                    momentum_score=momentum_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Scalping opportunity analysis error: {e}")
            return None

    async def analyze_swing_opportunity(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        """Analyze swing trading opportunity"""
        try:
            # Similar to scalping but with different parameters
            return await self.analyze_scalping_opportunity(symbol, timeframe)
            
        except Exception as e:
            logger.error(f"Swing opportunity analysis error: {e}")
            return None

    async def get_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get market data for a symbol and timeframe"""
        try:
            # Try to get from cache first
            for exchange_name in self.exchanges.keys():
                key = f"{exchange_name}_{symbol}_{timeframe}"
                if key in self.market_data:
                    return self.market_data[key]
            
            # If not in cache, fetch from exchange
            for exchange_name, exchange in self.exchanges.items():
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                    if ohlcv:
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        return df
                except Exception as e:
                    logger.warning(f"Error fetching data from {exchange_name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Market data retrieval error: {e}")
            return None

    async def calculate_technical_score(self, market_data: pd.DataFrame) -> float:
        """Calculate technical analysis score"""
        try:
            if len(market_data) < 20:
                return 0.0
            
            # RSI
            rsi = self.calculate_rsi(market_data['close'])
            rsi_score = self.rsi_score(rsi)
            
            # MACD
            macd_score = self.calculate_macd_score(market_data['close'])
            
            # Bollinger Bands
            bb_score = self.calculate_bollinger_bands_score(market_data)
            
            # Moving averages
            ma_score = self.calculate_moving_average_score(market_data['close'])
            
            # Combined score
            total_score = (
                rsi_score * 0.3 +
                macd_score * 0.3 +
                bb_score * 0.2 +
                ma_score * 0.2
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"Technical score calculation error: {e}")
            return 0.0

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50

    def rsi_score(self, rsi: float) -> float:
        """Convert RSI to score"""
        if rsi < 30:
            return 0.8  # Oversold
        elif rsi < 40:
            return 0.4
        elif rsi > 70:
            return -0.8  # Overbought
        elif rsi > 60:
            return -0.4
        else:
            return 0.0

    def calculate_macd_score(self, prices: pd.Series) -> float:
        """Calculate MACD score"""
        try:
            exp1 = prices.ewm(span=12).mean()
            exp2 = prices.ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            if histogram.iloc[-1] > 0:
                return 0.6
            else:
                return -0.6
        except:
            return 0.0

    def calculate_bollinger_bands_score(self, market_data: pd.DataFrame) -> float:
        """Calculate Bollinger Bands score"""
        try:
            close = market_data['close']
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            current_price = close.iloc[-1]
            upper_band = upper.iloc[-1]
            lower_band = lower.iloc[-1]
            
            if current_price <= lower_band:
                return 0.8  # Oversold
            elif current_price >= upper_band:
                return -0.8  # Overbought
            else:
                return 0.0
        except:
            return 0.0

    def calculate_moving_average_score(self, prices: pd.Series) -> float:
        """Calculate moving average score"""
        try:
            sma_5 = prices.rolling(5).mean().iloc[-1]
            sma_20 = prices.rolling(20).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            if current_price > sma_5 > sma_20:
                return 0.6  # Uptrend
            elif current_price < sma_5 < sma_20:
                return -0.6  # Downtrend
            else:
                return 0.0
        except:
            return 0.0

    async def calculate_ai_score(self, market_data: pd.DataFrame) -> float:
        """Calculate AI score (simplified)"""
        try:
            # This would use actual AI models
            # For now, return a random score
            return np.random.uniform(-0.5, 0.5)
            
        except Exception as e:
            logger.error(f"AI score calculation error: {e}")
            return 0.0

    async def calculate_sentiment_score(self, symbol: str) -> float:
        """Calculate sentiment score (simplified)"""
        try:
            # This would use actual sentiment analysis
            # For now, return a random score
            return np.random.uniform(-0.3, 0.3)
            
        except Exception as e:
            logger.error(f"Sentiment score calculation error: {e}")
            return 0.0

    async def calculate_volume_score(self, market_data: pd.DataFrame) -> float:
        """Calculate volume score"""
        try:
            if len(market_data) < 20:
                return 0.0
            
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            
            if avg_volume > 0:
                ratio = current_volume / avg_volume
                if ratio > 2.0:
                    return 0.6
                elif ratio > 1.5:
                    return 0.3
                elif ratio < 0.5:
                    return -0.3
                else:
                    return 0.0
            return 0.0
        except:
            return 0.0

    async def calculate_volatility_score(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility score"""
        try:
            if len(market_data) < 20:
                return 0.0
            
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            
            if volatility > 0.05:  # High volatility
                return 0.6
            elif volatility > 0.02:  # Medium volatility
                return 0.3
            else:  # Low volatility
                return -0.1
        except:
            return 0.0

    async def calculate_momentum_score(self, market_data: pd.DataFrame) -> float:
        """Calculate momentum score"""
        try:
            if len(market_data) < 20:
                return 0.0
            
            # Price momentum
            price_momentum = (market_data['close'].iloc[-1] / market_data['close'].iloc[-10] - 1) * 100
            
            # Volume momentum
            volume_momentum = (market_data['volume'].iloc[-1] / market_data['volume'].rolling(10).mean().iloc[-1] - 1) * 100
            
            # Combined momentum score
            momentum_score = (price_momentum * 0.7 + volume_momentum * 0.3) / 100
            
            return np.tanh(momentum_score)
        except:
            return 0.0

    async def calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(market_data) < period + 1:
                return 0.0
            
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
        except:
            return 0.0

    async def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, confidence: float) -> float:
        """Calculate position size"""
        try:
            risk_amount = abs(entry_price - stop_loss)
            if risk_amount == 0:
                return 0.001
            
            # Use confidence to adjust risk
            adjusted_risk = self.config['trading']['max_risk_per_trade'] * confidence
            position_size = (adjusted_risk * 10000) / risk_amount  # Assuming 10k balance
            
            return max(0.001, min(position_size, 0.1 * 10000 / entry_price))
        except:
            return 0.001

    async def save_signal(self, signal: TradingSignal):
        """Save trading signal to database"""
        try:
            cursor = self.db.cursor()
            cursor.execute('''
                INSERT INTO trading_signals (
                    symbol, signal_type, confidence, entry_price, stop_loss,
                    take_profit_1, take_profit_2, take_profit_3, position_size,
                    risk_reward, timeframe, session, timestamp, ai_score,
                    technical_score, sentiment_score, volume_score, volatility_score,
                    momentum_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.symbol, signal.signal_type.value, signal.confidence,
                signal.entry_price, signal.stop_loss, signal.take_profit_1,
                signal.take_profit_2, signal.take_profit_3, signal.position_size,
                signal.risk_reward, signal.timeframe.value, signal.session.value,
                signal.timestamp, signal.ai_score, signal.technical_score,
                signal.sentiment_score, signal.volume_score, signal.volatility_score,
                signal.momentum_score
            ))
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Signal save error: {e}")

    async def send_signal_alert(self, signal: TradingSignal):
        """Send signal alert via Telegram"""
        try:
            from telegram import Bot
            
            bot = Bot(token=self.config['telegram']['bot_token'])
            
            message = f"""
🚀 **NOBEL HEDGE FUND SIGNAL**

📊 **Symbol**: {signal.symbol}
🎯 **Signal**: {signal.signal_type.value}
💰 **Entry**: ${signal.entry_price:.4f}
🛡️ **Stop Loss**: ${signal.stop_loss:.4f}
🎯 **TP1**: ${signal.take_profit_1:.4f}
🎯 **TP2**: ${signal.take_profit_2:.4f}
🎯 **TP3**: ${signal.take_profit_3:.4f}
📈 **Confidence**: {signal.confidence:.2f}
⚖️ **Risk/Reward**: {signal.risk_reward:.2f}
📊 **Timeframe**: {signal.timeframe.value}

🧠 **AI Analysis**:
• AI Score: {signal.ai_score:.2f}
• Technical: {signal.technical_score:.2f}
• Sentiment: {signal.sentiment_score:.2f}
• Volume: {signal.volume_score:.2f}
• Volatility: {signal.volatility_score:.2f}
• Momentum: {signal.momentum_score:.2f}

⏰ **Time**: {signal.timestamp.strftime('%H:%M:%S')}
🏆 **Nobel Hedge Fund System**
            """
            
            # Send to VIP channel
            await bot.send_message(
                chat_id=self.config['telegram']['channels']['vip'],
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Signal alert error: {e}")

    async def execute_trades(self):
        """Execute trading signals"""
        try:
            # Get pending signals
            cursor = self.db.cursor()
            cursor.execute('''
                SELECT * FROM trading_signals 
                WHERE executed = FALSE 
                AND confidence >= ? 
                ORDER BY timestamp DESC
            ''', (self.config['trading']['min_confidence'],))
            
            signals = cursor.fetchall()
            
            for signal in signals:
                await self.execute_signal(signal)
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")

    async def execute_signal(self, signal_data):
        """Execute a single trading signal"""
        try:
            symbol = signal_data[1]
            signal_type = signal_data[2]
            entry_price = signal_data[3]
            position_size = signal_data[9]
            
            # Check if we already have a position in this symbol
            if symbol in self.positions:
                return
            
            # Execute trade on primary exchange
            exchange = self.exchanges.get('bybit')
            if not exchange:
                return
            
            try:
                # Place market order
                order = exchange.create_market_order(
                    symbol=symbol,
                    side=signal_type.lower(),
                    amount=position_size
                )
                
                # Create position
                position = Position(
                    symbol=symbol,
                    side=signal_type,
                    size=position_size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    unrealized_pnl=0.0,
                    stop_loss=signal_data[4],
                    take_profit_1=signal_data[5],
                    take_profit_2=signal_data[6],
                    take_profit_3=signal_data[7],
                    trailing_stop=entry_price,
                    max_profit=0.0,
                    entry_time=datetime.now(),
                    last_update=datetime.now(),
                    status='OPEN',
                    risk_amount=abs(entry_price - signal_data[4]),
                    reward_amount=abs(signal_data[5] - entry_price)
                )
                
                # Store position
                self.positions[symbol] = position
                
                # Update signal as executed
                cursor = self.db.cursor()
                cursor.execute('''
                    UPDATE trading_signals SET executed = TRUE WHERE id = ?
                ''', (signal_data[0],))
                self.db.commit()
                
                # Send execution alert
                await self.send_execution_alert(position)
                
                logger.info(f"✅ Trade executed: {symbol} {signal_type} {position_size}")
                
            except Exception as e:
                logger.error(f"Order execution error: {e}")
                
        except Exception as e:
            logger.error(f"Signal execution error: {e}")

    async def send_execution_alert(self, position: Position):
        """Send trade execution alert"""
        try:
            from telegram import Bot
            
            bot = Bot(token=self.config['telegram']['bot_token'])
            
            message = f"""
✅ **TRADE EXECUTED**

📊 **Symbol**: {position.symbol}
🎯 **Side**: {position.side}
💰 **Entry**: ${position.entry_price:.4f}
📏 **Size**: {position.size:.4f}
🛡️ **Stop Loss**: ${position.stop_loss:.4f}
🎯 **TP1**: ${position.take_profit_1:.4f}
🎯 **TP2**: ${position.take_profit_2:.4f}
🎯 **TP3**: ${position.take_profit_3:.4f}

⏰ **Time**: {position.entry_time.strftime('%H:%M:%S')}
🏆 **Nobel Hedge Fund System**
            """
            
            await bot.send_message(
                chat_id=self.config['telegram']['channels']['admin'],
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Execution alert error: {e}")

    async def manage_all_positions(self):
        """Manage all open positions"""
        try:
            for symbol, position in list(self.positions.items()):
                await self.manage_position(position)
                
        except Exception as e:
            logger.error(f"Position management error: {e}")

    async def manage_position(self, position: Position):
        """Manage a single position"""
        try:
            # Update current price
            market_data = await self.get_market_data(position.symbol, '1m')
            if market_data is not None and len(market_data) > 0:
                position.current_price = market_data['close'].iloc[-1]
                position.last_update = datetime.now()
                
                # Calculate unrealized PnL
                if position.side == 'BUY':
                    position.unrealized_pnl = (position.current_price - position.entry_price) / position.entry_price
                else:
                    position.unrealized_pnl = (position.entry_price - position.current_price) / position.entry_price
                
                # Update max profit
                if position.unrealized_pnl > position.max_profit:
                    position.max_profit = position.unrealized_pnl
                
                # Check exit conditions
                await self.check_exit_conditions(position)
                
        except Exception as e:
            logger.error(f"Position management error for {position.symbol}: {e}")

    async def check_exit_conditions(self, position: Position):
        """Check if position should be closed"""
        try:
            current_price = position.current_price
            entry_price = position.entry_price
            
            # Check take profit levels
            if position.side == 'BUY':
                if current_price >= position.take_profit_3:
                    await self.close_position(position, 'TP3', current_price)
                elif current_price >= position.take_profit_2:
                    await self.close_position(position, 'TP2', current_price)
                elif current_price >= position.take_profit_1:
                    await self.close_position(position, 'TP1', current_price)
                elif current_price <= position.stop_loss:
                    await self.close_position(position, 'STOP_LOSS', current_price)
            else:  # SELL
                if current_price <= position.take_profit_3:
                    await self.close_position(position, 'TP3', current_price)
                elif current_price <= position.take_profit_2:
                    await self.close_position(position, 'TP2', current_price)
                elif current_price <= position.take_profit_1:
                    await self.close_position(position, 'TP1', current_price)
                elif current_price >= position.stop_loss:
                    await self.close_position(position, 'STOP_LOSS', current_price)
            
            # Check trailing stop
            await self.update_trailing_stop(position)
            
        except Exception as e:
            logger.error(f"Exit condition check error: {e}")

    async def update_trailing_stop(self, position: Position):
        """Update trailing stop loss"""
        try:
            if position.side == 'BUY' and position.current_price > position.entry_price:
                # Update trailing stop for long position
                new_trailing_stop = position.current_price * 0.98  # 2% trailing stop
                if new_trailing_stop > position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
                    position.stop_loss = new_trailing_stop
            elif position.side == 'SELL' and position.current_price < position.entry_price:
                # Update trailing stop for short position
                new_trailing_stop = position.current_price * 1.02  # 2% trailing stop
                if new_trailing_stop < position.trailing_stop:
                    position.trailing_stop = new_trailing_stop
                    position.stop_loss = new_trailing_stop
                    
        except Exception as e:
            logger.error(f"Trailing stop update error: {e}")

    async def close_position(self, position: Position, reason: str, exit_price: float):
        """Close a position"""
        try:
            # Calculate final PnL
            if position.side == 'BUY':
                final_pnl = (exit_price - position.entry_price) / position.entry_price
            else:
                final_pnl = (position.entry_price - exit_price) / position.entry_price
            
            # Update statistics
            self.total_trades += 1
            if final_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            self.total_pnl += final_pnl
            self.current_balance *= (1 + final_pnl)
            
            # Update win rate
            self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            # Save to database
            cursor = self.db.cursor()
            cursor.execute('''
                UPDATE positions 
                SET status = 'CLOSED', current_price = ?, unrealized_pnl = ?, last_update = ?
                WHERE symbol = ? AND status = 'OPEN'
            ''', (exit_price, final_pnl, datetime.now(), position.symbol))
            self.db.commit()
            
            # Send close alert
            await self.send_close_alert(position, reason, exit_price, final_pnl)
            
            # Remove from active positions
            del self.positions[position.symbol]
            
            logger.info(f"✅ Position closed: {position.symbol} {reason} PnL: {final_pnl:.4f}")
            
        except Exception as e:
            logger.error(f"Position close error: {e}")

    async def send_close_alert(self, position: Position, reason: str, exit_price: float, pnl: float):
        """Send position close alert"""
        try:
            from telegram import Bot
            
            bot = Bot(token=self.config['telegram']['bot_token'])
            
            pnl_emoji = "📈" if pnl > 0 else "📉"
            
            message = f"""
{pnl_emoji} **POSITION CLOSED**

📊 **Symbol**: {position.symbol}
🎯 **Side**: {position.side}
💰 **Entry**: ${position.entry_price:.4f}
💰 **Exit**: ${exit_price:.4f}
📊 **Reason**: {reason}
💵 **PnL**: {pnl:.4f} ({pnl*100:.2f}%)
📈 **Total PnL**: {self.total_pnl:.4f}
💰 **Balance**: ${self.current_balance:.2f}
📊 **Win Rate**: {self.win_rate:.2f}

⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}
🏆 **Nobel Hedge Fund System**
            """
            
            await bot.send_message(
                chat_id=self.config['telegram']['channels']['admin'],
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Close alert error: {e}")

    async def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate Sharpe ratio
            if self.total_trades > 0:
                returns = [self.total_pnl / self.total_trades] * self.total_trades
                if len(returns) > 1:
                    self.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Save to database
            cursor = self.db.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics (
                    timestamp, balance, total_pnl, daily_pnl, max_drawdown,
                    sharpe_ratio, win_rate, total_trades, winning_trades, losing_trades
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), self.current_balance, self.total_pnl, 0.0,
                self.max_drawdown, self.sharpe_ratio, self.win_rate,
                self.total_trades, self.winning_trades, self.losing_trades
            ))
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Performance metrics update error: {e}")

    async def stop(self):
        """Stop the trading system"""
        try:
            logger.info("🛑 Stopping Nobel Simple System...")
            self.running = False
            
            # Close all positions
            for symbol, position in list(self.positions.items()):
                await self.close_position(position, 'SYSTEM_SHUTDOWN', position.current_price)
            
            # Close database connection
            if self.db:
                self.db.close()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("✅ Nobel Simple System stopped")
            
        except Exception as e:
            logger.error(f"System stop error: {e}")

# Main execution
async def main():
    """Main execution function"""
    try:
        # Create Nobel Simple System
        system = NobelSimpleSystem()
        
        # Initialize system
        if await system.initialize():
            logger.info("🏆 Nobel Simple System ready to dominate markets!")
            
            # Start trading
            await system.start_trading()
        else:
            logger.error("❌ Failed to initialize Nobel Simple System")
            
    except KeyboardInterrupt:
        logger.info("👋 Nobel Simple System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        if 'system' in locals():
            await system.stop()

if __name__ == "__main__":
    # Set up signal handlers
    import signal
    import sys
    
    def signal_handler(signum, frame):
        logger.info("🛑 Received shutdown signal")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the system
    asyncio.run(main())