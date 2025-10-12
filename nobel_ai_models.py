#!/usr/bin/env python3
"""
Nobel Prize Hedge Fund AI/ML Models
===================================

Advanced AI/ML models for market prediction, signal generation,
and risk management with continuous learning capabilities.
"""

import numpy as np
import pandas as pd
import joblib
import pickle
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
import asyncio

# ML Libraries
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Advanced ML
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import optuna
from optuna.integration import LightGBMPruningCallback

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten,
    Dropout, BatchNormalization, Attention, MultiHeadAttention,
    Input, Concatenate, Reshape, TimeDistributed
)
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

# Feature Engineering
import talib
from ta import add_all_ta_features
from ta.utils import dropna
import ta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class NobelAIModels:
    """
    Nobel Prize-level AI/ML models for trading
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.ensemble_models = {}
        self.deep_learning_models = {}
        self.time_series_models = {}
        self.performance_metrics = {}
        
        # Model directories
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all AI/ML models"""
        try:
            logger.info("🧠 Initializing Nobel AI Models...")
            
            # Price prediction models
            self.models['price_prediction'] = {
                'random_forest': RandomForestRegressor(
                    n_estimators=200, max_depth=15, random_state=42,
                    n_jobs=-1, warm_start=True
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=150, learning_rate=0.1, max_depth=8,
                    random_state=42
                ),
                'extra_trees': ExtraTreesRegressor(
                    n_estimators=200, max_depth=15, random_state=42,
                    n_jobs=-1
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, n_jobs=-1
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbose=-1
                ),
                'catboost': CatBoostRegressor(
                    iterations=200, depth=8, learning_rate=0.1,
                    random_seed=42, verbose=False
                ),
                'neural_network': MLPRegressor(
                    hidden_layer_sizes=(200, 100, 50),
                    activation='relu', solver='adam',
                    learning_rate='adaptive', max_iter=1000,
                    random_state=42
                ),
                'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=0.1)
            }
            
            # Signal classification models
            self.models['signal_classification'] = {
                'random_forest': RandomForestClassifier(
                    n_estimators=200, max_depth=15, random_state=42,
                    n_jobs=-1, class_weight='balanced'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.1, max_depth=8,
                    random_state=42
                ),
                'extra_trees': ExtraTreesClassifier(
                    n_estimators=200, max_depth=15, random_state=42,
                    n_jobs=-1, class_weight='balanced'
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, n_jobs=-1
                ),
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, n_jobs=-1, verbose=-1,
                    class_weight='balanced'
                ),
                'catboost': CatBoostClassifier(
                    iterations=200, depth=8, learning_rate=0.1,
                    random_seed=42, verbose=False
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(200, 100, 50),
                    activation='relu', solver='adam',
                    learning_rate='adaptive', max_iter=1000,
                    random_state=42
                ),
                'svc': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'naive_bayes': GaussianNB(),
                'knn': KNeighborsClassifier(n_neighbors=5)
            }
            
            # Volatility prediction models
            self.models['volatility_prediction'] = {
                'random_forest': RandomForestRegressor(
                    n_estimators=150, max_depth=12, random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=6,
                    random_state=42
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=150, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1
                ),
                'neural_network': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu', solver='adam',
                    learning_rate='adaptive', max_iter=1000,
                    random_state=42
                )
            }
            
            # Sentiment analysis models
            self.models['sentiment_analysis'] = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42,
                    n_jobs=-1, class_weight='balanced'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=6,
                    random_state=42
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    activation='relu', solver='adam',
                    learning_rate='adaptive', max_iter=1000,
                    random_state=42
                ),
                'naive_bayes': GaussianNB(),
                'svc': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
            }
            
            # Risk assessment models
            self.models['risk_assessment'] = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=6,
                    random_state=42
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1
                )
            }
            
            # Initialize scalers
            for model_type in self.models.keys():
                self.scalers[model_type] = {
                    'standard': StandardScaler(),
                    'minmax': MinMaxScaler(),
                    'robust': RobustScaler()
                }
            
            # Initialize feature selectors
            for model_type in self.models.keys():
                self.feature_selectors[model_type] = {
                    'k_best': SelectKBest(f_regression, k=50),
                    'mutual_info': SelectKBest(mutual_info_regression, k=50),
                    'pca': PCA(n_components=0.95)
                }
            
            # Initialize ensemble models
            self.initialize_ensemble_models()
            
            # Initialize deep learning models
            self.initialize_deep_learning_models()
            
            # Initialize time series models
            self.initialize_time_series_models()
            
            logger.info("✅ Nobel AI Models initialized successfully")
            
        except Exception as e:
            logger.error(f"AI model initialization error: {e}")
            raise
    
    def initialize_ensemble_models(self):
        """Initialize ensemble models"""
        try:
            # Price prediction ensemble
            price_models = [
                ('rf', self.models['price_prediction']['random_forest']),
                ('gb', self.models['price_prediction']['gradient_boosting']),
                ('xgb', self.models['price_prediction']['xgboost']),
                ('lgb', self.models['price_prediction']['lightgbm']),
                ('cat', self.models['price_prediction']['catboost'])
            ]
            
            self.ensemble_models['price_prediction'] = VotingRegressor(
                estimators=price_models,
                weights=[0.2, 0.2, 0.2, 0.2, 0.2]
            )
            
            # Signal classification ensemble
            signal_models = [
                ('rf', self.models['signal_classification']['random_forest']),
                ('gb', self.models['signal_classification']['gradient_boosting']),
                ('xgb', self.models['signal_classification']['xgboost']),
                ('lgb', self.models['signal_classification']['lightgbm']),
                ('cat', self.models['signal_classification']['catboost'])
            ]
            
            self.ensemble_models['signal_classification'] = VotingClassifier(
                estimators=signal_models,
                voting='soft',
                weights=[0.2, 0.2, 0.2, 0.2, 0.2]
            )
            
            logger.info("✅ Ensemble models initialized")
            
        except Exception as e:
            logger.error(f"Ensemble model initialization error: {e}")
    
    def initialize_deep_learning_models(self):
        """Initialize deep learning models"""
        try:
            # LSTM for price prediction
            self.deep_learning_models['lstm_price'] = self.build_lstm_model(
                input_shape=(60, 50),  # 60 timesteps, 50 features
                output_units=1,
                model_type='regression'
            )
            
            # GRU for volatility prediction
            self.deep_learning_models['gru_volatility'] = self.build_gru_model(
                input_shape=(30, 30),  # 30 timesteps, 30 features
                output_units=1,
                model_type='regression'
            )
            
            # CNN for pattern recognition
            self.deep_learning_models['cnn_patterns'] = self.build_cnn_model(
                input_shape=(100, 1),  # 100 timesteps, 1 feature
                output_units=3,  # 3 classes: buy, sell, hold
                model_type='classification'
            )
            
            # Transformer for attention-based prediction
            self.deep_learning_models['transformer'] = self.build_transformer_model(
                input_shape=(60, 50),
                output_units=1,
                model_type='regression'
            )
            
            logger.info("✅ Deep learning models initialized")
            
        except Exception as e:
            logger.error(f"Deep learning model initialization error: {e}")
    
    def build_lstm_model(self, input_shape: Tuple, output_units: int, model_type: str) -> Model:
        """Build LSTM model"""
        try:
            model = Sequential([
                Input(shape=input_shape),
                LSTM(128, return_sequences=True, dropout=0.2),
                LSTM(64, return_sequences=True, dropout=0.2),
                LSTM(32, dropout=0.2),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(output_units, activation='linear' if model_type == 'regression' else 'softmax')
            ])
            
            optimizer = Adam(learning_rate=0.001)
            loss = 'mse' if model_type == 'regression' else 'categorical_crossentropy'
            metrics = ['mae'] if model_type == 'regression' else ['accuracy']
            
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            return model
            
        except Exception as e:
            logger.error(f"LSTM model build error: {e}")
            return None
    
    def build_gru_model(self, input_shape: Tuple, output_units: int, model_type: str) -> Model:
        """Build GRU model"""
        try:
            model = Sequential([
                Input(shape=input_shape),
                GRU(128, return_sequences=True, dropout=0.2),
                GRU(64, return_sequences=True, dropout=0.2),
                GRU(32, dropout=0.2),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(output_units, activation='linear' if model_type == 'regression' else 'softmax')
            ])
            
            optimizer = Adam(learning_rate=0.001)
            loss = 'mse' if model_type == 'regression' else 'categorical_crossentropy'
            metrics = ['mae'] if model_type == 'regression' else ['accuracy']
            
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            return model
            
        except Exception as e:
            logger.error(f"GRU model build error: {e}")
            return None
    
    def build_cnn_model(self, input_shape: Tuple, output_units: int, model_type: str) -> Model:
        """Build CNN model"""
        try:
            model = Sequential([
                Input(shape=input_shape),
                Conv1D(64, 3, activation='relu'),
                MaxPooling1D(2),
                Conv1D(128, 3, activation='relu'),
                MaxPooling1D(2),
                Conv1D(256, 3, activation='relu'),
                GlobalMaxPooling1D(),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(output_units, activation='linear' if model_type == 'regression' else 'softmax')
            ])
            
            optimizer = Adam(learning_rate=0.001)
            loss = 'mse' if model_type == 'regression' else 'categorical_crossentropy'
            metrics = ['mae'] if model_type == 'regression' else ['accuracy']
            
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            return model
            
        except Exception as e:
            logger.error(f"CNN model build error: {e}")
            return None
    
    def build_transformer_model(self, input_shape: Tuple, output_units: int, model_type: str) -> Model:
        """Build Transformer model"""
        try:
            inputs = Input(shape=input_shape)
            
            # Multi-head attention
            attention = MultiHeadAttention(
                num_heads=8, key_dim=64, dropout=0.1
            )(inputs, inputs)
            
            # Add & Norm
            attention = Dropout(0.1)(attention)
            attention = Dense(input_shape[1])(attention)
            attention = tf.keras.layers.Add()([inputs, attention])
            attention = tf.keras.layers.LayerNormalization()(attention)
            
            # Feed forward
            ffn = Dense(256, activation='relu')(attention)
            ffn = Dropout(0.1)(ffn)
            ffn = Dense(input_shape[1])(ffn)
            ffn = tf.keras.layers.Add()([attention, ffn])
            ffn = tf.keras.layers.LayerNormalization()(ffn)
            
            # Global average pooling
            pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn)
            
            # Output layers
            output = Dense(128, activation='relu')(pooled)
            output = Dropout(0.3)(output)
            output = Dense(64, activation='relu')(output)
            output = Dropout(0.2)(output)
            output = Dense(output_units, activation='linear' if model_type == 'regression' else 'softmax')(output)
            
            model = Model(inputs=inputs, outputs=output)
            
            optimizer = Adam(learning_rate=0.001)
            loss = 'mse' if model_type == 'regression' else 'categorical_crossentropy'
            metrics = ['mae'] if model_type == 'regression' else ['accuracy']
            
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            return model
            
        except Exception as e:
            logger.error(f"Transformer model build error: {e}")
            return None
    
    def initialize_time_series_models(self):
        """Initialize time series models"""
        try:
            self.time_series_models = {
                'arima': ARIMA,
                'sarima': SARIMAX,
                'exponential_smoothing': ExponentialSmoothing
            }
            
            logger.info("✅ Time series models initialized")
            
        except Exception as e:
            logger.error(f"Time series model initialization error: {e}")
    
    def prepare_features(self, data: pd.DataFrame, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML models"""
        try:
            # Technical indicators
            features = self.calculate_technical_indicators(data)
            
            # Price features
            features = self.calculate_price_features(features)
            
            # Volume features
            features = self.calculate_volume_features(features)
            
            # Volatility features
            features = self.calculate_volatility_features(features)
            
            # Momentum features
            features = self.calculate_momentum_features(features)
            
            # Trend features
            features = self.calculate_trend_features(features)
            
            # Pattern features
            features = self.calculate_pattern_features(features)
            
            # Market microstructure features
            features = self.calculate_microstructure_features(features)
            
            # Remove NaN values
            features = features.dropna()
            
            if len(features) < 10:
                logger.warning("Insufficient data for feature preparation")
                return np.array([]), np.array([])
            
            # Separate features and target
            feature_columns = [col for col in features.columns if col not in ['target', 'future_price', 'future_return']]
            X = features[feature_columns].values
            y = features['target'].values if 'target' in features.columns else np.zeros(len(features))
            
            # Feature scaling
            scaler = self.scalers[model_type]['standard']
            X_scaled = scaler.fit_transform(X)
            
            # Feature selection
            selector = self.feature_selectors[model_type]['k_best']
            X_selected = selector.fit_transform(X_scaled, y)
            
            return X_selected, y
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return np.array([]), np.array([])
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            df = data.copy()
            
            # Price-based indicators
            df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
            df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
            
            df['ema_5'] = talib.EMA(df['close'], timeperiod=5)
            df['ema_10'] = talib.EMA(df['close'], timeperiod=10)
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            
            # RSI
            df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
            df['rsi_21'] = talib.RSI(df['close'], timeperiod=21)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Williams %R
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'])
            
            # CCI
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
            
            # ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
            
            # ADX
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'])
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'])
            
            # OBV
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
            # MFI
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
            
            return df
            
        except Exception as e:
            logger.error(f"Technical indicators calculation error: {e}")
            return data
    
    def calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based features"""
        try:
            df = data.copy()
            
            # Returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price changes
            df['price_change'] = df['close'] - df['open']
            df['price_range'] = df['high'] - df['low']
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            
            # Price ratios
            df['close_open_ratio'] = df['close'] / df['open']
            df['high_close_ratio'] = df['high'] / df['close']
            df['low_close_ratio'] = df['low'] / df['close']
            
            # Price momentum
            for period in [5, 10, 20, 50]:
                df[f'price_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                df[f'price_roc_{period}'] = df['close'].pct_change(period)
            
            return df
            
        except Exception as e:
            logger.error(f"Price features calculation error: {e}")
            return data
    
    def calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features"""
        try:
            df = data.copy()
            
            # Volume ratios
            for period in [5, 10, 20, 50]:
                df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
            
            # Volume momentum
            for period in [5, 10, 20]:
                df[f'volume_momentum_{period}'] = df['volume'] / df['volume'].shift(period) - 1
            
            # Volume-price features
            df['volume_price_trend'] = df['volume'] * df['returns']
            df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
            
            # On-balance volume
            df['obv_ratio'] = df['obv'] / df['obv'].rolling(20).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Volume features calculation error: {e}")
            return data
    
    def calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility features"""
        try:
            df = data.copy()
            
            # Historical volatility
            for period in [5, 10, 20, 50]:
                df[f'volatility_{period}'] = df['returns'].rolling(period).std()
                df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(50).mean()
            
            # GARCH-like features
            df['volatility_clustering'] = df['returns'].rolling(5).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0)
            
            # Volatility of volatility
            df['vol_vol'] = df['volatility_20'].rolling(20).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Volatility features calculation error: {e}")
            return data
    
    def calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features"""
        try:
            df = data.copy()
            
            # Rate of change
            for period in [5, 10, 20, 50]:
                df[f'roc_{period}'] = df['close'].pct_change(period)
            
            # Momentum
            for period in [5, 10, 20]:
                df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            
            # Price velocity
            df['price_velocity'] = df['close'].diff().diff()
            df['price_acceleration'] = df['price_velocity'].diff()
            
            return df
            
        except Exception as e:
            logger.error(f"Momentum features calculation error: {e}")
            return data
    
    def calculate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend features"""
        try:
            df = data.copy()
            
            # Trend strength
            for period in [5, 10, 20, 50]:
                df[f'trend_strength_{period}'] = np.where(df['close'] > df[f'sma_{period}'], 1, -1)
            
            # Trend consistency
            df['trend_consistency'] = df[['trend_strength_5', 'trend_strength_10', 'trend_strength_20']].sum(axis=1)
            
            # Support and resistance
            df['resistance'] = df['high'].rolling(20).max()
            df['support'] = df['low'].rolling(20).min()
            df['resistance_distance'] = (df['resistance'] - df['close']) / df['close']
            df['support_distance'] = (df['close'] - df['support']) / df['close']
            
            return df
            
        except Exception as e:
            logger.error(f"Trend features calculation error: {e}")
            return data
    
    def calculate_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate pattern recognition features"""
        try:
            df = data.copy()
            
            # Candlestick patterns
            df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
            df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
            
            # Pattern strength
            pattern_columns = ['doji', 'hammer', 'hanging_man', 'engulfing', 'harami', 'morning_star', 'evening_star']
            df['pattern_strength'] = df[pattern_columns].abs().sum(axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Pattern features calculation error: {e}")
            return data
    
    def calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features"""
        try:
            df = data.copy()
            
            # Bid-ask spread proxy
            df['spread_proxy'] = (df['high'] - df['low']) / df['close']
            
            # Price impact
            df['price_impact'] = df['returns'] / np.log(1 + df['volume'])
            
            # Order flow imbalance
            df['order_flow'] = (df['close'] - df['open']) / (df['high'] - df['low'])
            
            # Tick direction
            df['tick_direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
            
            # Tick momentum
            df['tick_momentum'] = df['tick_direction'].rolling(5).sum()
            
            return df
            
        except Exception as e:
            logger.error(f"Microstructure features calculation error: {e}")
            return data
    
    def train_models(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Dict:
        """Train all models for a given type"""
        try:
            if len(X) == 0 or len(y) == 0:
                logger.warning("No data available for training")
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            results = {}
            
            # Train individual models
            for model_name, model in self.models[model_type].items():
                try:
                    logger.info(f"Training {model_name} for {model_type}...")
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    if model_type == 'price_prediction' or model_type == 'volatility_prediction' or model_type == 'risk_assessment':
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results[model_name] = {'mse': mse, 'r2': r2}
                    else:
                        accuracy = accuracy_score(y_test, y_pred)
                        results[model_name] = {'accuracy': accuracy}
                    
                    # Save model
                    model_path = self.model_dir / f"{model_type}_{model_name}.pkl"
                    joblib.dump(model, model_path)
                    
                    logger.info(f"✅ {model_name} trained successfully")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue
            
            # Train ensemble model
            if model_type in self.ensemble_models:
                try:
                    logger.info(f"Training ensemble model for {model_type}...")
                    self.ensemble_models[model_type].fit(X_train, y_train)
                    
                    y_pred_ensemble = self.ensemble_models[model_type].predict(X_test)
                    
                    if model_type == 'price_prediction' or model_type == 'volatility_prediction' or model_type == 'risk_assessment':
                        mse = mean_squared_error(y_test, y_pred_ensemble)
                        r2 = r2_score(y_test, y_pred_ensemble)
                        results['ensemble'] = {'mse': mse, 'r2': r2}
                    else:
                        accuracy = accuracy_score(y_test, y_pred_ensemble)
                        results['ensemble'] = {'accuracy': accuracy}
                    
                    # Save ensemble model
                    ensemble_path = self.model_dir / f"{model_type}_ensemble.pkl"
                    joblib.dump(self.ensemble_models[model_type], ensemble_path)
                    
                    logger.info(f"✅ Ensemble model trained successfully")
                    
                except Exception as e:
                    logger.error(f"Error training ensemble model: {e}")
            
            # Store performance metrics
            self.performance_metrics[model_type] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return {}
    
    def predict(self, X: np.ndarray, model_type: str, model_name: str = 'ensemble') -> np.ndarray:
        """Make predictions using trained models"""
        try:
            if model_name == 'ensemble' and model_type in self.ensemble_models:
                return self.ensemble_models[model_type].predict(X)
            elif model_name in self.models[model_type]:
                return self.models[model_type][model_name].predict(X)
            else:
                logger.error(f"Model {model_name} not found for type {model_type}")
                return np.array([])
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.array([])
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_type: str, model_name: str) -> Dict:
        """Optimize hyperparameters using Optuna"""
        try:
            def objective(trial):
                if model_name == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                    }
                    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                elif model_name == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                    }
                    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                else:
                    return 0.0
                
                # Cross-validation
                scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization error: {e}")
            return {}
    
    def save_models(self):
        """Save all trained models"""
        try:
            for model_type, models in self.models.items():
                for model_name, model in models.items():
                    model_path = self.model_dir / f"{model_type}_{model_name}.pkl"
                    joblib.dump(model, model_path)
            
            # Save scalers
            for model_type, scalers in self.scalers.items():
                for scaler_name, scaler in scalers.items():
                    scaler_path = self.model_dir / f"{model_type}_{scaler_name}_scaler.pkl"
                    joblib.dump(scaler, scaler_path)
            
            # Save feature selectors
            for model_type, selectors in self.feature_selectors.items():
                for selector_name, selector in selectors.items():
                    selector_path = self.model_dir / f"{model_type}_{selector_name}_selector.pkl"
                    joblib.dump(selector, selector_path)
            
            logger.info("✅ All models saved successfully")
            
        except Exception as e:
            logger.error(f"Model saving error: {e}")
    
    def load_models(self):
        """Load all trained models"""
        try:
            for model_type in self.models.keys():
                for model_name in self.models[model_type].keys():
                    model_path = self.model_dir / f"{model_type}_{model_name}.pkl"
                    if model_path.exists():
                        self.models[model_type][model_name] = joblib.load(model_path)
            
            # Load scalers
            for model_type in self.scalers.keys():
                for scaler_name in self.scalers[model_type].keys():
                    scaler_path = self.model_dir / f"{model_type}_{scaler_name}_scaler.pkl"
                    if scaler_path.exists():
                        self.scalers[model_type][scaler_name] = joblib.load(scaler_path)
            
            # Load feature selectors
            for model_type in self.feature_selectors.keys():
                for selector_name in self.feature_selectors[model_type].keys():
                    selector_path = self.model_dir / f"{model_type}_{selector_name}_selector.pkl"
                    if selector_path.exists():
                        self.feature_selectors[model_type][selector_name] = joblib.load(selector_path)
            
            logger.info("✅ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading error: {e}")
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for all models"""
        return self.performance_metrics
    
    def retrain_models(self, new_data: pd.DataFrame):
        """Retrain models with new data"""
        try:
            logger.info("🔄 Retraining models with new data...")
            
            for model_type in self.models.keys():
                X, y = self.prepare_features(new_data, model_type)
                if len(X) > 0 and len(y) > 0:
                    self.train_models(X, y, model_type)
            
            logger.info("✅ Models retrained successfully")
            
        except Exception as e:
            logger.error(f"Model retraining error: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize AI models
    config = {
        'ai': {
            'retrain_interval': 3600,
            'feature_window': 100,
            'prediction_horizon': 10
        }
    }
    
    ai_models = NobelAIModels(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Train models
    for model_type in ai_models.models.keys():
        X, y = ai_models.prepare_features(sample_data, model_type)
        if len(X) > 0 and len(y) > 0:
            results = ai_models.train_models(X, y, model_type)
            print(f"{model_type}: {results}")
    
    # Save models
    ai_models.save_models()
    
    print("✅ Nobel AI Models example completed")