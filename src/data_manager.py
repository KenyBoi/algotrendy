"""
Data manager for fetching and processing financial market data.
Supports stocks, futures, and handles caching for performance.
Optimized for futures day trading with intraday data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
from pathlib import Path
import pickle
from typing import List, Optional, Dict, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, DATA_DIR

logger = logging.getLogger(__name__)

# Futures contract specifications
FUTURES_CONTRACTS = {
    'ES': {  # E-mini S&P 500
        'name': 'E-mini S&P 500',
        'multiplier': 50,
        'tick_size': 0.25,
        'tick_value': 12.50,
        'margin_initial': 1320,  # per contract
        'trading_hours': '09:30-16:00 ET',
        'exchange': 'CME'
    },
    'NQ': {  # E-mini Nasdaq-100
        'name': 'E-mini Nasdaq-100',
        'multiplier': 20,
        'tick_size': 0.25,
        'tick_value': 5.00,
        'margin_initial': 1870,
        'trading_hours': '09:30-16:00 ET',
        'exchange': 'CME'
    },
    'RTY': {  # E-mini Russell 2000
        'name': 'E-mini Russell 2000',
        'multiplier': 50,
        'tick_size': 0.10,
        'tick_value': 5.00,
        'margin_initial': 1180,
        'trading_hours': '09:30-16:00 ET',
        'exchange': 'CME'
    },
    'CL': {  # WTI Crude Oil
        'name': 'WTI Crude Oil',
        'multiplier': 1000,
        'tick_size': 0.01,
        'tick_value': 10.00,
        'margin_initial': 5175,
        'trading_hours': '09:00-14:30 ET',
        'exchange': 'NYMEX'
    },
    'GC': {  # Gold
        'name': 'Gold',
        'multiplier': 100,
        'tick_size': 0.10,
        'tick_value': 10.00,
        'margin_initial': 8250,
        'trading_hours': '08:20-13:30 ET',
        'exchange': 'COMEX'
    }
}

class DataManager:
    """Handles data fetching, processing, and caching for stocks and futures"""

    def __init__(self):
        self.cache_dir = DATA_DIR / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.futures_cache_dir = DATA_DIR / "futures_cache"
        self.futures_cache_dir.mkdir(exist_ok=True)
        
    def fetch_data(self, symbol: str, period: str = "2y", interval: str = "1d",
                    asset_type: str = "stock", chart_style: str = "time") -> pd.DataFrame:
        """
        Fetch stock or futures data from Yahoo Finance with caching and chart style support

        Args:
            symbol: Stock symbol (e.g., 'AAPL') or futures symbol (e.g., 'ES=F')
            period: Period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            asset_type: "stock" or "futures"
            chart_style: Chart style - "time", "tick", "range", "volume", "renko+", "line"

        Returns:
            DataFrame with OHLCV data
        """
        cache_dir = self.futures_cache_dir if asset_type == "futures" else self.cache_dir
        cache_file = cache_dir / f"{symbol}_{period}_{interval}_{chart_style}.pkl"

        # Check if cached data exists and is recent (less than 1 hour old for stocks, 15 min for futures)
        cache_age_limit = timedelta(minutes=15) if asset_type == "futures" else timedelta(hours=1)

        if cache_file.exists():
            try:
                cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - cache_time < cache_age_limit:
                    logger.info(f"Loading cached {asset_type} data for {symbol}")
                    return pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"Error reading cache for {symbol}: {e}")

        try:
            logger.info(f"Fetching fresh {asset_type} data for {symbol} (period={period}, interval={interval})")

            # Handle futures symbols - Yahoo Finance uses =F suffix for futures
            if asset_type == "futures" and not symbol.endswith('=F'):
                symbol = f"{symbol}=F"

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ValueError(f"No data found for {asset_type} symbol {symbol}")

            # Clean column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')

            # For futures, add contract information
            if asset_type == "futures":
                contract_symbol = symbol.replace('=F', '')
                if contract_symbol in FUTURES_CONTRACTS:
                    contract_info = FUTURES_CONTRACTS[contract_symbol]
                    df['contract_symbol'] = contract_symbol
                    df['contract_multiplier'] = contract_info['multiplier']
                    df['tick_size'] = contract_info['tick_size']
                    df['tick_value'] = contract_info['tick_value']

            # Apply chart style transformations if not time-based
            if chart_style != "time":
                df = self._apply_chart_style(df, chart_style, interval)

            # Cache the data
            df.to_pickle(cache_file)
            logger.info(f"Cached {asset_type} data for {symbol} saved")

            return df

        except Exception as e:
            logger.error(f"Error fetching {asset_type} data for {symbol}: {e}")
            raise

    def fetch_futures_data(self, symbol: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
        """
        Fetch futures data optimized for day trading

        Args:
            symbol: Futures symbol (e.g., 'ES' for E-mini S&P 500)
            period: Period for data (optimized for shorter periods for day trading)
            interval: Intraday interval (1m, 5m, 15m, 30m, 1h)

        Returns:
            DataFrame with futures OHLCV data and contract info
        """
        return self.fetch_data(symbol, period, interval, asset_type="futures")

    def get_futures_contract_info(self, symbol: str) -> Dict:
        """
        Get futures contract specifications

        Args:
            symbol: Futures symbol (e.g., 'ES')

        Returns:
            Dictionary with contract specifications
        """
        if symbol in FUTURES_CONTRACTS:
            return FUTURES_CONTRACTS[symbol].copy()
        else:
            raise ValueError(f"Unknown futures contract: {symbol}")

    def get_active_futures_contracts(self) -> List[str]:
        """
        Get list of supported futures contracts

        Returns:
            List of futures symbols
        """
        return list(FUTURES_CONTRACTS.keys())
    
    def calculate_technical_indicators(self, df: pd.DataFrame, asset_type: str = "stock") -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators optimized for stocks or futures

        Args:
            df: DataFrame with OHLCV data
            asset_type: "stock" or "futures" (affects indicator parameters)

        Returns:
            DataFrame with technical indicators added
        """
        try:
            data = df.copy()

            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            logger.info(f"Calculating {asset_type} technical indicators...")

            # Adjust window sizes based on asset type and timeframe
            # Futures day trading needs shorter windows for intraday signals
            if asset_type == "futures":
                sma_windows = [5, 10, 20]  # Shorter for intraday
                rsi_window = 9  # Shorter RSI for futures
                bb_window = 10  # Shorter Bollinger Bands
                atr_window = 7  # Shorter ATR
            else:
                sma_windows = [10, 20, 50]  # Standard for stocks
                rsi_window = 14
                bb_window = 20
                atr_window = 14

            # Price-based indicators
            for i, window in enumerate(sma_windows):
                data[f'sma_{window}'] = ta.trend.sma_indicator(data['close'], window=window)

            data['ema_12'] = ta.trend.ema_indicator(data['close'], window=12)
            data['ema_26'] = ta.trend.ema_indicator(data['close'], window=26)

            # MACD with adjusted parameters for futures
            macd_window_fast = 8 if asset_type == "futures" else 12
            macd_window_slow = 17 if asset_type == "futures" else 26
            macd_window_signal = 6 if asset_type == "futures" else 9

            macd = ta.trend.MACD(data['close'],
                               window_fast=macd_window_fast,
                               window_slow=macd_window_slow,
                               window_sign=macd_window_signal)
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['macd_diff'] = macd.macd_diff()

            # RSI with adjusted window
            data['rsi'] = ta.momentum.rsi(data['close'], window=rsi_window)

            # Bollinger Bands with adjusted window
            bb = ta.volatility.BollingerBands(data['close'], window=bb_window)
            data['bb_upper'] = bb.bollinger_hband()
            data['bb_middle'] = bb.bollinger_mavg()
            data['bb_lower'] = bb.bollinger_lband()
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])

            # Stochastic Oscillator
            data['stoch_k'] = ta.momentum.stoch(data['high'], data['low'], data['close'])
            data['stoch_d'] = ta.momentum.stoch_signal(data['high'], data['low'], data['close'])

            # Average True Range (ATR) - crucial for futures position sizing
            data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=atr_window)

            # Commodity Channel Index (CCI)
            data['cci'] = ta.trend.cci(data['high'], data['low'], data['close'])

            # Williams %R
            data['williams_r'] = ta.momentum.williams_r(data['high'], data['low'], data['close'])

            # Volume indicators
            # Volume SMA (Simple Moving Average of Volume) - implement manually since ta.volume.volume_sma doesn't exist
            volume_window = 10 if asset_type == "futures" else 20
            data['volume_sma'] = data['volume'].rolling(window=volume_window).mean()

            # VWAP (Volume Weighted Average Price) - implement manually since ta function may have issues
            # Manual VWAP calculation: (typical price * volume).cumsum() / volume.cumsum()
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            data['vwap'] = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()

            # Intraday momentum (important for day trading)
            data['price_change'] = data['close'].pct_change()
            data['high_low_ratio'] = data['high'] / data['low']
            data['close_open_ratio'] = data['close'] / data['open']

            # Volatility measures - more important for futures
            vol_windows = [5, 10, 20] if asset_type == "futures" else [10, 20, 50]
            for window in vol_windows:
                data[f'volatility_{window}'] = data['price_change'].rolling(window).std()
                data[f'volume_volatility_{window}'] = data['volume'].pct_change().rolling(window).std()

            # Support and resistance levels
            data['support_20'] = data['low'].rolling(window=20, min_periods=1).min()
            data['resistance_20'] = data['high'].rolling(window=20, min_periods=1).max()

            # Trend indicators
            data['price_above_sma20'] = (data['close'] > data['sma_20']).astype(int)
            data['price_above_sma50'] = (data['close'] > data['sma_50']).astype(int) if 'sma_50' in data.columns else 0
            data['sma_trend'] = (data['sma_10'] > data['sma_20']).astype(int) if 'sma_10' in data.columns else 0

            # Futures-specific indicators
            if asset_type == "futures":
                # Intraday range expansion
                data['range_expansion'] = data['high_low_ratio'].rolling(5).mean()

                # Momentum divergence
                data['momentum_5'] = data['close'].pct_change(5)
                data['momentum_10'] = data['close'].pct_change(10)

                # Volume-price trend
                data['volume_price_trend'] = (data['price_change'] * data['volume']).rolling(5).mean()

                # Opening range breakout (first 30 minutes)
                if hasattr(data.index, 'time'):
                    data['is_opening_range'] = data.index.time < pd.Timestamp('10:00').time()
                    data['opening_high'] = data[data['is_opening_range']]['high'].max()
                    data['opening_low'] = data[data['is_opening_range']]['low'].min()
                    data['above_opening_range'] = (data['close'] > data['opening_high']).astype(int)
                    data['below_opening_range'] = (data['close'] < data['opening_low']).astype(int)

            logger.info(f"Added {len([col for col in data.columns if col not in df.columns])} {asset_type} technical indicators")

            return data

        except Exception as e:
            logger.error(f"Error calculating {asset_type} technical indicators: {e}")
            raise
    
    def create_features(self, df: pd.DataFrame, asset_type: str = "stock",
                       lookback_periods: List[int] = None) -> pd.DataFrame:
        """
        Create additional features for machine learning optimized for stocks or futures

        Args:
            df: DataFrame with price and indicator data
            asset_type: "stock" or "futures"
            lookback_periods: List of periods to create lagged features

        Returns:
            DataFrame with ML features
        """
        try:
            data = df.copy()

            # Adjust lookback periods based on asset type and timeframe
            if lookback_periods is None:
                if asset_type == "futures":
                    # Shorter periods for intraday futures trading
                    lookback_periods = [1, 3, 5, 10, 15]
                else:
                    lookback_periods = [1, 3, 5, 10, 20]

            # Price momentum features
            for period in lookback_periods:
                data[f'return_{period}p'] = data['close'].pct_change(period)
                data[f'volatility_{period}p'] = data['price_change'].rolling(period).std()
                data[f'high_{period}p'] = (data['close'] == data['close'].rolling(period).max()).astype(int)
                data[f'low_{period}p'] = (data['close'] == data['close'].rolling(period).min()).astype(int)

            # Technical indicator signals
            if asset_type == "futures":
                # More sensitive thresholds for futures
                data['rsi_overbought'] = (data['rsi'] > 75).astype(int)
                data['rsi_oversold'] = (data['rsi'] < 25).astype(int)
                data['cci_overbought'] = (data['cci'] > 150).astype(int)
                data['cci_oversold'] = (data['cci'] < -150).astype(int)
            else:
                data['rsi_overbought'] = (data['rsi'] > 70).astype(int)
                data['rsi_oversold'] = (data['rsi'] < 30).astype(int)

            data['bb_squeeze'] = (data['bb_width'] < data['bb_width'].rolling(20).quantile(0.1)).astype(int)
            data['macd_bullish'] = (data['macd'] > data['macd_signal']).astype(int)
            data['macd_bearish'] = (data['macd'] < data['macd_signal']).astype(int)

            # Volume patterns
            vol_window = 10 if asset_type == "futures" else 20
            data['volume_spike'] = (data['volume'] > data['volume'].rolling(vol_window).mean() * 1.5).astype(int)
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(vol_window).mean()

            # Market structure features
            if asset_type == "futures":
                # More sensitive gap detection for futures
                data['gap_up'] = (data['open'] > data['close'].shift(1) * 1.005).astype(int)
                data['gap_down'] = (data['open'] < data['close'].shift(1) * 0.995).astype(int)
            else:
                data['gap_up'] = (data['open'] > data['close'].shift(1) * 1.02).astype(int)
                data['gap_down'] = (data['open'] < data['close'].shift(1) * 0.98).astype(int)

            # Time-based features (more important for futures day trading)
            if hasattr(data.index, 'hour'):
                data['hour'] = data.index.hour
                data['minute'] = data.index.minute
                data['hour_of_day'] = data.index.hour + data.index.minute / 60.0

                # Futures trading session features
                if asset_type == "futures":
                    # Regular trading hours (9:30 AM - 4:00 PM ET)
                    data['is_regular_hours'] = ((data.index.hour >= 9) & (data.index.hour < 16)).astype(int)
                    # Opening range (first 30 minutes)
                    data['is_opening_range'] = ((data.index.hour == 9) & (data.index.minute <= 30)).astype(int)
                    # Closing range (last 30 minutes)
                    data['is_closing_range'] = ((data.index.hour == 15) & (data.index.minute >= 30)).astype(int)

            if hasattr(data.index, 'dayofweek'):
                data['day_of_week'] = data.index.dayofweek
                data['is_monday'] = (data.index.dayofweek == 0).astype(int)
                data['is_friday'] = (data.index.dayofweek == 4).astype(int)

            # Futures-specific features
            if asset_type == "futures":
                # Intraday momentum
                data['intraday_momentum'] = data['close'] - data.groupby(data.index.date)['open'].transform('first')

                # Range expansion/contraction
                data['range_ratio'] = data['high_low_ratio'] / data['high_low_ratio'].rolling(20).mean()

                # Volatility regime
                data['high_volatility'] = (data['atr'] > data['atr'].rolling(50).quantile(0.8)).astype(int)
                data['low_volatility'] = (data['atr'] < data['atr'].rolling(50).quantile(0.2)).astype(int)

                # Trend strength
                data['trend_strength'] = abs(data['close'] - data['close'].shift(20)) / data['atr'].rolling(20).mean()

            logger.info(f"Created {len([col for col in data.columns if col not in df.columns])} {asset_type} ML features")

            return data

        except Exception as e:
            logger.error(f"Error creating {asset_type} features: {e}")
            raise
    
    def create_targets(self, df: pd.DataFrame, prediction_horizon: int = None,
                      profit_threshold: float = None, asset_type: str = "stock") -> pd.DataFrame:
        """
        Create target variables for prediction optimized for stocks or futures

        Args:
            df: DataFrame with price data
            prediction_horizon: Periods ahead to predict (adjusted for timeframe)
            profit_threshold: Minimum profit threshold for positive signal
            asset_type: "stock" or "futures"

        Returns:
            DataFrame with target variables
        """
        try:
            data = df.copy()

            # Adjust parameters based on asset type
            if prediction_horizon is None:
                prediction_horizon = 5 if asset_type == "stock" else 12  # Shorter horizon for futures

            if profit_threshold is None:
                profit_threshold = 0.02 if asset_type == "stock" else 0.005  # Lower threshold for futures due to leverage

            # Future returns
            data['future_return'] = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)

            # Risk-adjusted returns (important for futures)
            if 'atr' in data.columns:
                data['risk_adjusted_return'] = data['future_return'] / (data['atr'] / data['close'])
            else:
                data['risk_adjusted_return'] = data['future_return']

            # Binary classification targets
            data['target_binary'] = (data['future_return'] > profit_threshold).astype(int)

            # Multi-class targets with different thresholds for futures
            if asset_type == "futures":
                # More granular classification for futures due to leverage
                conditions = [
                    data['future_return'] > profit_threshold * 4,  # Strong buy (2%+ expected return)
                    data['future_return'] > profit_threshold * 2,  # Buy (1%+ expected return)
                    data['future_return'] > -profit_threshold,     # Hold
                    data['future_return'] > -profit_threshold * 2, # Sell
                    True  # Strong sell
                ]
                choices = [4, 3, 2, 1, 0]
            else:
                # Standard stock classification
                conditions = [
                    data['future_return'] > profit_threshold * 2,  # Strong buy
                    data['future_return'] > profit_threshold,      # Buy
                    data['future_return'] > -profit_threshold,     # Hold
                    data['future_return'] > -profit_threshold * 2, # Sell
                    True  # Strong sell
                ]
                choices = [4, 3, 2, 1, 0]

            data['target_multiclass'] = np.select(conditions, choices)

            # Regression target (normalized future return)
            data['target_regression'] = data['future_return']

            # Futures-specific targets
            if asset_type == "futures":
                # Intraday momentum targets
                data['target_intraday'] = data['close'].pct_change(6).shift(-6)  # 30min ahead for 5min data

                # Volatility-adjusted targets
                data['target_vol_adj'] = data['future_return'] / data['volatility_20'].fillna(0.01)

                # Directional accuracy target (for classification)
                data['target_direction'] = (data['future_return'] > 0).astype(int)

            logger.info(f"Created {asset_type} target variables with {prediction_horizon}p prediction horizon")

            return data

        except Exception as e:
            logger.error(f"Error creating {asset_type} targets: {e}")
            raise
    
    def prepare_dataset(self, symbol: str, period: str = "2y", interval: str = "1d",
                        asset_type: str = "stock", chart_style: str = "time") -> pd.DataFrame:
        """
        Complete data preparation pipeline for stocks or futures

        Args:
            symbol: Stock symbol (e.g., 'AAPL') or futures symbol (e.g., 'ES')
            period: Data period
            interval: Data interval
            asset_type: "stock" or "futures"
            chart_style: Chart style ("time", "tick", "range", "volume", "renko+", "line")

        Returns:
            Complete dataset ready for ML
        """
        try:
            logger.info(f"Preparing {asset_type} dataset for {symbol}")

            # Fetch raw data
            df = self.fetch_data(symbol, period, interval, asset_type, chart_style)

            # Add technical indicators (optimized for asset type)
            df = self.calculate_technical_indicators(df, asset_type)

            # Create ML features (optimized for asset type)
            df = self.create_features(df, asset_type)

            # Create target variables (adjusted for asset type)
            df = self.create_targets(df, asset_type=asset_type)

            # Drop rows with missing values
            initial_rows = len(df)
            df = df.dropna()
            final_rows = len(df)

            logger.info(f"{asset_type.title()} dataset prepared: {initial_rows} -> {final_rows} rows after cleaning")

            return df

        except Exception as e:
            logger.error(f"Error preparing {asset_type} dataset for {symbol}: {e}")
            raise

    def prepare_futures_dataset(self, symbol: str, period: str = "60d", interval: str = "5m",
                               chart_style: str = "time") -> pd.DataFrame:
        """
        Prepare futures dataset optimized for day trading

        Args:
            symbol: Futures symbol (e.g., 'ES')
            period: Data period (shorter for day trading)
            interval: Intraday interval
            chart_style: Chart style ("time", "tick", "range", "volume", "renko+", "line")

        Returns:
            Futures dataset ready for ML
        """
        return self.prepare_dataset(symbol, period, interval, asset_type="futures", chart_style=chart_style)

    def _apply_chart_style(self, df: pd.DataFrame, chart_style: str, interval: str) -> pd.DataFrame:
        """
        Apply chart style transformations to the data

        Args:
            df: Raw OHLCV DataFrame
            chart_style: Chart style to apply
            interval: Interval parameter (used for tick count, range size, etc.)

        Returns:
            Transformed DataFrame
        """
        try:
            logger.info(f"Applying {chart_style} chart style transformation...")

            if chart_style == "time":
                # No transformation needed for time-based charts
                return df
            elif chart_style == "tick":
                # Tick-based: aggregate every N trades
                try:
                    tick_count = int(interval.replace('tick', '')) if 'tick' in interval else 100
                except:
                    tick_count = 100  # Default 100 ticks

                return self._aggregate_by_ticks(df, tick_count)

            elif chart_style == "range":
                # Range-based: aggregate when price moves by N points
                try:
                    range_size = float(interval.replace('range', '').replace('p', '')) if 'range' in interval else 1.0
                except:
                    range_size = 1.0  # Default $1 range

                return self._aggregate_by_range(df, range_size)

            elif chart_style == "volume":
                # Volume-based: aggregate by volume
                try:
                    volume_size = int(interval.replace('vol', '')) if 'vol' in interval else 1000
                except:
                    volume_size = 1000  # Default 1000 contracts

                return self._aggregate_by_volume(df, volume_size)

            elif chart_style == "renko+":
                # Renko+ (tick-based Renko): Renko bars based on tick movements
                try:
                    brick_size = float(interval.replace('renko', '').replace('p', '')) if 'renko' in interval else 1.0
                except:
                    brick_size = 1.0  # Default 1 point bricks

                return self._create_renko_bars(df, brick_size, tick_based=True)

            elif chart_style == "line":
                # Line chart: simple price line (less useful for ML)
                df_line = df.copy()
                df_line['close'] = df['close']  # Just use close prices
                return df_line

            else:
                logger.warning(f"Unknown chart style: {chart_style}, using time-based")
                return df

        except Exception as e:
            logger.error(f"Error applying {chart_style} chart style: {e}")
            return df

    def _aggregate_by_ticks(self, df: pd.DataFrame, tick_count: int) -> pd.DataFrame:
        """
        Aggregate data by tick count (simulate tick-based bars)

        Args:
            df: OHLCV DataFrame
            tick_count: Number of ticks to aggregate

        Returns:
            Aggregated DataFrame
        """
        # For simulation, we'll use volume as a proxy for tick count
        # In real trading, this would be based on actual tick data
        df_agg = df.copy()

        # Create groups based on cumulative volume (proxy for ticks)
        df_agg['tick_group'] = (df_agg['volume'].cumsum() // tick_count).astype(int)

        # Aggregate by tick groups
        aggregated = df_agg.groupby('tick_group').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Reset index to timestamp
        if 'timestamp' in df_agg.columns:
            aggregated.index = df_agg.groupby('tick_group')['timestamp'].first()
        else:
            aggregated.index = df_agg.groupby('tick_group').apply(lambda x: x.index[0])

        logger.info(f"Aggregated {len(df)} bars into {len(aggregated)} tick-based bars")
        return aggregated

    def _aggregate_by_range(self, df: pd.DataFrame, range_size: float) -> pd.DataFrame:
        """
        Aggregate data by price range

        Args:
            df: OHLCV DataFrame
            range_size: Price range size for aggregation

        Returns:
            Aggregated DataFrame
        """
        bars = []
        current_bar = None

        for idx, row in df.iterrows():
            price = row['close']

            if current_bar is None:
                # Start new bar
                current_bar = {
                    'timestamp': idx,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': row['volume']
                }
            else:
                # Update current bar
                current_bar['high'] = max(current_bar['high'], price)
                current_bar['low'] = min(current_bar['low'], price)
                current_bar['close'] = price
                current_bar['volume'] += row['volume']

                # Check if range threshold is reached
                price_range = current_bar['high'] - current_bar['low']
                if price_range >= range_size:
                    bars.append(current_bar)
                    current_bar = {
                        'timestamp': idx,
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': row['volume']
                    }

        # Add final bar if exists
        if current_bar:
            bars.append(current_bar)

        result_df = pd.DataFrame(bars)
        if not result_df.empty:
            result_df.set_index('timestamp', inplace=True)

        logger.info(f"Created {len(result_df)} range-based bars (range size: {range_size})")
        return result_df

    def _aggregate_by_volume(self, df: pd.DataFrame, volume_size: int) -> pd.DataFrame:
        """
        Aggregate data by volume

        Args:
            df: OHLCV DataFrame
            volume_size: Volume size for aggregation

        Returns:
            Aggregated DataFrame
        """
        df_agg = df.copy()

        # Create groups based on cumulative volume
        df_agg['volume_group'] = (df_agg['volume'].cumsum() // volume_size).astype(int)

        # Aggregate by volume groups
        aggregated = df_agg.groupby('volume_group').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Reset index to timestamp
        if 'timestamp' in df_agg.columns:
            aggregated.index = df_agg.groupby('volume_group')['timestamp'].first()
        else:
            aggregated.index = df_agg.groupby('volume_group').apply(lambda x: x.index[0])

        logger.info(f"Aggregated {len(df)} bars into {len(aggregated)} volume-based bars")
        return aggregated

    def _create_renko_bars(self, df: pd.DataFrame, brick_size: float, tick_based: bool = False) -> pd.DataFrame:
        """
        Create Renko bars from price data

        Args:
            df: OHLCV DataFrame
            brick_size: Size of each Renko brick
            tick_based: Whether to use tick-based logic (Renko+)

        Returns:
            DataFrame with Renko bars
        """
        renko_bars = []
        current_level = df['close'].iloc[0]
        direction = 0  # 0: neutral, 1: up, -1: down

        for idx, row in df.iterrows():
            price = row['close']

            while True:
                if direction == 0:
                    # Determine initial direction
                    if price >= current_level + brick_size:
                        # Up brick
                        renko_bars.append({
                            'timestamp': idx,
                            'open': current_level,
                            'high': current_level + brick_size,
                            'low': current_level,
                            'close': current_level + brick_size,
                            'volume': row['volume'],
                            'direction': 1
                        })
                        current_level += brick_size
                        direction = 1
                    elif price <= current_level - brick_size:
                        # Down brick
                        renko_bars.append({
                            'timestamp': idx,
                            'open': current_level,
                            'high': current_level,
                            'low': current_level - brick_size,
                            'close': current_level - brick_size,
                            'volume': row['volume'],
                            'direction': -1
                        })
                        current_level -= brick_size
                        direction = -1
                    else:
                        break
                elif direction == 1:
                    # Continuing up
                    if price >= current_level + brick_size:
                        renko_bars.append({
                            'timestamp': idx,
                            'open': current_level,
                            'high': current_level + brick_size,
                            'low': current_level,
                            'close': current_level + brick_size,
                            'volume': row['volume'],
                            'direction': 1
                        })
                        current_level += brick_size
                    elif price <= current_level - brick_size:
                        # Reversal
                        renko_bars.append({
                            'timestamp': idx,
                            'open': current_level,
                            'high': current_level,
                            'low': current_level - brick_size,
                            'close': current_level - brick_size,
                            'volume': row['volume'],
                            'direction': -1
                        })
                        current_level -= brick_size
                        direction = -1
                    else:
                        break
                elif direction == -1:
                    # Continuing down
                    if price <= current_level - brick_size:
                        renko_bars.append({
                            'timestamp': idx,
                            'open': current_level,
                            'high': current_level,
                            'low': current_level - brick_size,
                            'close': current_level - brick_size,
                            'volume': row['volume'],
                            'direction': -1
                        })
                        current_level -= brick_size
                    elif price >= current_level + brick_size:
                        # Reversal
                        renko_bars.append({
                            'timestamp': idx,
                            'open': current_level,
                            'high': current_level + brick_size,
                            'low': current_level,
                            'close': current_level + brick_size,
                            'volume': row['volume'],
                            'direction': 1
                        })
                        current_level += brick_size
                        direction = 1
                    else:
                        break

        result_df = pd.DataFrame(renko_bars)
        if not result_df.empty:
            result_df.set_index('timestamp', inplace=True)

        logger.info(f"Created {len(result_df)} Renko bars (brick size: {brick_size})")
        return result_df

if __name__ == "__main__":
    # Example usage
    dm = DataManager()
    
    # Test with a single symbol
    try:
        df = dm.prepare_dataset("AAPL")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Save sample data
        sample_file = DATA_DIR / "sample_aapl_data.csv"
        df.to_csv(sample_file)
        print(f"Sample data saved to: {sample_file}")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        print(f"Error: {e}")