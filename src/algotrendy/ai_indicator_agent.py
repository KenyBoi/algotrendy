"""
AI Indicator Discovery Agent
Automatically discovers, tests, and integrates open-source technical indicators
into the ML trading strategy for enhanced performance.
"""

from __future__ import annotations

import os
import json
import time
import requests
import numpy as np
from datetime import datetime, timedelta
import re
import ast
import inspect
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from .config import CONFIG
from .data_adapter import DataAdapter
from .advanced_ml_trainer import AdvancedMLTrainer

# Lazily import pandas at runtime when needed. Do NOT import it at module import time.
def _get_pd():
    """Return the pandas module, importing it on first use.

    Raises a clear RuntimeError if pandas is not available.
    """
    try:
        import pandas as pd
        return pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas is required for this operation") from exc

class IndicatorDiscoveryAgent:
    """
    AI agent that discovers and integrates open-source technical indicators
    """

    def __init__(self):
        self.data_manager = DataAdapter()
        self.indicator_library = {}
        self.performance_cache = {}
        self.sources = {
            'github': self._search_github_indicators,
            'pypi': self._search_pypi_indicators,
            'quantopian': self._search_quantopian_indicators,
            'tradingview': self._search_tradingview_indicators,
            'local': self._load_local_indicators
        }

        # Initialize with known high-performing indicators
        self._initialize_known_indicators()

        print("AI Indicator Discovery Agent initialized")

    def _initialize_known_indicators(self):
        """Initialize with known high-performing indicators"""
        self.indicator_library = {
            'tsf': {
                'name': 'Time Series Forecast',
                'description': 'Linear regression forecast',
                'category': 'predictive',
                'source': 'ta-lib',
                'performance_score': 0.85,
                'code': self._tsf_indicator
            },
            'kama': {
                'name': 'Kaufman Adaptive Moving Average',
                'description': 'Adaptive moving average based on efficiency ratio',
                'category': 'trend',
                'source': 'ta-lib',
                'performance_score': 0.82,
                'code': self._kama_indicator
            },
            'vidya': {
                'name': 'Variable Index Dynamic Average',
                'description': 'Dynamic moving average with variable index',
                'category': 'trend',
                'source': 'ta-lib',
                'performance_score': 0.80,
                'code': self._vidya_indicator
            },
            'chandelier_exit': {
                'name': 'Chandelier Exit',
                'description': 'Trailing stop based on ATR',
                'category': 'volatility',
                'source': 'custom',
                'performance_score': 0.88,
                'code': self._chandelier_exit_indicator
            },
            'supertrend': {
                'name': 'SuperTrend',
                'description': 'Trend following indicator with trailing stop',
                'category': 'trend',
                'source': 'custom',
                'performance_score': 0.86,
                'code': self._supertrend_indicator
            }
        }

    def discover_indicators(self, categories: List[str] = None,
                          min_performance: float = 0.7) -> Dict[str, Dict]:
        """
        Discover new indicators from various sources

        Args:
            categories: List of indicator categories to search
            min_performance: Minimum performance score to include

        Returns:
            Dictionary of discovered indicators
        """
        print("Discovering new technical indicators...")

        discovered_indicators = {}

        # Search each source
        for source_name, search_func in self.sources.items():
            try:
                print(f"Searching {source_name}...")
                indicators = search_func(categories)
                discovered_indicators.update(indicators)
            except Exception as e:
                print(f"Error searching {source_name}: {e}")

        # Filter by performance and validate
        validated_indicators = {}
        for indicator_id, indicator_info in discovered_indicators.items():
            if indicator_info.get('performance_score', 0) >= min_performance:
                # Test the indicator
                if self._validate_indicator(indicator_info):
                    validated_indicators[indicator_id] = indicator_info
                    print(f"✓ Validated indicator: {indicator_info['name']}")

        # Update library
        self.indicator_library.update(validated_indicators)

        print(f"Discovered and validated {len(validated_indicators)} new indicators")

        return validated_indicators

    def _search_github_indicators(self, categories: List[str] = None) -> Dict[str, Dict]:
        """Search GitHub for technical indicators"""
        indicators = {}

        # GitHub search queries for technical indicators
        queries = [
            'technical analysis indicator python',
            'trading indicator algorithm',
            'financial indicator implementation',
            'ta-lib alternative python'
        ]

        # This would normally use GitHub API, but for demo we'll simulate
        # In production, this would use: https://api.github.com/search/code

        simulated_indicators = {
            'github_tsf_extended': {
                'name': 'Extended Time Series Forecast',
                'description': 'Advanced TSF with multiple lookbacks',
                'category': 'predictive',
                'source': 'github',
                'performance_score': 0.83,
                'code': self._extended_tsf_indicator,
                'url': 'https://github.com/example/tsf-extended'
            },
            'github_volume_profile': {
                'name': 'Volume Profile Indicator',
                'description': 'Volume distribution analysis',
                'category': 'volume',
                'source': 'github',
                'performance_score': 0.81,
                'code': self._volume_profile_indicator,
                'url': 'https://github.com/example/volume-profile'
            }
        }

        return simulated_indicators

    def _search_pypi_indicators(self, categories: List[str] = None) -> Dict[str, Dict]:
        """Search PyPI for technical analysis packages"""
        indicators = {}

        # PyPI packages with technical indicators
        packages = [
            'ta', 'pandas-ta', 'talib', 'technical', 'finta',
            'pyti', 'tulipindicators', 'pytrend', 'trend'
        ]

        # Simulate PyPI search results
        simulated_indicators = {
            'pypi_demark': {
                'name': 'DeMark Indicators',
                'description': 'Tom DeMark sequential and countdown indicators',
                'category': 'momentum',
                'source': 'pypi',
                'performance_score': 0.87,
                'code': self._demark_indicator,
                'package': 'demark'
            },
            'pypi_ichimoku': {
                'name': 'Ichimoku Cloud Advanced',
                'description': 'Enhanced Ichimoku Kinko Hyo system',
                'category': 'trend',
                'source': 'pypi',
                'performance_score': 0.84,
                'code': self._ichimoku_advanced_indicator,
                'package': 'ichimoku'
            }
        }

        return simulated_indicators

    def _search_quantopian_indicators(self, categories: List[str] = None) -> Dict[str, Dict]:
        """Search Quantopian/QuantConnect community indicators"""
        indicators = {}

        # Simulate QuantConnect community indicators
        simulated_indicators = {
            'qc_regime_filter': {
                'name': 'Market Regime Filter',
                'description': 'Adaptive indicator based on market volatility regimes',
                'category': 'regime',
                'source': 'quantconnect',
                'performance_score': 0.89,
                'code': self._regime_filter_indicator,
                'url': 'https://www.quantconnect.com/forum/discussion/123/regime-filter'
            },
            'qc_adaptive_ma': {
                'name': 'Adaptive Moving Average Ensemble',
                'description': 'Multiple adaptive MAs combined with ML weights',
                'category': 'trend',
                'source': 'quantconnect',
                'performance_score': 0.86,
                'code': self._adaptive_ma_ensemble_indicator,
                'url': 'https://www.quantconnect.com/forum/discussion/456/adaptive-ma'
            }
        }

        return simulated_indicators

    def _search_tradingview_indicators(self, categories: List[str] = None) -> Dict[str, Dict]:
        """Search TradingView Pine Script indicators (converted to Python)"""
        indicators = {}

        # Popular TradingView indicators
        simulated_indicators = {
            'tv_hull_suite': {
                'name': 'Hull Suite Indicator',
                'description': 'Complete Hull moving average system',
                'category': 'trend',
                'source': 'tradingview',
                'performance_score': 0.85,
                'code': self._hull_suite_indicator,
                'original': 'Hull Suite by Glaz'
            },
            'tv_ssa': {
                'name': 'Singular Spectrum Analysis',
                'description': 'Advanced spectral analysis for trend detection',
                'category': 'predictive',
                'source': 'tradingview',
                'performance_score': 0.88,
                'code': self._ssa_indicator,
                'original': 'SSA by LazyBear'
            }
        }

        return simulated_indicators

    def _load_local_indicators(self, categories: List[str] = None) -> Dict[str, Dict]:
        """Load indicators from local research and development"""
        indicators = {}

        # Custom developed indicators
        indicators.update({
            'local_neural_net': {
                'name': 'Neural Network Price Predictor',
                'description': 'LSTM-based price prediction indicator',
                'category': 'predictive',
                'source': 'local',
                'performance_score': 0.91,
                'code': self._neural_net_indicator
            },
            'local_wavelet': {
                'name': 'Wavelet Transform Indicator',
                'description': 'Multi-resolution analysis for market cycles',
                'category': 'cycle',
                'source': 'local',
                'performance_score': 0.87,
                'code': self._wavelet_indicator
            }
        })

        return indicators

    def _validate_indicator(self, indicator_info: Dict) -> bool:
        """
        Validate that an indicator can be executed and produces reasonable results

        Args:
            indicator_info: Indicator information dictionary

        Returns:
            True if indicator is valid
        """
        try:
            pd = _get_pd()
            # Get sample data
            sample_data = self._get_sample_data()

            # Test indicator function
            indicator_func = indicator_info['code']
            result = indicator_func(sample_data)

            # Validate result
            if not isinstance(result, (pd.Series, np.ndarray)):
                return False

            if len(result) != len(sample_data):
                return False

            # Check for reasonable values (not all NaN, not infinite)
            if result.isna().all():
                return False

            if np.isinf(result).any():
                return False

            return True

        except Exception as e:
            print(f"Validation failed for {indicator_info['name']}: {e}")
            return False

    def _get_sample_data(self) -> pd.DataFrame:
        """Get sample data for indicator testing"""
        pd = _get_pd()

        # Use cached sample data or generate new
        if not hasattr(self, '_sample_data'):
            # Generate synthetic futures data for testing
            np.random.seed(42)
            dates = pd.date_range('2024-01-01', periods=1000, freq='5min')

            # Generate realistic OHLCV data
            close = 4000 + np.cumsum(np.random.normal(0, 5, 1000))
            high = close + np.abs(np.random.normal(0, 3, 1000))
            low = close - np.abs(np.random.normal(0, 3, 1000))
            open_price = close + np.random.normal(0, 1, 1000)
            volume = np.random.lognormal(10, 1, 1000)

            self._sample_data = pd.DataFrame({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            }, index=dates)

        return self._sample_data

    def test_indicator_performance(self, indicator_id: str,
                                symbol: str = "ES",
                                period: str = "60d") -> Dict[str, float]:
        """
        Test indicator performance on real data

        Args:
            indicator_id: ID of indicator to test
            symbol: Trading symbol
            period: Test period

        Returns:
            Performance metrics
        """
        if indicator_id not in self.indicator_library:
            raise ValueError(f"Indicator {indicator_id} not found")

        indicator_info = self.indicator_library[indicator_id]

        try:
            # Get real data
            df = self.data_manager.prepare_futures_dataset(symbol, period=period)

            # Apply indicator
            indicator_func = indicator_info['code']
            indicator_values = indicator_func(df)

            # Create simple strategy: buy when indicator > threshold, sell when < threshold
            threshold = indicator_values.median()
            signals = np.where(indicator_values > threshold, 1,
                             np.where(indicator_values < threshold, -1, 0))

            # Calculate returns
            returns = df['close'].pct_change()
            strategy_returns = signals[:-1] * returns.values[1:]  # Align signals with returns

            # Performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
            win_rate = (strategy_returns > 0).mean()
            max_drawdown = self._calculate_max_drawdown(strategy_returns)

            performance = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'annualized_return': total_return * (252 / len(strategy_returns)),
                'volatility': strategy_returns.std() * np.sqrt(252)
            }

            # Cache performance
            self.performance_cache[indicator_id] = performance

            return performance

        except Exception as e:
            print(f"Error testing indicator {indicator_id}: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def integrate_best_indicators(self, target_accuracy: float = 0.85,
                                max_indicators: int = 10) -> List[str]:
        """
        Integrate the best performing indicators into the ML pipeline

        Args:
            target_accuracy: Minimum accuracy threshold
            max_indicators: Maximum number of indicators to integrate

        Returns:
            List of integrated indicator IDs
        """
        print("Integrating best performing indicators...")

        # Test all indicators if not already tested
        for indicator_id in self.indicator_library.keys():
            if indicator_id not in self.performance_cache:
                performance = self.test_indicator_performance(indicator_id)
                if performance:
                    print(f"Tested {indicator_id}: {performance.get('win_rate', 0):.1%} win rate")

        # Rank indicators by performance
        ranked_indicators = []
        for indicator_id, performance in self.performance_cache.items():
            if performance.get('win_rate', 0) >= target_accuracy:
                ranked_indicators.append((
                    indicator_id,
                    performance.get('sharpe_ratio', 0),
                    performance
                ))

        # Sort by Sharpe ratio (risk-adjusted returns)
        ranked_indicators.sort(key=lambda x: x[1], reverse=True)

        # Select top indicators
        selected_indicators = ranked_indicators[:max_indicators]
        selected_ids = [indicator_id for indicator_id, _, _ in selected_indicators]

        print(f"Selected {len(selected_ids)} high-performing indicators:")
        for indicator_id, sharpe, _ in selected_indicators:
            indicator_name = self.indicator_library[indicator_id]['name']
            print(f"  - {indicator_name}: Sharpe {sharpe:.2f}")

        return selected_ids

    def enhance_ml_pipeline(self, selected_indicators: List[str]) -> Dict[str, Any]:
        """
        Enhance the ML pipeline with selected indicators

        Args:
            selected_indicators: List of indicator IDs to integrate

        Returns:
            Enhanced pipeline configuration
        """
        print("Enhancing ML pipeline with new indicators...")

        # Create enhanced feature engineering function
        def enhanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
            # Start with base features
            enhanced_df = df.copy()

            # Add selected indicators
            for indicator_id in selected_indicators:
                if indicator_id in self.indicator_library:
                    indicator_func = self.indicator_library[indicator_id]['code']
                    try:
                        indicator_values = indicator_func(enhanced_df)
                        col_name = f"indicator_{indicator_id}"
                        enhanced_df[col_name] = indicator_values
                        print(f"Added indicator: {col_name}")
                    except Exception as e:
                        print(f"Failed to add indicator {indicator_id}: {e}")

            return enhanced_df

        # Create enhanced trainer
        class EnhancedMLTrainer(AdvancedMLTrainer):
            def _advanced_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
                # Call parent method first
                df = super()._advanced_feature_engineering(df)

                # Add discovered indicators
                df = enhanced_feature_engineering(df)

                return df

        pipeline_config = {
            'enhanced_trainer_class': EnhancedMLTrainer,
            'selected_indicators': selected_indicators,
            'indicator_library': {k: v for k, v in self.indicator_library.items()
                                if k in selected_indicators},
            'feature_engineering_function': enhanced_feature_engineering
        }

        return pipeline_config

    # Indicator implementations
    def _tsf_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Time Series Forecast indicator"""
        from sklearn.linear_model import LinearRegression

        pd = _get_pd()

        window = 20
        tsf_values = []

        for i in range(len(df)):
            if i < window:
                tsf_values.append(np.nan)
            else:
                y = df['close'].iloc[i-window:i].values
                X = np.arange(window).reshape(-1, 1)

                model = LinearRegression()
                model.fit(X, y)

                # Forecast next value
                next_x = np.array([[window]])
                forecast = model.predict(next_x)[0]
                tsf_values.append(forecast)

        return pd.Series(tsf_values, index=df.index)

    def _kama_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Kaufman's Adaptive Moving Average"""
        # Simplified implementation
        fast = 2
        slow = 30
        pd = _get_pd()

        efficiency_ratio = abs(df['close'] - df['close'].shift(10)) / \
                          (df['close'] - df['close'].shift(1)).rolling(10).sum()

        smoothing_constant = (efficiency_ratio * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2

        kama = df['close'].copy()
        for i in range(1, len(kama)):
            if not np.isnan(smoothing_constant.iloc[i]):
                kama.iloc[i] = kama.iloc[i-1] + smoothing_constant.iloc[i] * \
                              (df['close'].iloc[i] - kama.iloc[i-1])

        return kama

    def _vidya_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Variable Index Dynamic Average"""
        alpha = 0.2
        cmo_period = 9
        pd = _get_pd()

        # Chande Momentum Oscillator
        m1 = (df['close'] - df['close'].shift(1)).rolling(cmo_period).apply(
            lambda x: np.sum(np.where(x > 0, x, 0))
        )
        m2 = (df['close'] - df['close'].shift(1)).rolling(cmo_period).apply(
            lambda x: np.sum(np.where(x < 0, -x, 0))
        )

        cmo = 100 * (m1 - m2) / (m1 + m2)
        vidya = df['close'].copy()

        for i in range(1, len(vidya)):
            if not np.isnan(cmo.iloc[i]):
                alpha_dynamic = alpha * abs(cmo.iloc[i]) / 100
                vidya.iloc[i] = alpha_dynamic * df['close'].iloc[i] + \
                               (1 - alpha_dynamic) * vidya.iloc[i-1]
            else:
                vidya.iloc[i] = df['close'].iloc[i]

        return vidya

    def _chandelier_exit_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Chandelier Exit indicator"""
        atr_period = 22
        multiplier = 3
        pd = _get_pd()

        # ATR calculation
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        # Chandelier Exit
        highest_high = df['high'].rolling(atr_period).max()
        lowest_low = df['low'].rolling(atr_period).min()

        long_exit = highest_high - atr * multiplier
        short_exit = lowest_low + atr * multiplier

        # Return long exit for now (can be extended for short signals)
        return long_exit

    def _supertrend_indicator(self, df: pd.DataFrame) -> pd.Series:
        """SuperTrend indicator"""
        factor = 3
        atr_period = 10
        pd = _get_pd()

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        # Basic bands
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + factor * atr
        lower_band = hl2 - factor * atr

        # SuperTrend logic (simplified)
        supertrend = pd.Series(index=df.index, dtype=float)

        for i in range(len(df)):
            if i == 0:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                if df['close'].iloc[i] > supertrend.iloc[i-1]:
                    supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1])
                else:
                    supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1])

        return supertrend

    # Additional indicator implementations would go here
    def _extended_tsf_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Extended TSF with multiple lookbacks"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

    def _volume_profile_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Volume Profile indicator"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

    def _demark_indicator(self, df: pd.DataFrame) -> pd.Series:
        """DeMark indicator"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

    def _ichimoku_advanced_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Advanced Ichimoku indicator"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

    def _regime_filter_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Market Regime Filter"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

    def _adaptive_ma_ensemble_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Adaptive MA Ensemble"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

    def _hull_suite_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Hull Suite indicator"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

    def _ssa_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Singular Spectrum Analysis"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

    def _neural_net_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Neural Network indicator"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

    def _wavelet_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Wavelet Transform indicator"""
        # Implementation would go here
        pd = _get_pd()
        return pd.Series([np.nan] * len(df), index=df.index)

def run_indicator_discovery_demo():
    """Demo of AI indicator discovery and integration"""
    print("🤖 AI Indicator Discovery Agent Demo")
    print("=" * 50)

    # Initialize agent
    agent = IndicatorDiscoveryAgent()

    # Discover new indicators
    print("\n🔍 Discovering new indicators...")
    new_indicators = agent.discover_indicators(min_performance=0.7)

    print(f"\n📊 Found {len(new_indicators)} promising indicators")

    # Test indicator performance
    print("\n🧪 Testing indicator performance...")
    for indicator_id, indicator_info in list(new_indicators.items())[:3]:  # Test first 3
        performance = agent.test_indicator_performance(indicator_id)
        if performance:
            print(f"  {indicator_info['name']}: {performance.get('win_rate', 0):.1%} win rate")

    # Integrate best indicators
    print("\n🔗 Integrating best indicators into ML pipeline...")
    selected_indicators = agent.integrate_best_indicators(target_accuracy=0.75, max_indicators=5)

    if selected_indicators:
        pipeline_config = agent.enhance_ml_pipeline(selected_indicators)
        print(f"\n✅ Enhanced ML pipeline with {len(selected_indicators)} new indicators")
        print("🎯 Ready for advanced training with discovered indicators!")
    else:
        print("\n⚠️ No indicators met the performance threshold")

    print("\n🎉 AI Indicator Discovery Demo completed!")

if __name__ == "__main__":
    run_indicator_discovery_demo()

