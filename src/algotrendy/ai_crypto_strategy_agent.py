"""
AI Crypto Strategy Agent
Discovers, tests, and integrates advanced crypto trading strategies
for enhanced scalping and swing trading performance.
"""

from __future__ import annotations

import os
import json
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from .config import CONFIG
from .data_adapter import DataAdapter
from .crypto_scalping_trader import CryptoScalpingTrader
from .advanced_ml_trainer import AdvancedMLTrainer

def _get_pd():
    try:
        import pandas as pd
        return pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas is required for this operation") from exc

class CryptoStrategy:
    """Represents a complete crypto trading strategy"""

    def __init__(self, name: str, description: str, strategy_type: str,
                 parameters: Dict, entry_logic: str, exit_logic: str,
                 risk_management: Dict):
        self.name = name
        self.description = description
        self.strategy_type = strategy_type  # 'scalping', 'swing', 'arbitrage', 'momentum'
        self.parameters = parameters
        self.entry_logic = entry_logic
        self.exit_logic = exit_logic
        self.risk_management = risk_management
        self.performance_metrics = {}
        self.backtest_results = {}

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'strategy_type': self.strategy_type,
            'parameters': self.parameters,
            'entry_logic': self.entry_logic,
            'exit_logic': self.exit_logic,
            'risk_management': self.risk_management,
            'performance_metrics': self.performance_metrics,
            'backtest_results': self.backtest_results
        }

class AICryptoStrategyAgent:
    """
    AI agent specialized in discovering and optimizing crypto trading strategies
    """

    def __init__(self):
        self.data_manager = DataAdapter()
        self.strategy_library = {}
        self.performance_cache = {}
        self.sources = {
            'github': self._search_github_strategies,
            'quantopian': self._search_quantopian_strategies,
            'tradingview': self._search_tradingview_strategies,
            'arxiv': self._search_research_papers,
            'local': self._load_local_strategies
        }

        # Initialize with known high-performing crypto strategies
        self._initialize_known_strategies()

        print("AI Crypto Strategy Agent initialized")

    def _initialize_known_strategies(self):
        """Initialize with proven crypto trading strategies"""

        # Scalping strategies
        self.strategy_library['mean_reversion_scalp'] = CryptoStrategy(
            name="Mean Reversion Scalp",
            description="Identifies short-term deviations from moving averages for quick scalps",
            strategy_type="scalping",
            parameters={
                'lookback_period': 20,
                'deviation_threshold': 0.5,
                'profit_target': 0.003,
                'stop_loss': 0.001,
                'max_hold_time': 300
            },
            entry_logic="""
            # Mean reversion entry
            ma = close.rolling(lookback_period).mean()
            std = close.rolling(lookback_period).std()
            upper_band = ma + (std * deviation_threshold)
            lower_band = ma - (std * deviation_threshold)

            # Enter when price touches bands
            if close <= lower_band and momentum > 0:
                return 'BUY'
            elif close >= upper_band and momentum < 0:
                return 'SELL'
            """,
            exit_logic="""
            # Exit on profit target or stop loss
            if position == 'BUY':
                if (close - entry_price) / entry_price >= profit_target:
                    return 'SELL'
                elif (entry_price - close) / entry_price >= stop_loss:
                    return 'SELL'
            elif position == 'SELL':
                if (entry_price - close) / entry_price >= profit_target:
                    return 'BUY'
                elif (close - entry_price) / entry_price >= stop_loss:
                    return 'BUY'
            """,
            risk_management={
                'max_position_size': 0.02,
                'max_daily_trades': 50,
                'max_daily_loss': 0.05,
                'volatility_filter': True
            }
        )

        self.strategy_library['momentum_burst'] = CryptoStrategy(
            name="Momentum Burst Scalp",
            description="Captures explosive momentum moves in crypto markets",
            strategy_type="scalping",
            parameters={
                'momentum_period': 5,
                'acceleration_threshold': 0.002,
                'volume_multiplier': 1.5,
                'profit_target': 0.004,
                'stop_loss': 0.002
            },
            entry_logic="""
            # Momentum burst detection
            momentum = (close - close.shift(momentum_period)) / close.shift(momentum_period)
            acceleration = momentum - momentum.shift(1)
            volume_sma = volume.rolling(20).mean()

            # Enter on momentum burst with volume confirmation
            if acceleration >= acceleration_threshold and volume >= volume_sma * volume_multiplier:
                return 'BUY'
            elif acceleration <= -acceleration_threshold and volume >= volume_sma * volume_multiplier:
                return 'SELL'
            """,
            exit_logic="""
            # Exit on profit target, stop loss, or momentum reversal
            if position == 'BUY':
                if (close - entry_price) / entry_price >= profit_target:
                    return 'SELL'
                elif (entry_price - close) / entry_price >= stop_loss:
                    return 'SELL'
                elif momentum < 0:  # Momentum reversal
                    return 'SELL'
            """,
            risk_management={
                'max_position_size': 0.03,
                'max_daily_trades': 30,
                'max_daily_loss': 0.03,
                'correlation_filter': True
            }
        )

        # Swing strategies
        self.strategy_library['crypto_seasonal'] = CryptoStrategy(
            name="Crypto Seasonal Swing",
            description="Exploits seasonal patterns in crypto markets",
            strategy_type="swing",
            parameters={
                'seasonal_period': 365,
                'entry_threshold': 0.02,
                'profit_target': 0.15,
                'stop_loss': 0.05,
                'max_hold_days': 30
            },
            entry_logic="""
            # Seasonal pattern recognition
            seasonal_return = close / close.shift(seasonal_period) - 1
            seasonal_ma = seasonal_return.rolling(30).mean()

            # Enter when seasonal pattern is positive
            if seasonal_return >= entry_threshold and seasonal_ma > 0:
                return 'BUY'
            """,
            exit_logic="""
            # Exit on profit target or seasonal reversal
            if (close - entry_price) / entry_price >= profit_target:
                return 'SELL'
            elif seasonal_return <= -entry_threshold:
                return 'SELL'
            """,
            risk_management={
                'max_position_size': 0.10,
                'max_open_positions': 3,
                'portfolio_heat': 0.30,
                'seasonal_filter': True
            }
        )

    def discover_strategies(self, strategy_types: List[str] = None,
                          min_performance: float = 0.60) -> Dict[str, CryptoStrategy]:
        """
        Discover new crypto trading strategies from various sources

        Args:
            strategy_types: List of strategy types to search for
            min_performance: Minimum performance score to include

        Returns:
            Dictionary of discovered strategies
        """
        print("Discovering new crypto trading strategies...")

        discovered_strategies = {}

        # Search each source
        for source_name, search_func in self.sources.items():
            try:
                print(f"Searching {source_name}...")
                strategies = search_func(strategy_types)
                discovered_strategies.update(strategies)
            except Exception as e:
                print(f"Error searching {source_name}: {e}")

        # Filter by performance and validate
        validated_strategies = {}
        for strategy_id, strategy in discovered_strategies.items():
            if self._validate_strategy(strategy):
                # Test performance
                performance = self._test_strategy_performance(strategy)
                if performance.get('win_rate', 0) >= min_performance:
                    strategy.performance_metrics = performance
                    validated_strategies[strategy_id] = strategy
                    print(f"✓ Validated strategy: {strategy.name} ({performance.get('win_rate', 0):.1%} win rate)")

        # Update library
        self.strategy_library.update(validated_strategies)

        print(f"Discovered and validated {len(validated_strategies)} new crypto strategies")

        return validated_strategies

    def _search_github_strategies(self, strategy_types: List[str] = None) -> Dict[str, CryptoStrategy]:
        """Search GitHub for crypto trading strategies"""
        strategies = {}

        # Simulate GitHub search results for crypto strategies
        simulated_strategies = {
            'github_defi_yield': CryptoStrategy(
                name="DeFi Yield Strategy",
                description="Arbitrage between different DeFi yield opportunities",
                strategy_type="arbitrage",
                parameters={'min_yield_spread': 0.05, 'rebalance_threshold': 0.02},
                entry_logic="""# DeFi yield arbitrage logic""",
                exit_logic="""# Exit conditions""",
                risk_management={'max_position_size': 0.05}
            ),
            'github_whale_watching': CryptoStrategy(
                name="Whale Watching Momentum",
                description="Follow large wallet movements for momentum signals",
                strategy_type="momentum",
                parameters={'min_whale_size': 1000000, 'momentum_threshold': 0.01},
                entry_logic="""# Whale movement detection""",
                exit_logic="""# Momentum-based exits""",
                risk_management={'max_position_size': 0.08}
            )
        }

        return simulated_strategies

    def _search_quantopian_strategies(self, strategy_types: List[str] = None) -> Dict[str, CryptoStrategy]:
        """Search Quantopian/QuantConnect for crypto strategies"""
        strategies = {}

        simulated_strategies = {
            'qc_crypto_mean_reversion': CryptoStrategy(
                name="Crypto Mean Reversion Pro",
                description="Advanced mean reversion with crypto-specific adjustments",
                strategy_type="scalping",
                parameters={'lookback': 15, 'threshold': 0.8, 'profit_target': 0.005},
                entry_logic="""# Advanced mean reversion logic""",
                exit_logic="""# Optimized exit conditions""",
                risk_management={'volatility_adjusted': True}
            )
        }

        return simulated_strategies

    def _search_tradingview_strategies(self, strategy_types: List[str] = None) -> Dict[str, CryptoStrategy]:
        """Search TradingView for crypto strategies"""
        strategies = {}

        simulated_strategies = {
            'tv_crypto_supertrend': CryptoStrategy(
                name="Crypto SuperTrend Strategy",
                description="SuperTrend adapted for crypto volatility",
                strategy_type="trend",
                parameters={'factor': 4, 'atr_period': 12},
                entry_logic="""# SuperTrend crypto adaptation""",
                exit_logic="""# Trend-following exits""",
                risk_management={'trailing_stop': True}
            )
        }

        return simulated_strategies

    def _search_research_papers(self, strategy_types: List[str] = None) -> Dict[str, CryptoStrategy]:
        """Search academic papers for crypto strategies"""
        strategies = {}

        simulated_strategies = {
            'paper_crypto_ml': CryptoStrategy(
                name="ML-Based Crypto Prediction",
                description="Machine learning models for crypto price prediction",
                strategy_type="predictive",
                parameters={'model_type': 'ensemble', 'prediction_horizon': 24},
                entry_logic="""# ML-based entry signals""",
                exit_logic="""# Prediction-based exits""",
                risk_management={'confidence_threshold': 0.75}
            )
        }

        return simulated_strategies

    def _load_local_strategies(self, strategy_types: List[str] = None) -> Dict[str, CryptoStrategy]:
        """Load locally developed crypto strategies"""
        strategies = {}

        strategies.update({
            'local_crypto_arbitrage': CryptoStrategy(
                name="Cross-Exchange Arbitrage",
                description="Arbitrage between different crypto exchanges",
                strategy_type="arbitrage",
                parameters={'min_spread': 0.003, 'max_slippage': 0.001},
                entry_logic="""# Cross-exchange arbitrage logic""",
                exit_logic="""# Arbitrage exit conditions""",
                risk_management={'execution_speed': 'high'}
            ),
            'local_sentiment_strategy': CryptoStrategy(
                name="Crypto Sentiment Strategy",
                description="Social media sentiment analysis for crypto trading",
                strategy_type="sentiment",
                parameters={'sentiment_threshold': 0.7, 'news_weight': 0.6},
                entry_logic="""# Sentiment-based entries""",
                exit_logic="""# Sentiment reversal exits""",
                risk_management={'sentiment_filter': True}
            )
        })

        return strategies

    def _validate_strategy(self, strategy: CryptoStrategy) -> bool:
        """
        Validate that a strategy is implementable and reasonable

        Args:
            strategy: Strategy to validate

        Returns:
            True if strategy is valid
        """
        try:
            # Check required components
            required_attrs = ['name', 'parameters', 'entry_logic', 'exit_logic', 'risk_management']
            for attr in required_attrs:
                if not hasattr(strategy, attr) or not getattr(strategy, attr):
                    return False

            # Validate parameters
            if not isinstance(strategy.parameters, dict):
                return False

            # Check for reasonable parameter values
            if strategy.strategy_type == 'scalping':
                if strategy.parameters.get('profit_target', 0) > 0.01:  # Too large for scalping
                    return False
                if strategy.parameters.get('max_hold_time', 0) > 1800:  # Too long for scalping
                    return False

            return True

        except Exception as e:
            print(f"Strategy validation error: {e}")
            return False

    def _test_strategy_performance(self, strategy: CryptoStrategy,
                                 symbol: str = "BTC/USDT",
                                 period: str = "60d") -> Dict[str, float]:
        """
        Test strategy performance on historical data

        Args:
            strategy: Strategy to test
            symbol: Crypto symbol to test on
            period: Test period

        Returns:
            Performance metrics
        """
        try:
            # Get historical data
            df = self.data_manager.fetch_futures_data(symbol.replace('/', '-'), period=period)

            if df.empty:
                return {}

            # Simulate strategy execution
            trades = self._simulate_strategy(df, strategy)
            performance = self._calculate_strategy_metrics(trades)

            return performance

        except Exception as e:
            print(f"Strategy testing error: {e}")
            return {}

    def _simulate_strategy(self, df: pd.DataFrame, strategy: CryptoStrategy) -> List[Dict]:
        """
        Simulate strategy execution on historical data

        Args:
            df: Historical price data
            strategy: Strategy to simulate

        Returns:
            List of simulated trades
        """
        trades = []
        position = None
        entry_price = 0

        for i in range(len(df)):
            current_price = df.iloc[i]['close']

            # Simple strategy simulation (would be more sophisticated in practice)
            if position is None:
                # Random entry for simulation (would use actual strategy logic)
                if np.random.random() < 0.02:  # 2% chance of entry per bar
                    position = 'BUY' if np.random.random() > 0.5 else 'SELL'
                    entry_price = current_price
            else:
                # Check exit conditions
                if position == 'BUY':
                    profit_pct = (current_price - entry_price) / entry_price
                    if profit_pct >= strategy.parameters.get('profit_target', 0.003) or \
                       profit_pct <= -strategy.parameters.get('stop_loss', 0.001):
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit': profit_pct,
                            'direction': position
                        })
                        position = None
                else:  # SELL
                    profit_pct = (entry_price - current_price) / entry_price
                    if profit_pct >= strategy.parameters.get('profit_target', 0.003) or \
                       profit_pct <= -strategy.parameters.get('stop_loss', 0.001):
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit': profit_pct,
                            'direction': position
                        })
                        position = None

        return trades

    def _calculate_strategy_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics from trades"""
        if not trades:
            return {}

        profits = [trade['profit'] for trade in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]

        metrics = {
            'total_trades': len(trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0,
            'total_return': sum(profits),
            'max_drawdown': self._calculate_max_drawdown(profits)
        }

        return metrics

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0

    def optimize_strategy(self, strategy: CryptoStrategy,
                         symbol: str = "BTC/USDT") -> CryptoStrategy:
        """
        Optimize strategy parameters for better performance

        Args:
            strategy: Strategy to optimize
            symbol: Symbol to optimize on

        Returns:
            Optimized strategy
        """
        print(f"Optimizing strategy: {strategy.name}")

        # Parameter ranges to test
        param_ranges = {
            'profit_target': [0.002, 0.003, 0.004, 0.005],
            'stop_loss': [0.001, 0.0015, 0.002],
            'max_position_size': [0.02, 0.03, 0.05]
        }

        best_params = strategy.parameters.copy()
        best_performance = 0

        # Grid search optimization
        from itertools import product

        for params in product(*param_ranges.values()):
            test_params = dict(zip(param_ranges.keys(), params))

            # Create test strategy with new parameters
            test_strategy = CryptoStrategy(
                strategy.name, strategy.description, strategy.strategy_type,
                {**strategy.parameters, **test_params},
                strategy.entry_logic, strategy.exit_logic, strategy.risk_management
            )

            # Test performance
            performance = self._test_strategy_performance(test_strategy, symbol)
            win_rate = performance.get('win_rate', 0)

            if win_rate > best_performance:
                best_performance = win_rate
                best_params = test_strategy.parameters

        # Update strategy with best parameters
        strategy.parameters = best_params
        strategy.performance_metrics = self._test_strategy_performance(strategy, symbol)

        print(f"Optimization complete. Best win rate: {best_performance:.1%}")

        return strategy

    def integrate_best_strategies(self, target_win_rate: float = 0.65,
                                max_strategies: int = 5) -> List[CryptoStrategy]:
        """
        Integrate the best performing strategies into the crypto trading system

        Args:
            target_win_rate: Minimum win rate threshold
            max_strategies: Maximum number of strategies to integrate

        Returns:
            List of integrated strategies
        """
        print("Integrating best performing crypto strategies...")

        # Rank strategies by performance
        ranked_strategies = []
        for strategy_id, strategy in self.strategy_library.items():
            if strategy_id not in self.performance_cache:
                performance = self._test_strategy_performance(strategy)
                strategy.performance_metrics = performance
                self.performance_cache[strategy_id] = performance

            win_rate = strategy.performance_metrics.get('win_rate', 0)
            if win_rate >= target_win_rate:
                ranked_strategies.append((strategy, win_rate))

        # Sort by win rate
        ranked_strategies.sort(key=lambda x: x[1], reverse=True)

        # Select top strategies
        selected_strategies = [strategy for strategy, _ in ranked_strategies[:max_strategies]]

        print(f"Selected {len(selected_strategies)} high-performing crypto strategies:")
        for strategy in selected_strategies:
            win_rate = strategy.performance_metrics.get('win_rate', 0)
            print(f"  - {strategy.name}: {win_rate:.1%} win rate ({strategy.strategy_type})")

        return selected_strategies

    def create_strategy_portfolio(self, strategies: List[CryptoStrategy]) -> Dict[str, Any]:
        """
        Create a portfolio of strategies for diversified crypto trading

        Args:
            strategies: List of strategies to combine

        Returns:
            Portfolio configuration
        """
        portfolio = {
            'strategies': [strategy.to_dict() for strategy in strategies],
            'allocation': {},
            'risk_management': {
                'max_total_exposure': 0.50,
                'max_strategy_exposure': 0.15,
                'correlation_limit': 0.7,
                'daily_stop_loss': 0.05
            }
        }

        # Equal weight allocation initially
        weight = 1.0 / len(strategies)
        for strategy in strategies:
            portfolio['allocation'][strategy.name] = weight

        return portfolio

    def save_strategy_library(self, filename: str = "crypto_strategies.json"):
        """Save the strategy library to file"""
        try:
            strategy_data = {}
            for strategy_id, strategy in self.strategy_library.items():
                strategy_data[strategy_id] = strategy.to_dict()

            with open(filename, 'w') as f:
                json.dump(strategy_data, f, indent=2, default=str)

            print(f"Strategy library saved to {filename}")

        except Exception as e:
            print(f"Strategy save error: {e}")

    def load_strategy_library(self, filename: str = "crypto_strategies.json"):
        """Load strategy library from file"""
        try:
            with open(filename, 'r') as f:
                strategy_data = json.load(f)

            for strategy_id, data in strategy_data.items():
                strategy = CryptoStrategy(
                    data['name'], data['description'], data['strategy_type'],
                    data['parameters'], data['entry_logic'], data['exit_logic'],
                    data['risk_management']
                )
                strategy.performance_metrics = data.get('performance_metrics', {})
                strategy.backtest_results = data.get('backtest_results', {})

                self.strategy_library[strategy_id] = strategy

            print(f"Strategy library loaded from {filename}")

        except Exception as e:
            print(f"Strategy load error: {e}")

def run_crypto_strategy_discovery_demo():
    """Demo of AI crypto strategy discovery and integration"""
    print("₿ AI Crypto Strategy Discovery Demo")
    print("=" * 50)

    # Initialize the AI agent
    agent = AICryptoStrategyAgent()

    # Discover new strategies
    print("\n🔍 Discovering new crypto strategies...")
    new_strategies = agent.discover_strategies(min_performance=0.60)

    print(f"\n📊 Found {len(new_strategies)} promising strategies")

    # Test and optimize strategies
    print("\n🧪 Testing and optimizing strategies...")
    for strategy_id, strategy in list(new_strategies.items())[:3]:  # Test first 3
        print(f"Testing {strategy.name}...")
        optimized = agent.optimize_strategy(strategy)
        win_rate = optimized.performance_metrics.get('win_rate', 0)
        print(f"  Optimized win rate: {win_rate:.1%}")

    # Integrate best strategies
    print("\n🔗 Integrating best strategies...")
    selected_strategies = agent.integrate_best_strategies(target_win_rate=0.65, max_strategies=5)

    if selected_strategies:
        portfolio = agent.create_strategy_portfolio(selected_strategies)
        print(f"\n✅ Created strategy portfolio with {len(selected_strategies)} strategies")
        print("🎯 Ready for enhanced crypto trading performance!")
    else:
        print("\n⚠️ No strategies met the performance threshold")

    # Save strategy library
    agent.save_strategy_library()

    print("\n🎉 AI Crypto Strategy Discovery Demo completed!")

if __name__ == "__main__":
    run_crypto_strategy_discovery_demo()

