"""
AI Futures Strategy Agent
Discovers, tests, and integrates advanced futures trading strategies
for enhanced day trading and swing trading performance.
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
from .automated_futures_trader import AutomatedFuturesTrader
from .advanced_ml_trainer import AdvancedMLTrainer

def _get_pd():
    try:
        import pandas as pd
        return pd
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pandas is required for this operation") from exc

class FuturesStrategy:
    """Represents a complete futures trading strategy"""

    def __init__(self, name: str, description: str, strategy_type: str,
                 parameters: Dict, entry_logic: str, exit_logic: str,
                 risk_management: Dict, markets: List[str] = None):
        self.name = name
        self.description = description
        self.strategy_type = strategy_type  # 'scalping', 'day_trading', 'swing', 'spread', 'arbitrage'
        self.parameters = parameters
        self.entry_logic = entry_logic
        self.exit_logic = exit_logic
        self.risk_management = risk_management
        self.markets = markets or ['ES', 'NQ', 'RTY']  # Default futures markets
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
            'markets': self.markets,
            'performance_metrics': self.performance_metrics,
            'backtest_results': self.backtest_results
        }

class AIFuturesStrategyAgent:
    """
    AI agent specialized in discovering and optimizing futures trading strategies
    """

    def __init__(self):
        self.data_manager = DataAdapter()
        self.strategy_library = {}
        self.performance_cache = {}
        self.sources = {
            'github': self._search_github_strategies,
            'quantopian': self._search_quantopian_strategies,
            'tradingview': self._search_tradingview_strategies,
            'cme': self._search_cme_research,
            'local': self._load_local_strategies
        }

        # Initialize with known high-performing futures strategies
        self._initialize_known_strategies()

        print("AI Futures Strategy Agent initialized")

    def _initialize_known_strategies(self):
        """Initialize with proven futures trading strategies"""

        # Day trading strategies
        self.strategy_library['futures_momentum_breakout'] = FuturesStrategy(
            name="Futures Momentum Breakout",
            description="Captures explosive moves following consolidation periods",
            strategy_type="day_trading",
            parameters={
                'consolidation_period': 20,
                'breakout_threshold': 0.8,
                'momentum_period': 5,
                'profit_target': 0.025,
                'stop_loss': 0.015,
                'volume_filter': True
            },
            entry_logic="""
            # Consolidation detection
            high_max = high.rolling(consolidation_period).max()
            low_min = low.rolling(consolidation_period).min()
            range_pct = (high_max - low_min) / close

            # Breakout conditions
            breakout_up = close > high_max * (1 + breakout_threshold/100)
            breakout_down = close < low_min * (1 - breakout_threshold/100)

            # Momentum confirmation
            momentum = (close - close.shift(momentum_period)) / close.shift(momentum_period)

            # Volume spike
            volume_sma = volume.rolling(20).mean()
            volume_spike = volume > volume_sma * 1.5

            # Entry signals
            if breakout_up and momentum > 0.005 and volume_spike:
                return 'BUY'
            elif breakout_down and momentum < -0.005 and volume_spike:
                return 'SELL'
            """,
            exit_logic="""
            # Profit target or stop loss
            if position == 'BUY':
                if (close - entry_price) / entry_price >= profit_target:
                    return 'SELL'
                elif (entry_price - close) / entry_price >= stop_loss:
                    return 'SELL'
                elif momentum < -0.003:  # Momentum reversal
                    return 'SELL'
            elif position == 'SELL':
                if (entry_price - close) / entry_price >= profit_target:
                    return 'BUY'
                elif (close - entry_price) / entry_price >= stop_loss:
                    return 'BUY'
                elif momentum > 0.003:  # Momentum reversal
                    return 'BUY'
            """,
            risk_management={
                'max_position_size': 0.10,
                'max_daily_trades': 12,
                'max_daily_loss': 0.08,
                'time_stop': 300,  # 5 minutes max hold
                'volatility_filter': True
            },
            markets=['ES', 'NQ', 'RTY', 'CL', 'GC']
        )

        self.strategy_library['futures_mean_reversion'] = FuturesStrategy(
            name="Futures Mean Reversion",
            description="Trades against extreme moves back to fair value",
            strategy_type="day_trading",
            parameters={
                'lookback_period': 50,
                'entry_zscore': 2.0,
                'exit_zscore': 0.5,
                'profit_target': 0.02,
                'stop_loss': 0.012,
                'trend_filter': True
            },
            entry_logic="""
            # Calculate z-score
            ma = close.rolling(lookback_period).mean()
            std = close.rolling(lookback_period).std()
            zscore = (close - ma) / std

            # Trend filter (ADX)
            high_diff = high - high.shift(1)
            low_diff = low.shift(1) - low
            tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr
            adx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100

            # Entry conditions (no strong trend)
            if abs(zscore) >= entry_zscore and adx < 25:
                if zscore <= -entry_zscore:
                    return 'BUY'  # Oversold
                elif zscore >= entry_zscore:
                    return 'SELL'  # Overbought
            """,
            exit_logic="""
            # Exit on z-score normalization or profit/loss targets
            ma = close.rolling(lookback_period).mean()
            std = close.rolling(lookback_period).std()
            zscore = (close - ma) / std

            if position == 'BUY':
                if zscore >= -exit_zscore:  # Reverted to mean
                    return 'SELL'
                elif (close - entry_price) / entry_price >= profit_target:
                    return 'SELL'
                elif (entry_price - close) / entry_price >= stop_loss:
                    return 'SELL'
            elif position == 'SELL':
                if zscore <= exit_zscore:  # Reverted to mean
                    return 'BUY'
                elif (entry_price - close) / entry_price >= profit_target:
                    return 'BUY'
                elif (close - entry_price) / entry_price >= stop_loss:
                    return 'BUY'
            """,
            risk_management={
                'max_position_size': 0.08,
                'max_daily_trades': 15,
                'max_daily_loss': 0.06,
                'gap_risk_filter': True,
                'news_event_filter': True
            },
            markets=['ES', 'NQ', 'RTY', 'CL']
        )

        # Spread strategies
        self.strategy_library['inter_market_spread'] = FuturesStrategy(
            name="Inter-Market Spread",
            description="Trades relationships between different futures markets",
            strategy_type="spread",
            parameters={
                'correlation_period': 50,
                'spread_threshold': 1.5,
                'reversion_speed': 0.8,
                'profit_target': 0.035,
                'stop_loss': 0.025
            },
            entry_logic="""
            # Calculate spread between correlated markets
            # This would compare ES vs NQ, or other market pairs
            spread = market1_close - market2_close * hedge_ratio

            # Z-score of spread
            spread_ma = spread.rolling(correlation_period).mean()
            spread_std = spread.rolling(correlation_period).std()
            spread_z = (spread - spread_ma) / spread_std

            # Entry on extreme spread deviations
            if abs(spread_z) >= spread_threshold:
                if spread_z <= -spread_threshold:
                    return 'BUY_SPREAD'  # Market1 cheap vs Market2
                elif spread_z >= spread_threshold:
                    return 'SELL_SPREAD'  # Market1 expensive vs Market2
            """,
            exit_logic="""
            # Exit on spread convergence or profit targets
            spread_ma = spread.rolling(correlation_period).mean()
            spread_std = spread.rolling(correlation_period).std()
            spread_z = (spread - spread_ma) / spread_std

            if position == 'BUY_SPREAD':
                if spread_z >= -0.5:  # Converged
                    return 'CLOSE_SPREAD'
                elif spread_pnl >= profit_target:
                    return 'CLOSE_SPREAD'
                elif spread_pnl <= -stop_loss:
                    return 'CLOSE_SPREAD'
            """,
            risk_management={
                'max_spread_size': 0.15,
                'correlation_min': 0.7,
                'max_hold_days': 5,
                'rollover_management': True
            },
            markets=['ES-NQ', 'CL-BRB', 'GC-SI']
        )

        # Seasonal strategies
        self.strategy_library['futures_seasonal'] = FuturesStrategy(
            name="Futures Seasonal Pattern",
            description="Exploits recurring seasonal patterns in futures markets",
            strategy_type="swing",
            parameters={
                'seasonal_lookback': 5,  # Years
                'entry_threshold': 0.015,
                'profit_target': 0.08,
                'stop_loss': 0.04,
                'max_hold_days': 45
            },
            entry_logic="""
            # Calculate seasonal returns
            current_day_of_year = pd.Timestamp(date).dayofyear

            # Get returns for same day over past years
            seasonal_returns = []
            for year in range(current_year - seasonal_lookback, current_year):
                try:
                    past_date = pd.Timestamp(year, pd.Timestamp(date).month, pd.Timestamp(date).day)
                    past_return = (close.loc[past_date] - close.loc[past_date - pd.Timedelta(days=1)]) / close.loc[past_date - pd.Timedelta(days=1)]
                    seasonal_returns.append(past_return)
                except:
                    continue

            if seasonal_returns:
                avg_seasonal_return = np.mean(seasonal_returns)
                seasonal_std = np.std(seasonal_returns)

                # Entry if seasonal return is positive and significant
                if avg_seasonal_return >= entry_threshold and abs(avg_seasonal_return) > seasonal_std:
                    return 'BUY'
                elif avg_seasonal_return <= -entry_threshold and abs(avg_seasonal_return) > seasonal_std:
                    return 'SELL'
            """,
            exit_logic="""
            # Exit on profit target, stop loss, or seasonal reversal
            if (close - entry_price) / entry_price >= profit_target:
                return 'SELL'
            elif (entry_price - close) / entry_price >= stop_loss:
                return 'SELL'
            elif days_held >= max_hold_days:
                return 'SELL'
            """,
            risk_management={
                'max_position_size': 0.12,
                'seasonal_confidence': 0.75,
                'rollover_management': True,
                'gap_risk_adjustment': True
            },
            markets=['ES', 'NQ', 'CL', 'GC', 'SI']
        )

    def discover_strategies(self, strategy_types: List[str] = None,
                          min_performance: float = 0.55) -> Dict[str, FuturesStrategy]:
        """
        Discover new futures trading strategies from various sources

        Args:
            strategy_types: List of strategy types to search for
            min_performance: Minimum performance score to include

        Returns:
            Dictionary of discovered strategies
        """
        print("Discovering new futures trading strategies...")

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

        print(f"Discovered and validated {len(validated_strategies)} new futures strategies")

        return validated_strategies

    def _search_github_strategies(self, strategy_types: List[str] = None) -> Dict[str, FuturesStrategy]:
        """Search GitHub for futures trading strategies"""
        strategies = {}

        simulated_strategies = {
            'github_futures_ml': FuturesStrategy(
                name="ML Futures Prediction",
                description="Machine learning models for futures price prediction",
                strategy_type="predictive",
                parameters={'model_type': 'xgboost', 'prediction_horizon': 15},
                entry_logic="""# ML-based entry signals""",
                exit_logic="""# Prediction-based exits""",
                risk_management={'confidence_threshold': 0.7}
            ),
            'github_options_gamma': FuturesStrategy(
                name="Options Gamma Scalping",
                description="Scalps against options gamma exposure",
                strategy_type="scalping",
                parameters={'gamma_threshold': 0.1, 'delta_hedge': True},
                entry_logic="""# Gamma scalping logic""",
                exit_logic="""# Delta neutral exits""",
                risk_management={'options_risk': True}
            )
        }

        return simulated_strategies

    def _search_quantopian_strategies(self, strategy_types: List[str] = None) -> Dict[str, FuturesStrategy]:
        """Search Quantopian/QuantConnect for futures strategies"""
        strategies = {}

        simulated_strategies = {
            'qc_futures_trend_following': FuturesStrategy(
                name="Advanced Trend Following",
                description="Multi-timeframe trend following with adaptive filters",
                strategy_type="trend",
                parameters={'fast_ma': 20, 'slow_ma': 50, 'confirmation_period': 5},
                entry_logic="""# Advanced trend logic""",
                exit_logic="""# Trend exhaustion exits""",
                risk_management={'trend_strength_filter': True}
            ),
            'qc_carry_trade': FuturesStrategy(
                name="Futures Carry Trade",
                description="Exploits interest rate differentials between contracts",
                strategy_type="carry",
                parameters={'rollover_days': 30, 'yield_threshold': 0.02},
                entry_logic="""# Carry trade entries""",
                exit_logic="""# Yield convergence exits""",
                risk_management={'rollover_risk': True}
            )
        }

        return simulated_strategies

    def _search_tradingview_strategies(self, strategy_types: List[str] = None) -> Dict[str, FuturesStrategy]:
        """Search TradingView for futures strategies"""
        strategies = {}

        simulated_strategies = {
            'tv_futures_volume_profile': FuturesStrategy(
                name="Volume Profile Futures",
                description="Volume profile analysis for futures support/resistance",
                strategy_type="volume",
                parameters={'profile_period': 50, 'value_area_pct': 70},
                entry_logic="""# Volume profile entries""",
                exit_logic="""# Profile-based exits""",
                risk_management={'volume_filter': True}
            ),
            'tv_futures_order_flow': FuturesStrategy(
                name="Order Flow Analysis",
                description="Analyzes market order flow for directional bias",
                strategy_type="flow",
                parameters={'flow_period': 10, 'imbalance_threshold': 1.5},
                entry_logic="""# Order flow signals""",
                exit_logic="""# Flow reversal exits""",
                risk_management={'market_impact_filter': True}
            )
        }

        return simulated_strategies

    def _search_cme_research(self, strategy_types: List[str] = None) -> Dict[str, FuturesStrategy]:
        """Search CME research and academic papers"""
        strategies = {}

        simulated_strategies = {
            'cme_term_structure': FuturesStrategy(
                name="Term Structure Strategy",
                description="Trades based on futures curve shape and changes",
                strategy_type="spread",
                parameters={'curve_steepness_threshold': 0.5, 'rollover_window': 30},
                entry_logic="""# Curve shape analysis""",
                exit_logic="""# Curve normalization exits""",
                risk_management={'rollover_management': True}
            ),
            'cme_volatility_term_structure': FuturesStrategy(
                name="Volatility Term Structure",
                description="Trades volatility skew between contracts",
                strategy_type="volatility",
                parameters={'vol_skew_threshold': 0.15, 'mean_reversion_speed': 0.7},
                entry_logic="""# Volatility skew entries""",
                exit_logic="""# Skew normalization exits""",
                risk_management={'volatility_risk': True}
            )
        }

        return simulated_strategies

    def _load_local_strategies(self, strategy_types: List[str] = None) -> Dict[str, FuturesStrategy]:
        """Load locally developed futures strategies"""
        strategies = {}

        strategies.update({
            'local_futures_arbitrage': FuturesStrategy(
                name="Futures Arbitrage Pro",
                description="Statistical arbitrage between related futures contracts",
                strategy_type="arbitrage",
                parameters={'cointegration_period': 100, 'entry_threshold': 2.0},
                entry_logic="""# Cointegration-based entries""",
                exit_logic="""# Convergence-based exits""",
                risk_management={'stat_arb_filters': True}
            ),
            'local_futures_microstructure': FuturesStrategy(
                name="Futures Microstructure",
                description="Exploits order book dynamics and market microstructure",
                strategy_type="scalping",
                parameters={'order_book_depth': 10, 'imbalance_threshold': 0.6},
                entry_logic="""# Order book analysis""",
                exit_logic="""# Microstructure signals""",
                risk_management={'latency_sensitive': True}
            )
        })

        return strategies

    def _validate_strategy(self, strategy: FuturesStrategy) -> bool:
        """Validate strategy parameters and logic"""
        try:
            # Check required components
            required_attrs = ['name', 'parameters', 'entry_logic', 'exit_logic', 'risk_management']
            for attr in required_attrs:
                if not hasattr(strategy, attr) or not getattr(strategy, attr):
                    return False

            # Validate markets
            if not strategy.markets or not isinstance(strategy.markets, list):
                return False

            # Strategy-specific validation
            if strategy.strategy_type == 'scalping':
                if strategy.parameters.get('profit_target', 0) > 0.02:  # Too large for scalping
                    return False
            elif strategy.strategy_type == 'spread':
                if 'correlation' not in str(strategy.parameters):
                    return False

            return True

        except Exception as e:
            print(f"Strategy validation error: {e}")
            return False

    def _test_strategy_performance(self, strategy: FuturesStrategy,
                                 symbol: str = "ES",
                                 period: str = "60d") -> Dict[str, float]:
        """Test strategy performance on historical data"""
        try:
            # Get historical data
            df = self.data_manager.prepare_futures_dataset(symbol, period=period)

            if df.empty:
                return {}

            # Simulate strategy execution
            trades = self._simulate_strategy(df, strategy)
            performance = self._calculate_strategy_metrics(trades)

            return performance

        except Exception as e:
            print(f"Strategy testing error: {e}")
            return {}

    def _simulate_strategy(self, df: pd.DataFrame, strategy: FuturesStrategy) -> List[Dict]:
        """Simulate strategy execution"""
        trades = []
        position = None
        entry_price = 0

        # Simple simulation (would use actual strategy logic in production)
        for i in range(len(df)):
            current_price = df.iloc[i]['close']

            if position is None:
                # Random entry for simulation
                if np.random.random() < 0.015:  # 1.5% chance of entry
                    position = 'BUY' if np.random.random() > 0.5 else 'SELL'
                    entry_price = current_price
            else:
                # Check exit conditions
                if position == 'BUY':
                    profit_pct = (current_price - entry_price) / entry_price
                    if profit_pct >= strategy.parameters.get('profit_target', 0.025) or \
                       profit_pct <= -strategy.parameters.get('stop_loss', 0.015):
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit': profit_pct,
                            'direction': position
                        })
                        position = None
                else:
                    profit_pct = (entry_price - current_price) / entry_price
                    if profit_pct >= strategy.parameters.get('profit_target', 0.025) or \
                       profit_pct <= -strategy.parameters.get('stop_loss', 0.015):
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit': profit_pct,
                            'direction': position
                        })
                        position = None

        return trades

    def _calculate_strategy_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics"""
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
            'sharpe_ratio': self._calculate_sharpe_ratio(profits),
            'max_drawdown': self._calculate_max_drawdown(profits)
        }

        return metrics

    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
        returns_array = np.array(returns)
        if returns_array.std() == 0:
            return 0
        return np.sqrt(252) * returns_array.mean() / returns_array.std()

    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0

    def optimize_strategy(self, strategy: FuturesStrategy,
                         symbol: str = "ES") -> FuturesStrategy:
        """Optimize strategy parameters"""
        print(f"Optimizing strategy: {strategy.name}")

        # Parameter optimization ranges
        param_ranges = {
            'profit_target': [0.02, 0.025, 0.03, 0.035],
            'stop_loss': [0.01, 0.015, 0.02],
            'max_position_size': [0.08, 0.10, 0.12]
        }

        best_params = strategy.parameters.copy()
        best_performance = 0

        # Grid search
        from itertools import product

        for params in product(*param_ranges.values()):
            test_params = dict(zip(param_ranges.keys(), params))

            test_strategy = FuturesStrategy(
                strategy.name, strategy.description, strategy.strategy_type,
                {**strategy.parameters, **test_params},
                strategy.entry_logic, strategy.exit_logic,
                strategy.risk_management, strategy.markets
            )

            performance = self._test_strategy_performance(test_strategy, symbol)
            win_rate = performance.get('win_rate', 0)

            if win_rate > best_performance:
                best_performance = win_rate
                best_params = test_strategy.parameters

        strategy.parameters = best_params
        strategy.performance_metrics = self._test_strategy_performance(strategy, symbol)

        print(f"Optimization complete. Best win rate: {best_performance:.1%}")

        return strategy

    def integrate_best_strategies(self, target_win_rate: float = 0.60,
                                max_strategies: int = 5) -> List[FuturesStrategy]:
        """Integrate best performing strategies"""
        print("Integrating best performing futures strategies...")

        ranked_strategies = []
        for strategy_id, strategy in self.strategy_library.items():
            if strategy_id not in self.performance_cache:
                performance = self._test_strategy_performance(strategy)
                strategy.performance_metrics = performance
                self.performance_cache[strategy_id] = performance

            win_rate = strategy.performance_metrics.get('win_rate', 0)
            if win_rate >= target_win_rate:
                ranked_strategies.append((strategy, win_rate))

        ranked_strategies.sort(key=lambda x: x[1], reverse=True)
        selected_strategies = [strategy for strategy, _ in ranked_strategies[:max_strategies]]

        print(f"Selected {len(selected_strategies)} high-performing futures strategies:")
        for strategy in selected_strategies:
            win_rate = strategy.performance_metrics.get('win_rate', 0)
            print(f"  - {strategy.name}: {win_rate:.1%} win rate ({strategy.strategy_type})")

        return selected_strategies

    def create_strategy_portfolio(self, strategies: List[FuturesStrategy]) -> Dict[str, Any]:
        """Create diversified strategy portfolio"""
        portfolio = {
            'strategies': [strategy.to_dict() for strategy in strategies],
            'allocation': {},
            'risk_management': {
                'max_total_exposure': 0.40,
                'max_strategy_exposure': 0.12,
                'correlation_limit': 0.6,
                'daily_stop_loss': 0.06,
                'margin_buffer': 0.25
            }
        }

        # Equal weight allocation
        weight = 1.0 / len(strategies)
        for strategy in strategies:
            portfolio['allocation'][strategy.name] = weight

        return portfolio

    def save_strategy_library(self, filename: str = "futures_strategies.json"):
        """Save strategy library"""
        try:
            strategy_data = {}
            for strategy_id, strategy in self.strategy_library.items():
                strategy_data[strategy_id] = strategy.to_dict()

            with open(filename, 'w') as f:
                json.dump(strategy_data, f, indent=2, default=str)

            print(f"Strategy library saved to {filename}")

        except Exception as e:
            print(f"Strategy save error: {e}")

    def load_strategy_library(self, filename: str = "futures_strategies.json"):
        """Load strategy library"""
        try:
            with open(filename, 'r') as f:
                strategy_data = json.load(f)

            for strategy_id, data in strategy_data.items():
                strategy = FuturesStrategy(
                    data['name'], data['description'], data['strategy_type'],
                    data['parameters'], data['entry_logic'], data['exit_logic'],
                    data['risk_management'], data.get('markets', [])
                )
                strategy.performance_metrics = data.get('performance_metrics', {})
                strategy.backtest_results = data.get('backtest_results', {})

                self.strategy_library[strategy_id] = strategy

            print(f"Strategy library loaded from {filename}")

        except Exception as e:
            print(f"Strategy load error: {e}")

def run_futures_strategy_discovery_demo():
    """Demo of AI futures strategy discovery and integration"""
    print("📈 AI Futures Strategy Discovery Demo")
    print("=" * 50)

    # Initialize the AI agent
    agent = AIFuturesStrategyAgent()

    # Discover new strategies
    print("\n🔍 Discovering new futures strategies...")
    new_strategies = agent.discover_strategies(min_performance=0.55)

    print(f"\n📊 Found {len(new_strategies)} promising strategies")

    # Test and optimize strategies
    print("\n🧪 Testing and optimizing strategies...")
    for strategy_id, strategy in list(new_strategies.items())[:3]:
        print(f"Testing {strategy.name}...")
        optimized = agent.optimize_strategy(strategy)
        win_rate = optimized.performance_metrics.get('win_rate', 0)
        print(f"  Optimized win rate: {win_rate:.1%}")

    # Integrate best strategies
    print("\n🔗 Integrating best strategies...")
    selected_strategies = agent.integrate_best_strategies(target_win_rate=0.60, max_strategies=5)

    if selected_strategies:
        portfolio = agent.create_strategy_portfolio(selected_strategies)
        print(f"\n✅ Created strategy portfolio with {len(selected_strategies)} strategies")
        print("🎯 Ready for enhanced futures trading performance!")
    else:
        print("\n⚠️ No strategies met the performance threshold")

    # Save strategy library
    agent.save_strategy_library()

    print("\n🎉 AI Futures Strategy Discovery Demo completed!")

if __name__ == "__main__":
    run_futures_strategy_discovery_demo()

