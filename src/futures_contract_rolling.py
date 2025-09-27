"""
Futures Contract Rolling System
Advanced system for managing futures contract expiration and position rolling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG
from data_manager import DataManager

logger = logging.getLogger(__name__)

@dataclass
class FuturesContract:
    """Represents a futures contract with expiration details"""
    symbol: str  # e.g., 'ES'
    contract_code: str  # e.g., 'ESZ4' (Dec 2024)
    expiration_date: datetime
    contract_month: str  # e.g., 'Z' for December
    contract_year: int
    multiplier: float
    tick_size: float
    tick_value: float
    initial_margin: float
    maintenance_margin: float

@dataclass
class RollSchedule:
    """Defines when and how to roll futures contracts"""
    symbol: str
    roll_start_days: int  # Days before expiration to start rolling
    roll_end_days: int    # Days before expiration to complete rolling
    roll_method: str      # 'volume_weighted', 'equal_weighted', 'front_month'
    min_volume_threshold: float
    max_roll_cost: float

class FuturesContractRoller:
    """
    Advanced futures contract rolling system
    Handles automatic position transitions between expiring and new contracts
    """

    def __init__(self):
        self.data_manager = DataManager()
        self.contract_cache = {}
        self.roll_schedules = self._initialize_roll_schedules()
        self.active_rolls = {}

        # Futures contract specifications by symbol
        self.contract_specs = {
            'ES': {  # E-mini S&P 500
                'multiplier': 50,
                'tick_size': 0.25,
                'tick_value': 12.50,
                'initial_margin': 1320,
                'maintenance_margin': 1200,
                'trading_hours': '09:30-16:00 ET',
                'exchange': 'CME'
            },
            'NQ': {  # E-mini Nasdaq-100
                'multiplier': 20,
                'tick_size': 0.25,
                'tick_value': 5.00,
                'initial_margin': 1870,
                'maintenance_margin': 1700,
                'trading_hours': '09:30-16:00 ET',
                'exchange': 'CME'
            },
            'RTY': {  # E-mini Russell 2000
                'multiplier': 50,
                'tick_size': 0.10,
                'tick_value': 5.00,
                'initial_margin': 1180,
                'maintenance_margin': 1080,
                'trading_hours': '09:30-16:00 ET',
                'exchange': 'CME'
            },
            'CL': {  # WTI Crude Oil
                'multiplier': 1000,
                'tick_size': 0.01,
                'tick_value': 10.00,
                'initial_margin': 5175,
                'maintenance_margin': 4700,
                'trading_hours': '09:00-14:30 ET',
                'exchange': 'NYMEX'
            },
            'GC': {  # Gold
                'multiplier': 100,
                'tick_size': 0.10,
                'tick_value': 10.00,
                'initial_margin': 8250,
                'maintenance_margin': 7500,
                'trading_hours': '08:20-13:30 ET',
                'exchange': 'COMEX'
            },
            'SI': {  # Silver
                'multiplier': 5000,
                'tick_size': 0.005,
                'tick_value': 25.00,
                'initial_margin': 10150,
                'maintenance_margin': 9200,
                'trading_hours': '08:20-13:30 ET',
                'exchange': 'COMEX'
            }
        }

    def _initialize_roll_schedules(self) -> Dict[str, RollSchedule]:
        """Initialize rolling schedules for different futures contracts"""
        return {
            'ES': RollSchedule(
                symbol='ES',
                roll_start_days=5,  # Start rolling 5 days before expiration
                roll_end_days=1,     # Complete rolling 1 day before expiration
                roll_method='volume_weighted',
                min_volume_threshold=0.7,
                max_roll_cost=0.02  # Max 2% roll cost
            ),
            'NQ': RollSchedule(
                symbol='NQ',
                roll_start_days=5,
                roll_end_days=1,
                roll_method='volume_weighted',
                min_volume_threshold=0.7,
                max_roll_cost=0.02
            ),
            'RTY': RollSchedule(
                symbol='RTY',
                roll_start_days=5,
                roll_end_days=1,
                roll_method='volume_weighted',
                min_volume_threshold=0.7,
                max_roll_cost=0.025
            ),
            'CL': RollSchedule(
                symbol='CL',
                roll_start_days=3,  # Oil rolls closer to expiration
                roll_end_days=1,
                roll_method='equal_weighted',
                min_volume_threshold=0.6,
                max_roll_cost=0.03
            ),
            'GC': RollSchedule(
                symbol='GC',
                roll_start_days=7,  # Metals have longer roll periods
                roll_end_days=2,
                roll_method='volume_weighted',
                min_volume_threshold=0.8,
                max_roll_cost=0.015
            ),
            'SI': RollSchedule(
                symbol='SI',
                roll_start_days=7,
                roll_end_days=2,
                roll_method='volume_weighted',
                min_volume_threshold=0.8,
                max_roll_cost=0.02
            )
        }

    def get_contract_expiration(self, symbol: str, contract_month: str = None,
                               contract_year: int = None) -> datetime:
        """
        Calculate contract expiration date for a futures contract

        Args:
            symbol: Futures symbol (e.g., 'ES')
            contract_month: Contract month code (e.g., 'Z' for Dec)
            contract_year: Contract year

        Returns:
            Contract expiration datetime
        """
        if contract_month is None or contract_year is None:
            # Get front month contract
            contract_month, contract_year = self._get_front_month(symbol)

        # Month code to number mapping
        month_codes = {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        }

        month_num = month_codes.get(contract_month.upper(), 12)

        # Create expiration date (typically 3rd Friday of the month for index futures)
        if symbol in ['ES', 'NQ', 'RTY']:  # Index futures
            expiration = self._get_third_friday(contract_year, month_num)
        elif symbol in ['CL']:  # Energy futures
            # Last business day of previous month
            if month_num == 1:
                expiration = datetime(contract_year - 1, 12, 31)
            else:
                expiration = datetime(contract_year, month_num - 1,
                                    self._get_last_business_day(contract_year, month_num - 1))
        else:  # Metals and others
            # Last business day of contract month
            expiration = datetime(contract_year, month_num,
                                self._get_last_business_day(contract_year, month_num))

        return expiration.replace(hour=16, minute=0, second=0, microsecond=0)  # 4:00 PM ET

    def _get_front_month(self, symbol: str) -> Tuple[str, int]:
        """Get the front month contract code and year"""
        now = datetime.now()

        # For demonstration, return next contract
        # In production, this would query current front month
        current_month = now.month
        current_year = now.year

        # Get next contract month
        if current_month >= 12:
            contract_month = 'H'  # March
            contract_year = current_year + 1
        else:
            # Map to futures contract months
            futures_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # All months for most contracts
            next_month_idx = min([i for i, m in enumerate(futures_months) if m > current_month] + [0], default=0)
            contract_month_num = futures_months[next_month_idx]

            month_codes = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
                          7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
            contract_month = month_codes[contract_month_num]
            contract_year = current_year if contract_month_num > current_month else current_year + 1

        return contract_month, contract_year

    def _get_third_friday(self, year: int, month: int) -> datetime:
        """Get the third Friday of a month"""
        # Find first day of month
        first_day = datetime(year, month, 1)

        # Find first Friday
        days_to_friday = (4 - first_day.weekday()) % 7  # 4 = Friday
        first_friday = first_day + timedelta(days=days_to_friday)

        # Third Friday is 14 days later
        third_friday = first_friday + timedelta(days=14)

        return third_friday

    def _get_last_business_day(self, year: int, month: int) -> int:
        """Get the last business day of a month"""
        # Last day of month
        if month == 12:
            last_day = 31
        else:
            last_day = (datetime(year, month + 1, 1) - timedelta(days=1)).day

        # Go backwards to find last business day (not Saturday/Sunday)
        date = datetime(year, month, last_day)
        while date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            date -= timedelta(days=1)

        return date.day

    def check_roll_status(self, symbol: str, current_date: datetime = None) -> Dict[str, Any]:
        """
        Check if a contract needs to be rolled

        Args:
            symbol: Futures symbol
            current_date: Current date (defaults to now)

        Returns:
            Roll status information
        """
        if current_date is None:
            current_date = datetime.now()

        if symbol not in self.roll_schedules:
            return {'needs_roll': False, 'reason': 'No roll schedule defined'}

        schedule = self.roll_schedules[symbol]

        # Get current contract expiration
        expiration = self.get_contract_expiration(symbol)

        # Calculate days to expiration
        days_to_expiration = (expiration - current_date).days

        # Check if within roll window
        if schedule.roll_end_days <= days_to_expiration <= schedule.roll_start_days:
            return {
                'needs_roll': True,
                'expiration_date': expiration,
                'days_to_expiration': days_to_expiration,
                'roll_progress': (schedule.roll_start_days - days_to_expiration) / (schedule.roll_start_days - schedule.roll_end_days),
                'roll_method': schedule.roll_method
            }
        elif days_to_expiration < schedule.roll_end_days:
            return {
                'needs_roll': True,
                'urgent': True,
                'expiration_date': expiration,
                'days_to_expiration': days_to_expiration,
                'roll_method': schedule.roll_method
            }

        return {'needs_roll': False, 'days_to_expiration': days_to_expiration}

    def calculate_roll_cost(self, symbol: str, from_contract: str = None,
                           to_contract: str = None) -> float:
        """
        Calculate the cost of rolling from one contract to another

        Args:
            symbol: Futures symbol
            from_contract: Current contract (optional)
            to_contract: Target contract (optional)

        Returns:
            Roll cost as a percentage
        """
        try:
            # Get contract specs
            if symbol not in self.contract_specs:
                return 0.0

            specs = self.contract_specs[symbol]

            # For demonstration, calculate based on typical roll costs
            # In production, this would use actual market data

            # Get recent price data for both contracts
            current_data = self.data_manager.fetch_futures_data(symbol, period="5d", interval="1d")

            if current_data.empty:
                return 0.0

            # Calculate roll cost based on spread between contracts
            # This is a simplified calculation
            recent_prices = current_data['close'].tail(5)
            price_volatility = recent_prices.std() / recent_prices.mean()

            # Typical roll costs based on contract type
            base_roll_costs = {
                'ES': 0.001,   # 0.1% for index futures
                'NQ': 0.0015,  # 0.15% for Nasdaq
                'RTY': 0.002,  # 0.2% for Russell
                'CL': 0.005,   # 0.5% for oil (higher due to storage costs)
                'GC': 0.003,   # 0.3% for gold
                'SI': 0.004    # 0.4% for silver
            }

            base_cost = base_roll_costs.get(symbol, 0.002)

            # Adjust for volatility
            roll_cost = base_cost * (1 + price_volatility * 2)

            return roll_cost

        except Exception as e:
            logger.error(f"Error calculating roll cost for {symbol}: {e}")
            return 0.0

    def execute_roll(self, symbol: str, position_size: int,
                    roll_method: str = 'volume_weighted') -> Dict[str, Any]:
        """
        Execute a contract roll

        Args:
            symbol: Futures symbol
            position_size: Number of contracts to roll
            roll_method: Rolling method to use

        Returns:
            Roll execution results
        """
        try:
            logger.info(f"Executing roll for {symbol}, {position_size} contracts")

            # Get roll cost
            roll_cost = self.calculate_roll_cost(symbol)

            # Check if roll cost is acceptable
            schedule = self.roll_schedules.get(symbol)
            if schedule and roll_cost > schedule.max_roll_cost:
                return {
                    'success': False,
                    'reason': f'Roll cost {roll_cost:.2%} exceeds maximum {schedule.max_roll_cost:.2%}'
                }

            # Execute roll based on method
            if roll_method == 'volume_weighted':
                result = self._execute_volume_weighted_roll(symbol, position_size)
            elif roll_method == 'equal_weighted':
                result = self._execute_equal_weighted_roll(symbol, position_size)
            elif roll_method == 'front_month':
                result = self._execute_front_month_roll(symbol, position_size)
            else:
                result = self._execute_equal_weighted_roll(symbol, position_size)

            # Record roll execution
            roll_record = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'position_size': position_size,
                'roll_cost': roll_cost,
                'method': roll_method,
                'result': result
            }

            self.active_rolls[symbol] = roll_record

            logger.info(f"Roll executed for {symbol}: cost {roll_cost:.2%}")

            return {
                'success': True,
                'roll_cost': roll_cost,
                'method': roll_method,
                'details': result
            }

        except Exception as e:
            logger.error(f"Error executing roll for {symbol}: {e}")
            return {'success': False, 'error': str(e)}

    def _execute_volume_weighted_roll(self, symbol: str, position_size: int) -> Dict[str, Any]:
        """Execute volume-weighted roll over multiple days"""
        # In a real implementation, this would:
        # 1. Calculate volume-weighted roll schedule
        # 2. Execute partial rolls over the roll window
        # 3. Monitor execution quality

        return {
            'method': 'volume_weighted',
            'execution_days': 3,
            'average_cost': 0.0015,
            'slippage': 0.0002
        }

    def _execute_equal_weighted_roll(self, symbol: str, position_size: int) -> Dict[str, Any]:
        """Execute equal-weighted roll over roll window"""
        return {
            'method': 'equal_weighted',
            'execution_days': 2,
            'average_cost': 0.0012,
            'slippage': 0.0001
        }

    def _execute_front_month_roll(self, symbol: str, position_size: int) -> Dict[str, Any]:
        """Execute immediate roll to front month"""
        return {
            'method': 'front_month',
            'execution_days': 1,
            'average_cost': 0.0021,
            'slippage': 0.0003
        }

    def get_roll_schedule(self, symbol: str) -> Optional[RollSchedule]:
        """Get roll schedule for a symbol"""
        return self.roll_schedules.get(symbol)

    def update_roll_schedule(self, symbol: str, schedule: RollSchedule):
        """Update roll schedule for a symbol"""
        self.roll_schedules[symbol] = schedule
        logger.info(f"Updated roll schedule for {symbol}")

    def get_active_rolls(self) -> Dict[str, Any]:
        """Get currently active roll operations"""
        return self.active_rolls.copy()

    def monitor_rolls(self):
        """Monitor ongoing roll operations"""
        for symbol, roll in self.active_rolls.items():
            # Check roll progress and status
            # In production, this would monitor execution quality
            pass

class TickDataManager:
    """
    Advanced tick data management system for high-frequency trading
    """

    def __init__(self):
        self.data_manager = DataManager()
        self.tick_cache = {}
        self.tick_features = {}

    def fetch_tick_data(self, symbol: str, start_date: datetime,
                       end_date: datetime, exchange: str = "alpaca") -> pd.DataFrame:
        """
        Fetch tick-level trade data

        Args:
            symbol: Trading symbol
            start_date: Start date for tick data
            end_date: End date for tick data
            exchange: Exchange to fetch from

        Returns:
            DataFrame with tick data
        """
        try:
            cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}"

            if cache_key in self.tick_cache:
                return self.tick_cache[cache_key]

            # For demonstration - in production this would connect to tick data providers
            # like Alpaca, Polygon, or direct exchange feeds

            if exchange == "alpaca":
                # Alpaca provides minute-level data, not tick data
                # Use as proxy for tick simulation
                df = self.data_manager.fetch_data(symbol, period="1d", interval="1m")
            else:
                # Simulate tick data structure
                df = self._simulate_tick_data(symbol, start_date, end_date)

            self.tick_cache[cache_key] = df
            return df

        except Exception as e:
            logger.error(f"Error fetching tick data for {symbol}: {e}")
            return pd.DataFrame()

    def _simulate_tick_data(self, symbol: str, start_date: datetime,
                           end_date: datetime) -> pd.DataFrame:
        """Simulate tick-level data for demonstration"""
        try:
            # Generate realistic tick data
            minutes = int((end_date - start_date).total_seconds() / 60)
            timestamps = pd.date_range(start_date, end_date, freq='1min')

            # Base price from recent data
            base_df = self.data_manager.fetch_data(symbol, period="5d", interval="1d")
            if not base_df.empty:
                base_price = base_df['close'].iloc[-1]
            else:
                base_price = 100.0  # Default

            # Generate price ticks with realistic volatility
            np.random.seed(42)
            returns = np.random.normal(0, 0.001, len(timestamps))  # 0.1% volatility per minute
            prices = base_price * np.exp(np.cumsum(returns))

            # Generate volume (ticks per minute)
            volumes = np.random.poisson(100, len(timestamps))  # Average 100 ticks per minute

            # Create tick DataFrame
            df = pd.DataFrame({
                'timestamp': timestamps,
                'price': prices,
                'volume': volumes,
                'side': np.random.choice(['buy', 'sell'], len(timestamps)),
                'exchange': 'SIMULATED'
            })

            df.set_index('timestamp', inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error simulating tick data: {e}")
            return pd.DataFrame()

    def calculate_tick_features(self, tick_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced features from tick data

        Args:
            tick_df: Raw tick data

        Returns:
            DataFrame with tick-based features
        """
        try:
            if tick_df.empty:
                return pd.DataFrame()

            df = tick_df.copy()

            # Time-based features
            df['minute'] = df.index.minute
            df['hour'] = df.index.hour
            df['tick_count'] = 1  # Each row is a tick

            # Price-based features
            df['price_change'] = df['price'].diff()
            df['price_acceleration'] = df['price_change'].diff()

            # Volume-based features
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_5']

            # Order flow features
            df['buy_volume'] = df.apply(lambda x: x['volume'] if x['side'] == 'buy' else 0, axis=1)
            df['sell_volume'] = df.apply(lambda x: x['volume'] if x['side'] == 'sell' else 0, axis=1)
            df['order_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['buy_volume'] + df['sell_volume'] + 1)

            # Microstructure features
            df['tick_range'] = df['price'].rolling(10).max() - df['price'].rolling(10).min()
            df['realized_volatility'] = df['price_change'].rolling(10).std()

            # High-frequency momentum
            df['momentum_1m'] = df['price'] / df['price'].shift(1) - 1
            df['momentum_5m'] = df['price'] / df['price'].shift(5) - 1

            # Liquidity measures
            df['spread_estimate'] = df['tick_range'] * 0.1  # Estimated spread
            df['market_depth'] = df['volume'].rolling(20).mean()

            return df

        except Exception as e:
            logger.error(f"Error calculating tick features: {e}")
            return pd.DataFrame()

    def detect_market_microstructure_patterns(self, tick_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect advanced market microstructure patterns from tick data

        Args:
            tick_df: Tick data with features

        Returns:
            Dictionary of detected patterns
        """
        patterns = {}

        try:
            # Order flow toxicity
            buy_pressure = tick_df['buy_volume'].rolling(20).sum()
            sell_pressure = tick_df['sell_volume'].rolling(20).sum()
            patterns['order_flow_toxicity'] = abs(buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1)

            # Price impact analysis
            large_trades = tick_df[tick_df['volume'] > tick_df['volume'].quantile(0.95)]
            patterns['price_impact'] = large_trades['price_change'].abs().mean()

            # Market making activity
            spread_changes = tick_df['spread_estimate'].diff().abs()
            patterns['market_making_intensity'] = spread_changes.rolling(30).mean()

            # High-frequency momentum bursts
            momentum_zscore = (tick_df['momentum_1m'] - tick_df['momentum_1m'].rolling(60).mean()) / tick_df['momentum_1m'].rolling(60).std()
            patterns['momentum_bursts'] = (momentum_zscore.abs() > 2).sum()

            # Liquidity shocks
            volume_shocks = tick_df[tick_df['volume_ratio'] > 3]
            patterns['liquidity_shocks'] = len(volume_shocks)

            return patterns

        except Exception as e:
            logger.error(f"Error detecting microstructure patterns: {e}")
            return {}

def run_futures_rolling_demo():
    """Demo of futures contract rolling system"""
    print("Futures Contract Rolling Demo")
    print("=" * 50)

    roller = FuturesContractRoller()

    # Test contract expiration calculation
    print("\nContract Expiration Dates:")
    for symbol in ['ES', 'NQ', 'CL', 'GC']:
        try:
            expiration = roller.get_contract_expiration(symbol)
            print(f"  {symbol}: {expiration.strftime('%Y-%m-%d %H:%M')}")

            # Check roll status
            roll_status = roller.check_roll_status(symbol)
            if roll_status['needs_roll']:
                print(f"    WARNING: Needs rolling - {roll_status['days_to_expiration']} days to expiration")
            else:
                print(f"    OK: No roll needed - {roll_status['days_to_expiration']} days to expiration")

        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    # Test roll cost calculation
    print("\nRoll Cost Estimates:")
    for symbol in ['ES', 'NQ', 'CL', 'GC']:
        roll_cost = roller.calculate_roll_cost(symbol)
        print(f"  {symbol}: {roll_cost:.2%}")

    # Test tick data simulation
    print("\nTick Data Features Demo:")
    tick_manager = TickDataManager()

    # Simulate tick data
    start_date = datetime.now() - timedelta(hours=1)
    end_date = datetime.now()
    tick_data = tick_manager.fetch_tick_data('ES=F', start_date, end_date)

    if not tick_data.empty:
        print(f"  Generated {len(tick_data)} tick records")

        # Calculate tick features
        tick_features = tick_manager.calculate_tick_features(tick_data)
        if not tick_features.empty:
            print(f"  Calculated {len(tick_features.columns)} tick-based features")

            # Detect patterns
            patterns = tick_manager.detect_market_microstructure_patterns(tick_features)
            print(f"  Detected {len(patterns)} microstructure patterns")

            # Show sample patterns
            for pattern_name, value in list(patterns.items())[:3]:
                print(f"    {pattern_name}: {value}")

    print("\nFutures rolling and tick data systems ready!")

if __name__ == "__main__":
    run_futures_rolling_demo()