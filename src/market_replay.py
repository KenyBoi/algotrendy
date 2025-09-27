"""
Market Replay System for Algorithm Testing
Replays historical market data in real-time or accelerated time for testing trading algorithms
"""

import pandas as pd
import numpy as np
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
import queue

from config import CONFIG
from data_manager import DataManager

logger = logging.getLogger(__name__)

@dataclass
class ReplayEvent:
    """Market data event during replay"""
    timestamp: datetime
    symbol: str
    price_data: Dict[str, float]
    event_type: str = "price_update"

@dataclass
class ReplayConfig:
    """Configuration for market replay"""
    symbols: List[str]
    start_date: str
    end_date: str
    interval: str = "5m"
    speed_multiplier: float = 1.0  # 1.0 = real-time, 2.0 = 2x speed, etc.
    loop: bool = False  # Loop replay when finished
    shuffle_days: bool = False  # Randomize day order for robustness testing

class MarketReplay:
    """
    Real-time market data replay system for algorithm testing
    """

    def __init__(self, config: ReplayConfig):
        """
        Initialize market replay system

        Args:
            config: Replay configuration
        """
        self.config = config
        self.data_manager = DataManager()

        # Replay state
        self.is_running = False
        self.is_paused = False
        self.current_time = None
        self.start_time = None
        self.end_time = None

        # Data storage
        self.market_data = {}  # symbol -> DataFrame
        self.event_queue = queue.Queue()

        # Callbacks
        self.price_update_callbacks = []
        self.trading_hours_callbacks = []

        # Threading
        self.replay_thread = None
        self.stop_event = threading.Event()

        logger.info(f"Market Replay initialized for {len(config.symbols)} symbols")

    def load_data(self) -> bool:
        """
        Load historical data for all symbols

        Returns:
            True if data loaded successfully
        """
        try:
            logger.info("Loading historical market data...")

            for symbol in self.config.symbols:
                # Load data (use futures data if symbol ends with =F)
                asset_type = "futures" if symbol.endswith("=F") else "stock"
                clean_symbol = symbol.replace("=F", "")

                df = self.data_manager.fetch_data(
                    clean_symbol,
                    period="max",
                    interval=self.config.interval,
                    asset_type=asset_type
                )

                if df.empty:
                    logger.error(f"No data available for {symbol}")
                    return False

                # Filter date range
                df = df[(df.index >= self.config.start_date) & (df.index <= self.config.end_date)]

                if df.empty:
                    logger.error(f"No data in date range for {symbol}")
                    return False

                self.market_data[symbol] = df
                logger.info(f"Loaded {len(df)} bars for {symbol}")

            # Set replay time bounds
            all_timestamps = []
            for df in self.market_data.values():
                all_timestamps.extend(df.index)

            if all_timestamps:
                self.start_time = min(all_timestamps)
                self.end_time = max(all_timestamps)
                self.current_time = self.start_time

                logger.info(f"Replay time range: {self.start_time} to {self.end_time}")
                return True
            else:
                logger.error("No timestamps found in data")
                return False

        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            return False

    def add_price_callback(self, callback: Callable[[ReplayEvent], None]):
        """
        Add callback for price updates

        Args:
            callback: Function to call on price updates
        """
        self.price_update_callbacks.append(callback)

    def add_trading_hours_callback(self, callback: Callable[[bool], None]):
        """
        Add callback for trading hours changes

        Args:
            callback: Function to call when trading hours change (True=start, False=end)
        """
        self.trading_hours_callbacks.append(callback)

    def _is_trading_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is within trading hours"""
        # Simplified trading hours check (9:30 AM - 4:00 PM ET)
        # In production, use proper timezone handling
        hour = timestamp.hour
        minute = timestamp.minute

        current_minutes = hour * 60 + minute
        market_open_minutes = 9 * 60 + 30  # 9:30 AM
        market_close_minutes = 16 * 60      # 4:00 PM

        return market_open_minutes <= current_minutes <= market_close_minutes

    def _get_next_price_update(self) -> Optional[ReplayEvent]:
        """Get next price update event"""
        if self.current_time >= self.end_time:
            return None

        # Find next timestamp across all symbols
        next_time = None
        price_data = {}

        for symbol, df in self.market_data.items():
            # Find data at or after current time
            future_data = df[df.index >= self.current_time]
            if not future_data.empty:
                symbol_next_time = future_data.index[0]
                if next_time is None or symbol_next_time < next_time:
                    next_time = symbol_next_time

        if next_time is None:
            return None

        # Collect price data for all symbols at this timestamp
        for symbol, df in self.market_data.items():
            symbol_data = df[df.index == next_time]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                price_data[symbol] = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }

        self.current_time = next_time
        return ReplayEvent(
            timestamp=next_time,
            symbol=list(price_data.keys())[0],  # Primary symbol
            price_data=price_data
        )

    def _replay_loop(self):
        """Main replay loop"""
        logger.info("Starting market replay...")

        last_trading_hours = None

        while not self.stop_event.is_set():
            if self.is_paused:
                time.sleep(0.1)
                continue

            # Check trading hours
            current_trading_hours = self._is_trading_hours(self.current_time)
            if current_trading_hours != last_trading_hours:
                for callback in self.trading_hours_callbacks:
                    try:
                        callback(current_trading_hours)
                    except Exception as e:
                        logger.error(f"Error in trading hours callback: {e}")
                last_trading_hours = current_trading_hours

            # Only process price updates during trading hours
            if current_trading_hours:
                event = self._get_next_price_update()

                if event is None:
                    if self.config.loop:
                        logger.info("Replay finished, looping...")
                        self.current_time = self.start_time
                        continue
                    else:
                        logger.info("Replay finished")
                        break

                # Trigger price update callbacks
                for callback in self.price_update_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in price callback: {e}")

            # Calculate sleep time based on speed multiplier
            if current_trading_hours:
                # Sleep based on data interval and speed
                interval_seconds = self._interval_to_seconds(self.config.interval)
                sleep_time = interval_seconds / self.config.speed_multiplier
            else:
                # Fast-forward through non-trading hours
                sleep_time = 0.01  # Very fast

            time.sleep(min(sleep_time, 1.0))  # Cap at 1 second for responsiveness

        self.is_running = False
        logger.info("Market replay stopped")

    def _interval_to_seconds(self, interval: str) -> float:
        """Convert interval string to seconds"""
        interval_map = {
            '1m': 60,
            '2m': 120,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '1d': 86400
        }
        return interval_map.get(interval, 300)  # Default to 5 minutes

    def start_replay(self) -> bool:
        """
        Start market replay

        Returns:
            True if replay started successfully
        """
        if not self.market_data:
            logger.error("No data loaded. Call load_data() first.")
            return False

        if self.is_running:
            logger.warning("Replay already running")
            return False

        self.is_running = True
        self.is_paused = False
        self.stop_event.clear()

        self.replay_thread = threading.Thread(target=self._replay_loop, daemon=True)
        self.replay_thread.start()

        logger.info(f"Market replay started at {self.config.speed_multiplier}x speed")
        return True

    def pause_replay(self):
        """Pause replay"""
        self.is_paused = True
        logger.info("Market replay paused")

    def resume_replay(self):
        """Resume replay"""
        self.is_paused = False
        logger.info("Market replay resumed")

    def stop_replay(self):
        """Stop replay"""
        self.stop_event.set()
        if self.replay_thread:
            self.replay_thread.join(timeout=5)
        self.is_running = False
        logger.info("Market replay stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current replay status"""
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'current_time': self.current_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'progress': (self.current_time - self.start_time).total_seconds() / (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else 0,
            'speed_multiplier': self.config.speed_multiplier,
            'symbols': list(self.market_data.keys())
        }

    def set_speed(self, multiplier: float):
        """
        Set replay speed multiplier

        Args:
            multiplier: Speed multiplier (1.0 = real-time, 2.0 = 2x speed, etc.)
        """
        self.config.speed_multiplier = max(0.1, multiplier)  # Minimum 0.1x speed
        logger.info(f"Replay speed set to {self.config.speed_multiplier}x")

class ReplayTradingAlgorithm:
    """
    Example trading algorithm that works with market replay
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.positions = {}
        self.cash = 100000.0
        self.trades = []

        # Simple moving average strategy
        self.fast_period = 5
        self.slow_period = 20
        self.price_history = {symbol: [] for symbol in symbols}

    def on_price_update(self, event: ReplayEvent):
        """Handle price update events"""
        for symbol, prices in event.price_data.items():
            # Update price history
            self.price_history[symbol].append(prices['close'])

            # Keep only recent prices
            if len(self.price_history[symbol]) > self.slow_period + 10:
                self.price_history[symbol] = self.price_history[symbol][-self.slow_period-10:]

            # Simple moving average crossover strategy
            if len(self.price_history[symbol]) >= self.slow_period:
                fast_ma = np.mean(self.price_history[symbol][-self.fast_period:])
                slow_ma = np.mean(self.price_history[symbol][-self.slow_period:])

                current_price = prices['close']

                # Generate signals
                if fast_ma > slow_ma and symbol not in self.positions:
                    # Buy signal
                    shares = int(self.cash * 0.1 / current_price)  # Use 10% of cash
                    if shares > 0:
                        self.positions[symbol] = shares
                        self.cash -= shares * current_price
                        self.trades.append({
                            'timestamp': event.timestamp,
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': shares,
                            'price': current_price
                        })
                        print(f"BUY {shares} {symbol} @ ${current_price:.2f}")

                elif fast_ma < slow_ma and symbol in self.positions:
                    # Sell signal
                    shares = self.positions[symbol]
                    self.cash += shares * current_price
                    del self.positions[symbol]
                    self.trades.append({
                        'timestamp': event.timestamp,
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': shares,
                        'price': current_price
                    })
                    print(f"SELL {shares} {symbol} @ ${current_price:.2f}")

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        portfolio_value = self.cash
        for symbol, shares in self.positions.items():
            if symbol in current_prices:
                portfolio_value += shares * current_prices[symbol]
        return portfolio_value

def run_market_replay_demo():
    """Demo of market replay system"""
    print("🎬 Market Replay Demo")
    print("=" * 50)

    # Configure replay
    config = ReplayConfig(
        symbols=['AAPL', 'GOOGL'],
        start_date='2024-01-01',
        end_date='2024-01-31',
        interval='5m',
        speed_multiplier=100.0  # 100x speed for demo
    )

    # Initialize replay system
    replay = MarketReplay(config)

    # Load data
    if not replay.load_data():
        print("❌ Failed to load market data")
        return

    # Initialize trading algorithm
    trader = ReplayTradingAlgorithm(config.symbols)

    # Connect algorithm to replay
    replay.add_price_callback(trader.on_price_update)

    # Start replay
    print(f"Starting replay at {config.speed_multiplier}x speed...")
    replay.start_replay()

    # Monitor for a while
    try:
        for i in range(50):  # Monitor for ~50 updates
            time.sleep(0.1)
            status = replay.get_status()
            if status['current_time']:
                current_prices = {}
                for symbol in config.symbols:
                    if symbol in replay.market_data:
                        recent_data = replay.market_data[symbol][replay.market_data[symbol].index <= status['current_time']]
                        if not recent_data.empty:
                            current_prices[symbol] = recent_data['close'].iloc[-1]

                portfolio_value = trader.get_portfolio_value(current_prices)
                print(f"Time: {status['current_time']} | Portfolio: ${portfolio_value:,.2f} | Trades: {len(trader.trades)}")

    except KeyboardInterrupt:
        print("\nStopping replay...")

    # Stop replay
    replay.stop_replay()

    # Final results
    final_value = trader.get_portfolio_value(current_prices)
    print("\n📊 Final Results:")
    print(f"   Starting Capital: $100,000")
    print(f"   Final Portfolio: ${final_value:,.2f}")
    print(f"   Total Return: {((final_value - 100000) / 100000 * 100):.2f}%")
    print(f"   Total Trades: {len(trader.trades)}")

    print("\n✅ Market replay demo completed!")

if __name__ == "__main__":
    run_market_replay_demo()