"""
Automated Futures Day Trading System
Runs continuous ML-based futures trading with risk management
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from threading import Thread, Event

from config import CONFIG
from data_manager import DataManager
from simple_trader import SimpleXGBoostTrader
from alpaca_integration import AlpacaIntegratedTrader
from futures_contract_rolling import FuturesContractRoller, TickDataManager

logger = logging.getLogger(__name__)

class AutomatedFuturesTrader:
    """
    Automated futures day trading system with ML signals and risk management
    """

    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        """
        Initialize automated futures trader

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper = paper

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials required")

        # Initialize components
        self.data_manager = DataManager()
        self.contract_roller = FuturesContractRoller()
        self.tick_manager = TickDataManager()
        self.alpaca_trader = AlpacaIntegratedTrader(self.api_key, self.secret_key, self.paper)

        # Trading state
        self.symbols = []
        self.models = {}  # symbol -> trained model
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.is_trading = False
        self.stop_event = Event()

        # Risk management
        self.max_daily_trades = 20
        self.daily_profit_target = 0.03  # 3%
        self.daily_loss_limit = 0.05    # 5%
        self.max_position_size = 3       # Max contracts per position

        # Trading schedule (Eastern Time) - Regular market hours
        self.trading_start = "09:30"
        self.trading_end = "15:30"  # 3:30 PM ET
        self.data_update_interval = 60  # seconds

        logger.info("Automated Futures Trader initialized")

    def check_contract_rolls(self) -> Dict[str, Dict]:
        """
        Check if any futures contracts need to be rolled

        Returns:
            Dictionary of symbols needing rolls and their status
        """
        roll_status = {}

        for symbol in self.symbols:
            try:
                status = self.contract_roller.check_roll_status(symbol)
                if status['needs_roll']:
                    roll_status[symbol] = status
                    logger.info(f"Contract roll needed for {symbol}: {status['days_to_expiration']} days to expiration")
            except Exception as e:
                logger.error(f"Error checking roll status for {symbol}: {e}")

        return roll_status

    def execute_contract_rolls(self, roll_status: Dict[str, Dict]) -> List[Dict]:
        """
        Execute contract rolls for symbols that need them

        Args:
            roll_status: Dictionary from check_contract_rolls()

        Returns:
            List of roll execution results
        """
        roll_results = []

        for symbol, status in roll_status.items():
            try:
                # Get current position size
                positions = self.alpaca_trader.alpaca_trader.get_positions()
                current_position = 0

                for position in positions:
                    if position['symbol'].replace('=F', '') == symbol:
                        current_position = abs(int(float(position['qty'])))
                        break

                if current_position > 0:
                    # Execute roll
                    roll_result = self.contract_roller.execute_roll(symbol, current_position)
                    roll_results.append({
                        'symbol': symbol,
                        'position_size': current_position,
                        'roll_result': roll_result
                    })

                    if roll_result['success']:
                        logger.info(f"Successfully rolled {current_position} contracts of {symbol}, cost: {roll_result['roll_cost']:.2%}")
                    else:
                        logger.error(f"Failed to roll {symbol}: {roll_result.get('reason', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error executing roll for {symbol}: {e}")

        return roll_results

    def get_tick_based_signals(self) -> Dict[str, Dict]:
        """
        Generate signals using tick-based data for higher frequency trading

        Returns:
            Dictionary of tick-enhanced signals
        """
        signals = {}

        for symbol in self.symbols:
            if symbol not in self.models:
                continue

            try:
                # Get tick data for last hour
                end_date = datetime.now()
                start_date = end_date - timedelta(hours=1)

                tick_df = self.tick_manager.fetch_tick_data(f"{symbol}=F", start_date, end_date)

                if tick_df.empty:
                    continue

                # Calculate tick features
                tick_features_df = self.tick_manager.calculate_tick_features(tick_df)

                if tick_features_df.empty:
                    continue

                # Detect microstructure patterns
                patterns = self.tick_manager.detect_market_microstructure_patterns(tick_features_df)

                # Get latest OHLC data for signal generation
                df = self.data_manager.prepare_futures_dataset(symbol, period="1d", interval="1m")

                if df.empty or len(df) < 20:
                    continue

                # Generate features for latest data
                trader = self.models[symbol]
                X, _ = trader.prepare_features(df)

                if len(X) == 0:
                    continue

                # Get latest signal
                latest_features = X[-1:]
                signal = trader.predict(latest_features)[0]
                confidence = np.max(trader.predict_proba(latest_features)[0])

                # Enhance signal with tick data
                tick_enhanced_confidence = confidence

                # Boost confidence if tick patterns confirm signal
                if signal == 1:  # Buy signal
                    if patterns.get('order_flow_toxicity', 1) < 0.3:  # Low toxicity = good buying conditions
                        tick_enhanced_confidence = min(1.0, confidence * 1.2)
                elif signal == -1:  # Sell signal
                    if patterns.get('momentum_bursts', 0) > 5:  # High momentum bursts
                        tick_enhanced_confidence = min(1.0, confidence * 1.15)

                # Get current price
                current_price = df['close'].iloc[-1]

                signals[symbol] = {
                    'signal': int(signal),
                    'confidence': float(confidence),
                    'tick_enhanced_confidence': float(tick_enhanced_confidence),
                    'current_price': float(current_price),
                    'tick_patterns': patterns,
                    'timestamp': df.index[-1]
                }

            except Exception as e:
                logger.error(f"Error generating tick-based signal for {symbol}: {e}")

        return signals

        logger.info("Automated Futures Trader initialized")

    def train_models(self, symbols: List[str], days: int = 60) -> Dict:
        """
        Train ML models for each futures symbol

        Args:
            symbols: List of futures symbols (e.g., ['ES', 'NQ'])
            days: Days of historical data to use

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training models for {len(symbols)} futures symbols...")

        training_results = {}

        for symbol in symbols:
            try:
                logger.info(f"Training model for {symbol}...")

                # Fetch historical data
                df = self.data_manager.prepare_futures_dataset(symbol, period=f"{days}d", interval="5m")

                if df.empty or len(df) < 100:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue

                # Train model
                trader = SimpleXGBoostTrader()
                X, y = trader.prepare_features(df)
                metrics = trader.train(X, y)

                # Store model
                self.models[symbol] = trader

                training_results[symbol] = {
                    'model': trader,
                    'metrics': metrics,
                    'data_points': len(X),
                    'accuracy': metrics.get('test_accuracy', 0)
                }

                logger.info(f"Model trained for {symbol}: {metrics.get('test_accuracy', 0):.3f} accuracy")

            except Exception as e:
                logger.error(f"Error training model for {symbol}: {e}")
                training_results[symbol] = {'error': str(e)}

        return training_results

    def is_trading_hours(self) -> bool:
        """Check if current time is within regular market trading hours"""
        now = datetime.now()

        # Convert to Eastern Time (simplified - assumes system is in ET)
        # In production, use proper timezone conversion
        current_time = now.strftime("%H:%M")

        # Regular market hours: Monday-Friday, 9:30 AM to 3:30 PM ET
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Only trade Monday through Friday
        if weekday < 5:  # Monday-Friday (0-4)
            return self.trading_start <= current_time <= self.trading_end
        else:
            return False

    def check_daily_limits(self) -> bool:
        """
        Check if daily trading limits have been reached

        Returns:
            True if can continue trading, False if limits reached
        """
        # Check trade count limit
        if self.daily_trades >= self.max_daily_trades:
            logger.warning(f"Daily trade limit reached: {self.daily_trades}/{self.max_daily_trades}")
            return False

        # Check profit target
        if self.daily_pnl >= self.daily_profit_target:
            logger.info(f"Daily profit target reached: {self.daily_pnl:.2%}")
            return False

        # Check loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2%}")
            return False

        return True

    def generate_signals(self) -> Dict[str, Dict]:
        """
        Generate trading signals for all symbols

        Returns:
            Dictionary of symbol -> signal data
        """
        signals = {}

        for symbol in self.symbols:
            if symbol not in self.models:
                continue

            try:
                # Get latest data
                df = self.data_manager.prepare_futures_dataset(symbol, period="5d", interval="5m")

                if df.empty or len(df) < 20:
                    continue

                # Generate features for latest data
                trader = self.models[symbol]
                X, _ = trader.prepare_features(df)

                if len(X) == 0:
                    continue

                # Get latest signal
                latest_features = X[-1:]
                signal = trader.predict(latest_features)[0]
                confidence = np.max(trader.predict_proba(latest_features)[0])

                # Get current price
                current_price = df['close'].iloc[-1]

                signals[symbol] = {
                    'signal': int(signal),
                    'confidence': float(confidence),
                    'current_price': float(current_price),
                    'timestamp': df.index[-1]
                }

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        return signals

    def execute_trades(self, signals: Dict[str, Dict]) -> List[Dict]:
        """
        Execute trades based on signals (enhanced with tick data)

        Args:
            signals: Dictionary of signals from get_tick_based_signals()

        Returns:
            List of executed trades
        """
        executed_trades = []

        for symbol, signal_data in signals.items():
            signal = signal_data['signal']
            confidence = signal_data.get('tick_enhanced_confidence', signal_data['confidence'])

            # Only trade high-confidence signals (lower threshold for tick-enhanced signals)
            confidence_threshold = 0.60 if 'tick_enhanced_confidence' in signal_data else 0.65

            if abs(signal) != 1 or confidence < confidence_threshold:
                continue

            try:
                # Check position limits
                account = self.alpaca_trader.alpaca_trader.get_account_info()
                positions = self.alpaca_trader.alpaca_trader.get_positions()

                # Count current positions
                current_positions = len([p for p in positions if p['symbol'].replace('=F', '') in self.symbols])

                if current_positions >= len(self.symbols):
                    continue  # Max one position per symbol

                # Execute trade
                results = self.alpaca_trader.execute_strategy([f"{symbol}=F"], max_positions=1, asset_type="futures")

                if results.get('executed_trades'):
                    executed_trades.extend(results['executed_trades'])
                    self.daily_trades += len(results['executed_trades'])

                    # Update daily P&L (simplified)
                    for trade in results['executed_trades']:
                        if trade['action'] == 'BUY':
                            # Estimate P&L impact (simplified)
                            self.daily_pnl -= 0.0005  # Commission estimate
                        elif trade['action'] == 'SELL':
                            self.daily_pnl += 0.001  # Rough profit estimate

                logger.info(f"Executed {len(results.get('executed_trades', []))} trades for {symbol}")

            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")

        return executed_trades

    def trading_loop(self):
        """Main trading loop - operates during regular market hours"""
        logger.info("Starting automated futures trading loop (9:30 AM - 3:30 PM ET, Mon-Fri)...")

        while not self.stop_event.is_set():
            try:
                # Check if we should be trading
                if not self.is_trading_hours():
                    logger.debug("Outside trading hours, waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue

                if not self.check_daily_limits():
                    logger.info("Daily limits reached, stopping for today")
                    self.stop_trading()
                    break

                # Check for contract rolls
                roll_status = self.check_contract_rolls()
                if roll_status:
                    logger.info(f"Contract rolls needed for: {list(roll_status.keys())}")
                    roll_results = self.execute_contract_rolls(roll_status)
                    logger.info(f"Executed {len(roll_results)} contract rolls")

                # Generate signals (enhanced with tick data)
                signals = self.get_tick_based_signals()

                if signals:
                    logger.info(f"Generated tick-enhanced signals for {len(signals)} symbols")

                    # Log tick pattern insights
                    for symbol, signal_data in signals.items():
                        if signal_data.get('tick_patterns'):
                            patterns = signal_data['tick_patterns']
                            toxic = patterns.get('order_flow_toxicity', 0)
                            bursts = patterns.get('momentum_bursts', 0)
                            logger.debug(f"{symbol} tick patterns - Toxicity: {toxic:.2f}, Momentum bursts: {bursts}")

                    # Execute trades
                    executed_trades = self.execute_trades(signals)

                    if executed_trades:
                        logger.info(f"Executed {len(executed_trades)} trades")

                # Wait before next iteration
                time.sleep(self.data_update_interval)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute on error

        logger.info("Trading loop stopped")

    def start_trading(self, symbols: List[str], max_daily_trades: int = 20,
                     daily_profit_target: float = 0.03, daily_loss_limit: float = 0.05):
        """
        Start automated trading

        Args:
            symbols: List of futures symbols to trade
            max_daily_trades: Maximum trades per day
            daily_profit_target: Daily profit target (decimal)
            daily_loss_limit: Daily loss limit (decimal)
        """
        self.symbols = symbols
        self.max_daily_trades = max_daily_trades
        self.daily_profit_target = daily_profit_target
        self.daily_loss_limit = daily_loss_limit

        # Reset daily counters
        self.daily_pnl = 0.0
        self.daily_trades = 0

        # Train models
        training_results = self.train_models(symbols)

        successful_models = [s for s, r in training_results.items() if 'error' not in r]
        logger.info(f"Successfully trained models for {len(successful_models)}/{len(symbols)} symbols")

        if len(successful_models) == 0:
            raise ValueError("No models trained successfully")

        # Start trading
        self.is_trading = True
        self.stop_event.clear()

        # Start trading thread
        trading_thread = Thread(target=self.trading_loop, daemon=True)
        trading_thread.start()

        logger.info(f"🚀 Automated futures trading started for: {successful_models}")
        logger.info(f"Daily limits: {max_daily_trades} trades, {daily_profit_target:.1%} profit target, {daily_loss_limit:.1%} loss limit")

        return {
            'status': 'started',
            'symbols': successful_models,
            'training_results': training_results
        }

    def stop_trading(self):
        """Stop automated trading"""
        logger.info("Stopping automated futures trading...")
        self.is_trading = True
        self.stop_event.set()

        # Close all positions
        try:
            positions = self.alpaca_trader.alpaca_trader.get_positions()
            symbols_to_close = [p['symbol'] for p in positions if p['qty'] != 0]

            if symbols_to_close:
                logger.info(f"Closing positions: {symbols_to_close}")
                self.alpaca_trader.execute_strategy(symbols_to_close, asset_type="futures")

        except Exception as e:
            logger.error(f"Error closing positions: {e}")

    def get_status(self) -> Dict:
        """Get current trading status with contract rolling and tick data info"""
        # Get contract roll status
        roll_status = self.check_contract_rolls()

        return {
            'is_trading': self.is_trading,
            'symbols': self.symbols,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'is_trading_hours': self.is_trading_hours(),
            'contract_rolls_needed': list(roll_status.keys()),
            'tick_data_enabled': True,
            'account_info': self.alpaca_trader.alpaca_trader.get_account_info()
        }

if __name__ == "__main__":
    # Example usage
    print("🤖 Automated Futures Day Trading System")
    print("=" * 50)

    # Get API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        print("❌ Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        exit(1)

    # Initialize automated trader
    auto_trader = AutomatedFuturesTrader(api_key, secret_key, paper=True)

    # Start trading
    try:
        result = auto_trader.start_trading(
            symbols=['ES', 'NQ'],
            max_daily_trades=10,
            daily_profit_target=0.02,
            daily_loss_limit=0.03
        )

        print(f"✅ Trading started for: {result['symbols']}")

        # Monitor for a while (in production, this would run continuously)
        import time
        for i in range(10):  # Monitor for ~10 minutes
            time.sleep(60)
            status = auto_trader.get_status()
            print(f"Status: Trades={status['daily_trades']}, P&L={status['daily_pnl']:.2%}")

        # Stop trading
        auto_trader.stop_trading()
        print("✅ Trading stopped")

    except KeyboardInterrupt:
        print("\n🛑 Stopping trading...")
        auto_trader.stop_trading()
    except Exception as e:
        print(f"❌ Error: {e}")
        auto_trader.stop_trading()