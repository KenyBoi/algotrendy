"""
Crypto Scalping Trader
High-frequency, 24/7 crypto trading system optimized for small, frequent profits.
"""

import os
import sys
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
# Binance imports (optional)
try:
    from binance import BinanceWs
    BINANCE_WS_AVAILABLE = True
except ImportError:
    BINANCE_WS_AVAILABLE = False
    print("Warning: Binance WebSocket not available")
# Optional imports for crypto exchanges
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: CCXT not available for crypto trading")
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: websocket-client not available")
import json

from config import CONFIG
from data_manager import DataManager
# Optional ML trainer import
try:
    from advanced_ml_trainer import AdvancedMLTrainer
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    print("Warning: Advanced ML trainer not available")
try:
    from alpaca_integration import AlpacaIntegratedTrader
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("Warning: Alpaca integration not available")

class CryptoScalpingTrader:
    """
    High-frequency crypto scalping system for 24/7 automated trading
    """

    def __init__(self, exchange: str = "binance", symbols: List[str] = None):
        """
        Initialize crypto scalping trader

        Args:
            exchange: Exchange to trade on ('binance', 'coinbase', 'alpaca')
            symbols: List of crypto symbols to trade
        """
        self.exchange = exchange
        self.symbols = symbols or ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        self.is_running = False
        self.positions = {}
        self.account_balance = {}
        self.last_update = datetime.now()

        # Scalping parameters
        self.scalping_config = {
            'timeframe': '1m',  # 1-minute bars for scalping
            'profit_target': 0.002,  # 0.2% profit target
            'stop_loss': 0.001,  # 0.1% stop loss
            'max_position_size': 0.02,  # 2% of portfolio per trade
            'max_trades_per_hour': 20,  # Limit trade frequency
            'min_volume': 10000,  # Minimum 24h volume
            'max_spread': 0.001,  # Maximum acceptable spread
            'cooldown_period': 30,  # Seconds between trades per symbol
        }

        # Initialize exchange connections
        self.exchange_clients = {}
        self.websocket_connections = {}
        self._initialize_exchanges()

        # ML components
        self.ml_trainer = AdvancedMLTrainer(symbol="BTC", asset_type="crypto") if ADVANCED_ML_AVAILABLE else None
        self.scalping_models = {}
        self.ml_features = {}

        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        self.daily_pnl_history = []

        # Risk management
        self.risk_manager = CryptoRiskManager(self.scalping_config)

        print(f"Crypto Scalping Trader initialized for {exchange} with symbols: {self.symbols}")

    def _initialize_exchanges(self):
        """Initialize connections to crypto exchanges"""
        try:
            if self.exchange == "binance":
                api_key = os.getenv('BINANCE_API_KEY')
                api_secret = os.getenv('BINANCE_SECRET_KEY')

                if api_key and api_secret:
                    self.exchange_clients['binance'] = BinanceClient(api_key, api_secret)
                    print("Binance client initialized")
                else:
                    print("Warning: Binance API keys not found")

            elif self.exchange == "coinbase":
                # Initialize Coinbase Pro client
                self.exchange_clients['coinbase'] = ccxt.coinbasepro({
                    'apiKey': os.getenv('COINBASE_API_KEY'),
                    'secret': os.getenv('COINBASE_SECRET'),
                    'password': os.getenv('COINBASE_PASSPHRASE')
                })
                print("Coinbase Pro client initialized")

            elif self.exchange == "alpaca":
                # Use existing Alpaca integration for crypto
                self.exchange_clients['alpaca'] = AlpacaIntegratedTrader(
                    os.getenv('ALPACA_API_KEY'),
                    os.getenv('ALPACA_SECRET_KEY'),
                    paper=True
                )
                print("Alpaca crypto client initialized")

        except Exception as e:
            print(f"Error initializing {self.exchange}: {e}")

    def start_scalping(self):
        """Start the scalping operation"""
        print("🚀 Starting crypto scalping operation...")

        self.is_running = True

        # Start background threads
        threading.Thread(target=self._market_data_thread, daemon=True).start()
        threading.Thread(target=self._trading_thread, daemon=True).start()
        threading.Thread(target=self._risk_monitoring_thread, daemon=True).start()

        # WebSocket connections for real-time data
        self._start_websocket_connections()

        print("✅ Crypto scalping system is now active!")

    def stop_scalping(self):
        """Stop the scalping operation"""
        print("🛑 Stopping crypto scalping operation...")

        self.is_running = False

        # Close WebSocket connections
        for symbol, ws in self.websocket_connections.items():
            try:
                ws.close()
            except:
                pass

        # Close exchange connections
        for client in self.exchange_clients.values():
            try:
                if hasattr(client, 'close'):
                    client.close()
            except:
                pass

        print("✅ Crypto scalping system stopped")

    def _market_data_thread(self):
        """Background thread for market data processing"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    self._update_market_data(symbol)
                time.sleep(1)  # Update every second
            except Exception as e:
                print(f"Market data thread error: {e}")
                time.sleep(5)

    def _trading_thread(self):
        """Background thread for trade execution"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    if self._should_trade(symbol):
                        self._execute_scalp_trade(symbol)
                time.sleep(0.1)  # Very fast trading loop
            except Exception as e:
                print(f"Trading thread error: {e}")
                time.sleep(1)

    def _risk_monitoring_thread(self):
        """Background thread for risk monitoring"""
        while self.is_running:
            try:
                self.risk_manager.monitor_positions(self.positions)
                self.risk_manager.check_daily_limits()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                print(f"Risk monitoring error: {e}")
                time.sleep(30)

    def _start_websocket_connections(self):
        """Start WebSocket connections for real-time data"""
        try:
            if self.exchange == "binance":
                self._start_binance_websockets()
            elif self.exchange == "coinbase":
                self._start_coinbase_websockets()
        except Exception as e:
            print(f"WebSocket initialization error: {e}")

    def _start_binance_websockets(self):
        """Start Binance WebSocket connections"""
        try:
            # Use BinanceWs for WebSocket connections
            ws_client = BinanceWs()

            # Start ticker sockets for each symbol
            for symbol in self.symbols:
                binance_symbol = symbol.replace('/', '').lower()
                # Note: BinanceWs API might be different, simplified for now
                print(f"WebSocket connection for {binance_symbol} would be established here")

        except Exception as e:
            print(f"Binance WebSocket error: {e}")

    def _binance_ticker_callback(self, msg):
        """Handle Binance ticker updates"""
        try:
            if msg['e'] == '24hrTicker':
                symbol = msg['s']
                price = float(msg['c'])
                volume = float(msg['v'])

                # Update internal price data
                self._update_price_data(symbol, price, volume)

        except Exception as e:
            print(f"Binance callback error: {e}")

    def _start_coinbase_websockets(self):
        """Start Coinbase WebSocket connections"""
        # Implementation for Coinbase WebSockets
        pass

    def _update_market_data(self, symbol: str):
        """Update market data for a symbol"""
        try:
            # Get recent klines/candles
            if self.exchange == "binance":
                # Use CCXT for Binance data (more reliable)
                exchange = ccxt.binance({
                    'apiKey': os.getenv('BINANCE_API_KEY'),
                    'secret': os.getenv('BINANCE_SECRET_KEY'),
                })

                # Get OHLCV data (Open, High, Low, Close, Volume)
                ohlcv = exchange.fetch_ohlcv(symbol, '1m', limit=100)

                # Process OHLCV into DataFrame
                df = self._process_ohlcv(ohlcv, symbol)

                # Update ML features
                self._update_ml_features(symbol, df)

            elif self.exchange == "coinbase":
                # Coinbase data fetching
                pass

        except Exception as e:
            print(f"Market data update error for {symbol}: {e}")

    def _process_ohlcv(self, ohlcv: List, symbol: str) -> pd.DataFrame:
        """Process raw OHLCV data into DataFrame"""
        try:
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            print(f"OHLCV processing error: {e}")
            return pd.DataFrame()

    def _update_ml_features(self, symbol: str, df: pd.DataFrame):
        """Update ML features for scalping decisions"""
        try:
            if len(df) < 50:  # Need enough data
                return

            # Calculate scalping-specific features
            features = self._calculate_scalping_features(df)

            # Store for ML model
            self.ml_features[symbol] = features

        except Exception as e:
            print(f"ML features update error for {symbol}: {e}")

    def _calculate_scalping_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate features optimized for scalping"""
        try:
            # Price momentum (very short-term)
            df['returns_1m'] = df['close'].pct_change(1)
            df['returns_3m'] = df['close'].pct_change(3)
            df['returns_5m'] = df['close'].pct_change(5)

            # Volatility (1-minute realized)
            df['volatility_5m'] = df['returns_1m'].rolling(5).std()
            df['volatility_10m'] = df['returns_1m'].rolling(10).std()

            # Volume analysis
            df['volume_sma_5'] = df['volume'].rolling(5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_5']

            # Price action
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['close_open_ratio'] = df['close'] / df['open']

            # Microstructure features
            df['spread_estimate'] = df['high_low_range'] * 0.1  # Estimated spread
            df['price_acceleration'] = df['returns_1m'] - df['returns_1m'].shift(1)

            # Get latest values
            latest = df.iloc[-1]

            features = {
                'returns_1m': latest.get('returns_1m', 0),
                'returns_3m': latest.get('returns_3m', 0),
                'returns_5m': latest.get('returns_5m', 0),
                'volatility_5m': latest.get('volatility_5m', 0),
                'volatility_10m': latest.get('volatility_10m', 0),
                'volume_ratio': latest.get('volume_ratio', 1),
                'high_low_range': latest.get('high_low_range', 0),
                'close_open_ratio': latest.get('close_open_ratio', 1),
                'spread_estimate': latest.get('spread_estimate', 0),
                'price_acceleration': latest.get('price_acceleration', 0),
                'current_price': latest['close'],
                'volume': latest['volume']
            }

            return features

        except Exception as e:
            print(f"Scalping features calculation error: {e}")
            return {}

    def _should_trade(self, symbol: str) -> bool:
        """Determine if conditions are right for a scalp trade"""
        try:
            if symbol not in self.ml_features:
                return False

            features = self.ml_features[symbol]

            # Risk checks
            if not self.risk_manager.can_trade(symbol):
                return False

            # Market condition checks
            if features.get('spread_estimate', 1) > self.scalping_config['max_spread']:
                return False

            if features.get('volume_ratio', 0) < 0.5:  # Low volume
                return False

            # ML prediction (placeholder - would use trained model)
            # For now, simple momentum-based signal
            momentum = features.get('returns_3m', 0)
            volatility = features.get('volatility_5m', 1)

            # Scalping logic: Enter on momentum with controlled volatility
            if abs(momentum) > 0.001 and volatility < 0.005:  # 0.1% momentum, low volatility
                return True

            return False

        except Exception as e:
            print(f"Trade decision error for {symbol}: {e}")
            return False

    def _execute_scalp_trade(self, symbol: str):
        """Execute a scalping trade"""
        try:
            if symbol in self.positions:
                # Check for exit conditions
                self._check_exit_conditions(symbol)
                return

            # Entry conditions met
            features = self.ml_features.get(symbol, {})
            momentum = features.get('returns_3m', 0)

            # Determine direction
            if momentum > 0:
                side = 'buy'
                stop_loss = features['current_price'] * (1 - self.scalping_config['stop_loss'])
                take_profit = features['current_price'] * (1 + self.scalping_config['profit_target'])
            else:
                side = 'sell'
                stop_loss = features['current_price'] * (1 + self.scalping_config['stop_loss'])
                take_profit = features['current_price'] * (1 - self.scalping_config['profit_target'])

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, features['current_price']
            )

            if position_size > 0:
                # Execute trade
                order = self._place_order(symbol, side, position_size)

                if order:
                    # Record position
                    self.positions[symbol] = {
                        'side': side,
                        'entry_price': features['current_price'],
                        'quantity': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'entry_time': datetime.now(),
                        'order_id': order.get('orderId')
                    }

                    print(f"🎯 Scalp {side.upper()} {position_size} {symbol} @ {features['current_price']}")

        except Exception as e:
            print(f"Trade execution error for {symbol}: {e}")

    def _check_exit_conditions(self, symbol: str):
        """Check if position should be exited"""
        try:
            position = self.positions[symbol]
            current_price = self.ml_features[symbol]['current_price']

            exit_reason = None

            # Check stop loss
            if position['side'] == 'buy':
                if current_price <= position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price >= position['take_profit']:
                    exit_reason = 'take_profit'
            else:  # sell
                if current_price >= position['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price <= position['take_profit']:
                    exit_reason = 'take_profit'

            # Check time-based exit (scalping timeout)
            if (datetime.now() - position['entry_time']).seconds > 300:  # 5 minutes
                exit_reason = 'timeout'

            if exit_reason:
                # Exit position
                self._close_position(symbol, exit_reason)

        except Exception as e:
            print(f"Exit condition check error for {symbol}: {e}")

    def _close_position(self, symbol: str, reason: str):
        """Close a scalping position"""
        try:
            position = self.positions[symbol]
            current_price = self.ml_features[symbol]['current_price']

            # Calculate P&L
            if position['side'] == 'buy':
                pnl = (current_price - position['entry_price']) * position['quantity']
            else:
                pnl = (position['entry_price'] - current_price) * position['quantity']

            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['total_pnl'] += pnl

            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1

            # Calculate win rate
            if self.performance_metrics['total_trades'] > 0:
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['winning_trades'] /
                    self.performance_metrics['total_trades']
                )

            # Place closing order
            if position['side'] == 'buy':
                self._place_order(symbol, 'sell', position['quantity'])
            else:
                self._place_order(symbol, 'buy', position['quantity'])

            print(f"💰 Closed {symbol} position - P&L: ${pnl:.2f} ({reason})")

            # Remove position
            del self.positions[symbol]

            # Update risk manager
            self.risk_manager.update_after_trade(symbol, pnl)

        except Exception as e:
            print(f"Position close error for {symbol}: {e}")

    def _place_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """Place an order on the exchange"""
        try:
            if self.exchange == "binance":
                client = self.exchange_clients.get('binance')
                if client:
                    binance_symbol = symbol.replace('/', '')

                    order = client.create_order(
                        symbol=binance_symbol,
                        side=side.upper(),
                        type='MARKET',
                        quantity=quantity
                    )

                    return order

            elif self.exchange == "alpaca":
                # Use Alpaca crypto trading
                client = self.exchange_clients.get('alpaca')
                if client:
                    # Alpaca crypto order placement
                    pass

            return None

        except Exception as e:
            print(f"Order placement error: {e}")
            return None

    def _update_price_data(self, symbol: str, price: float, volume: float):
        """Update real-time price data"""
        if symbol not in self.ml_features:
            self.ml_features[symbol] = {}

        self.ml_features[symbol]['current_price'] = price
        self.ml_features[symbol]['volume'] = volume
        self.last_update = datetime.now()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        report = self.performance_metrics.copy()

        # Calculate additional metrics
        if report['total_trades'] > 0:
            report['avg_trade_pnl'] = report['total_pnl'] / report['total_trades']

            # Calculate Sharpe ratio (simplified)
            if len(self.daily_pnl_history) > 1:
                returns = np.array(self.daily_pnl_history)
                if returns.std() > 0:
                    report['sharpe_ratio'] = np.sqrt(365) * returns.mean() / returns.std()

        report['active_positions'] = len(self.positions)
        report['total_symbols'] = len(self.symbols)
        report['uptime'] = str(datetime.now() - self.start_time) if hasattr(self, 'start_time') else "N/A"

        return report

    def save_performance_data(self, filename: str = "crypto_scalping_performance.json"):
        """Save performance data to file"""
        try:
            report = self.get_performance_report()
            report['timestamp'] = datetime.now().isoformat()

            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            print(f"Performance data saved to {filename}")

        except Exception as e:
            print(f"Performance save error: {e}")


class CryptoRiskManager:
    """Risk management system optimized for crypto scalping"""

    def __init__(self, config: Dict):
        self.config = config
        self.daily_trade_count = {}
        self.symbol_cooldowns = {}
        self.portfolio_value = 10000  # Starting capital
        self.daily_loss_limit = 500   # $500 daily loss limit
        self.daily_pnl = 0

    def can_trade(self, symbol: str) -> bool:
        """Check if trading is allowed for a symbol"""
        now = datetime.now()

        # Check trade frequency limit
        if symbol not in self.daily_trade_count:
            self.daily_trade_count[symbol] = 0

        if self.daily_trade_count[symbol] >= self.config['max_trades_per_hour']:
            return False

        # Check cooldown period
        if symbol in self.symbol_cooldowns:
            if (now - self.symbol_cooldowns[symbol]).seconds < self.config['cooldown_period']:
                return False

        # Check daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            return False

        return True

    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Risk 0.5% of portfolio per trade
            risk_amount = self.portfolio_value * 0.005

            # Stop loss distance
            stop_distance = price * self.config['stop_loss']

            # Position size = risk amount / stop distance
            position_size = risk_amount / stop_distance

            # Cap at maximum position size
            max_size = self.portfolio_value * self.config['max_position_size'] / price
            position_size = min(position_size, max_size)

            return position_size

        except Exception as e:
            print(f"Position size calculation error: {e}")
            return 0

    def monitor_positions(self, positions: Dict):
        """Monitor open positions for risk"""
        for symbol, position in positions.items():
            # Check for excessive drawdown
            current_price = position.get('current_price', position['entry_price'])
            entry_price = position['entry_price']

            if position['side'] == 'buy':
                drawdown = (entry_price - current_price) / entry_price
            else:
                drawdown = (current_price - entry_price) / entry_price

            # Emergency exit if drawdown > 2%
            if drawdown > 0.02:
                print(f"🚨 Emergency exit for {symbol} - drawdown: {drawdown:.1%}")
                # Would trigger position close here

    def check_daily_limits(self):
        """Check daily risk limits"""
        # Reset daily counters if new day
        today = datetime.now().date()
        if not hasattr(self, '_last_reset') or self._last_reset != today:
            self.daily_trade_count = {}
            self.daily_pnl = 0
            self._last_reset = today

    def update_after_trade(self, symbol: str, pnl: float):
        """Update risk metrics after trade"""
        self.daily_trade_count[symbol] = self.daily_trade_count.get(symbol, 0) + 1
        self.symbol_cooldowns[symbol] = datetime.now()
        self.daily_pnl += pnl
        self.portfolio_value += pnl


def run_crypto_scalping_demo():
    """Demo of crypto scalping system"""
    print("₿ Crypto Scalping Demo")
    print("=" * 40)

    # Initialize scalping trader
    trader = CryptoScalpingTrader(exchange="binance", symbols=['BTC/USDT', 'ETH/USDT'])

    print("\n🔧 Scalping Configuration:")
    for key, value in trader.scalping_config.items():
        print(f"   {key}: {value}")

    print("\n📊 Risk Management:")
    print(f"   Daily Loss Limit: ${trader.risk_manager.daily_loss_limit}")
    print(f"   Max Position Size: {trader.scalping_config['max_position_size']*100}%")
    print(f"   Profit Target: {trader.scalping_config['profit_target']*100}%")
    print(f"   Stop Loss: {trader.scalping_config['stop_loss']*100}%")

    print("\n⚠️  Note: This is a demo. Real trading requires:")
    print("   - Valid API keys for chosen exchange")
    print("   - Sufficient account balance")
    print("   - Understanding of crypto market risks")
    print("   - Paper trading testing first")

    print("\n✅ Crypto scalping system ready for deployment!")

if __name__ == "__main__":
    run_crypto_scalping_demo()