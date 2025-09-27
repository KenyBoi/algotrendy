"""
Alpaca API Integration for AlgoTrendy XGBoost Trading System
Handles real market data fetching and trading execution via Alpaca API.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, AssetClass
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True

    # Try to import futures support (may not be available in all versions)
    try:
        from alpaca.data.historical import FuturesHistoricalDataClient
        from alpaca.data.requests import FuturesBarsRequest
        FUTURES_AVAILABLE = True
    except ImportError:
        FUTURES_AVAILABLE = False
        print("Warning: Futures support not available in this Alpaca version")

except ImportError:
    ALPACA_AVAILABLE = False
    FUTURES_AVAILABLE = False
    print("Warning: Alpaca packages not available. Install with: pip install alpaca-py")

from config import CONFIG

logger = logging.getLogger(__name__)

class AlpacaDataManager:
    """Manages market data from Alpaca API for stocks and futures"""

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca data client

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (True) or live trading (False)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca packages not installed")

        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper

        # Initialize data clients
        self.stock_data_client = StockHistoricalDataClient(api_key, secret_key)

        # Initialize futures data client if available
        if FUTURES_AVAILABLE:
            self.futures_data_client = FuturesHistoricalDataClient(api_key, secret_key)
        else:
            self.futures_data_client = None
            logger.warning("Futures data client not available")

        logger.info(f"Alpaca Data Manager initialized ({'Paper' if paper else 'Live'} mode)")
    
    def fetch_bars(self, symbol: str, timeframe: str = "1Day",
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    limit: int = 1000, asset_type: str = "stock") -> Dict:
        """
        Fetch OHLCV bars from Alpaca for stocks or futures

        Args:
            symbol: Stock symbol (e.g., 'AAPL') or futures symbol (e.g., 'ESU5')
            timeframe: Bar timeframe ('1Min', '5Min', '1Hour', '1Day')
            start_date: Start date for data
            end_date: End date for data
            limit: Maximum number of bars
            asset_type: "stock" or "futures"

        Returns:
            Dictionary with OHLCV data
        """
        try:
            # Map timeframe strings to Alpaca TimeFrame objects
            timeframe_map = {
                '1Min': TimeFrame.Minute,
                '5Min': TimeFrame(5, TimeFrame.Unit.Minute),
                '15Min': TimeFrame(15, TimeFrame.Unit.Minute),
                '1Hour': TimeFrame.Hour,
                '1Day': TimeFrame.Day,
                '1Tick': TimeFrame.Tick
            }

            if timeframe not in timeframe_map:
                raise ValueError(f"Unsupported timeframe: {timeframe}")

            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                if asset_type == "futures":
                    start_date = end_date - timedelta(days=60)  # Shorter for futures day trading
                else:
                    start_date = end_date - timedelta(days=365)  # 1 year of data

            logger.info(f"Fetching {asset_type} {symbol} bars ({timeframe}) from {start_date.date()} to {end_date.date()}")

            if asset_type == "futures":
                if not FUTURES_AVAILABLE or self.futures_data_client is None:
                    raise ValueError("Futures data client not available")

                # Create futures request
                request_params = FuturesBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=timeframe_map[timeframe],
                    start=start_date,
                    end=end_date,
                    limit=limit
                )

                bars = self.futures_data_client.get_futures_bars(request_params)
            else:
                # Create stock request
                request_params = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=timeframe_map[timeframe],
                    start=start_date,
                    end=end_date,
                    limit=limit
                )

                bars = self.stock_data_client.get_stock_bars(request_params)

            # Convert to our format
            if symbol in bars.data:
                bar_data = bars.data[symbol]

                data = {
                    'open': np.array([bar.open for bar in bar_data]),
                    'high': np.array([bar.high for bar in bar_data]),
                    'low': np.array([bar.low for bar in bar_data]),
                    'close': np.array([bar.close for bar in bar_data]),
                    'volume': np.array([bar.volume for bar in bar_data]),
                    'timestamps': [bar.timestamp for bar in bar_data]
                }

                logger.info(f"Retrieved {len(bar_data)} {asset_type} bars for {symbol}")
                return data
            else:
                logger.warning(f"No {asset_type} data found for {symbol}")
                return {}

        except Exception as e:
            logger.error(f"Error fetching Alpaca {asset_type} data for {symbol}: {e}")
            raise
    
    def get_latest_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get latest quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with bid/ask data
        """
        try:
            request_params = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_stock_latest_quote(request_params)
            
            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    'bid': quote.bid_price,
                    'ask': quote.ask_price,
                    'bid_size': quote.bid_size,
                    'ask_size': quote.ask_size,
                    'timestamp': quote.timestamp
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None
    
    def get_market_hours(self) -> Dict:
        """Get market hours information"""
        # This is a simplified version - Alpaca has a calendar API for more details
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_market_open = market_open <= now <= market_close and now.weekday() < 5
        
        return {
            'is_open': is_market_open,
            'next_open': market_open if now < market_open else market_open + timedelta(days=1),
            'next_close': market_close if now < market_close else market_close + timedelta(days=1)
        }

class AlpacaTrader:
    """Handles trading operations via Alpaca API"""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca trading client
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (True) or live trading (False)
        """
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca packages not installed")
            
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        
        # Initialize trading client
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        
        # Initialize data manager
        self.data_manager = AlpacaDataManager(api_key, secret_key, paper)
        
        logger.info(f"Alpaca Trader initialized ({'Paper' if paper else 'Live'} mode)")
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            
            return {
                'account_id': account.id,
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'day_trade_count': account.day_trade_count,
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            
            position_data = []
            for pos in positions:
                position_data.append({
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'market_value': float(pos.market_value),
                    'cost_basis': float(pos.cost_basis),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'current_price': float(pos.current_price),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'side': pos.side.value
                })
            
            return position_data
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def place_market_order(self, symbol: str, qty: int, side: str, asset_class: str = "stock") -> Optional[str]:
        """
        Place a market order for stocks or futures

        Args:
            symbol: Stock symbol (e.g., 'AAPL') or futures symbol (e.g., 'ESU5')
            qty: Quantity to trade (for futures, this is number of contracts)
            side: 'buy' or 'sell'
            asset_class: "stock" or "futures"

        Returns:
            Order ID if successful, None if failed
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            # Set asset class for futures
            asset_class_enum = AssetClass.FUTURES if asset_class == "futures" else AssetClass.STOCK

            market_order_data = MarketOrderRequest(
                symbol=symbol,
                qty=abs(qty),
                side=order_side,
                time_in_force=TimeInForce.DAY,
                asset_class=asset_class_enum
            )

            market_order = self.trading_client.submit_order(order_data=market_order_data)

            logger.info(f"{asset_class.title()} market order placed: {side.upper()} {qty} {symbol} (Order ID: {market_order.id})")
            return market_order.id

        except Exception as e:
            logger.error(f"Error placing {asset_class} market order: {e}")
            return None
    
    def place_limit_order(self, symbol: str, qty: int, side: str, limit_price: float, asset_class: str = "stock") -> Optional[str]:
        """
        Place a limit order for stocks or futures

        Args:
            symbol: Stock symbol (e.g., 'AAPL') or futures symbol (e.g., 'ESU5')
            qty: Quantity to trade (for futures, this is number of contracts)
            side: 'buy' or 'sell'
            limit_price: Limit price
            asset_class: "stock" or "futures"

        Returns:
            Order ID if successful, None if failed
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            # Set asset class for futures
            asset_class_enum = AssetClass.FUTURES if asset_class == "futures" else AssetClass.STOCK

            limit_order_data = LimitOrderRequest(
                symbol=symbol,
                qty=abs(qty),
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                asset_class=asset_class_enum
            )

            limit_order = self.trading_client.submit_order(order_data=limit_order_data)

            logger.info(f"{asset_class.title()} limit order placed: {side.upper()} {qty} {symbol} @ ${limit_price} (Order ID: {limit_order.id})")
            return limit_order.id

        except Exception as e:
            logger.error(f"Error placing {asset_class} limit order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_orders(self, status: str = None, limit: int = 50) -> List[Dict]:
        """
        Get orders
        
        Args:
            status: Order status filter ('open', 'closed', etc.)
            limit: Maximum number of orders to return
            
        Returns:
            List of order dictionaries
        """
        try:
            # Map status string to enum if provided
            status_filter = None
            if status:
                status_map = {
                    'open': [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.PENDING_NEW],
                    'closed': [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]
                }
                status_filter = status_map.get(status.lower())
            
            request_params = GetOrdersRequest(
                status=status_filter,
                limit=limit
            )
            
            orders = self.trading_client.get_orders(filter=request_params)
            
            order_data = []
            for order in orders:
                order_data.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'qty': float(order.qty),
                    'filled_qty': float(order.filled_qty),
                    'side': order.side.value,
                    'order_type': order.order_type.value,
                    'status': order.status.value,
                    'created_at': order.created_at,
                    'filled_at': order.filled_at,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
                })
            
            return order_data
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

class AlpacaIntegratedTrader:
    """Integrated trading system using Alpaca with XGBoost predictions"""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """Initialize integrated trader"""
        self.alpaca_trader = AlpacaTrader(api_key, secret_key, paper)
        self.data_manager = AlpacaDataManager(api_key, secret_key, paper)
        
        # Store for trained models
        self.models = {}
        
    def prepare_alpaca_data_for_ml(self, symbol: str, days: int = 252, asset_type: str = "stock") -> Optional[Dict]:
        """
        Fetch and prepare Alpaca data for ML model

        Args:
            symbol: Stock symbol (e.g., 'AAPL') or futures symbol (e.g., 'ESU5')
            days: Number of days of data to fetch
            asset_type: "stock" or "futures"

        Returns:
            Dictionary with prepared data
        """
        try:
            # Fetch data from Alpaca
            end_date = datetime.now()
            if asset_type == "futures":
                start_date = end_date - timedelta(days=min(days, 60))  # Shorter for futures
                timeframe = "1Hour"  # Intraday for futures
            else:
                start_date = end_date - timedelta(days=days + 50)  # Extra buffer for indicators
                timeframe = "1Day"

            data = self.data_manager.fetch_bars(symbol, timeframe, start_date, end_date, asset_type=asset_type)

            if not data or len(data['close']) < 30:
                logger.warning(f"Insufficient {asset_type} data for {symbol}")
                return None

            # Add basic technical indicators (simplified version)
            prices = data['close']

            # Simple moving averages
            sma_10 = np.convolve(prices, np.ones(10)/10, mode='valid')
            sma_20 = np.convolve(prices, np.ones(20)/20, mode='valid')

            # RSI calculation (simplified)
            returns = np.diff(prices)
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)

            avg_gains = np.convolve(gains, np.ones(14)/14, mode='valid')
            avg_losses = np.convolve(losses, np.ones(14)/14, mode='valid')

            rs = avg_gains / (avg_losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # Align all arrays to same length
            min_len = min(len(sma_20), len(rsi))

            prepared_data = {
                'close': prices[-min_len:],
                'open': data['open'][-min_len:],
                'high': data['high'][-min_len:],
                'low': data['low'][-min_len:],
                'volume': data['volume'][-min_len:],
                'sma_10': sma_10[-min_len:],
                'sma_20': sma_20[-min_len:],
                'rsi': rsi[-min_len:],
                'timestamps': data['timestamps'][-min_len:],
                'asset_type': asset_type
            }

            logger.info(f"Prepared {len(prepared_data['close'])} {asset_type} data points for {symbol}")
            return prepared_data

        except Exception as e:
            logger.error(f"Error preparing {asset_type} data for {symbol}: {e}")
            return None
    
    def generate_trading_signal(self, symbol: str, model=None) -> Dict:
        """
        Generate trading signal for a symbol
        
        Args:
            symbol: Stock symbol
            model: Trained model (if None, will use a simple strategy)
            
        Returns:
            Dictionary with signal information
        """
        try:
            # Get latest data
            data = self.prepare_alpaca_data_for_ml(symbol, days=30)
            
            if not data:
                return {'signal': 0, 'confidence': 0, 'reason': 'No data available'}
            
            # Simple strategy if no model provided
            if model is None:
                current_price = data['close'][-1]
                sma_20 = data['sma_20'][-1]
                rsi = data['rsi'][-1]
                
                # Simple rules-based signal
                if current_price > sma_20 and rsi < 70:
                    signal = 1  # Buy
                    confidence = 0.7
                    reason = f"Price above SMA20 (${current_price:.2f} > ${sma_20:.2f}), RSI not overbought ({rsi:.1f})"
                elif current_price < sma_20 and rsi > 30:
                    signal = -1  # Sell
                    confidence = 0.7
                    reason = f"Price below SMA20 (${current_price:.2f} < ${sma_20:.2f}), RSI not oversold ({rsi:.1f})"
                else:
                    signal = 0  # Hold
                    confidence = 0.5
                    reason = "No clear signal"
            else:
                # Use ML model (placeholder for when XGBoost model is integrated)
                signal = 0
                confidence = 0.5
                reason = "ML model prediction (not implemented yet)"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': reason,
                'current_price': data['close'][-1],
                'rsi': data['rsi'][-1],
                'sma_20': data['sma_20'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {'signal': 0, 'confidence': 0, 'reason': f'Error: {e}'}
    
    def execute_strategy(self, symbols: List[str], max_positions: int = 5, asset_type: str = "stock") -> Dict:
        """
        Execute trading strategy across multiple symbols for stocks or futures

        Args:
            symbols: List of symbols to trade
            max_positions: Maximum number of positions to hold
            asset_type: "stock" or "futures"

        Returns:
            Dictionary with execution results
        """
        try:
            # Check account info
            account = self.alpaca_trader.get_account_info()
            current_positions = self.alpaca_trader.get_positions()

            logger.info(f"Account equity: ${account['equity']:,.2f}")
            logger.info(f"Current {asset_type} positions: {len(current_positions)}")

            # Generate signals for all symbols
            signals = {}
            for symbol in symbols:
                signals[symbol] = self.generate_trading_signal(symbol)

            # Execute trades based on signals
            executed_trades = []

            for symbol, signal_data in signals.items():
                signal = signal_data['signal']
                confidence = signal_data['confidence']

                # Find existing position
                existing_position = None
                for pos in current_positions:
                    if pos['symbol'] == symbol:
                        existing_position = pos
                        break

                # Trading logic with asset-specific position sizing
                if signal == 1 and confidence > (0.7 if asset_type == "futures" else 0.6):  # Buy signal (higher threshold for futures)
                    if not existing_position and len(current_positions) < max_positions:
                        if asset_type == "futures":
                            # Futures position sizing based on margin requirements
                            contract_symbol = symbol.replace('=F', '')[:2]  # Extract base symbol (ES, NQ, etc.)
                            if contract_symbol in FUTURES_CONTRACTS:
                                contract_info = FUTURES_CONTRACTS[contract_symbol]
                                # Calculate contracts based on margin (use 5% of equity per contract)
                                max_contracts_per_position = int((account['equity'] * 0.05) / contract_info['margin_initial'])
                                qty = min(max_contracts_per_position, 5)  # Max 5 contracts per position
                            else:
                                qty = 1  # Default to 1 contract
                        else:
                            # Stock position sizing (use 10% of equity per position)
                            position_value = account['equity'] * 0.1
                            qty = int(position_value / signal_data['current_price'])

                        if qty > 0:
                            order_id = self.alpaca_trader.place_market_order(symbol, qty, 'buy', asset_type)
                            if order_id:
                                executed_trades.append({
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'qty': qty,
                                    'asset_type': asset_type,
                                    'order_id': order_id,
                                    'reason': signal_data['reason']
                                })

                elif signal == -1 and existing_position:  # Sell signal
                    qty = int(abs(existing_position['qty']))
                    order_id = self.alpaca_trader.place_market_order(symbol, qty, 'sell', asset_type)
                    if order_id:
                        executed_trades.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'qty': qty,
                            'asset_type': asset_type,
                            'order_id': order_id,
                            'reason': signal_data['reason']
                        })

            return {
                'signals': signals,
                'executed_trades': executed_trades,
                'account_info': account,
                'asset_type': asset_type
            }

        except Exception as e:
            logger.error(f"Error executing {asset_type} strategy: {e}")
            return {'error': str(e)}

if __name__ == "__main__":
    # Example usage - you'll need to set your Alpaca API credentials
    print("🔑 Alpaca API Integration Example")
    print("=" * 40)
    
    # You need to set these environment variables or replace with your actual keys
    api_key = os.getenv('ALPACA_API_KEY', 'your_api_key_here')
    secret_key = os.getenv('ALPACA_SECRET_KEY', 'your_secret_key_here')
    
    if api_key == 'your_api_key_here':
        print("❌ Please set your Alpaca API credentials in environment variables:")
        print("   ALPACA_API_KEY=your_actual_key")
        print("   ALPACA_SECRET_KEY=your_actual_secret")
        print("\nOr create a .env file with these values.")
    else:
        try:
            # Initialize integrated trader (paper trading mode)
            trader = AlpacaIntegratedTrader(api_key, secret_key, paper=True)
            
            # Test account info
            account = trader.alpaca_trader.get_account_info()
            print(f"✅ Connected to Alpaca (Paper Trading)")
            print(f"   Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"   Buying Power: ${account['buying_power']:,.2f}")
            
            # Test signal generation
            test_symbol = "AAPL"
            signal = trader.generate_trading_signal(test_symbol)
            print(f"\n📊 Signal for {test_symbol}:")
            print(f"   Signal: {signal['signal']} (confidence: {signal['confidence']:.2f})")
            print(f"   Reason: {signal['reason']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")