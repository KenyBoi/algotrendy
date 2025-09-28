"""
Complete Alpaca + ML Trading Demo
Uses the working simple_trader.py with real Alpaca market data
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Ensure project and examples directories are on sys.path
EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parent
SRC_DIR = PROJECT_ROOT / 'src'

for extra_path in (EXAMPLES_DIR, SRC_DIR):
    path_str = str(extra_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from algotrendy.config import CONFIG, setup_logging
from simple_trader import SimpleMLTrader, SyntheticMarketData

# For environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("💡 python-dotenv not available - using environment variables")

# Try to import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  Alpaca packages not available")

class AlpacaMLDemo:
    """Complete Alpaca + Machine Learning Demo"""
    
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML trader
        self.ml_trader = SimpleMLTrader()
        
        # Alpaca components
        self.trading_client = None
        self.data_client = None
        
        # Demo settings
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        self.paper_trading = True
        
    def setup_alpaca(self) -> bool:
        """Setup Alpaca API connection"""
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                self.logger.warning("Alpaca API credentials not found")
                return False
            
            if not ALPACA_AVAILABLE:
                self.logger.error("Alpaca packages not installed")
                return False
            
            # Initialize clients
            self.trading_client = TradingClient(api_key, secret_key, paper=self.paper_trading)
            self.data_client = StockHistoricalDataClient(api_key, secret_key)
            
            # Test connection
            account = self.trading_client.get_account()
            self.logger.info(f"✅ Connected to Alpaca - Portfolio: ${float(account.portfolio_value):,.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    def fetch_real_data(self, symbol: str, days: int = 252) -> Optional[Dict]:
        """Fetch real market data from Alpaca"""
        try:
            if not self.data_client:
                return None
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 50)  # Extra buffer
            
            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                limit=days + 50
            )
            
            # Fetch data
            bars = self.data_client.get_stock_bars(request)
            
            if symbol not in bars.data:
                self.logger.warning(f"No data found for {symbol}")
                return None
            
            bar_list = bars.data[symbol]
            
            if len(bar_list) < 50:
                self.logger.warning(f"Insufficient data for {symbol}: {len(bar_list)} bars")
                return None
            
            # Convert to our format
            data = {
                'open': [float(bar.open) for bar in bar_list],
                'high': [float(bar.high) for bar in bar_list],
                'low': [float(bar.low) for bar in bar_list],
                'close': [float(bar.close) for bar in bar_list],
                'volume': [int(bar.volume) for bar in bar_list]
            }
            
            self.logger.info(f"📈 Fetched {len(bar_list)} days of data for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def train_on_real_data(self, symbol: str) -> Optional[Dict]:
        """Train ML model on real market data"""
        try:
            # Fetch data
            data = self.fetch_real_data(symbol)
            if not data:
                return None
            
            # Prepare features
            X, y = self.ml_trader.prepare_features(data)
            
            # Train model
            metrics = self.ml_trader.train(X, y)
            
            self.logger.info(f"🤖 Trained model for {symbol} - Accuracy: {metrics['accuracy']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}")
            return None
    
    def generate_trading_signals(self, symbols: List[str]) -> Dict:
        """Generate trading signals for multiple symbols"""
        signals = {}
        
        for symbol in symbols:
            try:
                # Fetch recent data
                recent_data = self.fetch_real_data(symbol, days=60)
                if not recent_data:
                    continue
                
                # Prepare features for last few days
                X, _ = self.ml_trader.prepare_features(recent_data)
                
                if len(X) > 0:
                    # Get latest signal
                    latest_signal = self.ml_trader.predict(X[-1:])
                    probabilities = self.ml_trader.predict_proba(X[-1:])
                    
                    confidence = float(max(probabilities[0])) if len(probabilities) > 0 else 0.0
                    
                    signals[symbol] = {
                        'signal': int(latest_signal[0]),
                        'confidence': confidence,
                        'current_price': recent_data['close'][-1],
                        'timestamp': datetime.now()
                    }
                    
                    self.logger.info(f"📊 {symbol}: Signal={latest_signal[0]}, Confidence={confidence:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals
    
    def execute_paper_trades(self, signals: Dict) -> List[Dict]:
        """Execute paper trades based on signals"""
        executed_trades = []
        
        if not self.trading_client:
            self.logger.warning("Trading client not available")
            return executed_trades
        
        try:
            # Get account info
            account = self.trading_client.get_account()
            cash = float(account.cash)
            
            # Get current positions
            positions = self.trading_client.get_all_positions()
            position_dict = {pos.symbol: float(pos.qty) for pos in positions}
            
            self.logger.info(f"💰 Available cash: ${cash:,.2f}")
            self.logger.info(f"📊 Current positions: {len(positions)}")
            
            for symbol, signal_data in signals.items():
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                price = signal_data['current_price']
                
                current_position = position_dict.get(symbol, 0)
                
                # Buy signal with high confidence
                if signal == 1 and confidence > 0.6 and current_position == 0 and cash > 1000:
                    # Calculate position size (5% of portfolio)
                    portfolio_value = float(account.portfolio_value)
                    position_value = portfolio_value * 0.05
                    qty = int(position_value / price)
                    
                    if qty > 0:
                        try:
                            order_request = MarketOrderRequest(
                                symbol=symbol,
                                qty=qty,
                                side=OrderSide.BUY,
                                time_in_force=TimeInForce.DAY
                            )
                            
                            order = self.trading_client.submit_order(order_request)
                            
                            executed_trades.append({
                                'symbol': symbol,
                                'action': 'BUY',
                                'qty': qty,
                                'price': price,
                                'order_id': order.id,
                                'confidence': confidence
                            })
                            
                            self.logger.info(f"🟢 BUY order submitted: {qty} {symbol} @ ${price:.2f}")
                            
                        except Exception as e:
                            self.logger.error(f"Error submitting buy order for {symbol}: {e}")
                
                # Sell signal
                elif signal == -1 and current_position > 0:
                    qty = int(abs(current_position))
                    
                    try:
                        order_request = MarketOrderRequest(
                            symbol=symbol,
                            qty=qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.DAY
                        )
                        
                        order = self.trading_client.submit_order(order_request)
                        
                        executed_trades.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'qty': qty,
                            'price': price,
                            'order_id': order.id,
                            'confidence': confidence
                        })
                        
                        self.logger.info(f"🔴 SELL order submitted: {qty} {symbol} @ ${price:.2f}")
                        
                    except Exception as e:
                        self.logger.error(f"Error submitting sell order for {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")
        
        return executed_trades
    
    def run_full_demo(self):
        """Run complete Alpaca + ML demo"""
        print("🚀 ALPACA + MACHINE LEARNING TRADING DEMO")
        print("=" * 50)
        
        # Step 1: Try Alpaca connection
        print("\n1️⃣  Connecting to Alpaca API...")
        alpaca_connected = self.setup_alpaca()
        
        if not alpaca_connected:
            print("❌ Alpaca connection failed")
            print("🔄 Running with synthetic data...")
            self.run_synthetic_demo()
            return
        
        # Step 2: Train model on real data
        print("\n2️⃣  Training ML model on real market data...")
        training_symbol = self.symbols[0]  # Start with AAPL
        
        metrics = self.train_on_real_data(training_symbol)
        if metrics:
            print(f"✅ Model trained on {training_symbol}")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   Samples: {metrics['n_samples']}")
        else:
            print("❌ Model training failed - using synthetic data")
            self.run_synthetic_demo()
            return
        
        # Step 3: Generate signals
        print("\n3️⃣  Generating trading signals...")
        signals = self.generate_trading_signals(self.symbols[:3])  # Test with 3 symbols
        
        if signals:
            print(f"📊 Generated signals for {len(signals)} symbols")
            for symbol, data in signals.items():
                emoji = "🟢" if data['signal'] > 0 else "🔴" if data['signal'] < 0 else "⚪"
                print(f"   {symbol}: {emoji} {data['signal']} (Confidence: {data['confidence']:.3f})")
        
        # Step 4: Execute trades (paper only)
        print("\n4️⃣  Executing paper trades...")
        trades = self.execute_paper_trades(signals)
        
        if trades:
            print(f"⚡ Executed {len(trades)} trades:")
            for trade in trades:
                emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                print(f"   {emoji} {trade['action']} {trade['qty']} {trade['symbol']} @ ${trade['price']:.2f}")
        else:
            print("📝 No trades executed (no strong signals or insufficient conditions)")
        
        # Step 5: Show portfolio status
        print("\n5️⃣  Portfolio Summary...")
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            print(f"💰 Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"💵 Cash Available: ${float(account.cash):,.2f}")
            print(f"📊 Active Positions: {len(positions)}")
            
            if positions:
                print("   Current Holdings:")
                for pos in positions[:5]:  # Show top 5
                    pnl_emoji = "🟢" if float(pos.unrealized_pl) >= 0 else "🔴"
                    print(f"   - {pos.symbol}: {pos.qty} shares {pnl_emoji} ${float(pos.unrealized_pl):.2f}")
        
        except Exception as e:
            self.logger.error(f"Error getting portfolio info: {e}")
        
        print("\n✅ Demo completed successfully!")
        print("📈 System ready for live trading (switch paper=False when ready)")
    
    def run_synthetic_demo(self):
        """Fallback to synthetic data demo"""
        print("\n🔄 Running synthetic data demonstration...")
        
        # Generate synthetic data
        data = SyntheticMarketData.generate_price_series(days=300)
        
        # Train model
        X, y = self.ml_trader.prepare_features(data)
        metrics = self.ml_trader.train(X, y)
        
        print(f"✅ Synthetic Demo Results:")
        print(f"   Data points: {len(data['close'])}")
        print(f"   Model accuracy: {metrics['accuracy']:.3f}")
        print(f"   Features: {metrics['n_features']}")
        
        # Simple backtest
        signals = self.ml_trader.predict(X[-30:])  # Last 30 predictions
        buy_signals = sum(1 for s in signals if s == 1)
        sell_signals = sum(1 for s in signals if s == -1)
        
        print(f"   Recent signals: {buy_signals} BUY, {sell_signals} SELL")
        print("\n💡 To use real data:")
        print("   1. Sign up at https://alpaca.markets")
        print("   2. Set environment variables: ALPACA_API_KEY, ALPACA_SECRET_KEY")
        print("   3. Run this demo again!")

def main():
    """Main function"""
    demo = AlpacaMLDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main()
