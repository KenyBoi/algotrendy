"""
Real Alpaca + ML Demo using REST API
Combines working simple_trader.py with direct Alpaca REST calls
"""

import os
import sys
import json
import requests
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
from simple_trader import SimpleMLTrader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RealAlpacaMLDemo:
    """ML Demo using direct Alpaca REST API calls"""
    
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML trader
        self.ml_trader = SimpleMLTrader()
        
        # Alpaca configuration
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        
        # Headers for API calls
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        }
        
        # Demo symbols
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
    def test_connection(self) -> bool:
        """Test Alpaca API connection"""
        try:
            if not self.api_key or not self.secret_key:
                self.logger.error("API credentials not found")
                return False
            
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
            
            if response.status_code == 200:
                account_data = response.json()
                self.logger.info(f"✅ Connected to Alpaca - Portfolio: ${float(account_data['portfolio_value']):,.2f}")
                return True
            else:
                self.logger.error(f"Connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def fetch_stock_data(self, symbol: str, days: int = 252) -> Optional[Dict]:
        """Fetch stock data using Alpaca REST API"""
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days + 50)).strftime('%Y-%m-%d')
            
            # Get historical bars
            response = requests.get(
                f"{self.data_url}/v2/stocks/{symbol}/bars",
                headers=self.headers,
                params={
                    'timeframe': '1Day',
                    'start': start_date,
                    'end': end_date,
                    'limit': days + 50
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                bars = data.get('bars', [])
                
                if len(bars) >= 50:
                    # Convert to our format
                    stock_data = {
                        'open': [float(bar['o']) for bar in bars],
                        'high': [float(bar['h']) for bar in bars],
                        'low': [float(bar['l']) for bar in bars],
                        'close': [float(bar['c']) for bar in bars],
                        'volume': [int(bar['v']) for bar in bars]
                    }
                    
                    self.logger.info(f"📈 Fetched {len(bars)} days of data for {symbol}")
                    return stock_data
                else:
                    self.logger.warning(f"Insufficient data for {symbol}: {len(bars)} bars")
                    return None
            else:
                self.logger.error(f"Failed to fetch data for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def train_ml_model(self, symbol: str) -> Optional[Dict]:
        """Train ML model on real stock data"""
        try:
            self.logger.info(f"🤖 Training ML model for {symbol}")
            
            # Fetch real data
            data = self.fetch_stock_data(symbol)
            if not data:
                return None
            
            # Prepare features
            X, y = self.ml_trader.prepare_features(data)
            
            # Train model
            metrics = self.ml_trader.train(X, y)
            
            self.logger.info(f"✅ Model trained for {symbol} - Accuracy: {metrics['accuracy']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}")
            return None
    
    def generate_signals(self, symbols: List[str]) -> Dict:
        """Generate ML-based trading signals"""
        signals = {}
        
        for symbol in symbols:
            try:
                # Fetch recent data
                data = self.fetch_stock_data(symbol, days=60)
                if not data:
                    continue
                
                # Prepare features
                X, _ = self.ml_trader.prepare_features(data)
                
                if len(X) > 0:
                    # Get prediction
                    prediction = self.ml_trader.predict(X[-1:])
                    probabilities = self.ml_trader.predict_proba(X[-1:])
                    
                    confidence = float(max(probabilities[0])) if len(probabilities) > 0 else 0.0
                    current_price = data['close'][-1]
                    
                    signals[symbol] = {
                        'signal': int(prediction[0]),
                        'confidence': confidence,
                        'current_price': current_price,
                        'timestamp': datetime.now()
                    }
                    
                    self.logger.info(f"📊 {symbol}: Signal={prediction[0]}, Confidence={confidence:.3f}, Price=${current_price:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get account info: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"Failed to get positions: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def execute_paper_trade(self, symbol: str, side: str, qty: int) -> Optional[str]:
        """Execute a paper trade"""
        try:
            order_data = {
                'symbol': symbol,
                'qty': qty,
                'side': side.lower(),
                'type': 'market',
                'time_in_force': 'day'
            }
            
            response = requests.post(
                f"{self.base_url}/v2/orders",
                headers=self.headers,
                json=order_data
            )
            
            if response.status_code == 201:
                order = response.json()
                order_id = order.get('id')
                self.logger.info(f"✅ {side} order submitted: {qty} {symbol} (Order: {order_id[:8]}...)")
                return order_id
            else:
                self.logger.error(f"Order failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def run_full_demo(self):
        """Run complete real data demo"""
        print("🚀 REAL ALPACA + MACHINE LEARNING DEMO")
        print("=" * 45)
        
        # Step 1: Test connection
        print("\n1️⃣  Testing Alpaca API connection...")
        if not self.test_connection():
            print("❌ API connection failed - check your credentials")
            return
        
        # Step 2: Get account info
        print("\n2️⃣  Getting account information...")
        account = self.get_account_info()
        positions = self.get_positions()
        
        if account:
            print(f"💰 Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
            print(f"💵 Cash Available: ${float(account.get('cash', 0)):,.2f}")
            print(f"📊 Current Positions: {len(positions)}")
        
        # Step 3: Train ML model
        print(f"\n3️⃣  Training ML model on real {self.symbols[0]} data...")
        training_symbol = self.symbols[0]  # Use AAPL for training
        
        metrics = self.train_ml_model(training_symbol)
        if metrics:
            print(f"✅ Model trained successfully!")
            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   Training samples: {metrics['n_samples']}")
            
            # Feature importance
            importance = self.ml_trader.get_feature_importance()
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"   Top features: {', '.join([f[0] for f in top_features])}")
        else:
            print("❌ Model training failed")
            return
        
        # Step 4: Generate signals for all symbols
        print(f"\n4️⃣  Generating signals for {len(self.symbols)} symbols...")
        signals = self.generate_signals(self.symbols)
        
        if signals:
            print(f"📊 Generated {len(signals)} trading signals:")
            for symbol, signal_data in signals.items():
                signal_val = signal_data['signal']
                confidence = signal_data['confidence']
                price = signal_data['current_price']
                
                # Signal interpretation
                if signal_val == 1:
                    signal_text = "🟢 BUY"
                elif signal_val == -1:
                    signal_text = "🔴 SELL"
                else:
                    signal_text = "⚪ HOLD"
                
                print(f"   {symbol}: {signal_text} | ${price:.2f} | Confidence: {confidence:.3f}")
        
        # Step 5: Simulate trades (only if high confidence)
        print("\n5️⃣  Evaluating trade opportunities...")
        executed_trades = []
        
        for symbol, signal_data in signals.items():
            signal_val = signal_data['signal']
            confidence = signal_data['confidence']
            price = signal_data['current_price']
            
            # Only trade on high confidence signals
            if signal_val == 1 and confidence > 0.6:  # Strong BUY
                # Calculate position size (5% of portfolio)
                portfolio_value = float(account.get('portfolio_value', 100000))
                position_value = portfolio_value * 0.05
                qty = int(position_value / price)
                
                if qty > 0:
                    print(f"   📝 Would BUY {qty} shares of {symbol} @ ${price:.2f}")
                    # For demo, we'll just log the trade rather than execute
                    executed_trades.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'qty': qty,
                        'price': price,
                        'confidence': confidence
                    })
        
        if executed_trades:
            print(f"\n⚡ Trade recommendations ({len(executed_trades)}):")
            total_investment = 0
            for trade in executed_trades:
                investment = trade['qty'] * trade['price']
                total_investment += investment
                print(f"   🟢 {trade['action']} {trade['qty']} {trade['symbol']} "
                      f"@ ${trade['price']:.2f} = ${investment:,.2f}")
            
            print(f"\n💰 Total recommended investment: ${total_investment:,.2f}")
        else:
            print("   📝 No high-confidence trading opportunities found")
        
        # Step 6: Summary
        print(f"\n6️⃣  Demo Summary:")
        print(f"✅ Real market data: {len(signals)} symbols analyzed")
        print(f"✅ ML model accuracy: {metrics['accuracy']:.3f}")
        print(f"✅ Trading signals generated using live data")
        print(f"✅ Paper trading environment (no real money)")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"💡 Your system is ready for automated trading!")
        print(f"⚠️  Always test thoroughly before using real money!")

def main():
    """Main function"""
    demo = RealAlpacaMLDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main()
