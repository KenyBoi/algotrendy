"""
Alpaca + XGBoost Trading Demo
Demonstrates integration of real market data with ML-based trading system.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG, setup_logging
from alpaca_integration import AlpacaIntegratedTrader, ALPACA_AVAILABLE
from xgboost_trader import XGBoostTrader

# For environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("💡 Install python-dotenv for .env file support: pip install python-dotenv")

class AlpacaXGBoostDemo:
    """
    Comprehensive demo showing XGBoost integration with Alpaca API
    """
    
    def __init__(self):
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.alpaca_trader = None
        self.xgb_trader = XGBoostTrader()
        
        # Demo symbols
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
    def setup_alpaca_connection(self):
        """Setup Alpaca API connection"""
        try:
            # Try to get credentials from environment
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key or api_key == 'your_api_key_here':
                self.logger.warning("Alpaca API credentials not found in environment")
                return False
            
            if not ALPACA_AVAILABLE:
                self.logger.error("Alpaca packages not installed")
                return False
            
            # Initialize Alpaca trader (paper mode)
            self.alpaca_trader = AlpacaIntegratedTrader(
                api_key=api_key,
                secret_key=secret_key,
                paper=True  # Always use paper trading for demo
            )
            
            # Test connection
            account = self.alpaca_trader.alpaca_trader.get_account_info()
            self.logger.info(f"✅ Connected to Alpaca - Portfolio: ${account['portfolio_value']:,.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    def fetch_real_market_data(self, symbol: str) -> Dict:
        """Fetch real market data from Alpaca"""
        try:
            if not self.alpaca_trader:
                raise ValueError("Alpaca not connected")
            
            # Fetch data for ML training
            data = self.alpaca_trader.prepare_alpaca_data_for_ml(symbol, days=252)
            
            if not data:
                self.logger.warning(f"No data available for {symbol}")
                return {}
            
            self.logger.info(f"📈 Fetched {len(data['close'])} days of data for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return {}
    
    def train_models_on_real_data(self) -> Dict:
        """Train XGBoost models on real market data"""
        trained_models = {}
        
        for symbol in self.symbols[:2]:  # Start with 2 symbols for demo
            try:
                self.logger.info(f"🤖 Training XGBoost model for {symbol}")
                
                # Fetch real data
                data = self.fetch_real_market_data(symbol)
                if not data:
                    continue
                
                # Prepare features for XGBoost
                features = self.prepare_features_for_xgboost(data)
                if not features:
                    continue
                
                # Train model
                X, y = features['X'], features['y']
                self.xgb_trader.train(X, y)
                
                # Save model
                model_path = CONFIG.MODEL_DIR / f"{symbol}_xgboost.pkl"
                self.xgb_trader.save_model(str(model_path))
                
                trained_models[symbol] = {
                    'model_path': model_path,
                    'training_samples': len(X),
                    'accuracy': self.evaluate_model_accuracy(X, y)
                }
                
                self.logger.info(f"✅ Model trained for {symbol}: {len(X)} samples")
                
            except Exception as e:
                self.logger.error(f"Error training model for {symbol}: {e}")
        
        return trained_models
    
    def prepare_features_for_xgboost(self, data: Dict) -> Dict:
        """Prepare features for XGBoost training"""
        try:
            import numpy as np
            
            # Basic price features
            closes = np.array(data['close'])
            highs = np.array(data['high'])
            lows = np.array(data['low'])
            volumes = np.array(data['volume'])
            
            # Calculate additional features
            returns = np.diff(closes) / closes[:-1]
            volatility = np.array([np.std(returns[max(0, i-20):i]) if i >= 20 else 0 
                                 for i in range(len(returns))])
            
            # Price momentum
            momentum_5 = (closes[5:] - closes[:-5]) / closes[:-5]
            momentum_10 = (closes[10:] - closes[:-10]) / closes[:-10]
            
            # Volume features
            volume_ma = np.convolve(volumes, np.ones(20)/20, mode='valid')
            volume_ratio = volumes[19:] / (volume_ma + 1e-10)
            
            # Align all features to same length
            min_len = min(len(momentum_10), len(volatility), len(volume_ratio))
            
            # Create feature matrix
            X = np.column_stack([
                data['sma_10'][-min_len:] / closes[-min_len:],  # SMA ratio
                data['sma_20'][-min_len:] / closes[-min_len:],  # SMA ratio
                data['rsi'][-min_len:] / 100.0,  # Normalized RSI
                momentum_5[-min_len:],  # 5-day momentum
                momentum_10[-min_len:],  # 10-day momentum
                volatility[-min_len:],  # Volatility
                volume_ratio[-min_len:],  # Volume ratio
                (highs[-min_len:] - lows[-min_len:]) / closes[-min_len:]  # Daily range
            ])
            
            # Create labels (1 if next day return > 0.5%, -1 if < -0.5%, 0 otherwise)
            future_returns = returns[-min_len+1:] if len(returns) >= min_len else []
            y = np.array([1 if r > 0.005 else -1 if r < -0.005 else 0 
                         for r in future_returns])
            
            # Ensure X and y have same length
            min_samples = min(len(X), len(y))
            X = X[:min_samples]
            y = y[:min_samples]
            
            return {'X': X, 'y': y}
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return {}
    
    def evaluate_model_accuracy(self, X, y) -> float:
        """Evaluate model accuracy"""
        try:
            predictions = self.xgb_trader.predict(X)
            accuracy = np.mean(predictions == y)
            return accuracy
        except:
            return 0.0
    
    def run_live_strategy_demo(self) -> Dict:
        """Run live strategy demonstration"""
        try:
            if not self.alpaca_trader:
                return {'error': 'Alpaca not connected'}
            
            self.logger.info("🚀 Running live strategy demo")
            
            # Get current account status
            account = self.alpaca_trader.alpaca_trader.get_account_info()
            positions = self.alpaca_trader.alpaca_trader.get_positions()
            
            self.logger.info(f"📊 Account Status:")
            self.logger.info(f"   Portfolio Value: ${account['portfolio_value']:,.2f}")
            self.logger.info(f"   Cash Available: ${account['cash']:,.2f}")
            self.logger.info(f"   Current Positions: {len(positions)}")
            
            # Generate signals for demo symbols
            signals = {}
            for symbol in self.symbols[:3]:  # Test with 3 symbols
                signal_data = self.alpaca_trader.generate_trading_signal(symbol)
                signals[symbol] = signal_data
                
                self.logger.info(f"📈 {symbol}: Signal={signal_data['signal']}, "
                               f"Confidence={signal_data['confidence']:.2f}, "
                               f"Price=${signal_data['current_price']:.2f}")
            
            # Execute strategy (in paper trading mode)
            execution_result = self.alpaca_trader.execute_strategy(
                symbols=list(signals.keys()),
                max_positions=3
            )
            
            return {
                'account': account,
                'positions': positions,
                'signals': signals,
                'execution': execution_result
            }
            
        except Exception as e:
            self.logger.error(f"Error in live strategy demo: {e}")
            return {'error': str(e)}
    
    def display_results(self, results: Dict):
        """Display demo results"""
        print("\n" + "="*60)
        print("🎯 ALPACA + XGBOOST TRADING DEMO RESULTS")
        print("="*60)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        # Account information
        account = results.get('account', {})
        print(f"\n💰 Account Information:")
        print(f"   Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"   Cash Available: ${account.get('cash', 0):,.2f}")
        print(f"   Buying Power: ${account.get('buying_power', 0):,.2f}")
        
        # Current positions
        positions = results.get('positions', [])
        print(f"\n📊 Current Positions ({len(positions)}):")
        if positions:
            for pos in positions:
                pnl_color = "🟢" if pos['unrealized_pl'] >= 0 else "🔴"
                print(f"   {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f} "
                      f"{pnl_color} P&L: ${pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']*100:.1f}%)")
        else:
            print("   No current positions")
        
        # Trading signals
        signals = results.get('signals', {})
        print(f"\n🎯 Trading Signals:")
        for symbol, signal_data in signals.items():
            signal_emoji = "🟢" if signal_data['signal'] > 0 else "🔴" if signal_data['signal'] < 0 else "⚪"
            print(f"   {symbol}: {signal_emoji} Signal={signal_data['signal']} "
                  f"(Confidence: {signal_data['confidence']:.2f})")
            print(f"      Price: ${signal_data['current_price']:.2f}, RSI: {signal_data['rsi']:.1f}")
            print(f"      Reason: {signal_data['reason']}")
        
        # Execution results
        execution = results.get('execution', {})
        executed_trades = execution.get('executed_trades', [])
        print(f"\n⚡ Executed Trades ({len(executed_trades)}):")
        if executed_trades:
            for trade in executed_trades:
                action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                print(f"   {action_emoji} {trade['action']} {trade['qty']} {trade['symbol']} "
                      f"(Order: {trade['order_id'][:8]}...)")
                print(f"      Reason: {trade['reason']}")
        else:
            print("   No trades executed in this cycle")
        
        print(f"\n✅ Demo completed successfully!")
    
    def run_full_demo(self):
        """Run complete demonstration"""
        print("🚀 Starting Alpaca + XGBoost Trading Demo")
        print("="*50)
        
        # Step 1: Setup Alpaca connection
        print("\n1️⃣  Setting up Alpaca connection...")
        if not self.setup_alpaca_connection():
            print("❌ Could not connect to Alpaca API")
            print("💡 To run with real data, set up your Alpaca API credentials:")
            print("   1. Sign up at https://alpaca.markets")
            print("   2. Get your API keys from the dashboard")
            print("   3. Set environment variables or create .env file:")
            print("      ALPACA_API_KEY=your_key")
            print("      ALPACA_SECRET_KEY=your_secret")
            print("\n🔄 Running with synthetic data fallback...")
            self.run_synthetic_fallback()
            return
        
        # Step 2: Train models on real data
        print("\n2️⃣  Training XGBoost models on real market data...")
        trained_models = self.train_models_on_real_data()
        
        if trained_models:
            print(f"✅ Trained {len(trained_models)} models")
            for symbol, model_info in trained_models.items():
                print(f"   {symbol}: {model_info['training_samples']} samples, "
                      f"accuracy: {model_info['accuracy']:.3f}")
        
        # Step 3: Run live strategy
        print("\n3️⃣  Running live trading strategy...")
        results = self.run_live_strategy_demo()
        
        # Step 4: Display results
        self.display_results(results)
    
    def run_synthetic_fallback(self):
        """Run demo with synthetic data if Alpaca not available"""
        print("🔄 Running synthetic data demonstration...")
        
        # Import the working demo
        try:
            from working_demo import XGBoostTradingDemo
            
            # Run synthetic demo
            demo = XGBoostTradingDemo()
            demo.run_demo()
            
            print("\n💡 This was a synthetic data demo.")
            print("   Set up Alpaca API credentials to use real market data!")
            
        except Exception as e:
            print(f"❌ Error running synthetic fallback: {e}")

def main():
    """Main demo function"""
    demo = AlpacaXGBoostDemo()
    demo.run_full_demo()

if __name__ == "__main__":
    main()