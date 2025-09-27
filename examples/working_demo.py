"""
Working Demo of AlgoTrendy XGBoost Trading System
This demo uses synthetic market data to show the core functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

class SyntheticMarketData:
    """Generate realistic synthetic market data"""
    
    def __init__(self, n_days=1000, initial_price=100.0):
        self.n_days = n_days
        self.initial_price = initial_price
        
    def generate_price_data(self):
        """Generate OHLCV-like data with realistic patterns"""
        np.random.seed(42)
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0005, 0.02, self.n_days)  # Daily returns
        
        # Add some trend components
        trend = np.sin(np.arange(self.n_days) / 50) * 0.001
        returns += trend
        
        # Calculate prices
        prices = [self.initial_price]
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        prices = np.array(prices[1:])  # Remove initial price
        
        # Generate OHLC from close prices
        data = {}
        data['close'] = prices
        
        # Generate opens (close of previous day + small gap)
        gaps = np.random.normal(0, 0.005, len(prices))
        data['open'] = np.roll(prices, 1) * (1 + gaps)
        data['open'][0] = self.initial_price
        
        # Generate highs and lows
        high_mult = 1 + np.abs(np.random.normal(0, 0.01, len(prices)))
        low_mult = 1 - np.abs(np.random.normal(0, 0.01, len(prices)))
        
        data['high'] = np.maximum(data['open'], data['close']) * high_mult
        data['low'] = np.minimum(data['open'], data['close']) * low_mult
        
        # Generate volume
        base_volume = 1000000
        volume_noise = np.random.lognormal(0, 0.5, len(prices))
        data['volume'] = base_volume * volume_noise
        
        return data

class SimpleIndicators:
    """Calculate basic technical indicators"""
    
    @staticmethod
    def sma(prices, window):
        """Simple Moving Average"""
        return np.convolve(prices, np.ones(window)/window, mode='valid')
    
    @staticmethod
    def ema(prices, window):
        """Exponential Moving Average"""
        alpha = 2.0 / (window + 1.0)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    def rsi(prices, window=14):
        """Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(window)/window, mode='valid')
        avg_losses = np.convolve(losses, np.ones(window)/window, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class XGBoostTradingDemo:
    """Simplified XGBoost trading system for demonstration"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def create_features(self, data):
        """Create ML features from price data"""
        prices = data['close']
        volume = data['volume']
        
        # Price-based features
        returns_1d = np.diff(prices) / prices[:-1]
        returns_5d = (prices[5:] - prices[:-5]) / prices[:-5]
        
        # Technical indicators
        sma_10 = SimpleIndicators.sma(prices, 10)
        sma_20 = SimpleIndicators.sma(prices, 20)
        ema_12 = SimpleIndicators.ema(prices, 12)
        rsi = SimpleIndicators.rsi(prices, 14)
        
        # Price ratios (align array lengths)
        price_to_sma10 = prices[len(prices)-len(sma_10):] / sma_10
        price_to_sma20 = prices[len(prices)-len(sma_20):] / sma_20
        sma10_to_sma20 = sma_10[len(sma_10)-len(sma_20):] / sma_20
        
        # Volatility
        volatility_10 = np.array([np.std(returns_1d[max(0,i-10):i+1]) 
                                 for i in range(len(returns_1d))])
        
        # Volume features
        volume_sma = SimpleIndicators.sma(volume, 20)
        volume_ratio = volume[len(volume)-len(volume_sma):] / volume_sma
        
        # Combine features (align all to same length)
        min_length = min(len(price_to_sma20), len(sma10_to_sma20), 
                        len(rsi), len(volume_ratio), len(volatility_10)-20, len(returns_5d))
        
        features = np.column_stack([
            price_to_sma10[-min_length:],
            price_to_sma20[-min_length:],
            sma10_to_sma20[-min_length:],
            rsi[-min_length:],
            volume_ratio[-min_length:],
            volatility_10[-(min_length):],
            returns_1d[-min_length:],
            returns_5d[-min_length:]
        ])
        
        self.feature_names = [
            'price_to_sma10', 'price_to_sma20', 'sma10_to_sma20', 
            'rsi', 'volume_ratio', 'volatility_10', 'returns_1d', 'returns_5d'
        ]
        
        return features
    
    def create_targets(self, data, prediction_days=5):
        """Create target variables for prediction"""
        prices = data['close']
        
        # Future returns
        future_returns = []
        for i in range(len(prices) - prediction_days):
            ret = (prices[i + prediction_days] - prices[i]) / prices[i]
            future_returns.append(ret)
        
        future_returns = np.array(future_returns)
        
        # Binary classification target (profit > 2%)
        binary_target = (future_returns > 0.02).astype(int)
        
        return binary_target
    
    def train(self, data):
        """Train the XGBoost model"""
        print("🔄 Creating features and targets...")
        
        # Create features and targets
        X = self.create_features(data)
        y = self.create_targets(data)
        
        # Align X and y to same length
        min_length = min(len(X), len(y))
        X = X[-min_length:]
        y = y[-min_length:]
        
        print(f"📊 Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print("🤖 Training XGBoost model...")
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully!")
        print(f"📈 Test Accuracy: {accuracy:.3f}")
        
        # Feature importance
        importance = self.model.feature_importances_
        print("\n🔍 Top Features:")
        for i, (name, imp) in enumerate(zip(self.feature_names, importance)):
            print(f"   {i+1}. {name:<15}: {imp:.4f}")
        
        return {
            'accuracy': accuracy,
            'feature_importance': list(zip(self.feature_names, importance)),
            'predictions': y_pred,
            'actual': y_test
        }
    
    def backtest(self, data, initial_capital=100000):
        """Simple backtest"""
        print("\n📊 Running backtest...")
        
        X = self.create_features(data)
        
        # Generate signals
        predictions = self.model.predict_proba(X)[:, 1]  # Probability of profit
        signals = np.where(predictions > 0.6, 1, 0)  # Buy if >60% confidence
        
        # Simple backtest
        capital = initial_capital
        position = 0
        prices = data['close'][-len(signals):]  # Align with signals
        
        trades = []
        portfolio_values = [capital]
        
        for i in range(1, len(signals)):
            current_price = prices[i]
            
            # Entry signal
            if signals[i] == 1 and position == 0:
                # Buy
                shares = capital * 0.95 / current_price  # Use 95% of capital
                position = shares
                capital -= shares * current_price
                trades.append(('BUY', current_price, shares))
                
            # Exit after 5 days or if signal changes
            elif position > 0 and (signals[i] == 0 or i % 5 == 0):
                # Sell
                capital += position * current_price
                trades.append(('SELL', current_price, position))
                position = 0
            
            # Calculate portfolio value
            portfolio_value = capital + (position * current_price if position > 0 else 0)
            portfolio_values.append(portfolio_value)
        
        # Final stats
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        print(f"💰 Backtest Results:")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Number of Trades: {len(trades)//2}")
        
        return {
            'portfolio_values': portfolio_values,
            'total_return': total_return,
            'trades': trades
        }

def main():
    """Run the complete demo"""
    print("🚀 AlgoTrendy XGBoost Trading System Demo")
    print("=" * 50)
    
    # 1. Generate synthetic market data
    print("📈 Generating synthetic market data...")
    market_data = SyntheticMarketData(n_days=1000, initial_price=100.0)
    data = market_data.generate_price_data()
    
    print(f"   Generated {len(data['close'])} days of market data")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 2. Train XGBoost model
    print("\n🤖 Training XGBoost Model...")
    trader = XGBoostTradingDemo()
    results = trader.train(data)
    
    # 3. Run backtest
    backtest_results = trader.backtest(data)
    
    # 4. Plot results if matplotlib works
    try:
        plt.figure(figsize=(12, 8))
        
        # Price chart
        plt.subplot(2, 2, 1)
        plt.plot(data['close'])
        plt.title('Stock Price Over Time')
        plt.ylabel('Price ($)')
        
        # Portfolio value
        plt.subplot(2, 2, 2)
        plt.plot(backtest_results['portfolio_values'])
        plt.title('Portfolio Value')
        plt.ylabel('Value ($)')
        
        # Feature importance
        plt.subplot(2, 2, 3)
        features, importance = zip(*results['feature_importance'])
        plt.barh(range(len(features)), importance)
        plt.yticks(range(len(features)), features)
        plt.title('Feature Importance')
        
        # Prediction accuracy
        plt.subplot(2, 2, 4)
        plt.hist(results['predictions'], alpha=0.7, label='Predictions')
        plt.hist(results['actual'], alpha=0.7, label='Actual')
        plt.title('Prediction Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('algotrendy_demo_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Charts saved as 'algotrendy_demo_results.png'")
        
    except Exception as e:
        print(f"⚠️  Chart generation skipped: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps to enhance the system:")
    print("1. Connect to real market data APIs")
    print("2. Add more sophisticated technical indicators")
    print("3. Implement risk management rules")
    print("4. Add portfolio optimization")
    print("5. Create live trading interface")

if __name__ == "__main__":
    main()