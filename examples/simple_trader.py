"""
Simplified XGBoost Trader without pandas dependency
Works with raw numpy arrays for Python 3.13 compatibility
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from config import CONFIG

logger = logging.getLogger(__name__)

class SimpleXGBoostTrader:
    """
    Simplified XGBoost-style trader using RandomForest for Python 3.13 compatibility
    """
    
    def __init__(self):
        """Initialize the trader"""
        self.model = None
        self.feature_names = [
            'price_change_1d', 'price_change_5d', 'price_change_10d',
            'volume_ratio', 'volatility_20d', 'rsi', 'price_sma_ratio',
            'high_low_range'
        ]
        
        logger.info("Simple XGBoost Trader initialized")
    
    def prepare_features(self, price_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features from price data
        
        Args:
            price_data: Dictionary with 'close', 'high', 'low', 'volume', etc.
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            closes = np.array(price_data['close'])
            highs = np.array(price_data['high'])
            lows = np.array(price_data['low'])
            volumes = np.array(price_data['volume'])
            
            if len(closes) < 30:
                raise ValueError("Need at least 30 data points")
            
            # Calculate features
            features_list = []
            labels_list = []
            
            for i in range(20, len(closes) - 1):
                # Price change features
                price_change_1d = (closes[i] - closes[i-1]) / closes[i-1]
                price_change_5d = (closes[i] - closes[i-5]) / closes[i-5]
                price_change_10d = (closes[i] - closes[i-10]) / closes[i-10]
                
                # Volume ratio
                vol_ma = np.mean(volumes[i-20:i])
                volume_ratio = volumes[i] / vol_ma if vol_ma > 0 else 1
                
                # Volatility
                returns = np.diff(closes[i-20:i]) / closes[i-20:i-1]
                volatility_20d = np.std(returns)
                
                # RSI calculation (simplified)
                window_returns = np.diff(closes[i-14:i])
                gains = np.where(window_returns > 0, window_returns, 0)
                losses = np.where(window_returns < 0, -window_returns, 0)
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100
                
                # Price to SMA ratio
                sma_20 = np.mean(closes[i-20:i])
                price_sma_ratio = closes[i] / sma_20
                
                # High-low range
                high_low_range = (highs[i] - lows[i]) / closes[i]
                
                # Combine features
                features = np.array([
                    price_change_1d, price_change_5d, price_change_10d,
                    volume_ratio, volatility_20d, rsi / 100.0,
                    price_sma_ratio, high_low_range
                ])
                
                # Label: future return
                future_return = (closes[i+1] - closes[i]) / closes[i]
                
                # Classify: 1 if return > 1%, -1 if return < -1%, 0 otherwise
                if future_return > 0.01:
                    label = 1
                elif future_return < -0.01:
                    label = -1
                else:
                    label = 0
                
                features_list.append(features)
                labels_list.append(label)
            
            X = np.array(features_list)
            y = np.array(labels_list)
            
            logger.info(f"Prepared {len(X)} feature samples with {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Training metrics
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest (XGBoost alternative)
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'accuracy': accuracy,
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
            logger.info(f"Model trained - Test Accuracy: {accuracy:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability matrix
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'timestamp': datetime.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_scores = self.model.feature_importances_
        
        return {
            name: score for name, score in 
            zip(self.feature_names, importance_scores)
        }

# Synthetic data generator for testing
class SyntheticMarketData:
    """Generate realistic synthetic market data"""
    
    @staticmethod
    def generate_price_series(days: int = 252, start_price: float = 100.0) -> Dict:
        """Generate synthetic OHLCV data"""
        np.random.seed(42)  # For reproducible results
        
        # Generate price walk
        returns = np.random.normal(0.0005, 0.02, days)  # Small positive drift
        
        # Add some trend and momentum
        trend = np.linspace(0, 0.3, days)  # 30% annual trend
        momentum = np.convolve(returns, np.ones(5)/5, mode='same')  # 5-day momentum
        
        adjusted_returns = returns + trend/days + momentum * 0.1
        
        # Calculate prices
        prices = [start_price]
        for ret in adjusted_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        closes = np.array(prices)
        
        # Generate OHLV from closes
        highs = closes * (1 + np.abs(np.random.normal(0, 0.01, days)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.01, days)))
        
        # Opens based on previous close with gap
        opens = np.roll(closes, 1) * (1 + np.random.normal(0, 0.005, days))
        opens[0] = start_price
        
        # Volume with some correlation to price movement
        base_volume = 1000000
        volume_multiplier = 1 + np.abs(adjusted_returns) * 10  # Higher volume on big moves
        volumes = base_volume * volume_multiplier * np.random.lognormal(0, 0.3, days)
        
        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes.astype(int)
        }

def run_simple_demo():
    """Run a complete demo of the simplified trading system"""
    print("🚀 Simple XGBoost Trading Demo (Python 3.13 Compatible)")
    print("=" * 60)
    
    # Initialize trader
    trader = SimpleXGBoostTrader()
    
    # Generate synthetic data
    print("\n📊 Generating synthetic market data...")
    data = SyntheticMarketData.generate_price_series(days=500, start_price=100)
    
    print(f"   Generated {len(data['close'])} days of data")
    print(f"   Start price: ${data['close'][0]:.2f}")
    print(f"   End price: ${data['close'][-1]:.2f}")
    print(f"   Total return: {(data['close'][-1] / data['close'][0] - 1) * 100:.1f}%")
    
    # Prepare features
    print("\n🔧 Preparing ML features...")
    X, y = trader.prepare_features(data)
    
    # Show feature distribution
    buy_signals = np.sum(y == 1)
    sell_signals = np.sum(y == -1)
    hold_signals = np.sum(y == 0)
    
    print(f"   Features shape: {X.shape}")
    print(f"   Signal distribution: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD")
    
    # Train model
    print("\n🤖 Training ML model...")
    metrics = trader.train(X, y)
    
    print(f"   Training accuracy: {metrics['train_accuracy']:.3f}")
    print(f"   Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"   Features used: {metrics['n_features']}")
    
    # Feature importance
    importance = trader.get_feature_importance()
    print("\n📈 Feature Importance:")
    for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"   {feature}: {score:.3f}")
    
    # Backtest on last 60 days
    print("\n📊 Running backtest simulation...")
    
    # Use last 60 days for backtesting
    test_start = len(data['close']) - 80
    test_data = {key: val[test_start:] for key, val in data.items()}
    
    # Prepare features for backtesting
    X_test, _ = trader.prepare_features(test_data)
    
    # Generate signals
    signals = trader.predict(X_test)
    probabilities = trader.predict_proba(X_test)
    
    # Simple backtesting
    portfolio_value = 10000  # Start with $10k
    position = 0  # Number of shares
    cash = portfolio_value
    trades = []
    
    prices = test_data['close'][20:-1]  # Align with features
    
    for i, (signal, price) in enumerate(zip(signals, prices)):
        max_confidence = np.max(probabilities[i])
        
        if signal == 1 and max_confidence > 0.4 and position == 0:  # Buy
            shares = int(cash * 0.95 / price)  # Use 95% of cash
            if shares > 0:
                position = shares
                cash -= shares * price
                trades.append(('BUY', shares, price, portfolio_value))
        
        elif signal == -1 and position > 0:  # Sell
            cash += position * price
            trades.append(('SELL', position, price, cash + position * price))
            position = 0
        
        # Update portfolio value
        portfolio_value = cash + position * price
    
    # Final portfolio value
    if position > 0:
        portfolio_value = cash + position * prices[-1]
    else:
        portfolio_value = cash
    
    total_return = (portfolio_value - 10000) / 10000 * 100
    
    print(f"\n💰 Backtest Results:")
    print(f"   Initial capital: $10,000")
    print(f"   Final portfolio value: ${portfolio_value:,.2f}")
    print(f"   Total return: {total_return:.2f}%")
    print(f"   Number of trades: {len(trades)}")
    
    if trades:
        print(f"\n📋 Recent trades:")
        for i, (action, qty, price, value) in enumerate(trades[-5:]):
            print(f"   {i+1}. {action} {qty} shares @ ${price:.2f} (Portfolio: ${value:,.2f})")
    
    # Model persistence test
    print("\n💾 Testing model persistence...")
    model_path = "simple_model.pkl"
    trader.save_model(model_path)
    
    # Create new trader and load model
    new_trader = SimpleXGBoostTrader()
    new_trader.load_model(model_path)
    
    # Test prediction
    test_prediction = new_trader.predict(X_test[:1])
    print(f"   Loaded model prediction: {test_prediction[0]}")
    
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
    
    print("\n✅ Demo completed successfully!")
    print("🔗 Ready for Alpaca integration!")

if __name__ == "__main__":
    run_simple_demo()