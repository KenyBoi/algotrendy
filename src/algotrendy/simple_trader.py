"""
Simplified ML Trader using XGBoost (or RandomForest fallback)
Works with raw numpy arrays for Python 3.13 compatibility
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import os
import sys
import logging
from pathlib import Path
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

import numpy as np
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

from algotrendy.config import CONFIG

logger = logging.getLogger(__name__)

class SimpleMLTrader:
    """
    Simplified ML trader using XGBoost (or RandomForest fallback) for trading signals
    """
    
    def __init__(self):
        """Initialize the trader"""
        self.model = None
        self.label_mapping = None
        self.inverse_label_mapping = None
        self.classes_ = [-1, 0, 1]
        self.uses_xgboost = False
        self.feature_names = [
            'price_change_1d', 'price_change_5d', 'price_change_10d',
            'volume_ratio', 'volatility_20d', 'rsi', 'price_sma_ratio',
            'high_low_range'
        ]
        
        logger.info("Simple ML Trader initialized")
    

    def prepare_features(self, price_data: Union[Dict, 'pd.DataFrame'], return_index: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features from price data

        Args:
            price_data: Dictionary with 'close', 'high', 'low', 'volume', etc.

        Returns:
            Tuple of (features, labels) or (features, labels, index) when return_index=True
        """
        try:
            # Accept either a dict-of-arrays or a pandas DataFrame
            try:
                import pandas as pd
            except Exception:
                pd = None

            if pd is not None and isinstance(price_data, pd.DataFrame):
                df = price_data.copy()
                # Normalize column names to lowercase for robustness
                df.columns = df.columns.str.lower()

                if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns or 'volume' not in df.columns:
                    raise KeyError("DataFrame must contain 'open','high','low','close','volume' columns (case-insensitive)")

                closes = df['close'].to_numpy()
                highs = df['high'].to_numpy()
                lows = df['low'].to_numpy()
                volumes = df['volume'].to_numpy()
            else:
                closes = np.array(price_data['close'])
                highs = np.array(price_data['high'])
                lows = np.array(price_data['low'])
                volumes = np.array(price_data['volume'])

            if len(closes) < 30:
                raise ValueError("Need at least 30 data points")

            features_list = []
            labels_list = []
            feature_index = []

            for i in range(20, len(closes) - 1):
                price_change_1d = (closes[i] - closes[i - 1]) / closes[i - 1]
                price_change_5d = (closes[i] - closes[i - 5]) / closes[i - 5]
                price_change_10d = (closes[i] - closes[i - 10]) / closes[i - 10]

                vol_ma = np.mean(volumes[i - 20:i])
                volume_ratio = volumes[i] / vol_ma if vol_ma > 0 else 1

                returns = np.diff(closes[i - 20:i]) / closes[i - 20:i - 1]
                volatility_20d = np.std(returns)

                window_returns = np.diff(closes[i - 14:i])
                gains = np.where(window_returns > 0, window_returns, 0)
                losses = np.where(window_returns < 0, -window_returns, 0)
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0

                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100

                sma_20 = np.mean(closes[i - 20:i])
                price_sma_ratio = closes[i] / sma_20

                high_low_range = (highs[i] - lows[i]) / closes[i]

                features = np.array([
                    price_change_1d,
                    price_change_5d,
                    price_change_10d,
                    volume_ratio,
                    volatility_20d,
                    rsi / 100.0,
                    price_sma_ratio,
                    high_low_range,
                ])

                future_return = (closes[i + 1] - closes[i]) / closes[i]

                if future_return > 0.01:
                    label = 1
                elif future_return < -0.01:
                    label = -1
                else:
                    label = 0

                features_list.append(features)
                labels_list.append(label)
                feature_index.append(i)

            X = np.array(features_list)
            y = np.array(labels_list)

            feature_dates = None
            if return_index:
                # Prefer DataFrame index if provided
                if pd is not None and isinstance(price_data, pd.DataFrame):
                    index_array = price_data.index.to_numpy()
                    feature_dates = index_array[20:-1]
                elif 'index' in price_data:
                    index_array = np.array(price_data['index'])
                    feature_dates = index_array[20:-1]
                else:
                    feature_dates = np.array(feature_index)

            logger.info(f"Prepared {len(X)} feature samples with {X.shape[1]} features")
            if return_index:
                return X, y, feature_dates
            return X, y

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.classes_ = sorted(np.unique(y))

            if XGBOOST_AVAILABLE:
                self.uses_xgboost = True
                self.label_mapping = {label: idx for idx, label in enumerate(self.classes_)}
                self.inverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}

                y_train_encoded = np.vectorize(self.label_mapping.get)(y_train)
                y_test_encoded = np.vectorize(self.label_mapping.get)(y_test)

                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                logger.info("Using XGBoost for training")
                self.model.fit(X_train, y_train_encoded)

                train_score = self.model.score(X_train, y_train_encoded)
                test_score = self.model.score(X_test, y_test_encoded)

                y_pred_encoded = self.model.predict(X_test)
                y_pred = np.vectorize(self.inverse_label_mapping.get)(y_pred_encoded)
                y_test_original = np.vectorize(self.inverse_label_mapping.get)(y_test_encoded)
                accuracy = accuracy_score(y_test_original, y_pred)
            else:
                self.uses_xgboost = False
                self.label_mapping = None
                self.inverse_label_mapping = None

                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                logger.info("Using RandomForest for training (XGBoost not available)")
                self.model.fit(X_train, y_train)

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
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        predictions = self.model.predict(X)
        if self.inverse_label_mapping:
            predictions = np.vectorize(self.inverse_label_mapping.get)(predictions)
        return predictions
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        probabilities = self.model.predict_proba(X)

        if self.inverse_label_mapping and self.label_mapping:
            encoded_order = sorted(self.inverse_label_mapping.keys())
            reorder = [encoded_order.index(self.label_mapping[label]) for label in self.classes_]
            probabilities = probabilities[:, reorder]
        elif hasattr(self.model, "classes_") and getattr(self, "classes_", None):
            model_classes = list(self.model.classes_)
            reorder = [model_classes.index(label) for label in self.classes_ if label in model_classes]
            if reorder and len(reorder) == probabilities.shape[1]:
                probabilities = probabilities[:, reorder]

        # Sanitize probabilities: ensure non-negative and rows sum to 1 (avoid tiny floating rounding issues)
        try:
            probs = np.array(probabilities, dtype=float)
            probs = np.clip(probs, 0.0, None)
            row_sums = probs.sum(axis=1, keepdims=True)
            # Avoid division by zero (if a row is all zeros, fallback to uniform distribution)
            zero_rows = (row_sums == 0).flatten()
            if np.any(zero_rows):
                ncols = probs.shape[1]
                probs[zero_rows] = np.ones((zero_rows.sum(), ncols)) / float(ncols)
                row_sums = probs.sum(axis=1, keepdims=True)

            probs = probs / row_sums
            return probs
        except Exception:
            # If anything unexpected happens, return original probabilities (best-effort)
            return probabilities


    def save_model(self, path: str) -> None:
        """Persist the trained model and metadata to disk.

        Writes two files: <path>.joblib (model object) and <path>.json (metadata).
        If joblib is not available, falls back to writing a single pickle file at <path>.pkl
        for backward compatibility.
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        base = str(path)
        try:
            import joblib
            metadata = {
                'feature_names': self.feature_names,
                'label_mapping': self.label_mapping,
                'inverse_label_mapping': self.inverse_label_mapping,
                'classes_': self.classes_,
                'uses_xgboost': self.uses_xgboost
            }

            model_file = base if base.endswith('.joblib') else base + '.joblib'
            meta_file = base if base.endswith('.json') else base + '.json'

            # Save a single payload so loading is atomic and metadata isn't lost if separate files are missing
            payload = {
                'model': self.model,
                'feature_names': self.feature_names,
                'label_mapping': self.label_mapping,
                'inverse_label_mapping': self.inverse_label_mapping,
                'classes_': self.classes_,
                'uses_xgboost': self.uses_xgboost
            }

            joblib.dump(payload, model_file)
            # Also write metadata JSON for compatibility
            try:
                with open(meta_file, 'w', encoding='utf-8') as mf:
                    import json
                    json.dump(metadata, mf, ensure_ascii=False, indent=2)
            except Exception:
                pass

            logger.info(f"Model saved to {model_file} and metadata to {meta_file}")
            return

        except Exception:
            # Fallback to pickle for environments without joblib
            payload = {
                'model': self.model,
                'feature_names': self.feature_names,
                'label_mapping': self.label_mapping,
                'inverse_label_mapping': self.inverse_label_mapping,
                'classes_': self.classes_,
                'uses_xgboost': self.uses_xgboost
            }
            fallback_path = base if base.endswith('.pkl') else base + '.pkl'
            with open(fallback_path, 'wb') as f:
                pickle.dump(payload, f)
            logger.info(f"Joblib not available; model pickled to {fallback_path}")

    def load_model(self, path: str) -> None:
        """Load a persisted model from disk.

        Tries to load <path>.joblib + <path>.json pairing first, falls back to <path>.pkl pickle.
        """
        base = str(path)
        try:
            import joblib
            model_file = base if base.endswith('.joblib') else base + '.joblib'
            meta_file = base if base.endswith('.json') else base + '.json'

            # Load model
            loaded = joblib.load(model_file)
            # Support both legacy (model object) and new (payload dict) formats
            if isinstance(loaded, dict) and 'model' in loaded:
                payload = loaded
                self.model = payload.get('model')
                self.feature_names = payload.get('feature_names', self.feature_names)
                self.label_mapping = payload.get('label_mapping')
                self.inverse_label_mapping = payload.get('inverse_label_mapping')
                self.classes_ = payload.get('classes_', self.classes_)
                self.uses_xgboost = payload.get('uses_xgboost', False)
            else:
                # Legacy joblib-only model; try to load metadata JSON
                self.model = loaded
                try:
                    import json
                    with open(meta_file, 'r', encoding='utf-8') as mf:
                        metadata = json.load(mf)
                    self.feature_names = metadata.get('feature_names', self.feature_names)
                    self.label_mapping = metadata.get('label_mapping')
                    self.inverse_label_mapping = metadata.get('inverse_label_mapping')
                    self.classes_ = metadata.get('classes_', self.classes_)
                    self.uses_xgboost = metadata.get('uses_xgboost', False)
                except Exception:
                    logger.warning(f"Metadata file {meta_file} not found or invalid; continuing with model only")

            logger.info(f"Loaded model from {model_file}")
            return

        except Exception:
            # Fallback to old pickle format
            fallback_path = base if base.endswith('.pkl') else base + '.pkl'
            with open(fallback_path, 'rb') as f:
                payload = pickle.load(f)

            self.model = payload['model']
            self.feature_names = payload.get('feature_names', self.feature_names)
            self.label_mapping = payload.get('label_mapping')
            self.inverse_label_mapping = payload.get('inverse_label_mapping')
            self.classes_ = payload.get('classes_', self.classes_)
            self.uses_xgboost = payload.get('uses_xgboost', False)
            logger.info(f"Loaded legacy pickle model from {fallback_path}")

    def get_feature_importance(self) -> Dict:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance_scores = getattr(self.model, 'feature_importances_', None)
        if importance_scores is None:
            raise AttributeError('Model does not provide feature importances')

        return {
            name: float(score) for name, score in zip(self.feature_names, importance_scores)
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
    print("🚀 Simple ML Trading Demo (Python 3.13 Compatible)")
    print("=" * 60)
    
    # Initialize trader
    trader = SimpleMLTrader()
    
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
    new_trader = SimpleMLTrader()
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



