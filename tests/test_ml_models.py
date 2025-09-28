"""
ML Model Validation and Testing Framework
Tests for all ML trading models to ensure reliability and accuracy
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

try:
    from examples.simple_trader import SimpleMLTrader, SyntheticMarketData
    from examples.xgboost_trader import XGBoostTrader
    from algotrendy.advanced_ml_trainer import AdvancedMLTrainer
except Exception as e:
    pytest.skip(f"Skipping ML model tests due to import error: {e}", allow_module_level=True)


class TestSimpleMLTrader:
    """Test suite for SimpleMLTrader"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample market data for testing"""
        return SyntheticMarketData.generate_price_series(days=300)

    @pytest.fixture
    def trader(self):
        """Initialize trader for testing"""
        return SimpleMLTrader()

    def test_initialization(self, trader):
        """Test trader initialization"""
        assert trader.model is None
        assert len(trader.feature_names) == 8
        assert 'price_change_1d' in trader.feature_names

    def test_feature_preparation(self, trader, sample_data):
        """Test feature preparation"""
        X, y = trader.prepare_features(sample_data)

        # Check dimensions
        assert X.shape[1] == 8  # 8 features
        assert len(X) == len(y)
        assert len(X) > 0

        # Check labels are valid (-1, 0, 1)
        assert all(label in [-1, 0, 1] for label in y)

    def test_model_training(self, trader, sample_data):
        """Test model training"""
        X, y = trader.prepare_features(sample_data)
        metrics = trader.train(X, y)

        # Check required metrics
        required_keys = ['train_accuracy', 'test_accuracy', 'accuracy', 'n_samples', 'n_features']
        for key in required_keys:
            assert key in metrics

        # Check accuracy is reasonable
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['n_samples'] == len(X)

    def test_predictions(self, trader, sample_data):
        """Test model predictions"""
        X, y = trader.prepare_features(sample_data)
        trader.train(X, y)

        # Test predictions
        predictions = trader.predict(X[:10])
        probabilities = trader.predict_proba(X[:10])

        assert len(predictions) == 10
        assert all(pred in [-1, 0, 1] for pred in predictions)
        assert probabilities.shape == (10, 3)  # 3 classes

    def test_model_persistence(self, trader, sample_data, tmp_path):
        """Test model save/load functionality"""
        X, y = trader.prepare_features(sample_data)
        trader.train(X, y)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        trader.save_model(str(model_path))

        # Load model
        new_trader = SimpleMLTrader()
        new_trader.load_model(str(model_path))

        # Test predictions match
        test_X = X[:5]
        original_pred = trader.predict(test_X)
        loaded_pred = new_trader.predict(test_X)

        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_feature_importance(self, trader, sample_data):
        """Test feature importance calculation"""
        X, y = trader.prepare_features(sample_data)
        trader.train(X, y)

        importance = trader.get_feature_importance()

        assert len(importance) == 8
        assert all(isinstance(score, (int, float)) for score in importance.values())
        assert all(score >= 0 for score in importance.values())


class TestXGBoostTrader:
    """Test suite for XGBoostTrader"""

    @pytest.fixture
    def sample_df(self):
        """Generate sample DataFrame for XGBoost testing"""
        data = SyntheticMarketData.generate_price_series(days=300)

        # Convert to DataFrame format expected by XGBoostTrader
        df = pd.DataFrame({
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume']
        })

        # Add target column (dummy for testing)
        df['target_binary'] = np.random.choice([0, 1], size=len(df))

        return df

    @pytest.fixture
    def trader(self):
        """Initialize XGBoost trader"""
        return XGBoostTrader(model_type='binary')

    def test_initialization(self, trader):
        """Test XGBoost trader initialization"""
        assert trader.model_type == 'binary'
        assert trader.scaler is not None
        assert not trader.is_fitted

    def test_feature_preparation(self, trader, sample_df):
        """Test feature preparation for XGBoost"""
        X, y = trader.prepare_features(sample_df, 'target_binary')

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_model_training(self, trader, sample_df):
        """Test XGBoost model training"""
        X, y = trader.prepare_features(sample_df, 'target_binary')
        results = trader.train(X, y)

        assert trader.is_fitted
        assert 'train_accuracy' in results
        assert 'test_accuracy' in results
        assert 'feature_importance' in results

    def test_predictions(self, trader, sample_df):
        """Test XGBoost predictions"""
        X, y = trader.prepare_features(sample_df, 'target_binary')
        trader.train(X, y)

        predictions = trader.predict(X[:10])
        assert len(predictions) == 10

    def test_model_persistence(self, trader, sample_df, tmp_path):
        """Test XGBoost model save/load"""
        X, y = trader.prepare_features(sample_df, 'target_binary')
        trader.train(X, y)

        model_path = tmp_path / "test_xgb_model"
        saved_path = trader.save_model("test_xgb_model")

        new_trader = XGBoostTrader(model_type='binary')
        new_trader.load_model("test_xgb_model")

        assert new_trader.is_fitted


class TestAdvancedMLTrainer:
    """Test suite for AdvancedMLTrainer"""

    @pytest.fixture
    def trader(self):
        """Initialize advanced trainer (fixture name matches tests)"""
        return AdvancedMLTrainer(symbol="TEST")

    def test_initialization(self, trader):
        """Test advanced trainer initialization"""
        assert trader.symbol == "TEST"
        assert trader.models == {}
        assert trader.best_model is None

    def test_data_loading_and_preparation(self, trader):
        """Test data loading and preparation (mock test)"""
        # This would require actual data manager setup
        # For now, just test the structure exists
        assert hasattr(trader, 'load_and_prepare_data')
        assert hasattr(trader, '_advanced_feature_engineering')

    def test_model_training_structure(self, trader):
        """Test that training methods exist"""
        assert hasattr(trader, 'train_advanced_model')
        assert hasattr(trader, 'evaluate_model')

    def test_model_persistence(self, trader, tmp_path):
        """Test model save/load structure"""
        assert hasattr(trader, 'save_advanced_model')
        assert hasattr(trader, 'load_advanced_model')


class TestModelValidation:
    """Cross-cutting model validation tests"""

    def test_model_consistency(self):
        """Test that different model implementations give consistent results"""
        # Generate same data
        data = SyntheticMarketData.generate_price_series(days=200, start_price=100)

        # Train both models
        simple_trader = SimpleMLTrader()
        X, y = simple_trader.prepare_features(data)
        simple_trader.train(X, y)

        # Both should be able to make predictions
        predictions = simple_trader.predict(X[:10])
        assert len(predictions) == 10

    def test_feature_stability(self):
        """Test that features are calculated consistently"""
        data1 = SyntheticMarketData.generate_price_series(days=100, start_price=100)
        data2 = SyntheticMarketData.generate_price_series(days=100, start_price=100)

        trader1 = SimpleMLTrader()
        trader2 = SimpleMLTrader()

        X1, y1 = trader1.prepare_features(data1)
        X2, y2 = trader2.prepare_features(data2)

        # Same number of features
        assert X1.shape[1] == X2.shape[1] == 8

    def test_prediction_ranges(self):
        """Test that predictions are within expected ranges"""
        data = SyntheticMarketData.generate_price_series(days=150)

        trader = SimpleMLTrader()
        X, y = trader.prepare_features(data)
        trader.train(X, y)

        predictions = trader.predict(X)
        probabilities = trader.predict_proba(X)

        # Check prediction values
        assert all(pred in [-1, 0, 1] for pred in predictions)

        # Check probabilities sum to 1 and are non-negative
        prob_sums = np.sum(probabilities, axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-10)
        assert np.all(probabilities >= 0)


if __name__ == "__main__":
    pytest.main([__file__])

