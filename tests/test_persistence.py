import os
import tempfile
import numpy as np
import pytest

try:
    from examples.simple_trader import SimpleMLTrader, SyntheticMarketData
except Exception as e:
    pytest.skip(f"Skipping persistence tests due to import error: {e}", allow_module_level=True)


def test_joblib_persistence(tmp_path):
    trader = SimpleMLTrader()
    data = SyntheticMarketData.generate_price_series(days=200)
    X, y = trader.prepare_features(data)
    trader.train(X, y)

    base = str(tmp_path / "persist_test_model")
    # Save using new API
    trader.save_model(base)

    # Load into a fresh instance
    new_trader = SimpleMLTrader()
    new_trader.load_model(base)

    # Sanity check: predictions should be available
    preds = new_trader.predict(X[:5])
    assert len(preds) == 5
