import pytest

# Try to import the AdvancedMLTrainer. Some environments (CI or lightweight dev
# machines) may not have heavy optional dependencies like pandas installed.
# If import fails, tests that depend on this fixture will be skipped with a
# clear message instead of causing an import-time error.
try:
    from algotrendy.advanced_ml_trainer import AdvancedMLTrainer
    _conftest_import_error = None
except Exception as _conftest_import_err:
    AdvancedMLTrainer = None
    _conftest_import_error = _conftest_import_err


@pytest.fixture
def trainer():
    """Fixture to provide AdvancedMLTrainer for tests that expect a 'trainer' fixture.

    If the real trainer can't be imported due to missing optional deps, the
    fixture will skip the test with a helpful message.
    """
    if AdvancedMLTrainer is None:
        pytest.skip(f"AdvancedMLTrainer unavailable; skipping test (import error: {_conftest_import_error})")
    return AdvancedMLTrainer(symbol="TEST")
