import importlib

# Smoke test to ensure core package modules can be imported without optional heavy deps
# This prevents accidental regressions where top-level imports require pandas/numpy
MODULES_TO_CHECK = [
    "algotrendy",
    "algotrendy.data_manager",
    "algotrendy.ai_indicator_agent",
    "algotrendy.ai_futures_strategy_agent",
    "algotrendy.ai_crypto_strategy_agent",
    "algotrendy.backtester",
    "algotrendy.market_replay",
    "algotrendy.futures_contract_rolling",
    "algotrendy.feature_analysis",
    "algotrendy.quantconnect_integration",
]


def test_core_modules_import_without_pandas():
    for mod in MODULES_TO_CHECK:
        # Import will raise and fail the test if the module requires heavy deps at import-time
        importlib.import_module(mod)
