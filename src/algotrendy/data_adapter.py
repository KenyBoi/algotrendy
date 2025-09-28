"""Data adapter shim for AlgoTrendy.

This module provides a thin wrapper around the existing DataManager so the
rest of the codebase can depend on an adapter interface while the underlying
implementation is refactored or swapped later.

The wrapper intentionally mirrors DataManager's public API without changing
behavior.
"""
from typing import List, Dict, Any


class DataAdapter:
    """Thin adapter that delegates to DataManager.

    Use this class when you want to program against an adapter interface
    instead of directly instantiating the heavy DataManager implementation.
    """

    def __init__(self):
        # Delay importing/instantiating the potentially heavy DataManager
        # until a method is called. This keeps module import light-weight
        # and suitable for environments that don't have optional deps.
        self._dm = None

    def _ensure_dm(self):
        if self._dm is None:
            # Local import to avoid import-time dependency on pandas, yfinance, etc.
            from .data_manager import DataManager

            self._dm = DataManager()

    def fetch_data(self, symbol: str, period: str = "2y", interval: str = "1d",
                   asset_type: str = "stock", chart_style: str = "time"):
        self._ensure_dm()
        return self._dm.fetch_data(symbol, period=period, interval=interval,
                                   asset_type=asset_type, chart_style=chart_style)

    def fetch_futures_data(self, symbol: str, period: str = "60d", interval: str = "5m"):
        self._ensure_dm()
        return self._dm.fetch_futures_data(symbol, period=period, interval=interval)

    def prepare_dataset(self, symbol: str, period: str = "2y", interval: str = "1d",
                        asset_type: str = "stock", chart_style: str = "time"):
        self._ensure_dm()
        return self._dm.prepare_dataset(symbol, period=period, interval=interval,
                                        asset_type=asset_type, chart_style=chart_style)

    def prepare_futures_dataset(self, symbol: str, period: str = "60d", interval: str = "5m",
                                chart_style: str = "time"):
        self._ensure_dm()
        return self._dm.prepare_futures_dataset(symbol, period=period, interval=interval,
                                                chart_style=chart_style)

    def calculate_technical_indicators(self, df, asset_type: str = "stock"):
        self._ensure_dm()
        return self._dm.calculate_technical_indicators(df, asset_type=asset_type)

    def create_features(self, df, asset_type: str = "stock",
                        lookback_periods: List[int] = None):
        self._ensure_dm()
        return self._dm.create_features(df, asset_type=asset_type, lookback_periods=lookback_periods)

    def create_targets(self, df, prediction_horizon: int = None,
                       profit_threshold: float = None, asset_type: str = "stock"):
        self._ensure_dm()
        return self._dm.create_targets(df, prediction_horizon=prediction_horizon,
                                       profit_threshold=profit_threshold, asset_type=asset_type)

    def get_futures_contract_info(self, symbol: str) -> Dict[str, Any]:
        self._ensure_dm()
        return self._dm.get_futures_contract_info(symbol)

    def get_active_futures_contracts(self) -> List[str]:
        self._ensure_dm()
        return self._dm.get_active_futures_contracts()

__all__ = ["DataAdapter"]
