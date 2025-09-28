"""Automated retraining utilities for AlgoTrendy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List, Optional

from .model_monitor import ModelMonitor

try:
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover - Python <3.8 fallback (not expected)
    TYPE_CHECKING = False

if TYPE_CHECKING:  # pragma: no cover
    from .main import AlgoTrendyApp


class ModelRetrainer:
    """Evaluate trained models and trigger retraining when needed."""

    def __init__(
        self,
        app: 'AlgoTrendyApp',
        drift_threshold: float = 0.05,
        max_hours_between: float = 24.0,
    ) -> None:
        self.app = app
        self.monitor: ModelMonitor = app.monitor
        self.drift_threshold = drift_threshold
        self.max_hours_between = max_hours_between

    # ------------------------------------------------------------------
    def _latest_training_entry(self, symbol: str) -> Optional[Dict[str, Any]]:
        entries = self.monitor.get_recent_training(symbol, limit=1)
        return entries[0] if entries else None

    # ------------------------------------------------------------------
    def _hours_since_last_training(self, symbol: str) -> Optional[float]:
        entry = self._latest_training_entry(symbol)
        if not entry:
            return None
        ts = entry.get('timestamp') or entry.get('trained_at')
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except ValueError:
            return None
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0

    # ------------------------------------------------------------------
    def _drift_flagged(self, symbol: str) -> bool:
        entry = self.app.models.get(symbol)
        if not entry:
            return True
        drift_info = entry.get('drift', {})
        return bool(drift_info.get('drift_detected'))

    # ------------------------------------------------------------------
    def _accuracy_delta_exceeds_threshold(self, symbol: str, current_accuracy: Optional[float]) -> bool:
        if current_accuracy is None:
            return True
        history = self.monitor.get_recent_training(symbol, limit=5)
        if not history:
            return True
        baseline = sum(e['metrics'].get('test_accuracy', 0.0) for e in history) / len(history)
        return (baseline - current_accuracy) >= self.drift_threshold

    # ------------------------------------------------------------------
    def _stale_model(self, symbol: str) -> bool:
        hours = self._hours_since_last_training(symbol)
        return hours is None or hours >= self.max_hours_between

    # ------------------------------------------------------------------
    def evaluate_symbol(
        self,
        symbol: str,
        model_type: str = 'binary',
        force: bool = False,
    ) -> Dict[str, Any]:
        entry = self.app.models.get(symbol)
        metrics = entry.get('metrics', {}) if entry else {}
        current_accuracy = metrics.get('test_accuracy')

        needs_retrain = force or not entry
        if not needs_retrain:
            needs_retrain = (
                self._drift_flagged(symbol)
                or self._accuracy_delta_exceeds_threshold(symbol, current_accuracy)
                or self._stale_model(symbol)
            )

        if needs_retrain:
            result = self.app.train_single_symbol(symbol, model_type=model_type, save_model=True)
            status = 'retrained'
        else:
            result = entry
            status = 'skipped'

        return {
            'symbol': symbol,
            'status': status,
            'result': result,
        }

    # ------------------------------------------------------------------
    def run_cycle(
        self,
        symbols: Iterable[str],
        model_type: str = 'binary',
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        outcomes: List[Dict[str, Any]] = []
        for symbol in symbols:
            outcomes.append(self.evaluate_symbol(symbol, model_type=model_type, force=force))
        return outcomes
