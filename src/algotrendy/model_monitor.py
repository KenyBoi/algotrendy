"""Model monitoring utilities for AlgoTrendy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import RESULTS_DIR


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class ModelMonitor:
    """Lightweight persistence layer for model metrics and drift checks."""

    def __init__(self, metrics_file: Optional[Path] = None) -> None:
        self.metrics_dir = RESULTS_DIR / "monitoring"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = metrics_file or self.metrics_dir / "model_metrics.json"
        self._state: Dict[str, List[Dict[str, Any]]] = {"training": [], "backtest": []}
        self._load()

    # ------------------------------------------------------------------
    def record_training(self, symbol: str, metrics: Dict[str, Any], asset_type: str = "stock") -> None:
        entry = {
            "type": "training",
            "symbol": symbol,
            "asset_type": asset_type,
            "timestamp": _now_iso(),
            "metrics": metrics,
        }
        self._append("training", entry)

    # ------------------------------------------------------------------
    def record_backtest(self, symbol: str, results: Dict[str, Any], asset_type: str = "stock") -> None:
        entry = {
            "type": "backtest",
            "symbol": symbol,
            "asset_type": asset_type,
            "timestamp": _now_iso(),
            "metrics": results,
        }
        self._append("backtest", entry)

    # ------------------------------------------------------------------
    def get_recent_training(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        return [
            entry for entry in reversed(self._state.get("training", []))
            if entry["symbol"] == symbol
        ][:limit]

    # ------------------------------------------------------------------
    def check_accuracy_drift(
        self,
        symbol: str,
        current_accuracy: float,
        window: int = 5,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Simple drift heuristic comparing accuracy against recent history."""
        history = self.get_recent_training(symbol, limit=window)
        if not history:
            return {"drift_detected": False, "baseline_accuracy": None}

        baseline = sum(entry["metrics"].get("test_accuracy", 0.0) for entry in history) / len(history)
        drift = baseline - current_accuracy >= threshold
        return {
            "drift_detected": drift,
            "baseline_accuracy": baseline,
            "delta": current_accuracy - baseline,
        }

    # ------------------------------------------------------------------
    def _append(self, bucket: str, entry: Dict[str, Any]) -> None:
        self._state.setdefault(bucket, []).append(entry)
        self._save()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.metrics_file.exists():
            return
        try:
            data = json.loads(self.metrics_file.read_text())
            if isinstance(data, dict):
                self._state.update({k: list(v) for k, v in data.items() if isinstance(v, list)})
        except json.JSONDecodeError:
            # If the file is corrupted we start fresh but keep a backup copy
            backup = self.metrics_file.with_suffix(".bak")
            self.metrics_file.rename(backup)

    # ------------------------------------------------------------------
    def _save(self) -> None:
        self.metrics_file.write_text(json.dumps(self._state, indent=2))
