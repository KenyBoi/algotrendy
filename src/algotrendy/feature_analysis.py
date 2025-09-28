"""Feature importance reporting utilities for AlgoTrendy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from .config import RESULTS_DIR


@dataclass(frozen=True)
class FeatureImportanceSummary:
    symbol: str
    asset_type: str
    generated_at: datetime
    top_features: Any
    coverage: float


class FeatureImportanceReporter:
    """Persist and summarize feature importance for trained models."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or RESULTS_DIR / "reports" / "feature_importance"
        # Defer directory creation until actually needed to avoid import-time
        # filesystem side-effects in tests.
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def build_report(
        self,
        symbol: str,
        asset_type: str,
        feature_importance: Dict[str, float],
        metadata: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if not feature_importance:
            raise ValueError("No feature importance data available")

        import pandas as pd

        df = pd.DataFrame(
            {
                "feature": list(feature_importance.keys()),
                "importance": [float(score) for score in feature_importance.values()],
            }
        )
        total = df["importance"].sum()
        df["importance_pct"] = df["importance"] / total if total else 0.0
        df.sort_values("importance", ascending=False, inplace=True)
        df["rank"] = range(1, len(df) + 1)
        df["cumulative_importance"] = df["importance_pct"].cumsum()
        df["symbol"] = symbol
        df["asset_type"] = asset_type
        df["generated_at"] = datetime.utcnow().replace(microsecond=0)

        if metadata:
            for key, value in metadata.items():
                df[key] = value

        output_path = self.output_dir / f"{symbol.lower()}_feature_importance.csv"
        df.to_csv(output_path, index=False)
        return df

    # ------------------------------------------------------------------
    def summarize(
        self,
        report: pd.DataFrame,
        top_n: int = 5,
    ) -> FeatureImportanceSummary:
        if report.empty:
            raise ValueError("Feature importance report is empty")

        top_features = report.head(top_n).copy()
        coverage = float(top_features["importance_pct"].sum())
        generated_at = report["generated_at"].iloc[0]
        symbol = report["symbol"].iloc[0]
        asset_type = report["asset_type"].iloc[0]

        top_features = top_features[["rank", "feature", "importance", "importance_pct", "cumulative_importance"]]
        return FeatureImportanceSummary(
            symbol=symbol,
            asset_type=asset_type,
            generated_at=generated_at,
            top_features=top_features,
            coverage=coverage,
        )

    # ------------------------------------------------------------------
    def export_summary_markdown(
        self,
        summary: FeatureImportanceSummary,
        destination: Path | None = None,
    ) -> Path:
        destination = destination or (self.output_dir / f"{summary.symbol.lower()}_summary.md")
        try:
            table = summary.top_features.to_markdown(index=False)
        except Exception:
            table = summary.top_features.to_csv(index=False)
        lines = [
            f"# Feature Importance Summary - {summary.symbol}",
            "",
            f"- Asset type: **{summary.asset_type}**",
            f"- Generated at: {summary.generated_at.isoformat()}",
            f"- Coverage (top features): {summary.coverage:.1%}",
            "",
            table,
        ]
        destination.write_text("\n".join(lines), encoding="utf-8")
        return destination

    # ------------------------------------------------------------------
    def bulk_report(
        self,
        records: Iterable[Dict[str, Any]],
    ) -> Dict[str, pd.DataFrame]:
        reports: Dict[str, pd.DataFrame] = {}
        for record in records:
            symbol = record["symbol"]
            asset_type = record.get("asset_type", "stock")
            importance = record.get("importance", {})
            metadata = {k: v for k, v in record.items() if k not in {"symbol", "importance"}}
            reports[symbol] = self.build_report(symbol, asset_type, importance, metadata=metadata)
        return reports
