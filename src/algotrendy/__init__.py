"""AlgoTrendy core package.

This package intentionally avoids importing heavy submodules at import-time.
Call the small loader helpers below when you need the heavier functionality so
tests and lightweight tooling can import the package without pulling large
optional dependencies or triggering side-effects.
"""

from .config import CONFIG, setup_logging  # noqa: F401


def load_model_monitor():
	"""Lazily import and return the ModelMonitor class.

	Usage:
		ModelMonitor = load_model_monitor()
		mm = ModelMonitor(...)
	"""
	from .model_monitor import ModelMonitor

	return ModelMonitor


def load_feature_importance():
	"""Lazily import and return feature importance helpers."""
	from .feature_analysis import FeatureImportanceReporter, FeatureImportanceSummary

	return FeatureImportanceReporter, FeatureImportanceSummary


def load_retrainer():
	"""Lazily import and return the ModelRetrainer class."""
	from .retraining import ModelRetrainer

	return ModelRetrainer


__all__ = [
	"CONFIG",
	"setup_logging",
	"load_model_monitor",
	"load_feature_importance",
	"load_retrainer",
]

