Priority Issues — AlgoTrendy

Top Priority (blockers)
- Data representation mismatch between `DataManager` (pandas DataFrame) and `SimpleMLTrader` (dict/np arrays).
  - Impact: Causes empty/invalid feature arrays leading to sklearn failures during training and test failures.
  - Action: Implement an adapter or refactor the trader to accept DataFrames.

- Tests failing in `tests/test_ml_models.py` due to "at least one array or dtype is required" and probability-sum assertions.
  - Impact: CI is red and developer workflow is blocked.
  - Action: Fix data pipeline and add test coverage for edge cases.

High Priority
- Redis/asyncio import errors in `src/test_ai_orchestrator.py` (environment-dependent tests).
  - Action: Isolate or mock redis in tests or provide a test-only stub. Add tox/CI matrix to manage optional services.

- Model persistence uses `pickle` (security & cross-version compatibility concern).
  - Action: Switch to `joblib` for model objects and store metadata in JSON.

Medium Priority
- Logging uses emojis and caused encoding errors on Windows consoles.
  - Action: Make emojis optional and ensure console handler uses UTF-8 fallback.

- No pinned `requirements.txt` or CI workflow.
  - Action: Create `requirements.txt` and a GitHub Actions workflow to run tests on push.

Low Priority / Nice-to-have
- Add type hints and run `mypy` for modules where API stability matters.
- Add more unit tests for backtester P&L, futures margin flows, and edge cases.
- Replace ad-hoc caching/pickle artifacts with a small cache module and documented structure.

Suggested owners & ETA
- You (repo owner): Confirm design choice for canonical data representation (DataFrame vs numpy).
- Me: I can implement the DataFrame adapter refactor, add persistence changes, and create `requirements.txt` + CI.
  - ETA for adapter + tests: 1-2 days
  - ETA for persistence + tests: 1 day
  - ETA for CI + requirements: 1 day

How to triage new issues
- Reproduce locally and run `pytest -q` to capture failing tests.
- Add failing test first (red), implement fix (green), keep PR small.
- For environment-dependent tests (redis, external APIs), use mocks or `pytest` markers to skip in CI unless configured.
