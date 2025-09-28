Development Plan — AlgoTrendy

Goal: Stabilize the ML + data pipeline, fix failing tests, and prepare the repository for CI and safe paper/live trading. Deliver a clear, prioritized roadmap, small milestones, and actionable tasks.

Scope
- Core modules: DataManager, SimpleMLTrader, Backtester
- Integration: real_alpaca_demo, Alpaca REST integration
- Infrastructure: tests, CI, requirements, logging, model persistence

Milestones
1. Pipeline Unification & Test Fixes (Days 0-2)
   - Objective: Make the data flow deterministic and fix unit tests that fail due to data-format mismatches.
   - Tasks:
     - Refactor `SimpleMLTrader.prepare_features` to accept pandas DataFrame and/or provide an adapter function `df_to_feature_arrays(df)`.
     - Add input validation to `SimpleMLTrader.prepare_features` and `DataManager.prepare_dataset`.
     - Add small unit tests for edge cases (short series, NaN-filled data).
     - Run pytest and fix any other top-level failures.

2. Model Persistence & Reproducibility (Days 2-3)
   - Objective: Replace raw pickle with joblib + JSON metadata; add model versioning.
   - Tasks:
     - Implement `save_model(path)` → writes `model.joblib` + `model_meta.json`.
     - Implement `load_model(path)` counterpart.
     - Add tests for persistence and model roundtrip compatibility.

3. Logging & CLI polish (Days 3-4)
   - Objective: Make logging robust across OSes, add CLI switches to enable/disable emojis and verbose logs.
   - Tasks:
     - Move emoji logs behind `CONFIG.use_emoji` flag.
     - Keep UTF-8 console handler as fallback for Windows.
     - Add `--quiet`, `--verbose` arguments in examples/ demos.

4. CI, Requirements & Packaging (Days 4-6)
   - Objective: Create a reproducible development environment and CI.
   - Tasks:
     - Add `requirements.txt` (pinned) and `dev-requirements.txt` (pytest, flake8, mypy).
     - Add GitHub Actions to run pytest on push for Windows and Ubuntu.
     - Add `.github/workflows/ci.yml`.

5. Safety & Execution (Week 2)
   - Objective: Prepare for automated paper trading and risk management checks.
   - Tasks:
     - Add a dry-run executor and safety checks before any real orders.
     - Add a human-in-the-loop confirmation step for live trading.

How I will work
- Make small, incremental commits for each task and run tests after each change.
- Prefer small, reversible edits (non-breaking) and add tests for new behavior.
- Ask for clarification only when a design choice is ambiguous (e.g., DataFrame vs numpy canonical type).

Success criteria
- All existing tests pass on CI.
- Data pipeline unification (single canonical data type across modules).
- Robust model save/load (joblib + metadata).
- Examples run without Unicode/logging crashes on Windows.

Notes
- Python 3.13 may have third-party package build issues; prefer using wheels/pinned versions in `requirements.txt`.
- Keep secrets out of repo; document how to configure Alpaca keys in README and CI secrets.

6. Frontend Modernization (Week 3)
   - Objective: Upgrade the web UI with modular, server-driven components.
   - Tasks:
     - Integrate HTMX for partial page updates in the FastAPI frontend.
     - Introduce Tailwind CSS for rapid styling.
     - Create an initial HTMX-powered dashboard view (e.g., live model metrics).
     - Document conventions for future UI modules in `docs/frontend.md`.
