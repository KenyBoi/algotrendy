This folder contains a small, fast smoke test suite that does not import any heavy ML or optional dependencies.

Usage:

# Activate your venv and run pytest for just this folder
& .\.venv\Scripts\Activate.ps1; pytest -q isolated_tests

These tests exercise only the frontend FastAPI app endpoints (SPA discovery and proxy close/reopen) so they run quickly in minimal environments.
