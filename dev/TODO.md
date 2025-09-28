# Development TODOs

This file tracks development tasks, their status, and notes for future work.

## Backlog / Tabled Items

| ID | Task | Description | Status | Priority | Owner | Notes |
|---:|---|---|---|---:|---|---|
| T-001 | Lazy pandas annotations conversion | Convert remaining `pd.DataFrame` / `pd.Series` type annotations to be import-safe (use `from __future__ import annotations`, `if TYPE_CHECKING`, or `Any` / string annotations). Add a centralized `_get_pd()` helper where helpful. | Tabled | Low | Unassigned | Found during import-hardening work; leave for later as lower priority than QuantConnect and frontend tasks. |

---

How to un-table:

- Move the row from `Tabled` to `In Progress` and assign an Owner.
- Suggested quick plan when resuming: run a repo scan for `pd.DataFrame|pd.Series|-> pd.` and fix files in small batches (2–4 files), add tests to `isolated_tests/` to guard regressions, and run CI.

If you want this added to a different tracker (issues, GitHub project, etc.) tell me where and I can create it there as well.
