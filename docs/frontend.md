# Frontend Conventions

This document outlines the lightweight UI stack now used by the AlgoTrendy frontend.

## Technology

- **HTMX** (`https://unpkg.com/htmx.org@1.9.12`) for server-driven interactivity
  - Add `hx-get`, `hx-trigger`, and `hx-target` attributes to HTML elements to fetch partial templates from FastAPI.
  - HTMX responses live under `frontend/templates/partials/`.
- **Tailwind CSS** (`https://cdn.tailwindcss.com`) for utility-based styling layered on top of existing CSS.
- **FastAPI + Jinja2** remains the base web framework.

## Adding a New HTMX Panel

1. Create a partial template in `frontend/templates/partials/`.
2. Add an endpoint in `frontend/app.py` that returns the partial via `templates.TemplateResponse`.
3. Drop an element with `hx-get="/htmx/<route>"` into `dashboard.html` (or another page).
4. Optionally add `hx-trigger="every 30s"` or similar for polling.

## Styling Guidelines

- Prefer Tailwind utility classes for new components (e.g., `bg-white shadow rounded-lg`).
- Existing legacy styles (in `/static/css`) can remain until migrated.
- When mixing Tailwind with legacy CSS, scope new sections with fresh class names to avoid conflicts.

## Live Example

The “Live Model Snapshot” card on the dashboard demonstrates the pattern:

- HTML container in `dashboard.html` with HTMX attributes.
- Partial template `partials/models_table.html` for the table markup.
- Endpoint `/htmx/models` in `frontend/app.py` fetching live model data and returning the partial.

Follow the same approach for future widgets (performance charts, alerts, etc.).
