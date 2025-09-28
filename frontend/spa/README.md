AlgoTrendy SPA (prototype)

This folder contains a minimal Vite + React scaffold to build a single-page app that consumes the AlgoTrendy frontend API.

Quick start (requires Node.js >= 18):

```powershell
cd frontend/spa
npm install
npm run dev
```

The dev server runs on http://localhost:5173 by default and will call the backend proxy endpoints served from the FastAPI frontend on http://localhost:5000 (same host, different port). CORS is enabled in the FastAPI frontend for local development.
