# Frontend (Vite + React + TypeScript)

Research SPA for the algo-trading stack. Talks to the FastAPI backend in `../api`.

## Stack

- Vite + React 18 + TypeScript
- TanStack Query for HTTP state
- axios for the API client
- Tailwind CSS v3 for styling

## Quickstart

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

The dev server expects the API at `http://localhost:8000`. Override with:

```bash
VITE_API_URL=http://localhost:9000 npm run dev
```

To boot the API (from the repo root):

```bash
uvicorn api.app:app --reload --port 8000
```

## What's wired in Task 1b

- Sidebar: ticker multi-select (from `/tickers`), date range, strategy dropdown
  (from `/strategies`), auto-rendered params form (prefilled from
  `default_params`), Single/Sweep toggle, Run button.
- Run posts to `/backtest` and dumps the JSON response into a `<pre>` in the
  main panel + the browser console.

## Not yet (Task 1c)

- Equity curve / drawdown charts
- Metrics grid
- Sweep results table + heatmap
- Smoke tests (Vitest / Playwright)

## Scripts

- `npm run dev` — start the dev server.
- `npm run build` — type-check then build to `dist/`.
- `npm run preview` — serve the built bundle.
- `npm run lint` — run eslint.
