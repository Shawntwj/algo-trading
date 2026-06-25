#!/usr/bin/env bash
# Dev launcher — runs FastAPI (port 8000) and Vite (port 5173) in parallel.
# Ctrl-C tears down both children. See `make dev`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[dev] starting FastAPI on :8000"
uvicorn api.app:app --reload --port 8000 &
API_PID=$!

echo "[dev] starting Vite on :5173"
npm --prefix frontend run dev &
WEB_PID=$!

cleanup() {
  echo
  echo "[dev] shutting down (api=$API_PID web=$WEB_PID)"
  kill "$API_PID" "$WEB_PID" 2>/dev/null || true
  wait "$API_PID" "$WEB_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# macOS ships bash 3.2 (no `wait -n`); poll instead.
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 1
done
echo "[dev] one process exited; tearing down the other"
exit 0
