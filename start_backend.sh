#!/usr/bin/env bash
# Start PathSnap FastAPI Backend on port 8000
set -e
echo "Starting PathSnap FastAPI Backend on http://0.0.0.0:8000 ..."
.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
