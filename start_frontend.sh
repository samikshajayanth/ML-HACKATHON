#!/usr/bin/env bash
# Start PathSnap Streamlit UI on port 8501
set -e
echo "Starting PathSnap Streamlit Frontend on http://localhost:8501 ..."
.venv/bin/streamlit run frontend/app.py --server.port 8501 --server.headless false
