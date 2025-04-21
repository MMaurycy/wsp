#!/bin/sh
set -e                                  # zakończ przy błędzie

echo "Starting Uvicorn server in background..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload &

sleep 5                                 # daj czas na start API

echo "Starting Streamlit server in foreground..."
python -m streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0
