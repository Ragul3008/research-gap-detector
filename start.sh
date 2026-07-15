#!/bin/bash
set -e

echo "=== Starting AI Research Novelty & Gap Detector ==="

# Seed data and build FAISS index
echo "[1/3] Seeding paper data..."
python scripts/seed_data.py

echo "[2/3] Building FAISS index..."
python scripts/build_index.py

# Start FastAPI in background on port 8000
echo "[3/3] Starting FastAPI backend on port 8000..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "API is ready!"
        break
    fi
    sleep 2
done

# Start Streamlit frontend on port 7860
echo "Starting Streamlit frontend on port 7860..."
exec python -m streamlit run frontend/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
