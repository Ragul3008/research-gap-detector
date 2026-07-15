FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gap_detector

# System dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 gcc g++ curl postgresql postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed data/embeddings

EXPOSE 7860

# Start script: PostgreSQL → seed → index → FastAPI (background) → Streamlit (foreground)
COPY <<'EOF' /app/start.sh
#!/bin/bash
set -e

# Start PostgreSQL
service postgresql start
su - postgres -c "psql -c \"ALTER USER postgres PASSWORD 'postgres';\"" 2>/dev/null || true
su - postgres -c "psql -c \"CREATE DATABASE gap_detector;\"" 2>/dev/null || true

# Seed data and build FAISS index
python scripts/seed_data.py
python scripts/build_index.py

# Start FastAPI in background on port 8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
sleep 5

# Start Streamlit frontend on port 7860
python -m streamlit run frontend/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
EOF

RUN chmod +x /app/start.sh

CMD ["/bin/bash", "/app/start.sh"]
