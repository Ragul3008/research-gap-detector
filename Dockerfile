FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# System dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 gcc g++ \
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

# Start app — seed + index built at runtime, not build time
CMD python scripts/seed_data.py && \
    python scripts/build_index.py && \
    python -m streamlit run app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
