<<<<<<< HEAD
# ─────────────────────────────────────────────
# Hugging Face Docker Deployment
# AI Research Novelty & Gap Detector v3.0
# For Arts, Science & Engineering PhD Scholars
# ─────────────────────────────────────────────

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    gcc \
    g++ \
=======
FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
>>>>>>> aa4dafb06eb9ac91886b1d5fb981f1a07ae66c3d
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

<<<<<<< HEAD
# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/embeddings

# Generate dataset
RUN python scripts/seed_data.py

# Build FAISS index (this downloads the model — takes a few minutes)
RUN python scripts/build_index.py

# Expose port for Hugging Face
EXPOSE 7860

# Start Streamlit app
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
=======
COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
>>>>>>> aa4dafb06eb9ac91886b1d5fb981f1a07ae66c3d
