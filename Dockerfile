# Hugging Face Docker Deployment
# AI Research Novelty & Gap Detector v3.0

FROM python:3.10-slim

WORKDIR /app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TOKENIZERS_PARALLELISM=false

# System deps (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port (HF expects 7860)
EXPOSE 7860

# Run app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]