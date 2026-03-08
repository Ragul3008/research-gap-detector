"""
config/settings.py
-------------------
Central configuration for the AI Research Novelty & Gap Detector.
All paths, model names, and thresholds are defined here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Base paths
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
EMBEDDINGS_DIR  = DATA_DIR / "embeddings"

PAPERS_CSV      = RAW_DIR / "papers.csv"
PROCESSED_CSV   = PROCESSED_DIR / "papers_processed.csv"
FAISS_INDEX_PATH= EMBEDDINGS_DIR / "faiss_index.bin"
METADATA_PATH   = EMBEDDINGS_DIR / "metadata.pkl"

# ─────────────────────────────────────────────
# Embedding model
# ─────────────────────────────────────────────
# Primary: BAAI/bge-large-en-v1.5 (best quality)
# Fallback: all-MiniLM-L6-v2    (lightweight, fast)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 1024   # bge-large-en dimension; 384 for MiniLM
BATCH_SIZE    = 32

# ─────────────────────────────────────────────
# FAISS retrieval
# ─────────────────────────────────────────────
TOP_K_RESULTS   = 10   # how many similar papers to return
FAISS_NLIST     = 50   # IVF clusters (tune for large corpora)
USE_IVF         = False # set True when corpus > 10 000 papers

# ─────────────────────────────────────────────
# Novelty thresholds (cosine similarity)
# ─────────────────────────────────────────────
NOVELTY_LOW_THRESHOLD    = 0.75   # avg sim > 0.75  → LOW novelty
NOVELTY_MEDIUM_THRESHOLD = 0.55   # avg sim 0.55-0.75 → MEDIUM
# avg sim < 0.55 → HIGH novelty

# ─────────────────────────────────────────────
# BERTopic gap detection
# ─────────────────────────────────────────────
BERTOPIC_MIN_TOPIC_SIZE = 3
BERTOPIC_N_GRAM_RANGE   = (1, 2)
TOP_N_GAP_TOPICS        = 5

# ─────────────────────────────────────────────
# LLM explanation — Google Gemini
# ─────────────────────────────────────────────
# Set GEMINI_API_KEY in your environment or .env file.
# Get a free key at: https://aistudio.google.com
# If not set, the system uses template-based explanation (works offline).
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL          = "gemini-1.5-flash"   # or "gemini-1.5-pro" for higher quality
LLM_MAX_TOKENS     = 1024
USE_LLM_EXPLANATION= bool(GEMINI_API_KEY)

# ─────────────────────────────────────────────
# API settings
# ─────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
