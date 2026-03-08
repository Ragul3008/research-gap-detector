"""
scripts/build_index.py
-----------------------
End-to-end data pipeline:
  1. Load raw papers.csv
  2. Preprocess text
  3. Generate embeddings
  4. Build and save FAISS index
  5. Save metadata

Run once before starting the API server:
  python scripts/build_index.py
"""

import sys
import os
import logging
import pickle
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils.preprocessing import preprocess_dataframe
from models.embedding_model import encode_texts
from retrieval.faiss_index import build_index, save_index
from config.settings import (
    PAPERS_CSV, PROCESSED_CSV, PROCESSED_DIR,
    EMBEDDINGS_DIR, BATCH_SIZE
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline():
    start = time.time()
    logger.info("=" * 60)
    logger.info("AI Research Novelty & Gap Detector — Index Build Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Load raw data ────────────────
    if not PAPERS_CSV.exists():
        logger.error(
            f"Raw dataset not found at {PAPERS_CSV}.\n"
            "Run: python scripts/seed_data.py  (to generate dummy data)\n"
            "Or place your papers.csv in data/raw/"
        )
        sys.exit(1)

    df = pd.read_csv(PAPERS_CSV)
    logger.info(f"Loaded {len(df)} papers from {PAPERS_CSV}")

    # ── Step 2: Preprocess ───────────────────
    logger.info("Preprocessing text …")
    df_processed = preprocess_dataframe(df)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_processed.to_csv(PROCESSED_CSV, index=False)
    logger.info(f"Processed data saved → {PROCESSED_CSV}")

    # ── Step 3: Generate embeddings ──────────
    texts = df_processed["combined_text"].tolist()
    logger.info(f"Generating embeddings for {len(texts)} papers …")
    logger.info("(This may take a few minutes on CPU.)")

    embeddings = encode_texts(
        texts,
        batch_size=BATCH_SIZE,
        normalize=True,
        show_progress=True,
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Save raw embeddings for future use
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    np.save(EMBEDDINGS_DIR / "embeddings.npy", embeddings)
    logger.info(f"Embeddings saved → {EMBEDDINGS_DIR / 'embeddings.npy'}")

    # ── Step 4: Build FAISS index ────────────
    logger.info("Building FAISS index …")
    index = build_index(embeddings)

    # ── Step 5: Prepare and save metadata ────
    metadata_cols = [
        "id", "title", "abstract", "keywords",
        "year", "domain", "region", "method", "theme"
    ]
    available_cols = [c for c in metadata_cols if c in df_processed.columns]
    metadata = df_processed[available_cols].to_dict(orient="records")

    save_index(index, metadata)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"✅ Pipeline complete in {elapsed:.1f}s")
    logger.info(f"   Papers indexed: {index.ntotal}")
    logger.info(f"   FAISS index: {os.path.getsize(str(EMBEDDINGS_DIR / 'faiss_index.bin')) / 1024:.1f} KB")
    logger.info("   Ready to start the API server.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
