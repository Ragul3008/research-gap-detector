"""
retrieval/faiss_index.py
-------------------------
Build, save, and load a FAISS index for fast nearest-neighbour retrieval.
Supports both flat (exact) and IVF (approximate) indices.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import faiss
import pandas as pd

from config.settings import (
    FAISS_INDEX_PATH, METADATA_PATH,
    EMBEDDING_DIM, USE_IVF, FAISS_NLIST, EMBEDDINGS_DIR
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Index construction
# ─────────────────────────────────────────────

def build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an exact inner-product index (= cosine sim on L2-normalised vecs).
    Best for corpora < 100 000 papers.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    logger.info(f"Built FlatIP index with {index.ntotal} vectors (dim={dim})")
    return index


def build_ivf_index(embeddings: np.ndarray, nlist: int = FAISS_NLIST) -> faiss.IndexIVFFlat:
    """
    Build an IVF index for large corpora (>100 000 papers).
    Requires training before adding vectors.
    """
    dim        = embeddings.shape[1]
    quantizer  = faiss.IndexFlatIP(dim)
    index      = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings.astype("float32"))
    index.add(embeddings.astype("float32"))
    index.nprobe = max(1, nlist // 10)
    logger.info(f"Built IVFFlat index: {index.ntotal} vectors, nlist={nlist}")
    return index


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Select index type based on corpus size and settings."""
    if USE_IVF and len(embeddings) > 1000:
        return build_ivf_index(embeddings)
    return build_flat_index(embeddings)


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────

def save_index(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
) -> None:
    """Persist FAISS index and paper metadata to disk."""
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Index saved → {index_path}")
    logger.info(f"Metadata saved → {metadata_path} ({len(metadata)} records)")


def load_index(
    index_path: Path = FAISS_INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load FAISS index and paper metadata from disk."""
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run scripts/build_index.py first."
        )
    index    = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    logger.info(f"Index loaded from {index_path} ({index.ntotal} vectors)")
    return index, metadata


# ─────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────

def search_index(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    query_vector: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search the FAISS index and return top-k similar papers.

    Args:
        index        : Loaded FAISS index.
        metadata     : List of paper metadata dicts.
        query_vector : shape (1, dim) float32 numpy array.
        top_k        : Number of results to return.

    Returns:
        List of dicts with paper info + 'similarity_score'.
    """
    query_vector = query_vector.astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:           # FAISS returns -1 for empty slots
            continue
        paper = metadata[idx].copy()
        # Inner product on L2-normalised vecs == cosine similarity
        paper["similarity_score"] = float(np.clip(dist, 0.0, 1.0))
        results.append(paper)

    return results
