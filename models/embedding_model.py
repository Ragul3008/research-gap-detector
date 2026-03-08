"""
models/embedding_model.py
--------------------------
Singleton wrapper around SentenceTransformers.
Handles model loading, batched encoding, and normalisation.
"""

import logging
import numpy as np
from typing import List, Union

from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL_NAME, BATCH_SIZE

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Singleton pattern — one model instance per process
# ─────────────────────────────────────────────
_model_instance: Union[SentenceTransformer, None] = None


def get_model() -> SentenceTransformer:
    """
    Returns the global SentenceTransformer instance.
    Loads from HuggingFace Hub on first call; cached thereafter.
    """
    global _model_instance
    if _model_instance is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")
    return _model_instance


# ─────────────────────────────────────────────
# Encoding functions
# ─────────────────────────────────────────────

def encode_texts(
    texts: List[str],
    batch_size: int = BATCH_SIZE,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Encode a list of text strings into dense vectors.

    Args:
        texts         : List of strings to encode.
        batch_size    : Mini-batch size for encoding.
        normalize     : L2-normalise embeddings (needed for cosine FAISS).
        show_progress : Show tqdm progress bar.

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = get_model()

    # BGE models recommend a query prefix for retrieval
    if "bge" in EMBEDDING_MODEL_NAME.lower():
        texts = [f"Represent this sentence: {t}" for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    logger.debug(f"Encoded {len(texts)} texts → shape {embeddings.shape}")
    return embeddings


def encode_query(
    query_text: str,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode a single query string.
    BGE models use a specific query instruction prefix.

    Returns:
        numpy array of shape (1, embedding_dim)
    """
    model = get_model()

    # BGE query instruction (improves retrieval quality)
    if "bge" in EMBEDDING_MODEL_NAME.lower():
        query_text = f"Represent this sentence for searching relevant passages: {query_text}"

    embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embedding  # shape: (1, dim)
