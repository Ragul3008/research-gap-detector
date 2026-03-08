"""
retrieval/retriever.py
-----------------------
High-level RAG retriever that glues together:
  embedding_model → faiss_index → ranked results
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np

from models.embedding_model import encode_query
from retrieval.faiss_index import load_index, search_index
from utils.preprocessing import preprocess_query
from config.settings import TOP_K_RESULTS

logger = logging.getLogger(__name__)


class ResearchRetriever:
    """
    Stateful retriever: loads the FAISS index once and
    serves multiple queries efficiently.
    """

    def __init__(self):
        logger.info("Initialising ResearchRetriever …")
        self.index, self.metadata = load_index()
        logger.info(f"Retriever ready — corpus size: {len(self.metadata)}")

    def retrieve(
        self,
        title: str,
        abstract: Optional[str] = "",
        keywords: Optional[str] = "",
        top_k: int = TOP_K_RESULTS,
    ) -> List[Dict[str, Any]]:
        """
        End-to-end retrieval pipeline.

        1. Preprocess input text
        2. Encode to dense vector
        3. Search FAISS index
        4. Return ranked results with metadata

        Args:
            title    : Research title (required).
            abstract : Research abstract (optional).
            keywords : Comma-separated keywords (optional).
            top_k    : Number of results.

        Returns:
            List of paper dicts sorted by similarity_score (descending).
        """
        # Step 1 – preprocess
        combined_text = preprocess_query(title, abstract, keywords)
        logger.debug(f"Query text (truncated): {combined_text[:120]} …")

        # Step 2 – encode
        query_vector = encode_query(combined_text)   # shape: (1, dim)

        # Step 3 – FAISS search
        results = search_index(
            self.index, self.metadata, query_vector, top_k=top_k
        )

        # Step 4 – sort descending (FAISS usually returns sorted, but enforce)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

        logger.info(
            f"Retrieved {len(results)} papers for query: '{title[:60]}…'"
        )
        return results

    def get_corpus_size(self) -> int:
        return len(self.metadata)


# ─────────────────────────────────────────────
# Convenience singleton
# ─────────────────────────────────────────────
_retriever_instance: Optional[ResearchRetriever] = None


def get_retriever() -> ResearchRetriever:
    """Return global singleton retriever (loads index once)."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ResearchRetriever()
    return _retriever_instance
