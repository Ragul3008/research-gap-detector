"""
retrieval/retriever.py
-----------------------
High-level RAG retriever that implements hybrid retrieval:
  - FAISS (dense semantic search)
  - BM25 (sparse keyword search)
  - Reciprocal Rank Fusion (RRF) to merge lists
  - Cross-Encoder re-ranking for accuracy (CUDA supported)
"""

import logging
import re
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from models.embedding_model import encode_query
from retrieval.faiss_index import load_index, search_index
from utils.preprocessing import preprocess_query
from config.settings import TOP_K_RESULTS, EMBEDDINGS_DIR

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Cross-Encoder Singleton
# ─────────────────────────────────────────────
_cross_encoder_instance: Optional[CrossEncoder] = None

def get_cross_encoder() -> CrossEncoder:
    """Returns the global CrossEncoder instance for re-ranking."""
    global _cross_encoder_instance
    if _cross_encoder_instance is None:
        from config.settings import CROSS_ENCODER_MODEL_NAME
        logger.info(f"Loading CrossEncoder model: {CROSS_ENCODER_MODEL_NAME}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device for CrossEncoder: {device}")
        _cross_encoder_instance = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device)
        logger.info("CrossEncoder model loaded successfully.")
    return _cross_encoder_instance


# ─────────────────────────────────────────────
# Tokenizer helper
# ─────────────────────────────────────────────
def tokenize_text(text: str) -> List[str]:
    """Simple regex word tokenizer for BM25 search."""
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


# ─────────────────────────────────────────────
# Retriever Implementation
# ─────────────────────────────────────────────
class ResearchRetriever:
    """
    Stateful retriever: loads FAISS index, metadata, and embeddings.npy once
    and serves hybrid queries with Cross-Encoder re-ranking.
    """

    def __init__(self):
        logger.info("Initialising ResearchRetriever with Hybrid Search …")
        self.index, self.metadata = load_index()
        
        # Load raw embeddings if available to compute exact cosine similarity for BM25 results
        self.embeddings = None
        emb_path = EMBEDDINGS_DIR / "embeddings.npy"
        if emb_path.exists():
            try:
                self.embeddings = np.load(emb_path)
                logger.info(f"Loaded raw embeddings matrix of shape {self.embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed to load raw embeddings: {e}")
        
        # Create mapping of paper key to index
        self.paper_to_index = {}
        for idx, p in enumerate(self.metadata):
            key = str(p.get("id", p.get("title", "")))
            self.paper_to_index[key] = idx

        # Build BM25 index
        self.bm25 = None
        if self.metadata:
            logger.info("Building BM25 index over corpus …")
            try:
                from rank_bm25 import BM25Okapi
                bm25_corpus = []
                for p in self.metadata:
                    combined = f"{p.get('title','')} {p.get('abstract','')} {p.get('keywords','')}"
                    bm25_corpus.append(tokenize_text(combined))
                self.bm25 = BM25Okapi(bm25_corpus)
                logger.info("BM25 index built successfully.")
            except ImportError:
                logger.warning("rank_bm25 not installed. Sparse search disabled.")
            except Exception as e:
                logger.error(f"Failed to build BM25 index: {e}")

    def retrieve(
        self,
        title: str,
        abstract: Optional[str] = "",
        keywords: Optional[str] = "",
        top_k: int = TOP_K_RESULTS,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval pipeline.
        1. Query dense search (FAISS)
        2. Query sparse search (BM25)
        3. Reciprocal Rank Fusion (RRF)
        4. Cross-Encoder re-ranking
        5. Exact cosine similarity alignment
        """
        corpus_size = len(self.metadata)
        if corpus_size == 0:
            return []

        # Target retrieval size for fusion step
        m = min(50, corpus_size)

        # 1. Preprocess & Dense FAISS Retrieval
        combined_text = preprocess_query(title, abstract, keywords)
        query_vector = encode_query(combined_text)  # shape (1, dim)
        faiss_results = search_index(self.index, self.metadata, query_vector, top_k=m)

        # 2. Sparse BM25 Retrieval
        query_tokens = tokenize_text(combined_text)
        bm25_results = []
        if self.bm25 and query_tokens:
            bm25_scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(bm25_scores)[::-1][:m]
            for idx in top_indices:
                paper = self.metadata[idx].copy()
                paper["bm25_score"] = float(bm25_scores[idx])
                bm25_results.append(paper)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        rrf_scores = {}
        key_to_paper = {}

        def get_paper_key(p: Dict[str, Any]) -> str:
            return str(p.get("id", p.get("title", "")))

        for rank, p in enumerate(faiss_results):
            key = get_paper_key(p)
            key_to_paper[key] = p
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, p in enumerate(bm25_results):
            key = get_paper_key(p)
            key_to_paper[key] = p
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)

        fused_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        fused_candidates = [key_to_paper[k] for k in fused_keys[:25]]

        # Ensure all candidate papers have exact cosine similarity computed
        for paper in fused_candidates:
            key = get_paper_key(paper)
            idx = self.paper_to_index.get(key)
            if idx is not None and self.embeddings is not None:
                paper["similarity_score"] = float(np.clip(np.dot(query_vector[0], self.embeddings[idx]), 0.0, 1.0))
            else:
                paper["similarity_score"] = paper.get("similarity_score", 0.0)
            # Add helper percentage key
            paper["similarity_pct"] = round(paper["similarity_score"] * 100, 1)

        # 4. Cross-Encoder Re-ranking
        if fused_candidates:
            try:
                cross_encoder = get_cross_encoder()
                pairs = []
                for doc in fused_candidates:
                    # Input query compared to doc text (concatenated title and abstract)
                    doc_text = f"{doc.get('title','')} {doc.get('abstract','')}"
                    pairs.append([combined_text, doc_text[:1000]])
                
                ce_scores = cross_encoder.predict(pairs)
                for doc, score in zip(fused_candidates, ce_scores):
                    doc["re_rank_score"] = float(score)

                # Re-rank based on Cross-Encoder score descending
                fused_candidates.sort(key=lambda x: x["re_rank_score"], reverse=True)
            except Exception as e:
                logger.error(f"Cross-Encoder re-ranking failed: {e}. Falling back to FAISS ranking.")
                fused_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Return the top_k requested results
        results = fused_candidates[:top_k]
        logger.info(f"Retrieved {len(results)} hybrid re-ranked papers for query: '{title[:60]}…'")
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
