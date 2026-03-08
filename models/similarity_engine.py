"""
models/similarity_engine.py
-----------------------------
Computes fine-grained similarity analysis between the
user query and retrieved papers.
"""

import logging
from typing import List, Dict, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Core similarity calculations
# ─────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.
    (Redundant if vectors are already L2-normalised, but kept for safety.)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_similarity_stats(
    results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Derive aggregate similarity statistics from retrieval results.

    Args:
        results: List of paper dicts, each with 'similarity_score'.

    Returns:
        Dict with keys: max_sim, min_sim, mean_sim, median_sim, top3_mean_sim
    """
    if not results:
        return {
            "max_sim": 0.0, "min_sim": 0.0,
            "mean_sim": 0.0, "median_sim": 0.0,
            "top3_mean_sim": 0.0
        }

    scores = [r["similarity_score"] for r in results]
    top3   = scores[:3]

    return {
        "max_sim":      round(float(np.max(scores)),    4),
        "min_sim":      round(float(np.min(scores)),    4),
        "mean_sim":     round(float(np.mean(scores)),   4),
        "median_sim":   round(float(np.median(scores)), 4),
        "top3_mean_sim":round(float(np.mean(top3)),     4),
    }


def rank_results(
    results: List[Dict[str, Any]],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Return top-N results sorted by similarity score (desc).
    Adds a human-readable 'rank' field.
    """
    sorted_results = sorted(
        results, key=lambda x: x["similarity_score"], reverse=True
    )[:top_n]

    for i, r in enumerate(sorted_results, start=1):
        r["rank"] = i
        # Convert score to percentage for display
        r["similarity_pct"] = round(r["similarity_score"] * 100, 1)

    return sorted_results


def find_duplicate_risk(
    results: List[Dict[str, Any]],
    threshold: float = 0.90
) -> List[Dict[str, Any]]:
    """
    Flag papers with similarity score above the duplication threshold.
    Returns a list of near-duplicate papers (empty if none found).
    """
    return [
        r for r in results
        if r["similarity_score"] >= threshold
    ]


def summarise_domain_overlap(
    results: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Count how many retrieved papers belong to each domain.
    Useful for understanding the existing research landscape.
    """
    domain_counts: Dict[str, int] = {}
    for r in results:
        domain = r.get("domain", "Unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    # Sort by count descending
    return dict(
        sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    )


def summarise_year_distribution(
    results: List[Dict[str, Any]]
) -> Dict[int, int]:
    """Return year → count mapping for retrieved papers."""
    year_counts: Dict[int, int] = {}
    for r in results:
        year = int(r.get("year", 0))
        year_counts[year] = year_counts.get(year, 0) + 1
    return dict(sorted(year_counts.items()))
