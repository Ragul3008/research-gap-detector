"""
models/novelty_engine.py
-------------------------
Calculates a multi-dimensional Novelty Score for the user's
research idea based on similarity statistics, domain coverage,
recency, and topic uniqueness.
"""

import logging
import math
from typing import List, Dict, Any, Tuple

import numpy as np

from config.settings import (
    NOVELTY_LOW_THRESHOLD,
    NOVELTY_MEDIUM_THRESHOLD
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Score label → numeric mapping
# ─────────────────────────────────────────────
NOVELTY_LABELS = {
    "HIGH":   3,
    "MEDIUM": 2,
    "LOW":    1,
}

NOVELTY_COLORS = {
    "HIGH":   "#2ecc71",   # green
    "MEDIUM": "#f39c12",   # orange
    "LOW":    "#e74c3c",   # red
}

NOVELTY_DESCRIPTIONS = {
    "HIGH": (
        "Your research idea appears to be highly novel. "
        "Very few similar studies exist in the literature. "
        "This topic has strong potential for original contribution."
    ),
    "MEDIUM": (
        "Your research idea shows moderate novelty. "
        "Related work exists, but your specific focus or context "
        "introduces new elements. Refine your scope for stronger originality."
    ),
    "LOW": (
        "Your research idea has a low novelty score. "
        "Highly similar studies already exist. Consider changing the "
        "region, population, method, or theoretical framework."
    ),
}


# ─────────────────────────────────────────────
# Sub-score calculations
# ─────────────────────────────────────────────

def _similarity_subscore(mean_sim: float) -> float:
    """
    Maps average cosine similarity to a [0, 1] novelty sub-score.
    Higher similarity → lower novelty.
    """
    # Invert: low similarity = high novelty
    return 1.0 - float(np.clip(mean_sim, 0.0, 1.0))


def _coverage_subscore(
    results: List[Dict[str, Any]],
    user_domain: str = ""
) -> float:
    """
    How broadly is this topic covered across domains?
    High coverage (many domains) → lower novelty for this specific domain.
    """
    if not results:
        return 1.0  # no results → maximally novel

    domains = [r.get("domain", "") for r in results]
    unique_domains = len(set(d for d in domains if d))

    # If existing papers span many different domains, the idea is broadly
    # explored — reduce novelty slightly.
    # Score: 1.0 if only 1 domain, declining slowly with diversity.
    coverage_penalty = min(unique_domains / 10.0, 0.5)
    return max(0.0, 1.0 - coverage_penalty)


def _recency_subscore(results: List[Dict[str, Any]]) -> float:
    """
    Are the similar papers recent (last 3 years) or old?
    Recent heavy coverage → lower novelty.
    Old coverage → higher novelty (chance to update).
    """
    if not results:
        return 1.0

    from datetime import datetime
    current_year = datetime.now().year
    recent_cutoff = current_year - 3

    recent_count = sum(
        1 for r in results
        if int(r.get("year", 0)) >= recent_cutoff
    )
    recent_ratio = recent_count / len(results)

    # Many recent papers → lower novelty
    return 1.0 - (recent_ratio * 0.4)   # max 40% penalty for recency


def _top_similarity_penalty(top_score: float) -> float:
    """
    If the single most similar paper is extremely close (> 0.90),
    apply an additional penalty regardless of averages.
    """
    if top_score >= 0.92:
        return -0.20   # strong deduction
    elif top_score >= 0.85:
        return -0.10
    elif top_score >= 0.75:
        return -0.05
    return 0.0


# ─────────────────────────────────────────────
# Master novelty score
# ─────────────────────────────────────────────

def calculate_novelty_score(
    results: List[Dict[str, Any]],
    similarity_stats: Dict[str, float],
    user_domain: str = "",
) -> Dict[str, Any]:
    """
    Compute a weighted Novelty Score from multiple sub-dimensions.

    Weights:
        - Similarity sub-score   : 50%
        - Coverage sub-score     : 20%
        - Recency sub-score      : 20%
        - Top-sim penalty        : 10% adjustment

    Returns:
        Dict with:
            raw_score    : float [0, 1]
            percentage   : int [0, 100]
            label        : str  HIGH / MEDIUM / LOW
            color        : str  hex colour
            description  : str  human-readable explanation
            sub_scores   : dict breakdown
    """
    if not results:
        # No matches found → maximum novelty
        raw_score = 1.0
    else:
        mean_sim  = similarity_stats.get("mean_sim", 0.0)
        top_score = similarity_stats.get("max_sim",  0.0)

        sim_sub      = _similarity_subscore(mean_sim)
        coverage_sub = _coverage_subscore(results, user_domain)
        recency_sub  = _recency_subscore(results)
        penalty      = _top_similarity_penalty(top_score)

        raw_score = (
            0.50 * sim_sub +
            0.20 * coverage_sub +
            0.20 * recency_sub
        ) + penalty

        raw_score = float(np.clip(raw_score, 0.0, 1.0))

    # Determine label
    # Note: raw_score is NOVELTY (1 - similarity based)
    # HIGH novelty → raw_score > 0.45 (mean_sim < 0.55)
    # MEDIUM novelty → raw_score 0.25-0.45
    # LOW novelty → raw_score < 0.25
    if raw_score > 0.45:
        label = "HIGH"
    elif raw_score >= 0.25:
        label = "MEDIUM"
    else:
        label = "LOW"

    # Translate to a friendly 0-100 percentage
    pct = int(round(raw_score * 100))

    sub_scores = {}
    if results:
        sub_scores = {
            "similarity_sub":  round(_similarity_subscore(similarity_stats.get("mean_sim", 0.0)), 3),
            "coverage_sub":    round(_coverage_subscore(results, user_domain), 3),
            "recency_sub":     round(_recency_subscore(results), 3),
        }

    return {
        "raw_score":   round(raw_score, 4),
        "percentage":  pct,
        "label":       label,
        "color":       NOVELTY_COLORS[label],
        "description": NOVELTY_DESCRIPTIONS[label],
        "sub_scores":  sub_scores,
    }


def get_novelty_suggestions(
    label: str,
    similarity_stats: Dict[str, float],
    results: List[Dict[str, Any]],
) -> List[str]:
    """
    Return actionable suggestions based on novelty level.
    """
    suggestions = []

    if label == "LOW":
        suggestions.extend([
            "🔄 Change the geographical focus (e.g., shift from urban to rural, or a different state).",
            "👥 Target a different population group (e.g., women, tribal communities, elderly).",
            "🛠️ Adopt a different research methodology (e.g., switch from survey to ethnography).",
            "📅 Restrict to a specific time period or use longitudinal design.",
            "🔗 Combine two underexplored domains in your research framework.",
        ])
    elif label == "MEDIUM":
        suggestions.extend([
            "🎯 Narrow your focus to a specific sub-theme for greater originality.",
            "🌍 Emphasise the Indian regional context more explicitly.",
            "📊 Integrate quantitative and qualitative methods for a mixed approach.",
            "💡 Introduce a new theoretical framework from a different discipline.",
        ])
    else:  # HIGH
        suggestions.extend([
            "✅ Your topic is highly original — proceed with confidence.",
            "📖 Strengthen your theoretical framework with recent literature.",
            "🏆 Consider targeting high-impact journals — your topic is publishable.",
            "🤝 Identify interdisciplinary collaborators to strengthen the study.",
        ])

    return suggestions
