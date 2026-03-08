"""
models/gap_engine.py
---------------------
Research Gap Detection Engine using BERTopic + rule-based dimension analysis.

Strategy:
  1. Run BERTopic on abstracts of retrieved similar papers to find covered topics.
  2. Compare covered topics against a taxonomy of known research dimensions
     (region, population, method, theme, time period).
  3. Identify dimensions that appear rarely or not at all → research gaps.
  4. Generate specific, actionable gap statements.
"""

import logging
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Dimension taxonomies for Indian Arts & Science
# ─────────────────────────────────────────────

REGION_TAXONOMY = [
    "Tamil Nadu", "Kerala", "Maharashtra", "Uttar Pradesh", "West Bengal",
    "Karnataka", "Andhra Pradesh", "Rajasthan", "Gujarat", "Punjab",
    "Odisha", "Assam", "Bihar", "Telangana", "Madhya Pradesh",
    "North-East India", "tribal regions", "coastal communities",
    "rural", "urban", "peri-urban", "hilly regions"
]

POPULATION_TAXONOMY = [
    "women", "adolescents", "elderly", "scheduled castes", "scheduled tribes",
    "OBC", "migrant workers", "farmers", "teachers", "college students",
    "school dropouts", "self-help groups", "informal workers",
    "fisherfolk", "artisans", "transgender", "disabled persons",
    "children", "youth", "widows"
]

METHOD_TAXONOMY = [
    "qualitative", "quantitative", "mixed methods", "longitudinal",
    "ethnographic", "case study", "experimental", "quasi-experimental",
    "grounded theory", "action research", "discourse analysis",
    "content analysis", "meta-analysis", "systematic review",
    "participatory", "focus group", "interview-based", "survey-based"
]

TIME_TAXONOMY = [
    "pre-independence", "post-independence", "colonial", "1947-1990",
    "1991-2000", "2000-2010", "2010-2020", "post-2020", "COVID-19 period",
    "longitudinal", "historical"
]

THEME_TAXONOMY = [
    "livelihood", "social mobility", "caste", "gender", "mental health",
    "digital literacy", "climate change", "agriculture", "education",
    "political participation", "cultural identity", "economic empowerment",
    "health", "migration", "language policy", "religion",
    "environment", "biodiversity", "water", "urbanisation",
    "media", "governance", "poverty", "nutrition", "disability"
]

THEORY_TAXONOMY = [
    "feminist theory", "conflict theory", "postcolonial theory",
    "rational choice", "structuralism", "functionalism",
    "intersectionality", "capability approach", "social capital",
    "human development", "ecological modernisation", "gramsci",
    "foucault", "bourdieu", "giddens"
]


# ─────────────────────────────────────────────
# Text-based dimension scanning
# ─────────────────────────────────────────────

def _scan_dimension(
    texts: List[str],
    taxonomy: List[str],
) -> Dict[str, int]:
    """
    Count how many texts mention each taxonomy item.
    Simple lowercase string matching (fast, interpretable).
    """
    counts: Dict[str, int] = {item.lower(): 0 for item in taxonomy}
    for text in texts:
        text_lower = text.lower()
        for item in taxonomy:
            if item.lower() in text_lower:
                counts[item.lower()] += 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def _find_gaps_in_dimension(
    coverage: Dict[str, int],
    dimension_name: str,
    n_total_papers: int,
    min_coverage: int = 1,
) -> List[str]:
    """
    Return taxonomy items with coverage below min_coverage
    → these are the research gaps.
    """
    gaps = [
        item for item, count in coverage.items()
        if count <= min_coverage
    ]
    return gaps[:8]   # limit to top 8 gaps per dimension


# ─────────────────────────────────────────────
# BERTopic-based topic modelling
# ─────────────────────────────────────────────

def run_bertopic(
    texts: List[str],
    min_topic_size: int = 2,
    n_gram_range: Tuple[int, int] = (1, 2),
) -> Tuple[List[str], Dict]:
    """
    Run BERTopic on a list of texts to discover latent topics.

    Returns:
        topic_keywords : list of representative keywords per topic
        topic_info     : raw BERTopic topic_info dict
    """
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            ngram_range=n_gram_range,
            stop_words="english",
            min_df=1
        )
        topic_model = BERTopic(
            vectorizer_model=vectorizer,
            min_topic_size=min_topic_size,
            verbose=False,
            calculate_probabilities=False,
        )

        if len(texts) < 3:
            logger.warning("Too few texts for BERTopic — using fallback.")
            return _fallback_topic_keywords(texts), {}

        topics, _ = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()

        topic_keywords = []
        for topic_id in topic_info["Topic"].tolist():
            if topic_id == -1:   # outlier cluster
                continue
            kw_list = topic_model.get_topic(topic_id)
            if kw_list:
                top_words = [w for w, _ in kw_list[:5]]
                topic_keywords.append(", ".join(top_words))

        logger.info(f"BERTopic found {len(topic_keywords)} topics in corpus.")
        return topic_keywords, topic_info.to_dict()

    except ImportError:
        logger.warning("BERTopic not installed — using TF-IDF fallback.")
        return _fallback_topic_keywords(texts), {}
    except Exception as e:
        logger.error(f"BERTopic error: {e} — using fallback.")
        return _fallback_topic_keywords(texts), {}


def _fallback_topic_keywords(texts: List[str]) -> List[str]:
    """
    Simple TF-IDF-based keyword extraction when BERTopic is unavailable.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not texts:
            return []
        vec = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2))
        vec.fit(texts)
        return list(vec.get_feature_names_out())[:10]
    except Exception:
        return []


# ─────────────────────────────────────────────
# Master gap detection pipeline
# ─────────────────────────────────────────────

def detect_research_gaps(
    retrieved_papers: List[Dict[str, Any]],
    user_title: str = "",
    user_keywords: str = "",
) -> Dict[str, Any]:
    """
    Full research gap detection pipeline.

    Args:
        retrieved_papers : Top-k papers from FAISS retrieval.
        user_title       : User's proposed title.
        user_keywords    : User's keywords.

    Returns:
        Dict with gap analysis dimensions, covered topics, and gap list.
    """
    if not retrieved_papers:
        return {
            "covered_topics":   [],
            "gap_dimensions":   {},
            "gap_statements":   ["No similar papers found — this appears to be a completely new area."],
            "bertopic_topics":  [],
        }

    # Collect texts for analysis
    abstracts  = [p.get("abstract", "") for p in retrieved_papers]
    titles     = [p.get("title", "")    for p in retrieved_papers]
    all_texts  = [f"{t} {a}" for t, a in zip(titles, abstracts)]

    # ── Dimension coverage analysis ──────────
    region_cov   = _scan_dimension(all_texts, REGION_TAXONOMY)
    pop_cov      = _scan_dimension(all_texts, POPULATION_TAXONOMY)
    method_cov   = _scan_dimension(all_texts, METHOD_TAXONOMY)
    theme_cov    = _scan_dimension(all_texts, THEME_TAXONOMY)
    theory_cov   = _scan_dimension(all_texts, THEORY_TAXONOMY)
    time_cov     = _scan_dimension(all_texts, TIME_TAXONOMY)

    n = len(retrieved_papers)

    # ── Find gaps in each dimension ──────────
    region_gaps  = _find_gaps_in_dimension(region_cov,  "Region",     n)
    pop_gaps     = _find_gaps_in_dimension(pop_cov,     "Population", n)
    method_gaps  = _find_gaps_in_dimension(method_cov,  "Method",     n)
    theme_gaps   = _find_gaps_in_dimension(theme_cov,   "Theme",      n)
    theory_gaps  = _find_gaps_in_dimension(theory_cov,  "Theory",     n)
    time_gaps    = _find_gaps_in_dimension(time_cov,    "Time",       n)

    # ── BERTopic latent topic discovery ─────
    topic_keywords, _ = run_bertopic(abstracts)

    # ── Covered topics (top mentioned) ──────
    all_coverage = {
        **{k: v for k, v in region_cov.items() if v > 0},
        **{k: v for k, v in pop_cov.items()    if v > 0},
        **{k: v for k, v in theme_cov.items()  if v > 0},
    }
    covered_topics = sorted(all_coverage, key=all_coverage.get, reverse=True)[:10]

    # ── Generate natural language gap statements ──
    gap_statements = _generate_gap_statements(
        region_gaps, pop_gaps, method_gaps, theme_gaps, theory_gaps, time_gaps,
        user_title, user_keywords
    )

    return {
        "covered_topics":   covered_topics,
        "bertopic_topics":  topic_keywords[:6],
        "gap_dimensions": {
            "regional_gaps":    region_gaps[:5],
            "population_gaps":  pop_gaps[:5],
            "methodological_gaps": method_gaps[:5],
            "thematic_gaps":    theme_gaps[:5],
            "theoretical_gaps": theory_gaps[:4],
            "temporal_gaps":    time_gaps[:4],
        },
        "dimension_coverage": {
            "regions":     dict(list(region_cov.items())[:8]),
            "populations": dict(list(pop_cov.items())[:8]),
            "methods":     dict(list(method_cov.items())[:8]),
        },
        "gap_statements": gap_statements,
    }


def _generate_gap_statements(
    region_gaps, pop_gaps, method_gaps, theme_gaps, theory_gaps, time_gaps,
    user_title: str, user_keywords: str
) -> List[str]:
    """Build readable, specific research gap statements."""
    statements = []

    if region_gaps:
        rg = region_gaps[:2]
        statements.append(
            f"📍 Regional Gap: The literature largely overlooks "
            f"{' and '.join(rg)}. Studies specific to these regions are scarce."
        )

    if pop_gaps:
        pg = pop_gaps[:2]
        statements.append(
            f"👥 Population Gap: Research involving "
            f"{' and '.join(pg)} remains underrepresented. "
            "These groups warrant focused investigation."
        )

    if method_gaps:
        mg = method_gaps[:2]
        statements.append(
            f"🔬 Methodological Gap: Approaches such as "
            f"{' and '.join(mg)} have rarely been applied in this domain, "
            "offering an opportunity for methodological innovation."
        )

    if theme_gaps:
        tg = theme_gaps[:2]
        statements.append(
            f"💡 Thematic Gap: Topics related to "
            f"{' and '.join(tg)} are conspicuously absent from existing literature."
        )

    if theory_gaps:
        theo = theory_gaps[:2]
        statements.append(
            f"📚 Theoretical Gap: Frameworks such as "
            f"{' and '.join(theo)} have not been applied to this research area, "
            "offering a theoretical contribution opportunity."
        )

    if time_gaps:
        tg2 = time_gaps[:2]
        statements.append(
            f"📅 Temporal Gap: The {' and '.join(tg2)} period(s) remain "
            "under-examined. Longitudinal or historical perspectives could add value."
        )

    if not statements:
        statements.append(
            "🔍 Cross-disciplinary Gap: Existing research is concentrated in a "
            "single domain. Interdisciplinary approaches remain unexplored."
        )

    return statements


# ─────────────────────────────────────────────
# Title suggestion engine
# ─────────────────────────────────────────────

def suggest_improved_titles(
    original_title: str,
    gap_dimensions: Dict[str, List[str]],
    n_suggestions: int = 5,
) -> List[str]:
    """
    Generate refined research title suggestions based on detected gaps.
    """
    suggestions = []

    base = original_title.strip().rstrip(".")
    regions     = gap_dimensions.get("regional_gaps", [])
    populations = gap_dimensions.get("population_gaps", [])
    methods     = gap_dimensions.get("methodological_gaps", [])
    themes      = gap_dimensions.get("thematic_gaps", [])

    # Template-based title generation
    templates = [
        lambda: f"{base}: A {_pick(methods, 'Mixed-Methods')} Study in {_pick(regions, 'Rural India')}",
        lambda: f"Exploring {_pick(themes, 'Socio-Economic Challenges')} among {_pick(populations, 'Marginalised Communities')} — Evidence from {_pick(regions, 'South India')}",
        lambda: f"{base}: Perspectives from {_pick(populations, 'Women')} in {_pick(regions, 'North-East India')}",
        lambda: f"A {_pick(methods, 'Longitudinal')} Analysis of {_pick(themes, 'Social Mobility')} in {_pick(regions, 'Tribal Regions')} of India",
        lambda: f"Revisiting {base}: An Interdisciplinary Framework Using {_pick(methods, 'Qualitative')} Methods",
        lambda: f"{_pick(themes, 'Digital Literacy').title()} and {_pick(populations, 'Rural Communities')}: A Post-COVID Perspective from {_pick(regions, 'Tamil Nadu')}",
    ]

    for fn in templates[:n_suggestions]:
        try:
            suggestions.append(fn())
        except Exception:
            continue

    return suggestions


def _pick(lst: List[str], default: str) -> str:
    """Pick first item from list or return default."""
    clean = [x for x in lst if x]
    if clean:
        return clean[0].title()
    return default
