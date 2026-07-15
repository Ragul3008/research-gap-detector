"""
models/gap_engine.py
---------------------
Research Gap Detection Engine using BERTopic + taxonomy extraction.
Upgraded to v3.1: LLM-assisted extraction, caching, confidence scoring,
BERTopic cross-validation, and auditable gap scoring formulas.
"""

import logging
import re
import json
import hashlib
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    TAXONOMY_CACHE_PATH, GEMINI_API_KEY, LLM_MODEL, USE_LLM_EXPLANATION
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Dimension taxonomies for Indian Arts, Science & Engineering
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
# Cache Setup
# ─────────────────────────────────────────────
_taxonomy_cache: Dict[str, Dict[str, Any]] = {}

def load_taxonomy_cache():
    global _taxonomy_cache
    if TAXONOMY_CACHE_PATH.exists():
        try:
            with open(TAXONOMY_CACHE_PATH, "r", encoding="utf-8") as f:
                _taxonomy_cache = json.load(f)
            logger.info(f"Loaded taxonomy cache with {len(_taxonomy_cache)} entries.")
        except Exception as e:
            logger.error(f"Failed to load taxonomy cache: {e}")
            _taxonomy_cache = {}

def save_taxonomy_cache():
    try:
        TAXONOMY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TAXONOMY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_taxonomy_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save taxonomy cache: {e}")

def get_paper_hash(title: str, abstract: str) -> str:
    content = f"{title.strip().lower()}|||{abstract.strip().lower()}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# Load cache at module import
load_taxonomy_cache()


# ─────────────────────────────────────────────
# Taxonomy Extraction Logic
# ─────────────────────────────────────────────

def extract_taxonomy_with_gemini(title: str, abstract: str) -> Optional[Dict[str, Any]]:
    """Uses Gemini LLM to extract structured dimensions for a paper."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(LLM_MODEL)
        
        prompt = f"""You are a research taxonomy analyzer. Extract the academic dimensions from the following paper.
        
Title: {title}
Abstract: {abstract}

Provide your response strictly as a JSON object containing the keys: "region", "population", "method", "theory", "theme", "time_period".
For each key, output an object with two fields:
- "value": The extracted specific value (e.g., "Kerala", "women", "mixed methods", "capability approach", "digital literacy", "post-2020"). If not specified in the text, output an empty string "".
- "confidence": A float between 0.0 and 1.0 representing how confident you are in the extraction based on the text. If not specified, output 0.0.

Only output the raw JSON, no introductory or explanation text.
"""
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )
        text = response.text.strip()
        if text.startswith("```"):
            newline_idx = text.find("\n")
            if newline_idx != -1:
                text = text[newline_idx+1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
        data = json.loads(text)
        standard_keys = ["region", "population", "method", "theory", "theme", "time_period"]
        result = {}
        for key in standard_keys:
            val_obj = data.get(key, {})
            result[key] = {
                "value": str(val_obj.get("value", "")).strip(),
                "confidence": float(val_obj.get("confidence", 0.0))
            }
        return result
    except Exception as e:
        logger.error(f"Gemini taxonomy extraction failed for '{title[:30]}': {e}")
        return None


def extract_taxonomy_heuristics(title: str, abstract: str) -> Dict[str, Any]:
    """Fallback rule-based scanner using string containment & location weights."""
    title_lower = title.lower()
    abstract_lower = abstract.lower()
    
    result = {}
    taxonomies = {
        "region": REGION_TAXONOMY,
        "population": POPULATION_TAXONOMY,
        "method": METHOD_TAXONOMY,
        "theory": THEORY_TAXONOMY,
        "theme": THEME_TAXONOMY,
        "time_period": TIME_TAXONOMY,
    }
    
    for key, taxonomy in taxonomies.items():
        matched_val = ""
        confidence = 0.0
        
        for item in taxonomy:
            item_lower = item.lower()
            if item_lower in title_lower:
                matched_val = item
                confidence = 0.90  # Found in title
                break
            elif item_lower in abstract_lower:
                matched_val = item
                confidence = 0.60  # Found in abstract
                break
                
        result[key] = {
            "value": matched_val,
            "confidence": confidence
        }
    return result


def get_taxonomy_for_paper(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts paper taxonomy, checking the local on-disk cache first."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    h = get_paper_hash(title, abstract)
    
    if h in _taxonomy_cache:
        return _taxonomy_cache[h]
        
    result = None
    if USE_LLM_EXPLANATION and GEMINI_API_KEY:
        result = extract_taxonomy_with_gemini(title, abstract)
        
    if not result:
        result = extract_taxonomy_heuristics(title, abstract)
        
    _taxonomy_cache[h] = result
    save_taxonomy_cache()
    return result


def cross_validate_with_bertopic(val: str, bertopic_kws: List[str]) -> bool:
    """Returns True if the taxonomy term matches or overlaps with BERTopic keywords."""
    if not val or not bertopic_kws:
        return True
    val_words = set(re.findall(r"\w+", val.lower()))
    for topic_str in bertopic_kws:
        topic_words = set(re.findall(r"\w+", topic_str.lower()))
        if val_words & topic_words:
            return True
    return False


# ─────────────────────────────────────────────
# BERTopic Fallsbacks
# ─────────────────────────────────────────────

def run_bertopic(
    texts: List[str],
    min_topic_size: int = 2,
    n_gram_range: Tuple[int, int] = (1, 2),
) -> Tuple[Dict[int, str], List[int], Dict]:
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
            return _fallback_topic_keywords(texts), [i % 2 for i in range(len(texts))], {}

        topics, _ = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()

        topic_keywords = {}
        for topic_id in topic_info["Topic"].tolist():
            if topic_id == -1:
                continue
            kw_list = topic_model.get_topic(topic_id)
            if kw_list:
                top_words = [w for w, _ in kw_list[:5]]
                topic_keywords[int(topic_id)] = ", ".join(top_words)

        return topic_keywords, [int(t) for t in topics], topic_info.to_dict()

    except ImportError:
        return _fallback_topic_keywords(texts), [i % 2 for i in range(len(texts))], {}
    except Exception as e:
        logger.error(f"BERTopic error: {e} — using fallback.")
        return _fallback_topic_keywords(texts), [i % 2 for i in range(len(texts))], {}


def _fallback_topic_keywords(texts: List[str]) -> Dict[int, str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if not texts:
            return {}
        vec = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2))
        vec.fit(texts)
        kws = list(vec.get_feature_names_out())[:10]
        return {0: ", ".join(kws[:5]), 1: ", ".join(kws[5:10])}
    except Exception:
        return {}


# ─────────────────────────────────────────────
# Master gap detection pipeline
# ─────────────────────────────────────────────

def detect_research_gaps(
    retrieved_papers: List[Dict[str, Any]],
    user_title: str = "",
    user_keywords: str = "",
) -> Dict[str, Any]:
    """
    Computes research gaps over retrieved papers using a dynamic
    Auditable Gap Size formula: GapSize = 1.0 - AvgCoverage
    """
    if not retrieved_papers:
        return {
            "covered_topics": [],
            "gap_dimensions": {},
            "gap_statements": ["No similar papers found — this appears to be a completely new area."],
            "bertopic_topics": [],
            "auditable_formulas": {}
        }

    abstracts = [p.get("abstract", "") for p in retrieved_papers]
    topic_keywords, topics, _ = run_bertopic(abstracts)

    # Assign topic labels to papers
    for idx, paper in enumerate(retrieved_papers):
        tid = topics[idx] if idx < len(topics) else -1
        paper["topic_id"] = int(tid)
        paper["topic_keywords"] = topic_keywords.get(int(tid), "General research themes")

    # 1. Load taxonomy models for all retrieved papers
    paper_taxonomies = [get_taxonomy_for_paper(p) for p in retrieved_papers]

    # Adjust confidence via BERTopic validation
    for tax_obj in paper_taxonomies:
        for key in ["theme", "method"]:
            val_obj = tax_obj.get(key, {})
            val = val_obj.get("value", "")
            if val and not cross_validate_with_bertopic(val, list(topic_keywords.values())):
                val_obj["confidence"] = max(0.0, val_obj["confidence"] - 0.25)

    # 2. Setup Gap Size Calculations
    taxonomies = {
        "regional_gaps": ("region", REGION_TAXONOMY),
        "population_gaps": ("population", POPULATION_TAXONOMY),
        "methodological_gaps": ("method", METHOD_TAXONOMY),
        "thematic_gaps": ("theme", THEME_TAXONOMY),
        "theoretical_gaps": ("theory", THEORY_TAXONOMY),
        "temporal_gaps": ("time_period", TIME_TAXONOMY),
    }

    current_year = datetime.now().year
    gap_dimensions = {}
    auditable_formulas = {}

    for dim_name, (key, taxonomy) in taxonomies.items():
        dim_calculations = []
        
        for item in taxonomy:
            matched_papers_log = []
            coverage_sum = 0.0
            
            for idx, p in enumerate(retrieved_papers):
                tax = paper_taxonomies[idx]
                extracted_item = tax.get(key, {})
                extracted_val = extracted_item.get("value", "").lower()
                
                # Check for match (full word containment)
                is_match = False
                confidence = 0.0
                if extracted_val and (item.lower() in extracted_val or extracted_val in item.lower()):
                    is_match = True
                    confidence = extracted_item.get("confidence", 0.0)
                
                # Recency Weighting
                year = int(p.get("year", 0))
                recency_weight = 1.0 if year >= (current_year - 3) else 0.6
                
                weighted_coverage = confidence * recency_weight if is_match else 0.0
                coverage_sum += weighted_coverage
                
                if is_match:
                    matched_papers_log.append({
                        "paper_title": p.get("title", "")[:60],
                        "year": year,
                        "confidence": round(confidence, 2),
                        "recency_weight": recency_weight,
                        "weighted_coverage": round(weighted_coverage, 2)
                    })
            
            avg_coverage = coverage_sum / len(retrieved_papers)
            gap_size = 1.0 - avg_coverage
            
            dim_calculations.append({
                "item": item,
                "gap_size": round(gap_size, 3),
                "avg_coverage": round(avg_coverage, 3),
                "matched_papers": matched_papers_log
            })

        # Sort items in this dimension by Gap Size descending
        dim_calculations.sort(key=lambda x: x["gap_size"], reverse=True)
        auditable_formulas[dim_name] = {
            "formula": "GapSize = 1.0 - (Sum(Confidence * RecencyWeight) / len(papers))",
            "dimensions_log": dim_calculations
        }
        
        # High Gap Size items (e.g. size > 0.70) are the gaps
        gaps_list = [c["item"] for c in dim_calculations if c["gap_size"] >= 0.70]
        gap_dimensions[dim_name] = gaps_list[:5]  # Top 5 gaps

    # Generate gap statements
    statements = []
    if gap_dimensions.get("regional_gaps"):
        statements.append(
            f"📍 Regional Gap: The literature largely overlooks {', '.join(gap_dimensions['regional_gaps'][:2])}. "
            "Studies focusing on these regions are needed."
        )
    if gap_dimensions.get("population_gaps"):
        statements.append(
            f"👥 Population Gap: Research involving {', '.join(gap_dimensions['population_gaps'][:2])} is underrepresented, "
            "suggesting a clear demographic gap."
        )
    if gap_dimensions.get("methodological_gaps"):
        statements.append(
            f"🔬 Methodological Gap: Frameworks such as {', '.join(gap_dimensions['methodological_gaps'][:2])} have "
            "rarely been applied in this research area."
        )
    if gap_dimensions.get("thematic_gaps"):
        statements.append(
            f"💡 Thematic Gap: Topics related to {', '.join(gap_dimensions['thematic_gaps'][:2])} remain underexplored."
        )
    if gap_dimensions.get("theoretical_gaps"):
        statements.append(
            f"📚 Theoretical Gap: Theories like {', '.join(gap_dimensions['theoretical_gaps'][:2])} are not utilized."
        )
    if not statements:
        statements.append("🔍 General Gap: Limited cross-disciplinary or temporal research has been conducted.")

    # Covered topics (items with highest coverage)
    flat_coverage = []
    for dim_name in auditable_formulas:
        for x in auditable_formulas[dim_name]["dimensions_log"]:
            if x["avg_coverage"] > 0.1:
                flat_coverage.append((x["item"], x["avg_coverage"]))
    flat_coverage.sort(key=lambda x: x[1], reverse=True)
    covered_topics = [item for item, _ in flat_coverage[:10]]

    return {
        "covered_topics": covered_topics,
        "bertopic_topics": list(topic_keywords.values())[:6],
        "gap_dimensions": gap_dimensions,
        "gap_statements": statements,
        "auditable_formulas": auditable_formulas
    }


def suggest_improved_titles(
    original_title: str,
    gap_dimensions: Dict[str, List[str]],
    n_suggestions: int = 5,
) -> List[str]:
    suggestions = []
    base = original_title.strip().rstrip(".")
    regions = gap_dimensions.get("regional_gaps", [])
    populations = gap_dimensions.get("population_gaps", [])
    methods = gap_dimensions.get("methodological_gaps", [])
    themes = gap_dimensions.get("thematic_gaps", [])

    templates = [
        lambda: f"{base}: A {_pick(methods, 'Mixed-Methods')} Study in {_pick(regions, 'Rural India')}",
        lambda: f"Exploring {_pick(themes, 'Socio-Economic Challenges')} among {_pick(populations, 'Marginalised Communities')} — Evidence from {_pick(regions, 'South India')}",
        lambda: f"{base}: Perspectives from {_pick(populations, 'Women')} in {_pick(regions, 'North-East India')}",
        lambda: f"A {_pick(methods, 'Longitudinal')} Analysis of {_pick(themes, 'Social Mobility')} in {_pick(regions, 'Tribal Regions')} of India",
        lambda: f"Revisiting {base}: An Interdisciplinary Framework Using {_pick(methods, 'Qualitative')} Methods",
    ]

    for fn in templates[:n_suggestions]:
        try:
            suggestions.append(fn())
        except Exception:
            continue
    return suggestions


def _pick(lst: List[str], default: str) -> str:
    clean = [x for x in lst if x]
    if clean:
        return clean[0].title()
    return default
