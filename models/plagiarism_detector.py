"""
models/plagiarism_detector.py
------------------------------
Plagiarism Detection Engine.

Checks the user's research title/abstract against the retrieved papers
and computes a detailed plagiarism report with:
  - Overall plagiarism score (0-100%)
  - Sentence-level similarity matches
  - Exact phrase matches
  - Paraphrase detection
  - Risk level (HIGH / MEDIUM / LOW / SAFE)
  - Matched paper citations
"""

import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Risk level config
# ─────────────────────────────────────────────
RISK_CONFIG = {
    "HIGH":   {"min": 70, "color": "#e74c3c", "icon": "🔴",
                "message": "High plagiarism risk detected. Major rewriting required."},
    "MEDIUM": {"min": 40, "color": "#f39c12", "icon": "🟡",
                "message": "Moderate similarity found. Some sections need paraphrasing."},
    "LOW":    {"min": 15, "color": "#3498db", "icon": "🔵",
                "message": "Low similarity. Minor overlaps with existing literature."},
    "SAFE":   {"min": 0,  "color": "#2ecc71", "icon": "✅",
                "message": "No significant plagiarism detected. Your work appears original."},
}


# ─────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────

def _clean(text: str) -> str:
    """Lowercase and remove punctuation for comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _sentences(text: str) -> List[str]:
    """Split text into sentences."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in parts if len(s.split()) >= 5]


def _ngrams(text: str, n: int) -> List[str]:
    """Extract n-grams from cleaned text."""
    words = _clean(text).split()
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]


# ─────────────────────────────────────────────
# Matching engines
# ─────────────────────────────────────────────

def _exact_phrase_matches(
    user_text: str,
    paper_text: str,
    min_words: int = 6,
) -> List[Dict[str, Any]]:
    """
    Find exact multi-word phrase matches between user text and a paper.
    Returns list of matched phrases with their word count.
    """
    matches = []
    user_ngrams  = set(_ngrams(user_text,  min_words))
    paper_ngrams = set(_ngrams(paper_text, min_words))
    common       = user_ngrams & paper_ngrams

    for phrase in common:
        word_count = len(phrase.split())
        matches.append({
            "phrase":     phrase,
            "word_count": word_count,
            "severity":   "HIGH" if word_count >= 10 else "MEDIUM" if word_count >= 7 else "LOW"
        })

    return sorted(matches, key=lambda x: x["word_count"], reverse=True)[:5]


def _sentence_similarity(sent1: str, sent2: str) -> float:
    """
    Compute word-overlap based similarity between two sentences.
    Jaccard similarity on word sets.
    """
    w1 = set(_clean(sent1).split())
    w2 = set(_clean(sent2).split())
    if not w1 or not w2:
        return 0.0
    intersection = w1 & w2
    union        = w1 | w2
    # Remove very common stop words from scoring
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","is","are","was","were","be","been","this","that",
        "it","its","by","from","as","has","have","had","not","study",
        "research","paper","present","using","based","proposed"
    }
    meaningful = intersection - stopwords
    return len(meaningful) / max(len(union - stopwords), 1)


def _check_sentence_matches(
    user_sentences: List[str],
    paper: Dict[str, Any],
    threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Find sentences in user text that are highly similar to
    sentences in a retrieved paper.
    """
    paper_text      = f"{paper.get('title','')} {paper.get('abstract','')}"
    paper_sentences = _sentences(paper_text)
    matched         = []

    for u_sent in user_sentences:
        for p_sent in paper_sentences:
            sim = _sentence_similarity(u_sent, p_sent)
            if sim >= threshold:
                matched.append({
                    "user_sentence":  u_sent,
                    "paper_sentence": p_sent,
                    "similarity_pct": round(sim * 100, 1),
                    "paper_title":    paper.get("title", "")[:80],
                    "paper_year":     paper.get("year", ""),
                    "paper_domain":   paper.get("domain", ""),
                })

    return sorted(matched, key=lambda x: x["similarity_pct"], reverse=True)[:3]


def _compute_overall_score(
    retrieved_papers: List[Dict[str, Any]],
    user_text: str,
    similarity_stats: Dict[str, float],
) -> float:
    """
    Compute overall plagiarism percentage (0-100).

    Components:
      40% — Top FAISS cosine similarity (direct semantic match)
      35% — Sentence-level overlap with top papers
      25% — Exact phrase matches
    """
    # Component 1: FAISS similarity (already computed)
    top_sim = similarity_stats.get("max_sim", 0.0) * 100

    # Component 2: Sentence overlap with top 3 papers
    user_sents  = _sentences(user_text)
    overlap_scores = []
    for paper in retrieved_papers[:3]:
        paper_text = f"{paper.get('title','')} {paper.get('abstract','')}"
        paper_sents = _sentences(paper_text)
        if not paper_sents or not user_sents:
            continue
        sims = [
            _sentence_similarity(u, p)
            for u in user_sents
            for p in paper_sents
        ]
        if sims:
            overlap_scores.append(max(sims) * 100)

    sent_overlap = max(overlap_scores) if overlap_scores else 0.0

    # Component 3: Exact phrase matches
    phrase_score = 0.0
    for paper in retrieved_papers[:5]:
        paper_text = f"{paper.get('title','')} {paper.get('abstract','')}"
        phrases    = _exact_phrase_matches(user_text, paper_text)
        if phrases:
            phrase_score = max(phrase_score, len(phrases) * 10.0)
    phrase_score = min(phrase_score, 100.0)

    # Weighted combination
    overall = (
        0.40 * top_sim +
        0.35 * sent_overlap +
        0.25 * phrase_score
    )
    return round(min(overall, 100.0), 1)


def _get_risk_level(score: float) -> str:
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    elif score >= 15:
        return "LOW"
    return "SAFE"


# ─────────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────────

def _get_recommendations(
    risk: str,
    matched_papers: List[Dict],
    phrase_matches: List[Dict],
) -> List[str]:
    recs = []
    if risk == "HIGH":
        recs += [
            "🔄 Completely rewrite overlapping sections in your own words.",
            "📝 Use quotation marks and citations for any directly borrowed text.",
            "🎯 Change your research focus to a different region, population, or time period.",
            "🔬 Adopt a different methodology to differentiate your study.",
            "📚 Cite all matched papers properly in your references.",
        ]
    elif risk == "MEDIUM":
        recs += [
            "✏️ Paraphrase sentences that closely mirror existing literature.",
            "📖 Add proper citations wherever you reference similar studies.",
            "💡 Strengthen your original contribution section.",
            "🌍 Emphasise what makes your study context unique.",
        ]
    elif risk == "LOW":
        recs += [
            "✅ Your work shows low similarity — minor revisions may be sufficient.",
            "📝 Double-check any directly quoted passages are properly cited.",
            "🔍 Review the matched sections and ensure proper attribution.",
        ]
    else:
        recs += [
            "🎉 Excellent! Your research appears highly original.",
            "📚 Continue to cite all referenced works properly.",
            "✅ Proceed with confidence to your literature review.",
        ]
    return recs


# ─────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────

def detect_plagiarism(
    user_title: str,
    user_abstract: str,
    user_keywords: str,
    retrieved_papers: List[Dict[str, Any]],
    similarity_stats: Dict[str, float],
) -> Dict[str, Any]:
    """
    Full plagiarism detection pipeline.

    Args:
        user_title      : User's research title.
        user_abstract   : User's research abstract.
        user_keywords   : User's keywords.
        retrieved_papers: Top-k papers from FAISS retrieval.
        similarity_stats: Pre-computed similarity stats dict.

    Returns:
        Comprehensive plagiarism report dict.
    """
    # Combine user text
    user_text      = f"{user_title} {user_abstract} {user_keywords}".strip()
    user_sentences = _sentences(user_text)

    if not retrieved_papers:
        return {
            "plagiarism_score":  0.0,
            "risk_level":        "SAFE",
            "color":             "#2ecc71",
            "icon":              "✅",
            "message":           "No similar papers found — content appears fully original.",
            "matched_papers":    [],
            "sentence_matches":  [],
            "phrase_matches":    [],
            "recommendations":   ["✅ No similar work found. Your topic appears original."],
            "summary":           "No plagiarism detected.",
            "originality_score": 100.0,
        }

    # ── Overall score ────────────────────────
    score = _compute_overall_score(retrieved_papers, user_text, similarity_stats)
    risk  = _get_risk_level(score)
    cfg   = RISK_CONFIG[risk]

    # ── Per-paper sentence matches ────────────
    all_sentence_matches = []
    for paper in retrieved_papers[:5]:
        matches = _check_sentence_matches(user_sentences, paper)
        all_sentence_matches.extend(matches)

    # Sort by similarity
    all_sentence_matches = sorted(
        all_sentence_matches,
        key=lambda x: x["similarity_pct"],
        reverse=True
    )[:8]

    # ── Phrase matches ────────────────────────
    all_phrase_matches = []
    for paper in retrieved_papers[:5]:
        paper_text = f"{paper.get('title','')} {paper.get('abstract','')}"
        phrases    = _exact_phrase_matches(user_text, paper_text)
        for ph in phrases:
            ph["paper_title"] = paper.get("title", "")[:70]
            ph["paper_year"]  = paper.get("year", "")
            all_phrase_matches.append(ph)

    all_phrase_matches = sorted(
        all_phrase_matches,
        key=lambda x: x["word_count"],
        reverse=True
    )[:6]

    # ── Matched papers summary ────────────────
    matched_papers = []
    for p in retrieved_papers[:5]:
        sim = p.get("similarity_score", 0.0) * 100
        matched_papers.append({
            "title":          p.get("title", ""),
            "year":           p.get("year", ""),
            "domain":         p.get("domain", ""),
            "similarity_pct": round(sim, 1),
            "risk":           "HIGH" if sim >= 75 else "MEDIUM" if sim >= 55 else "LOW",
        })

    # ── Recommendations ───────────────────────
    recs = _get_recommendations(risk, matched_papers, all_phrase_matches)

    # ── Summary text ──────────────────────────
    summary = _build_summary(score, risk, cfg, matched_papers, all_sentence_matches)

    return {
        "plagiarism_score":  score,
        "originality_score": round(100.0 - score, 1),
        "risk_level":        risk,
        "color":             cfg["color"],
        "icon":              cfg["icon"],
        "message":           cfg["message"],
        "matched_papers":    matched_papers,
        "sentence_matches":  all_sentence_matches,
        "phrase_matches":    all_phrase_matches,
        "recommendations":   recs,
        "summary":           summary,
        "stats": {
            "total_sentences_checked": len(user_sentences),
            "sentences_flagged":       len(all_sentence_matches),
            "phrases_flagged":         len(all_phrase_matches),
            "papers_checked":          len(retrieved_papers[:5]),
        }
    }


def _build_summary(score, risk, cfg, matched_papers, sentence_matches) -> str:
    top_match = matched_papers[0] if matched_papers else {}
    lines = [
        f"{cfg['icon']} PLAGIARISM REPORT",
        "═" * 50,
        f"Plagiarism Score  : {score}%",
        f"Originality Score : {round(100-score, 1)}%",
        f"Risk Level        : {risk}",
        f"Status            : {cfg['message']}",
        "",
    ]
    if top_match:
        lines += [
            f"Highest Match     : {top_match.get('title','')[:60]}…",
            f"                    {top_match.get('similarity_pct',0):.1f}% similar "
            f"({top_match.get('year','')} · {top_match.get('domain','')})",
        ]
    if sentence_matches:
        lines += [
            "",
            f"Flagged Sentences : {len(sentence_matches)}",
            "Top flagged sentence:",
            f"  \"{sentence_matches[0].get('user_sentence','')[:100]}…\"",
            f"  Matches: \"{sentence_matches[0].get('paper_sentence','')[:100]}…\"",
            f"  Similarity: {sentence_matches[0].get('similarity_pct',0):.1f}%",
        ]
    lines += ["", "═" * 50,
              "Generated by AI Research Novelty & Gap Detector",
              "For PhD Scholars — India"]
    return "\n".join(lines)