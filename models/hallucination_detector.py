"""
models/hallucination_detector.py
---------------------------------
Hallucination Detection Engine.

Checks whether the LLM-generated explanation is grounded in the
actual retrieved papers. Flags claims that cannot be verified
against the evidence corpus.

Strategy:
  1. Extract key claims from LLM output
  2. Check each claim against retrieved paper metadata
  3. Compute a Grounding Score (0-100)
  4. Return flagged hallucinations with severity levels
"""

import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Severity levels
# ─────────────────────────────────────────────
SEVERITY_HIGH   = "HIGH"    # Claim directly contradicts evidence
SEVERITY_MEDIUM = "MEDIUM"  # Claim not supported by any retrieved paper
SEVERITY_LOW    = "LOW"     # Claim is plausible but unverifiable


def extract_claims(text: str) -> List[str]:
    """
    Extract individual sentences/claims from LLM output.
    Splits on sentence boundaries and filters short fragments.
    """
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter meaningful sentences (>10 words)
    claims = [s.strip() for s in sentences if len(s.split()) > 10]
    return claims


def _build_evidence_corpus(papers: List[Dict[str, Any]]) -> List[str]:
    """
    Build a flat list of evidence strings from retrieved papers.
    Each string combines title + abstract + keywords.
    """
    corpus = []
    for p in papers:
        evidence = f"{p.get('title','')} {p.get('abstract','')} {p.get('keywords','')} {p.get('domain','')} {p.get('region','')}".lower()
        corpus.append(evidence)
    return corpus


def _check_claim_against_evidence(
    claim: str,
    evidence_corpus: List[str],
    papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Check if a single claim is supported by the evidence corpus.

    Returns:
        dict with keys: claim, supported, confidence, matched_paper, issue
    """
    claim_lower = claim.lower()

    # Extract key noun phrases / numbers from the claim
    # Look for specific factual assertions
    numbers       = re.findall(r'\b\d+\b', claim)
    quoted_titles = re.findall(r"'([^']+)'", claim)
    year_mentions = re.findall(r'\b(19|20)\d{2}\b', claim)

    # ── Check 1: Quoted paper titles must exist ──
    for quoted in quoted_titles:
        found = any(quoted.lower() in ev for ev in evidence_corpus)
        if not found:
            return {
                "claim":         claim,
                "supported":     False,
                "confidence":    0.1,
                "issue":         f"Quoted title '{quoted}' not found in retrieved papers.",
                "severity":      SEVERITY_HIGH,
                "matched_paper": None,
            }

    # ── Check 2: Year mentions must match retrieved papers ──
    if year_mentions:
        paper_years = [str(p.get('year', '')) for p in papers]
        for yr in year_mentions:
            if yr not in paper_years and int(yr) > 2000:
                return {
                    "claim":         claim,
                    "supported":     False,
                    "confidence":    0.3,
                    "issue":         f"Year {yr} mentioned but no retrieved paper from this year.",
                    "severity":      SEVERITY_MEDIUM,
                    "matched_paper": None,
                }

    # ── Check 3: Domain/region keyword grounding ──
    domain_keywords = [
        "sociology", "history", "economics", "psychology", "education",
        "literature", "philosophy", "political", "anthropology", "geography",
        "chemistry", "physics", "mathematics", "botany", "zoology",
        "tamil nadu", "kerala", "maharashtra", "karnataka", "rajasthan",
        "gujarat", "punjab", "odisha", "assam", "bihar", "telangana",
        "women", "students", "farmers", "tribal", "rural", "urban",
        "scheduled castes", "scheduled tribes", "obc", "youth", "elderly"
    ]

    claim_domains = [kw for kw in domain_keywords if kw in claim_lower]

    if claim_domains:
        # At least one domain keyword must appear in evidence
        supported = any(
            any(kw in ev for kw in claim_domains)
            for ev in evidence_corpus
        )
        if not supported:
            return {
                "claim":         claim,
                "supported":     False,
                "confidence":    0.4,
                "issue":         f"Claim mentions '{claim_domains[0]}' but no retrieved paper supports this.",
                "severity":      SEVERITY_MEDIUM,
                "matched_paper": None,
            }

    # ── Check 4: Similarity percentage claims ──
    pct_claims = re.findall(r'(\d+(?:\.\d+)?)\s*%', claim)
    if pct_claims:
        for pct in pct_claims:
            val = float(pct)
            # Similarity scores should be between 0-100
            if val > 100:
                return {
                    "claim":         claim,
                    "supported":     False,
                    "confidence":    0.0,
                    "issue":         f"Similarity percentage {pct}% is impossible (>100%).",
                    "severity":      SEVERITY_HIGH,
                    "matched_paper": None,
                }

    # ── Default: claim appears grounded ──
    # Find best matching paper for attribution
    best_match = None
    best_overlap = 0
    claim_words = set(claim_lower.split())

    for i, ev in enumerate(evidence_corpus):
        ev_words  = set(ev.split())
        overlap   = len(claim_words & ev_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_match   = papers[i].get('title', '')

    confidence = min(0.5 + (best_overlap * 0.05), 1.0)

    return {
        "claim":         claim,
        "supported":     True,
        "confidence":    round(confidence, 2),
        "issue":         None,
        "severity":      None,
        "matched_paper": best_match,
    }


def compute_grounding_score(claim_results: List[Dict[str, Any]]) -> int:
    """
    Compute overall grounding score (0-100) from individual claim checks.
    Higher = more grounded = less hallucination.
    """
    if not claim_results:
        return 100

    total     = len(claim_results)
    supported = sum(1 for c in claim_results if c["supported"])

    # Weight by severity of unsupported claims
    penalty = 0
    for c in claim_results:
        if not c["supported"]:
            sev = c.get("severity", SEVERITY_LOW)
            if sev == SEVERITY_HIGH:
                penalty += 3
            elif sev == SEVERITY_MEDIUM:
                penalty += 2
            else:
                penalty += 1

    base_score = (supported / total) * 100
    final      = max(0, base_score - (penalty * 3))
    return int(round(final))


def get_grounding_label(score: int) -> Tuple[str, str]:
    """
    Returns (label, color) based on grounding score.
    """
    if score >= 80:
        return "WELL GROUNDED", "#2ecc71"
    elif score >= 60:
        return "MOSTLY GROUNDED", "#f39c12"
    elif score >= 40:
        return "PARTIALLY GROUNDED", "#e67e22"
    else:
        return "LIKELY HALLUCINATED", "#e74c3c"


# ─────────────────────────────────────────────
# Master hallucination detection pipeline
# ─────────────────────────────────────────────

def detect_hallucinations(
    llm_output: str,
    retrieved_papers: List[Dict[str, Any]],
    novelty_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full hallucination detection pipeline.

    Args:
        llm_output       : The explanation text generated by Gemini.
        retrieved_papers : Top-k papers from FAISS retrieval.
        novelty_data     : Novelty score dict for cross-checking.

    Returns:
        Dict with grounding_score, label, flagged_claims, safe_claims,
        hallucination_count, warnings.
    """
    if not llm_output or not llm_output.strip():
        return {
            "grounding_score":     100,
            "label":               "WELL GROUNDED",
            "color":               "#2ecc71",
            "flagged_claims":      [],
            "safe_claims":         [],
            "hallucination_count": 0,
            "warnings":            [],
            "summary":             "No LLM output to check.",
        }

    # Build evidence corpus
    evidence_corpus = _build_evidence_corpus(retrieved_papers)

    # Extract claims
    claims = extract_claims(llm_output)
    logger.info(f"Checking {len(claims)} claims for hallucination …")

    # Check each claim
    claim_results = []
    for claim in claims:
        result = _check_claim_against_evidence(claim, evidence_corpus, retrieved_papers)
        claim_results.append(result)

    # ── Cross-check novelty label ────────────
    novelty_label = novelty_data.get("label", "")
    novelty_pct   = novelty_data.get("percentage", 0)

    novelty_keywords = {
        "HIGH":   ["novel", "unique", "original", "new", "unexplored", "first"],
        "MEDIUM": ["moderate", "partial", "some", "related", "exists"],
        "LOW":    ["similar", "duplicate", "exists", "already", "overlap"],
    }

    expected_kws = novelty_keywords.get(novelty_label, [])
    output_lower = llm_output.lower()
    contradicts_novelty = False

    if novelty_label == "HIGH" and any(w in output_lower for w in ["low novelty", "already exists", "duplicate"]):
        contradicts_novelty = True
    elif novelty_label == "LOW" and any(w in output_lower for w in ["highly novel", "completely new", "no similar"]):
        contradicts_novelty = True

    warnings = []
    if contradicts_novelty:
        warnings.append(
            f"⚠️ LLM output contradicts the system's novelty score ({novelty_label}). "
            "The explanation may contain inaccurate novelty assessment."
        )

    # ── Compute scores ───────────────────────
    grounding_score          = compute_grounding_score(claim_results)
    label, color             = get_grounding_label(grounding_score)
    flagged                  = [c for c in claim_results if not c["supported"]]
    safe                     = [c for c in claim_results if c["supported"]]

    # ── Add generic warnings ─────────────────
    if len(retrieved_papers) < 3:
        warnings.append(
            "⚠️ Very few similar papers retrieved. LLM may have limited evidence to ground its claims."
        )
    if grounding_score < 60:
        warnings.append(
            "⚠️ Low grounding score detected. Treat the AI explanation with caution "
            "and verify claims against the listed similar papers."
        )

    summary = _build_hallucination_summary(grounding_score, label, flagged, warnings)

    return {
        "grounding_score":     grounding_score,
        "label":               label,
        "color":               color,
        "flagged_claims":      flagged,
        "safe_claims":         safe,
        "hallucination_count": len(flagged),
        "total_claims":        len(claims),
        "warnings":            warnings,
        "summary":             summary,
    }


def _build_hallucination_summary(
    score: int,
    label: str,
    flagged: List[Dict],
    warnings: List[str],
) -> str:
    """Build a human-readable hallucination summary."""
    if score >= 80:
        intro = (
            f"✅ Grounding Score: {score}/100 — {label}\n"
            "The AI explanation is well-supported by the retrieved literature. "
            "Claims align closely with the evidence corpus."
        )
    elif score >= 60:
        intro = (
            f"🟡 Grounding Score: {score}/100 — {label}\n"
            "Most claims are supported, but a few statements could not be "
            "directly verified against the retrieved papers."
        )
    else:
        intro = (
            f"🔴 Grounding Score: {score}/100 — {label}\n"
            "Several claims in the AI explanation could not be verified. "
            "Please cross-check with the listed similar papers before relying on this report."
        )

    if flagged:
        flag_str = "\n".join(
            f"  [{c.get('severity','?')}] {c.get('issue','Unknown issue')}"
            for c in flagged[:3]
        )
        intro += f"\n\nFlagged Issues:\n{flag_str}"

    if warnings:
        intro += "\n\nWarnings:\n" + "\n".join(f"  {w}" for w in warnings)

    return intro
