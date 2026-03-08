"""
models/explanation_engine.py
-----------------------------
LLM-powered explanation engine using Google Gemini.

Features:
  - Grounded prompting (feeds retrieved evidence directly to Gemini)
  - Chain-of-thought reasoning for gap analysis
  - Hallucination-resistant prompt design
  - Template fallback when API key is not set
"""

import logging
from typing import Dict, Any, List

from config.settings import (
    GEMINI_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, USE_LLM_EXPLANATION
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Grounded prompt builder
# ─────────────────────────────────────────────

def _build_grounded_prompt(analysis: Dict[str, Any]) -> str:
    """
    Build a hallucination-resistant grounded prompt.

    Key anti-hallucination techniques used:
    1. Provide ALL evidence explicitly in the prompt
    2. Instruct model to ONLY use provided evidence
    3. Ask model to cite paper numbers when making claims
    4. Instruct model to say 'not enough evidence' if unsure
    5. Chain-of-thought reasoning steps
    """
    inp         = analysis.get("input", {})
    title       = inp.get("title", "")
    abstract    = inp.get("abstract", "") or "Not provided"
    keywords    = inp.get("keywords", "") or "Not provided"
    domain      = inp.get("domain", "") or "Not specified"
    novelty     = analysis.get("novelty", {})
    gaps        = analysis.get("gaps", {})
    similar     = analysis.get("similar_papers", [])
    suggestions = analysis.get("title_suggestions", [])
    sim_stats   = analysis.get("similarity_stats", {})

    # Build numbered evidence list
    evidence_lines = []
    for i, p in enumerate(similar[:8], 1):
        evidence_lines.append(
            f"[Paper {i}] Title: {p.get('title','')}\n"
            f"           Year: {p.get('year','N/A')} | Domain: {p.get('domain','N/A')} | "
            f"Region: {p.get('region','N/A')}\n"
            f"           Similarity: {p.get('similarity_pct',0):.1f}%\n"
            f"           Abstract: {p.get('abstract','')[:200]}…"
        )
    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "No similar papers found."

    gap_stmts  = "\n".join(f"  - {g}" for g in gaps.get("gap_statements", [])[:5])
    title_sugg = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(suggestions[:3]))

    gap_dims = gaps.get("gap_dimensions", {})
    reg_gaps = ", ".join(gap_dims.get("regional_gaps", [])[:3])
    pop_gaps = ", ".join(gap_dims.get("population_gaps", [])[:3])
    met_gaps = ", ".join(gap_dims.get("methodological_gaps", [])[:3])

    prompt = f"""You are a senior academic research advisor helping an Indian PhD scholar evaluate their research topic.

═══════════════════════════════════════════════
SCHOLAR'S PROPOSED RESEARCH
═══════════════════════════════════════════════
Title    : {title}
Abstract : {abstract}
Keywords : {keywords}
Domain   : {domain}

═══════════════════════════════════════════════
SYSTEM ANALYSIS RESULTS (VERIFIED DATA)
═══════════════════════════════════════════════
Novelty Score  : {novelty.get('label','N/A')} ({novelty.get('percentage',0)}%)
Max Similarity : {sim_stats.get('max_sim',0)*100:.1f}%
Mean Similarity: {sim_stats.get('mean_sim',0)*100:.1f}%
Papers Found   : {len(similar)}

═══════════════════════════════════════════════
RETRIEVED EVIDENCE — USE ONLY THESE PAPERS
═══════════════════════════════════════════════
{evidence_block}

═══════════════════════════════════════════════
DETECTED RESEARCH GAPS (SYSTEM-VERIFIED)
═══════════════════════════════════════════════
{gap_stmts if gap_stmts else "No major gaps detected."}

Underexplored Regions    : {reg_gaps or 'None identified'}
Underrepresented Groups  : {pop_gaps or 'None identified'}
Unused Methodologies     : {met_gaps or 'None identified'}

═══════════════════════════════════════════════
SUGGESTED IMPROVED TITLES
═══════════════════════════════════════════════
{title_sugg if title_sugg else "No suggestions available."}

═══════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════
Write a 3-paragraph advisory report for the PhD scholar.

⚠️ STRICT RULES TO PREVENT HALLUCINATION:
1. ONLY use information from the RETRIEVED EVIDENCE section above.
2. When referencing a paper, cite it as [Paper 1], [Paper 2], etc.
3. If you are not sure about something, write "based on available evidence" or "further verification needed".
4. Do NOT invent paper titles, authors, years, or statistics not shown above.
5. Do NOT claim a paper exists unless it appears in the Retrieved Evidence section.
6. Use the EXACT novelty label shown: {novelty.get('label','N/A')}

PARAGRAPH 1 — Novelty Assessment:
Explain what the {novelty.get('label','N/A')} novelty score means, referencing the most similar papers by number.

PARAGRAPH 2 — Research Gaps:
Identify the 2-3 most important gaps from the verified gap data above. Be specific about regions, populations, or methods.

PARAGRAPH 3 — Recommendations:
Give 3 actionable next steps. Recommend one improved title from the list above and explain why.

Write in plain English for a PhD scholar beginning their research journey. Be encouraging but honest. (200-250 words)
"""
    return prompt


# ─────────────────────────────────────────────
# Gemini LLM call
# ─────────────────────────────────────────────

def generate_llm_explanation(analysis: Dict[str, Any]) -> str:
    """
    Call Google Gemini with a grounded, hallucination-resistant prompt.
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model  = genai.GenerativeModel(LLM_MODEL)
        prompt = _build_grounded_prompt(analysis)

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens = LLM_MAX_TOKENS,
                temperature       = 0.3,   # low temp = more factual, less hallucination
                top_p             = 0.85,
            )
        )

        output = response.text.strip()
        logger.info(f"Gemini response generated ({len(output.split())} words).")
        return output

    except ImportError:
        logger.warning("google-generativeai not installed. Run: pip install google-generativeai")
        return generate_template_explanation(analysis)
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}. Using template fallback.")
        return generate_template_explanation(analysis)


# ─────────────────────────────────────────────
# LLM-powered gap enhancement
# ─────────────────────────────────────────────

def enhance_gaps_with_llm(
    gap_statements: List[str],
    retrieved_papers: List[Dict[str, Any]],
    user_title: str,
) -> List[str]:
    """
    Use Gemini to enhance raw gap statements with deeper analysis.
    Falls back to original statements if API unavailable.
    """
    if not USE_LLM_EXPLANATION or not gap_statements:
        return gap_statements

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(LLM_MODEL)

        paper_titles = "\n".join(
            f"  - {p.get('title','')[:80]} ({p.get('year','N/A')})"
            for p in retrieved_papers[:5]
        )
        gaps_raw = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(gap_statements))

        prompt = f"""You are a research gap analyst.

Research Title: "{user_title}"

Similar papers already exist:
{paper_titles}

System-detected gaps:
{gaps_raw}

For each gap, write ONE specific, actionable sentence explaining:
- Exactly what is missing
- Why it matters for Indian PhD research
- What kind of study could fill it

RULES:
- Only reference gaps that are genuinely absent from the listed papers
- Do not invent statistics or claim specific numbers
- Keep each gap statement to 1-2 sentences
- Return exactly {len(gap_statements)} enhanced gap statements, one per line

Return only the enhanced gap statements, numbered 1 to {len(gap_statements)}.
"""
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=600,
                temperature=0.2,
            )
        )

        lines = [
            re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            for line in response.text.strip().split('\n')
            if line.strip() and len(line.strip()) > 20
        ]

        if len(lines) >= len(gap_statements) // 2:
            logger.info("Gap statements enhanced by Gemini.")
            return lines[:len(gap_statements)]
        return gap_statements

    except Exception as e:
        logger.warning(f"Gap enhancement failed: {e}. Using original gaps.")
        return gap_statements


# ─────────────────────────────────────────────
# Template fallback
# ─────────────────────────────────────────────

def generate_template_explanation(analysis: Dict[str, Any]) -> str:
    """
    Offline template-based report. Always works without any API key.
    """
    inp     = analysis.get("input", {})
    title   = inp.get("title", "your proposed topic")
    novelty = analysis.get("novelty", {})
    gaps    = analysis.get("gaps", {})
    similar = analysis.get("similar_papers", [])
    titles  = analysis.get("title_suggestions", [])

    label = novelty.get("label", "MEDIUM")
    pct   = novelty.get("percentage", 50)

    p1_intro = {
        "HIGH": (
            f"🎉 Excellent news! Your research — \"{title}\" — "
            f"achieved a HIGH Novelty Score of {pct}%. "
            "Very few similar studies exist in the database, indicating strong originality."
        ),
        "MEDIUM": (
            f"📊 Your research — \"{title}\" — "
            f"received a MEDIUM Novelty Score of {pct}%. "
            "Related work exists, but your specific focus introduces new elements."
        ),
        "LOW": (
            f"⚠️ Your research — \"{title}\" — "
            f"received a LOW Novelty Score of {pct}%. "
            "Highly similar studies already exist. Consider refining your focus."
        ),
    }[label]

    if similar:
        top = similar[0]
        p1_extra = (
            f" The most similar existing study is "
            f"'{top.get('title','')}' "
            f"({top.get('year','')}, {top.get('domain','')}) "
            f"at {top.get('similarity_pct',0):.0f}% similarity."
        )
    else:
        p1_extra = " No direct duplicates were found in the database."

    gap_stmts = gaps.get("gap_statements", [])
    para2 = (
        "🔍 Key Research Gaps Identified:\n\n"
        + "\n".join(f"  {s}" for s in gap_stmts[:4])
        + "\n\nThese represent your strongest opportunities for original contribution."
        if gap_stmts else
        "🔍 No major gaps detected. Consider narrowing your regional or population focus."
    )

    suggestions = novelty.get("suggestions", [])
    sug_text = (
        "\n".join(f"  • {s}" for s in suggestions[:3])
        if suggestions else
        "  • Narrow to a specific region or population.\n  • Adopt a mixed-methods approach."
    )

    title_text = (
        "\n".join(f"  {i+1}. {t}" for i, t in enumerate(titles[:3]))
        if titles else "  No suggestions available."
    )

    sep    = "\n" + "─" * 60 + "\n"
    report = (
        "📋 AI RESEARCH ANALYSIS REPORT\n"
        + "═" * 60 + "\n\n"
        + p1_intro + p1_extra
        + "\n\n" + sep
        + para2
        + "\n\n" + sep
        + "💡 Recommendations:\n\n"
        + sug_text
        + "\n\n📝 Suggested Improved Titles:\n\n"
        + title_text
        + "\n\n" + "═" * 60
        + "\n\nGenerated by AI Research Novelty & Gap Detector\n"
          "For Arts & Science PhD Scholars — India\n"
    )
    return report


# ─────────────────────────────────────────────
# Master entry point
# ─────────────────────────────────────────────

import re

def generate_explanation(analysis: Dict[str, Any]) -> str:
    """
    Generate explanation using Gemini (grounded) if API key is set,
    otherwise use the offline template engine.
    """
    if USE_LLM_EXPLANATION:
        logger.info("Generating grounded explanation via Google Gemini …")
        return generate_llm_explanation(analysis)
    else:
        logger.info("GEMINI_API_KEY not set — using template engine …")
        return generate_template_explanation(analysis)
