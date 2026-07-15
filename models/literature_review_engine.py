"""
models/literature_review_engine.py
-----------------------------------
Literature Review Drafting Engine.
Clusters papers using BERTopic topic labels, drafts synthesised paragraphs,
runs self-correcting grounding checks, and integrates research gaps.
"""

import logging
from collections import defaultdict
from typing import Dict, Any, List
import google.generativeai as genai

from config.settings import GEMINI_API_KEY, LLM_MODEL, USE_LLM_EXPLANATION
from models.hallucination_detector import detect_hallucinations

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Template Fallback Generator
# ─────────────────────────────────────────────

def generate_template_paragraph(cluster_kws: str, papers: List[Dict[str, Any]]) -> str:
    """Generates a structured paragraph based on templates when LLM is unavailable."""
    if not papers:
        return "No literature was retrieved for this subsection."
    
    sentences = [
        f"In the thematic area of {cluster_kws}, several studies have made significant contributions."
    ]
    for p in papers:
        tag = p["citation_tag"]
        title = p.get("title", "")
        year = p.get("year", "n.d.")
        domain = p.get("domain", "the field")
        sentences.append(
            f"Specifically, {tag} investigated '{title}' in {year}, highlighting critical factors relevant to the domain of {domain}."
        )
    sentences.append(
        "These papers collectively establish the current state-of-the-art and form the baseline for further investigations."
    )
    return " ".join(sentences)


# ─────────────────────────────────────────────
# Drafting and Self-Correction Loop
# ─────────────────────────────────────────────

def generate_with_correction(cluster_kws: str, papers: List[Dict[str, Any]]) -> str:
    """
    Drafts a paragraph using Gemini and runs a self-correction loop
    if any claims fail the grounding verification.
    """
    if not USE_LLM_EXPLANATION or not GEMINI_API_KEY:
        return generate_template_paragraph(cluster_kws, papers)
        
    novelty_data = {"label": "MEDIUM", "percentage": 50}
    paragraph = ""
    feedback = ""
    
    # Try up to 3 times (1 initial + 2 retries)
    for attempt in range(3):
        evidence = []
        for p in papers:
            evidence.append(
                f"{p['citation_tag']} Title: {p.get('title','')}\n"
                f"          Year: {p.get('year','N/A')} | Domain: {p.get('domain','N/A')}\n"
                f"          Abstract: {p.get('abstract','')[:400]}"
            )
        evidence_block = "\n\n".join(evidence)
        
        prompt = f"""You are an academic writer drafting a literature review chapter for a PhD thesis.
Write one cohesive, professional, and academic synthesis paragraph summarizing the following papers which cover the theme: "{cluster_kws}".

═══════════════════════════════════════════════
PAPERS TO CITE (USE ONLY THESE):
═══════════════════════════════════════════════
{evidence_block}

═══════════════════════════════════════════════
STRICT RULES:
1. Refer to the papers strictly by their citation tag, e.g., [Paper 1], [Paper 2], etc.
2. Synthesize their findings into a single paragraph. Do not write a list.
3. ONLY use details from the provided papers. Do not invent any outside claims, papers, or statistics.
4. If there are contradictions or complementary points, highlight them academically.
"""
        if feedback:
            prompt += f"\n\n⚠️ PREVIOUS ATTEMPT WAS REJECTED DUE TO HALLUCINATION:\n{feedback}\nPlease rewrite the paragraph, correcting these issues and adhering strictly to the evidence."
            
        prompt += "\n\nWrite in a formal, peer-reviewed journal style. (150-200 words)"
        
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(LLM_MODEL)
            response = model.generate_content(prompt)
            paragraph = response.text.strip()
            
            # Check grounding
            h_report = detect_hallucinations(paragraph, papers, novelty_data)
            if h_report.get("grounding_score", 100) >= 80:
                logger.info(f"Paragraph grounded successfully (Score: {h_report['grounding_score']}).")
                break
            else:
                issues = [c.get("issue") for c in h_report.get("flagged_claims", []) if c.get("issue")]
                feedback = "\n".join(f"- {i}" for i in issues)
                logger.warning(f"Hallucination detected on attempt {attempt+1}. Issues:\n{feedback}")
        except Exception as e:
            logger.error(f"Failed to generate paragraph on attempt {attempt+1}: {e}")
            paragraph = ""
            
    if not paragraph:
        return generate_template_paragraph(cluster_kws, papers)
    return paragraph


# ─────────────────────────────────────────────
# Literature Review Service
# ─────────────────────────────────────────────

class LiteratureReviewService:
    """Service to compile the literature review from retrieved papers."""
    
    @staticmethod
    def compile_review(
        query_title: str,
        retrieved_papers: List[Dict[str, Any]],
        gap_data: Dict[str, Any],
        section_preference: str = "thematic",
    ) -> Dict[str, Any]:
        """
        Executes the literature review drafting pipeline.
        1. Cluster papers using BERTopic topic assignments
        2. Draft synthesized paragraphs for each cluster
        3. Append Gap Analysis as closing section
        """
        if not retrieved_papers:
            return {
                "sections": [],
                "combined_markdown": "No literature found to compile review.",
                "citations": []
            }
            
        # Group papers by topic_id (reusing BERTopic outputs)
        clusters = defaultdict(list)
        for idx, paper in enumerate(retrieved_papers, 1):
            paper_copy = paper.copy()
            paper_copy["citation_tag"] = f"[Paper {idx}]"
            tid = paper.get("topic_id", -1)
            clusters[tid].append(paper_copy)
            
        # Build sections
        sections = []
        combined_md_parts = [
            f"# Literature Review: {query_title}",
            "This review is compiled from retrieved relevant papers, organized by thematic clustering."
        ]
        
        # Sort clusters by topic_id
        sorted_topics = sorted(clusters.keys())
        for tid in sorted_topics:
            cluster_papers = clusters[tid]
            cluster_kws = cluster_papers[0].get("topic_keywords", "General themes")
            
            logger.info(f"Synthesising literature review section for theme: '{cluster_kws}' ({len(cluster_papers)} papers) ...")
            paragraph = generate_with_correction(cluster_kws, cluster_papers)
            
            section_title = f"Thematic Synthesis: {cluster_kws.title()}"
            sections.append({
                "title": section_title,
                "paragraph": paragraph,
                "papers": [
                    {
                        "citation_tag": p["citation_tag"],
                        "title": p.get("title", ""),
                        "year": p.get("year", ""),
                        "domain": p.get("domain", "")
                    }
                    for p in cluster_papers
                ]
            })
            
            combined_md_parts.append(f"\n## {section_title}\n\n{paragraph}")
            
        # Add Research Gap section
        gap_statements = gap_data.get("gap_statements", [])
        gap_text = "\n".join(f"- {g}" for g in gap_statements[:4])
        
        gap_section_title = "Research Gap & Novelty Analysis"
        sections.append({
            "title": gap_section_title,
            "paragraph": gap_text,
            "papers": []
        })
        combined_md_parts.append(f"\n## {gap_section_title}\n\n{gap_text}")
        
        # Add Citations References Section
        combined_md_parts.append("\n## References & Citations")
        citations_list = []
        for idx, p in enumerate(retrieved_papers, 1):
            ref_str = f"[Paper {idx}] {p.get('title','')} ({p.get('year','N/A')}). Domain: {p.get('domain','N/A')}, Region: {p.get('region','N/A')}."
            combined_md_parts.append(ref_str)
            citations_list.append(ref_str)
            
        combined_markdown = "\n".join(combined_md_parts)
        
        return {
            "sections": sections,
            "combined_markdown": combined_markdown,
            "citations": citations_list
        }
