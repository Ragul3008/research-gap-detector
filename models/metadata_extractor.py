"""
models/metadata_extractor.py
----------------------------
Extracts Title, Abstract, Keywords, and Domain from paper text.
Uses Gemini LLM if available, otherwise falls back to heuristics.
"""

import re
import json
import logging
from typing import Dict, Any

from config.settings import GEMINI_API_KEY, LLM_MODEL, USE_LLM_EXPLANATION

logger = logging.getLogger(__name__)

DOMAINS = [
    "Sociology", "History", "Political Science", "Economics", "Psychology",
    "Environmental Science", "Literature", "Philosophy", "Anthropology",
    "Education", "Geography", "Women Studies", "Commerce", "Linguistics",
    "Botany", "Zoology", "Chemistry", "Physics", "Mathematics", "Statistics",
    "English Literature", "English Language Teaching", "Postcolonial Studies",
    "Comparative Literature", "Translation Studies", "Computer Science and Engineering",
    "Electronics and Communication Engineering", "Electrical Engineering",
    "Mechanical Engineering", "Civil Engineering", "Chemical Engineering",
    "Aerospace Engineering", "Biomedical Engineering", "Artificial Intelligence",
    "Machine Learning", "Deep Learning", "Internet of Things", "Cybersecurity",
    "Data Science", "Robotics and Automation", "VLSI Design", "Power Systems Engineering",
    "Renewable Energy Engineering", "Structural Engineering", "Environmental Engineering",
    "Nanotechnology", "Materials Science"
]


def extract_metadata_heuristics(text: str) -> Dict[str, Any]:
    """
    Extract metadata using rule-based text parsing heuristics.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # 1. Title Heuristics
    title_candidates = []
    for line in lines[:15]:
        lower_line = line.lower()
        # Skip headers, publishers, or small metadata lines
        if any(w in lower_line for w in ["journal", "vol.", "issue", "issn", "doi:", "http", "published", "page", "cite", "author", "abstract", "contents"]):
            continue
        # Limit line length to look like a title (between 15 and 150 chars)
        if 15 <= len(line) <= 150:
            title_candidates.append(line)
            if len(title_candidates) >= 2:
                break
                
    title = " ".join(title_candidates) if title_candidates else "Uploaded Research Paper"
    title = re.sub(r'\s+', ' ', title).strip()
    
    # 2. Abstract Heuristics
    abstract_lines = []
    abstract_started = False
    
    for line in lines[:60]:
        lower_line = line.lower()
        if not abstract_started:
            if lower_line.startswith("abstract") or lower_line.startswith("summary"):
                abstract_started = True
                clean_line = re.sub(r'^(abstract|summary)[\s\.:\-]+', '', line, flags=re.IGNORECASE)
                if clean_line.strip():
                    abstract_lines.append(clean_line.strip())
        else:
            # Look for stop headings
            if any(lower_line.startswith(stop) for stop in ["keywords", "key words", "introduction", "1. introduction", "i. introduction", "references"]):
                break
            # Add line if it's not a page number or random header
            if len(line) > 10:
                abstract_lines.append(line)
                
    abstract = " ".join(abstract_lines).strip()
    # Fallback if no abstract is found
    if not abstract:
        abstract = " ".join(lines[2:10])[:1200]
        
    # 3. Keywords Heuristics
    keywords = ""
    for line in lines[:80]:
        lower_line = line.lower()
        if "keywords" in lower_line or "key words" in lower_line:
            clean_line = re.sub(r'^.*(keywords|key words)[\s\.:\-]+', '', line, flags=re.IGNORECASE)
            keywords = clean_line.strip()
            break
            
    # 4. Domain Heuristics
    domain = ""
    full_lower_text = text.lower()
    domain_counts = {}
    for d in DOMAINS:
        count = full_lower_text.count(d.lower())
        if count > 0:
            domain_counts[d] = count
            
    if domain_counts:
        domain = max(domain_counts, key=domain_counts.get)
        
    return {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "domain": domain
    }


def extract_metadata_with_gemini(text: str) -> Dict[str, Any]:
    """
    Use Google Gemini to extract metadata from the first few pages of the paper.
    """
    import google.generativeai as genai
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(LLM_MODEL)
    
    # We take the first 8000 characters of the paper
    excerpt = text[:8000]
    
    prompt = f"""You are a research metadata extraction tool. Extract the Title, Abstract, Keywords, and Domain from the following academic paper text.

Text excerpt:
{excerpt}

Select the "domain" field STRICTLY from one of the following domains:
{DOMAINS}

Provide your response strictly as a JSON object, with the keys:
- "title": The title of the paper.
- "abstract": The abstract of the paper.
- "keywords": The keywords of the paper (as a comma-separated string).
- "domain": The selected domain (exact name from the list).

Do not output any introductory or explanation text, only the raw JSON.
"""
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=1000
        )
    )
    
    response_text = response.text.strip()
    
    # Clean up potential markdown formatting (```json ... ```)
    if response_text.startswith("```"):
        newline_idx = response_text.find("\n")
        if newline_idx != -1:
            response_text = response_text[newline_idx+1:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
    data = json.loads(response_text)
    
    # Validate returned fields
    extracted = {
        "title": data.get("title", "").strip(),
        "abstract": data.get("abstract", "").strip(),
        "keywords": data.get("keywords", "").strip(),
        "domain": data.get("domain", "").strip()
    }
    
    # Ensure domain is valid
    if extracted["domain"] not in DOMAINS:
        extracted["domain"] = ""
        
    return extracted


def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """
    Main metadata extraction endpoint. Tries Gemini, falls back to heuristics.
    """
    if not text.strip():
        return {
            "title": "",
            "abstract": "",
            "keywords": "",
            "domain": ""
        }
        
    if USE_LLM_EXPLANATION and GEMINI_API_KEY:
        try:
            logger.info("Using Gemini to extract paper metadata...")
            return extract_metadata_with_gemini(text)
        except Exception as e:
            logger.warning(f"Gemini metadata extraction failed: {e}. Falling back to heuristics.")
            
    logger.info("Using heuristics to extract paper metadata...")
    return extract_metadata_heuristics(text)
