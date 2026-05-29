"""
api/main.py
-----------
FastAPI backend — v3.0
Integrated: Gemini LLM + Hallucination Detection + Plagiarism Detection
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import time

from config.settings import API_HOST, API_PORT, API_RELOAD, TOP_K_RESULTS
from retrieval.retriever import get_retriever
from utils.pdf_extractor import extract_text_from_pdf
from models.metadata_extractor import extract_metadata_from_text
from models.similarity_engine import (
    compute_similarity_stats, rank_results,
    find_duplicate_risk, summarise_domain_overlap,
    summarise_year_distribution
)
from models.novelty_engine import calculate_novelty_score, get_novelty_suggestions
from models.gap_engine import detect_research_gaps, suggest_improved_titles
from models.explanation_engine import generate_explanation, enhance_gaps_with_llm
from models.hallucination_detector import detect_hallucinations
from models.plagiarism_detector import detect_plagiarism

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Research Novelty & Gap Detector API v3.0 …")
    try:
        retriever = get_retriever()
        app.state.retriever = retriever
        logger.info(f"Retriever ready. Corpus: {retriever.get_corpus_size()} papers")
    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
        app.state.retriever = None
    yield
    logger.info("Shutting down.")


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI Research Novelty & Gap Detector",
    description=(
        "For Arts, Science & Engineering PhD Scholars in India.\n"
        "Powered by Gemini LLM + FAISS + BERTopic + "
        "Hallucination Detection + Plagiarism Detection."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    title:             str           = Field(..., min_length=5, max_length=500)
    abstract:          Optional[str] = Field(None, max_length=3000)
    keywords:          Optional[str] = Field(None, max_length=500)
    domain:            Optional[str] = Field(None, max_length=100)
    top_k:             int           = Field(default=TOP_K_RESULTS, ge=3, le=20)
    check_plagiarism:  bool          = Field(default=True)
    full_text:         Optional[str] = Field(None)

    @validator("title")
    def title_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Title must not be empty.")
        return v.strip()


class SimilarRequest(BaseModel):
    title:    str
    abstract: Optional[str] = ""
    keywords: Optional[str] = ""
    top_k:    int = Field(default=5, ge=1, le=20)


class PlagiarismRequest(BaseModel):
    title:    str = Field(..., min_length=5)
    abstract: Optional[str] = Field(None, max_length=3000)
    keywords: Optional[str] = Field(None, max_length=500)
    top_k:    int = Field(default=10, ge=5, le=20)


# ─────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────

@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    elapsed  = round((time.time() - start) * 1000, 1)
    response.headers["X-Process-Time-ms"] = str(elapsed)
    return response


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health_check(request: Request):
    retriever = request.app.state.retriever
    from config.settings import USE_LLM_EXPLANATION, LLM_MODEL
    return {
        "status":              "healthy",
        "version":             "3.0.0",
        "corpus_size":         retriever.get_corpus_size() if retriever else 0,
        "index_ready":         retriever is not None,
        "llm_enabled":         USE_LLM_EXPLANATION,
        "llm_model":           LLM_MODEL if USE_LLM_EXPLANATION else "template",
        "plagiarism_enabled":  True,
        "hallucination_check": True,
    }


@app.get("/api/stats", tags=["System"])
async def get_stats(request: Request):
    retriever = request.app.state.retriever
    if not retriever:
        raise HTTPException(503, "Retriever not initialised.")
    meta    = retriever.metadata
    domains = {}
    years   = {}
    for m in meta:
        d = m.get("domain", "Unknown")
        y = int(m.get("year", 0))
        domains[d] = domains.get(d, 0) + 1
        years[y]   = years.get(y, 0) + 1
    return {
        "total_papers": len(meta),
        "domains": dict(sorted(domains.items(), key=lambda x: x[1], reverse=True)[:15]),
        "years":   dict(sorted(years.items())),
    }


@app.post("/api/similar", tags=["Retrieval"])
async def get_similar(req: SimilarRequest, request: Request):
    retriever = request.app.state.retriever
    if not retriever:
        raise HTTPException(503, "Retriever not initialised.")
    papers = retriever.retrieve(
        title=req.title, abstract=req.abstract or "",
        keywords=req.keywords or "", top_k=req.top_k,
    )
    return {"similar_papers": rank_results(papers, top_n=req.top_k)}


@app.post("/api/plagiarism", tags=["Plagiarism"])
async def check_plagiarism_only(req: PlagiarismRequest, request: Request):
    """
    Standalone plagiarism check endpoint.
    Returns only the plagiarism report without full analysis.
    """
    retriever = request.app.state.retriever
    if not retriever:
        raise HTTPException(503, "Retriever not initialised.")

    papers    = retriever.retrieve(
        title=req.title, abstract=req.abstract or "",
        keywords=req.keywords or "", top_k=req.top_k,
    )
    sim_stats = compute_similarity_stats(papers)
    plagiarism = detect_plagiarism(
        user_title      = req.title,
        user_abstract   = req.abstract or "",
        user_keywords   = req.keywords or "",
        retrieved_papers= papers,
        similarity_stats= sim_stats,
    )
    return {"status": "success", "plagiarism": plagiarism}


@app.post("/api/analyze", tags=["Analysis"])
async def analyze_research(req: AnalyzeRequest, request: Request):
    """
    Full pipeline v3.0:
      1.  Retrieve similar papers (FAISS)
      2.  Compute similarity statistics
      3.  Calculate novelty score
      4.  Detect research gaps (BERTopic + taxonomy)
      5.  Enhance gaps with Gemini LLM
      6.  Suggest improved titles
      7.  Generate grounded LLM explanation
      8.  Run hallucination detection
      9.  Run plagiarism detection
    """
    retriever = request.app.state.retriever
    if not retriever:
        raise HTTPException(503, "Retriever not initialised. Run build_index.py first.")

    try:
        # ── 1. Retrieval ──────────────────────
        papers = retriever.retrieve(
            title    = req.title,
            abstract = req.abstract or "",
            keywords = req.keywords or "",
            top_k    = req.top_k,
        )

        # ── 2. Similarity stats ───────────────
        sim_stats   = compute_similarity_stats(papers)
        ranked      = rank_results(papers, top_n=req.top_k)
        duplicates  = find_duplicate_risk(papers)
        domain_dist = summarise_domain_overlap(papers)
        year_dist   = summarise_year_distribution(papers)

        # ── 3. Novelty score ──────────────────
        novelty = calculate_novelty_score(papers, sim_stats, req.domain or "")
        novelty["suggestions"] = get_novelty_suggestions(
            novelty["label"], sim_stats, papers
        )

        # ── 4. Gap detection ──────────────────
        gaps = detect_research_gaps(
            retrieved_papers = papers,
            user_title       = req.title,
            user_keywords    = req.keywords or "",
        )

        # ── 5. LLM gap enhancement ────────────
        gaps["gap_statements"] = enhance_gaps_with_llm(
            gap_statements   = gaps.get("gap_statements", []),
            retrieved_papers = papers,
            user_title       = req.title,
        )

        # ── 6. Title suggestions ──────────────
        title_suggestions = suggest_improved_titles(
            original_title = req.title,
            gap_dimensions = gaps.get("gap_dimensions", {}),
        )

        # ── 7. LLM explanation ────────────────
        full_analysis = {
            "input":             req.dict(),
            "novelty":           novelty,
            "gaps":              gaps,
            "similarity_stats":  sim_stats,
            "similar_papers":    ranked,
            "title_suggestions": title_suggestions,
        }
        explanation = generate_explanation(full_analysis)

        # ── 8. Hallucination detection ────────
        hallucination = detect_hallucinations(
            llm_output       = explanation,
            retrieved_papers = papers,
            novelty_data     = novelty,
        )

        # ── 9. Plagiarism detection ───────────
        plagiarism = {}
        if req.check_plagiarism:
            # If full_text of the paper is provided, run plagiarism checking on the full text
            plag_text = req.full_text if req.full_text else (req.abstract or "")
            plagiarism = detect_plagiarism(
                user_title       = req.title,
                user_abstract    = plag_text,
                user_keywords    = req.keywords or "",
                retrieved_papers = papers,
                similarity_stats = sim_stats,
            )
            logger.info(
                f"Plagiarism check: {plagiarism.get('risk_level','?')} "
                f"({plagiarism.get('plagiarism_score',0):.1f}%)"
            )

        # ── Final response ────────────────────
        return {
            "status": "success",
            "input": {
                "title":    req.title,
                "abstract": req.abstract,
                "keywords": req.keywords,
                "domain":   req.domain,
                "full_text": req.full_text,
            },
            "novelty": {
                "label":       novelty["label"],
                "percentage":  novelty["percentage"],
                "raw_score":   novelty["raw_score"],
                "color":       novelty["color"],
                "description": novelty["description"],
                "sub_scores":  novelty.get("sub_scores", {}),
                "suggestions": novelty["suggestions"],
            },
            "similarity": {
                "stats":          sim_stats,
                "domain_dist":    domain_dist,
                "year_dist":      year_dist,
                "duplicate_risk": len(duplicates) > 0,
                "duplicates":     duplicates,
            },
            "similar_papers":    ranked,
            "gaps":              gaps,
            "title_suggestions": title_suggestions,
            "explanation":       explanation,
            "hallucination": {
                "grounding_score":     hallucination.get("grounding_score", 100),
                "label":               hallucination.get("label", ""),
                "color":               hallucination.get("color", "#2ecc71"),
                "hallucination_count": hallucination.get("hallucination_count", 0),
                "total_claims":        hallucination.get("total_claims", 0),
                "flagged_claims":      hallucination.get("flagged_claims", []),
                "warnings":            hallucination.get("warnings", []),
                "summary":             hallucination.get("summary", ""),
            },
            "plagiarism": plagiarism,
        }

    except FileNotFoundError as e:
        raise HTTPException(503, f"Index not found: {e}")
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        raise HTTPException(500, f"Analysis error: {str(e)}")


@app.post("/api/extract-metadata", tags=["Analysis"])
async def extract_metadata(file: UploadFile = File(...)):
    """
    Accepts an uploaded PDF or TXT research paper file, extracts raw text,
    and attempts to extract structured metadata (title, abstract, keywords, domain).
    """
    filename = file.filename or ""
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")
        
    try:
        file_bytes = await file.read()
        
        # 1. Extract text
        if filename.endswith(".pdf"):
            logger.info(f"Extracting text from PDF upload: {filename}")
            text = extract_text_from_pdf(file_bytes)
        else:
            logger.info(f"Decoding text from TXT upload: {filename}")
            text = file_bytes.decode("utf-8", errors="ignore")
            
        # 2. Extract metadata fields
        metadata = extract_metadata_from_text(text)
        
        return {
            "status": "success",
            "filename": filename,
            "metadata": metadata,
            "full_text": text
        }
    except Exception as e:
        logger.exception(f"Metadata extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse and extract metadata: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)