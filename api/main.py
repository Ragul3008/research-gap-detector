"""
api/main.py
-----------
FastAPI backend — v3.0
Integrated: Gemini LLM + Hallucination Detection + Plagiarism Detection
"""

import sys
import logging
import uuid
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import time

from config.settings import API_HOST, API_PORT, API_RELOAD, TOP_K_RESULTS, DATABASE_URL
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
from models.explanation_engine import generate_explanation, enhance_gaps_with_llm, suggest_relevant_authors
from models.hallucination_detector import detect_hallucinations
from models.plagiarism_detector import detect_plagiarism

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DB Helpers
# ─────────────────────────────────────────────

def init_db():
    """Create jobs table in PostgreSQL if not exists."""
    logger.info("Connecting to PostgreSQL to initialise jobs table...")
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS literature_review_jobs (
                job_id VARCHAR(100) PRIMARY KEY,
                status VARCHAR(50) NOT NULL,
                result TEXT,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
        logger.info("PostgreSQL database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL database: {e}")
    finally:
        if conn:
            conn.close()

def save_job(job_id: str, status: str, result: Optional[str] = None, error: Optional[str] = None):
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO literature_review_jobs (job_id, status, result, error, created_at, updated_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (job_id) DO UPDATE SET
                status = EXCLUDED.status,
                result = COALESCE(EXCLUDED.result, literature_review_jobs.result),
                error = COALESCE(EXCLUDED.error, literature_review_jobs.error),
                updated_at = CURRENT_TIMESTAMP;
        """, (job_id, status, result, error))
        conn.commit()
        cur.close()
    except Exception as e:
        logger.error(f"Failed to save job {job_id} to database: {e}")
    finally:
        if conn:
            conn.close()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT job_id, status, result, error, created_at, updated_at FROM literature_review_jobs WHERE job_id = %s", (job_id,))
        row = cur.fetchone()
        cur.close()
        if row:
            row["created_at"] = str(row["created_at"])
            row["updated_at"] = str(row["updated_at"])
            return dict(row)
        return None
    except Exception as e:
        logger.error(f"Failed to get job {job_id} from database: {e}")
        return None
    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Research Novelty & Gap Detector API v3.0 …")
    init_db()
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
    top_k:             int           = Field(default=TOP_K_RESULTS, ge=1, le=100)
    num_authors:       int           = Field(default=10, ge=1, le=100)
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

        # ── 6.5 Author suggestions ────────────
        author_suggestions = suggest_relevant_authors(
            title = req.title,
            abstract = req.abstract or "",
            keywords = req.keywords or "",
            domain = req.domain or "",
            num_authors = req.num_authors
        )

        # ── 7. LLM explanation ────────────────
        full_analysis = {
            "input":             req.dict(),
            "novelty":           novelty,
            "gaps":              gaps,
            "similarity_stats":  sim_stats,
            "similar_papers":    ranked,
            "title_suggestions": title_suggestions,
            "author_suggestions": author_suggestions,
        }
        explanation = generate_explanation(full_analysis)

        # ── 8. Hallucination detection ────────
        hallucination = detect_hallucinations(
            llm_output       = explanation,
            retrieved_papers = papers,
            novelty_data     = novelty,
            gap_data         = gaps,
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
            "author_suggestions": author_suggestions,
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


class LiteratureReviewRequest(BaseModel):
    title: str = Field(..., min_length=5, max_length=500)
    abstract: Optional[str] = Field(None, max_length=3000)
    keywords: Optional[str] = Field(None, max_length=500)
    domain: Optional[str] = Field(None, max_length=100)
    top_k: int = Field(default=8, ge=3, le=20)
    section_preference: str = Field(default="thematic")
    run_async: bool = Field(default=True)

def run_literature_review_task(
    job_id: str,
    title: str,
    abstract: Optional[str],
    keywords: Optional[str],
    domain: Optional[str],
    top_k: int,
    section_preference: str
):
    save_job(job_id, "RUNNING")
    try:
        retriever = get_retriever()
        papers = retriever.retrieve(
            title=title, abstract=abstract or "",
            keywords=keywords or "", top_k=top_k
        )
        gaps = detect_research_gaps(
            retrieved_papers=papers,
            user_title=title,
            user_keywords=keywords or ""
        )
        from models.literature_review_engine import LiteratureReviewService
        review = LiteratureReviewService.compile_review(
            query_title=title,
            retrieved_papers=papers,
            gap_data=gaps,
            section_preference=section_preference
        )
        save_job(job_id, "COMPLETED", result=json.dumps(review))
        logger.info(f"Background Job {job_id} completed successfully.")
    except Exception as e:
        logger.exception(f"Background Job {job_id} failed: {e}")
        save_job(job_id, "FAILED", error=str(e))

@app.post("/api/literature-review", tags=["Literature Review"])
async def create_literature_review(req: LiteratureReviewRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    save_job(job_id, "PENDING")
    
    if req.run_async:
        background_tasks.add_task(
            run_literature_review_task,
            job_id, req.title, req.abstract, req.keywords, req.domain, req.top_k, req.section_preference
        )
        return {"job_id": job_id, "status": "PENDING", "message": "Literature review compilation started in background."}
    else:
        run_literature_review_task(
            job_id, req.title, req.abstract, req.keywords, req.domain, req.top_k, req.section_preference
        )
        job = get_job(job_id)
        if job and job["status"] == "COMPLETED":
            return {"job_id": job_id, "status": "COMPLETED", "result": json.loads(job["result"])}
        else:
            raise HTTPException(500, f"Compilation failed: {job.get('error') if job else 'Unknown error'}")

@app.get("/api/literature-review/status/{job_id}", tags=["Literature Review"])
async def check_literature_review_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")
        
    response = {
        "job_id": job["job_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"]
    }
    if job["status"] == "COMPLETED" and job["result"]:
        response["result"] = json.loads(job["result"])
    elif job["status"] == "FAILED" and job["error"]:
        response["error"] = job["error"]
        
    return response


if __name__ == "__main__":
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)