# 🔬 AI Research Novelty & Gap Detector — System Architecture & Details (v3.1)

This document provides a comprehensive technical overview of the **AI Research Novelty & Gap Detector** system. It details the retrieval mechanics, gap analysis formulas, plagiarism detection, hallucination-checking, database job store, and the Next.js React client layout.

---

## 📋 Table of Contents
1. [System Overview & Purpose](#system-overview--purpose)
2. [Backend Engine Details](#backend-engine-details)
   - [Hybrid Retrieval & Re-ranking](#1-hybrid-retrieval--re-ranking)
   - [Novelty Scoring Mechanics](#2-novelty-scoring-mechanics)
   - [Taxonomy-Driven Gap Calculations](#3-taxonomy-driven-gap-calculations)
   - [Hallucination Grounding Validator](#4-hallucination-grounding-validator)
   - [Plagiarism Auditing](#5-plagiarism-auditing)
   - [PostgreSQL Literature Review Compiler](#6-postgresql-literature-review-compiler)
3. [Repository File Structure](#repository-file-structure)
4. [Next.js Client Structure](#next-js-client-structure)
5. [Setup & Execution Steps](#setup--execution-steps)

---

## System Overview & Purpose

Developing a PhD thesis topic requires verifying that the proposed study is novel, identifies a clear gap, and is grounded in existing literature. This system serves as a virtual co-advisor for PhD scholars in India across Arts, Science, and Engineering. 

It takes a research **Title**, **Abstract**, **Keywords**, and **Domain**, searches a database of indexed papers, and returns actionable metrics, gap warnings, plagiarism risks, and a synthesised literature review draft.

---

## Backend Engine Details

### 1. Hybrid Retrieval & Re-ranking
- **Dense Search**: Converts inputs into 1024-dimensional semantic embeddings using `BAAI/bge-large-en-v1.5` and queries a FAISS vector index (`FlatIP` inner-product cosine similarity).
- **Sparse Search**: Indexes titles, abstracts, and keywords using the BM25 Okapi algorithm to guarantee match accuracy on keyword tokens (e.g. specific regions or tools).
- **Reciprocal Rank Fusion (RRF)**: Merges dense and sparse search pools into a single candidate rank using:
  $$RRF(d) = \frac{1}{60 + Rank_{Dense}(d)} + \frac{1}{60 + Rank_{Sparse}(d)}$$
- **Cross-Encoder Re-ranker**: Pass the top fused results through `cross-encoder/ms-marco-MiniLM-L-6-v2` to score query-document pairings. Returns the top 10 re-ranked papers.

### 2. Novelty Scoring Mechanics
Combines three sub-scores to yield an overall percentage (Categorized as `HIGH`, `MEDIUM`, or `LOW`):
- **Similarity Sub-Score (50%)**: Derived from the Cross-Encoder score of the most similar paper. Apply a penalty if the top paper exceeds a critical similarity limit (e.g., duplicate risk).
- **Domain Coverage Sub-Score (30%)**: Measures how saturated the domain is with similar papers.
- **Recency Sub-Score (20%)**: penalizes topics where highly similar research was published within the last 3 years.

### 3. Taxonomy-Driven Gap Calculations
Matches titles and abstracts against taxonomy logs:
- **Taxonomies Scanned**: Region (states/tribes), Population (demographics), Method (study styles), Theme (challenges), Theory (scientific frameworks), and Time Period.
- **On-Disk Taxonomy Cache**: Cache taxonomy extractions in `data/embeddings/taxonomy_cache.json` keyed by SHA-256 content hashes to eliminate duplicate Gemini API calls.
- **Dynamic Auditable Gap Size Formula**:
  $$GapSize(x) = 1.0 - AvgCoverage(x)$$
  $$AvgCoverage(x) = \frac{1}{|Papers|} \sum_{p \in Papers} Confidence(x, p) \cdot RecencyWeight(p)$$
- **Recency Coefficient**: Evaluates temporal relevance, setting newer papers (2020+) to high weights and older papers (pre-2015) to lower weights.

### 4. Hallucination Grounding Validator
- Scans LLM advisory reports for sentences stating that certain parameters are "unexplored," "missing," or "gaps."
- Checks the average coverage score of these parameters in the actual database logs.
- If a claimed gap has `AvgCoverage > 0.5`, the claim is flagged as a high-severity hallucination contradiction.

### 5. Plagiarism Auditing
- **Sentence-Level Plagiarism**: Extracts sentence pairs and measures overlap using Jaccard Similarity (removing common stop words).
- **Phrase-Level Plagiarism**: Identifies exact sequential string matches exceeding 4 consecutive words.
- **Risk Level**: Evaluated as `HIGH`, `MODERATE`, or `LOW` based on maximum sentence overlap and phrase matches.

### 6. PostgreSQL Literature Review Compiler
- **Thematic Clustering**: Assigns papers to clusters using **BERTopic** topic indices.
- **Draft Synthesis**: Gemini synthesises a cohesive paragraph for each cluster incorporating inline citations (e.g. `[Paper 1]`).
- **Self-Correction Grounding Loop**: Submits drafted text to the Hallucination Validator. If hallucinated gap claims are flagged, it requests a retry from Gemini (up to 2 attempts) to adjust wording.
- **Multi-Worker Persistence**: Background compilation status is managed inside a PostgreSQL relational database table (`literature_review_jobs`) to guarantee consistency across multi-worker deployments (e.g., Gunicorn).

---

## Repository File Structure

```
research-gap-detector/
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI routes, schemas, database helpers
├── config/
│   ├── __init__.py
│   └── settings.py                 # Paths, parameters, and environmental bindings
├── data/
│   ├── raw/
│   │   └── papers.csv              # Seed database of research papers
│   ├── processed/
│   │   └── papers_processed.csv    # Tokenized and preprocessed text corpus
│   └── embeddings/
│       ├── embeddings.npy          # Raw generated vector array
│       ├── faiss_index.bin         # Compiled search index file
│       ├── metadata.pkl            # Index-aligned paper metadata records
│       └── taxonomy_cache.json     # Cache for taxonomy extractions
├── experiments/
│   └── evaluation.ipynb            # Jupyter evaluation pipeline comparing retrieval methods
├── frontend/
│   ├── app.py                      # Multi-tab Streamlit dashboard
│   └── app_hf.py                   # Streamlit layout for HuggingFace Spaces
├── models/
│   ├── __init__.py
│   ├── embedding_model.py          # BGE-Large-v1.5 embedding loader (GPU/CPU)
│   ├── explanation_engine.py       # Gemini advisory report compiler
│   ├── gap_engine.py               # Taxonomy calculations and audit logging
│   ├── hallucination_detector.py   # Grounding validation checking
│   ├── literature_review_engine.py # Thematic compilers and self-correction loop
│   ├── novelty_engine.py           # Novelty percentage calculators
│   └── plagiarism_detector.py     # Sentence Jaccard and phrase scanners
├── retrieval/
│   ├── __init__.py
│   ├── faiss_index.py              # FlatIP index compiler and loader
│   └── retriever.py                # BM25 + FAISS hybrid RRF execution
├── scripts/
│   ├── build_index.py              # Compilation pipeline index builder
│   └── seed_data.py                # Synthetic dataset seeder script
├── utils/
│   ├── __init__.py
│   ├── pdf_extractor.py            # PDF parser
│   └── preprocessing.py            # Stopwords and string cleaners
├── web/                            # Next.js 16 Web Dashboard Client
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── .env                            # API keys, DB URL, and settings
└── README.md
```

---

## Next.js Client Structure

Inside the `web/` folder, the TypeScript React application is configured as follows:

```
web/src/
├── app/
│   ├── layout.tsx                  # Font embedding (Crimson Pro & Atkinson), Query client providers
│   ├── globals.css                 # Import styling rules and theme bindings
│   ├── page.tsx                    # Ingestion panel & Analysis Form
│   ├── analysis/
│   │   └── [id]/
│   │       └── page.tsx            # Results dashboard with gauges, charts, Plagiarism panel
│   └── literature-review/
│       ├── page.tsx                # Literature Review setup configurations
│       └── [jobId]/
│           └── page.tsx            # Async review progress polling & Markdown rendering
│
├── components/
│   ├── forms/
│   │   ├── AnalysisForm.tsx
│   │   ├── FileUploadField.tsx     # Ingest proposal PDF to prefill values
│   │   └── FormField.tsx           # Shared inputs primitives
│   ├── results/
│   │   ├── NoveltyGauge.tsx        # Percentage needle gauge
│   │   ├── GapRadarChart.tsx       # Recharts 6-axes Radar Chart (Region, Pop, Method, Theme, Theory, Time)
│   │   ├── GapDataTable.tsx        # Accessible table displaying the exact calculations
│   │   ├── PlagiarismPanel.tsx     # Flags sentence Jaccard and phrase duplicates
│   │   ├── AdvisoryReport.tsx      # Renders Gemini report and highlights hallucinated gap warnings
│   │   ├── SuggestedTitles.tsx
│   │   └── ReportDownloadButton.tsx
│   └── literature-review/
│       ├── ReviewProgress.tsx      # Multi-stage PostgreSQL job progress bar
│       ├── ThematicSection.tsx     # Section paragraph with citation tag highlights
│       ├── CitationList.tsx        # List of mapped reference papers
│       └── GapSummarySection.tsx   # Closing thesis suggestions
│
├── lib/
│   ├── api.ts                      # Client-side API wrapper targeting NEXT_PUBLIC_API_URL
│   ├── types.ts                    # TypeScript request/response contracts matching backend
│   ├── queryClient.ts              # TanStack Query bindings
│   └── hooks/                      # Query hooks: useAnalyze, usePlagiarismCheck, useLiteratureReviewStatus
│
└── styles/
    └── tokens.css                  # Color palette variables (slate border, gold accent, navy primary)
```

---

## Setup & Execution Steps

### 1. Database Configuration
Ensure PostgreSQL is active and export the connection path in your `.env` file:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/gap_detector
```

### 2. Seeding & Indexing
Run the scripts in order to seed the data corpus and build the semantic FAISS index:
```bash
# Seed 100 papers for local testing (can increase parameters in seed_data.py)
venv\Scripts\python scripts/seed_data.py

# Generate dense vector embeddings and compile FAISS
venv\Scripts\python scripts/build_index.py
```

### 3. Start Backend Server
```bash
venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: `http://localhost:8000/docs`

### 4. Start Next.js Client
```bash
cd web
npm install
npm run dev
```
- Dashboards: `http://localhost:3000` (Main Novelty Scan) & `http://localhost:3000/literature-review` (Review Compile).
