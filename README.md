# 🔬 AI Research Novelty & Gap Detector
### For Arts & Science PhD Scholars in India

A production-ready AI system that helps Indian PhD scholars evaluate the originality of their research topics, detect literature gaps, and get actionable improvement suggestions.

---

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Setup Instructions](#setup-instructions)
5. [Running the System](#running-the-system)
6. [API Reference](#api-reference)
7. [Evaluation](#evaluation)
8. [Deployment](#deployment)
9. [Scalability](#scalability)
10. [Research Contribution](#research-contribution)

---

## System Overview

| Feature | Description |
|---|---|
| **Input** | Research title, abstract, keywords, domain |
| **Similarity Retrieval** | FAISS + SentenceTransformers (bge-large-en-v1.5) |
| **Novelty Score** | Multi-dimensional: HIGH / MEDIUM / LOW + percentage |
| **Gap Detection** | BERTopic + taxonomy scanning (6 dimensions) |
| **Title Suggestions** | 5 refined alternatives per query |
| **Explanation** | Natural language report (LLM or template) |
| **Backend** | FastAPI (async, production-grade) |
| **Frontend** | Streamlit with Plotly visualisations |

---

## Architecture

```
User Query (Title + Abstract + Keywords)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │ Preprocessing │───▶│  Embedding Model          │   │
│  │ (text clean) │    │  (bge-large-en-v1.5)     │   │
│  └──────────────┘    └──────────┬───────────────┘   │
│                                 │ query vector       │
│                      ┌──────────▼───────────────┐   │
│                      │   FAISS Index             │   │
│                      │   (cosine similarity)     │   │
│                      └──────────┬───────────────┘   │
│                                 │ top-k papers       │
│           ┌─────────────────────▼─────────────────┐ │
│           │           Analysis Engines              │ │
│           │  ┌──────────────┐  ┌────────────────┐  │ │
│           │  │ Novelty      │  │ Gap Detection  │  │ │
│           │  │ Engine       │  │ (BERTopic +    │  │ │
│           │  │ (multi-dim)  │  │  taxonomy)     │  │ │
│           │  └──────────────┘  └────────────────┘  │ │
│           │  ┌──────────────────────────────────┐  │ │
│           │  │  Explanation Engine (LLM/Template)│  │ │
│           │  └──────────────────────────────────┘  │ │
│           └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
         │
         ▼
  Streamlit Dashboard
  (Gauges · Tables · Charts · Gap Report · Title Suggestions)
```

---

## Project Structure

```
research-gap-detector/
├── config/
│   └── settings.py              # All configuration constants
├── data/
│   ├── raw/
│   │   └── papers.csv           # Input dataset
│   ├── processed/
│   │   └── papers_processed.csv # After preprocessing
│   └── embeddings/
│       ├── faiss_index.bin      # FAISS index
│       ├── metadata.pkl         # Paper metadata
│       └── embeddings.npy       # Raw embedding matrix
├── models/
│   ├── embedding_model.py       # SentenceTransformer wrapper
│   ├── similarity_engine.py     # Cosine sim, stats, ranking
│   ├── novelty_engine.py        # Multi-dim novelty scoring
│   ├── gap_engine.py            # BERTopic + taxonomy gap detection
│   └── explanation_engine.py    # LLM + template report generator
├── retrieval/
│   ├── faiss_index.py           # FAISS build/save/load/search
│   └── retriever.py             # High-level RAG retriever
├── api/
│   └── main.py                  # FastAPI application
├── frontend/
│   └── app.py                   # Streamlit UI
├── utils/
│   └── preprocessing.py         # Text cleaning pipeline
├── scripts/
│   ├── seed_data.py             # Generate dummy dataset
│   └── build_index.py           # End-to-end index build pipeline
├── experiments/
│   └── evaluation.ipynb         # Evaluation notebook (thesis ready)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- 4 GB RAM minimum (8 GB recommended for bge-large)
- Internet connection (first run downloads model ~1.3 GB)

### Step 1: Clone & Install

```bash
git clone <your-repo-url>
cd research-gap-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

```bash
# Option A: Generate the built-in dummy dataset (200 papers)
python scripts/seed_data.py

# Option B: Use your own dataset
# Place a CSV file at data/raw/papers.csv with columns:
# title, abstract, keywords, year, domain
```

### Step 3: Build the Index

```bash
# This downloads the embedding model and builds the FAISS index
# Takes ~3-5 minutes on CPU for 200 papers
python scripts/build_index.py
```

### Step 4: Configure (Optional)

Create a `.env` file for optional LLM explanations:
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...   # Optional: enables Claude AI explanations
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5   # or all-MiniLM-L6-v2 for speed
```

---

## Running the System

### Start the FastAPI Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# API docs: http://localhost:8000/docs
```

### Start the Streamlit Frontend

```bash
streamlit run frontend/app.py
# Opens: http://localhost:8501
```

### Quick API Test

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Digital Literacy among Rural Women in Tamil Nadu",
    "abstract": "This study examines digital skill gaps and adoption barriers.",
    "keywords": "digital literacy, rural women, Tamil Nadu",
    "domain": "Sociology"
  }'
```

---

## API Reference

### `POST /api/analyze`

Full analysis pipeline.

**Request:**
```json
{
  "title":    "string (required)",
  "abstract": "string (optional)",
  "keywords": "string (optional)",
  "domain":   "string (optional)",
  "top_k":    10
}
```

**Response:**
```json
{
  "status": "success",
  "novelty": {
    "label": "HIGH | MEDIUM | LOW",
    "percentage": 75,
    "description": "...",
    "suggestions": ["..."]
  },
  "similar_papers": [...],
  "gaps": {
    "gap_statements": ["..."],
    "gap_dimensions": {...}
  },
  "title_suggestions": ["..."],
  "explanation": "Full report text..."
}
```

### `GET /api/health`
Returns `{"status": "healthy", "corpus_size": 200, "index_ready": true}`

### `GET /api/stats`
Returns domain and year distribution of the corpus.

### `POST /api/similar`
Lightweight retrieval-only endpoint.

---

## Evaluation

Run the evaluation notebook:
```bash
jupyter notebook experiments/evaluation.ipynb
```

**Metrics evaluated:**
- Novelty label accuracy vs expert judgment
- Pearson & Spearman correlation (system vs expert score)
- MAE, RMSE
- Classification report (Precision/Recall/F1 per label)
- Gap detection Precision@K
- Per-query runtime analysis

---

## Deployment

### Docker (Recommended)

```bash
# Build and start both services
docker-compose up --build

# API:      http://localhost:8000
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

### Production (Manual)

```bash
# API with Gunicorn (4 workers)
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 --timeout 120

# Frontend (production)
streamlit run frontend/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
```

### Cloud (AWS/GCP/Azure)
- Deploy API on ECS/Cloud Run with auto-scaling
- Use EFS/GCS to persist `data/embeddings/` volume
- Place Streamlit behind CloudFront/CDN
- Use Redis for query caching in high-traffic scenarios

---

## Scalability

| Corpus Size | Recommended Index | Latency |
|---|---|---|
| < 10,000 papers | FlatIP (exact) | < 100ms |
| 10,000–500,000 | IVFFlat (set `USE_IVF=True`) | < 200ms |
| > 500,000 | HNSW or IVF-PQ | < 300ms |

For high traffic:
- Add Redis query cache (`TTL=3600s`)
- Scale API horizontally (stateless design)
- Pre-compute and cache popular query embeddings

---

## Research Contribution

This system introduces:
1. **Multi-dimensional novelty scoring** combining similarity, coverage, and recency sub-scores
2. **Taxonomy-driven gap detection** across 6 research dimensions (region, population, method, theme, theory, time)
3. **Explainable AI reports** suitable for non-technical PhD scholars
4. **Evaluation framework** with expert-judgment comparison suitable for publication

**Suitable for:**
- UGC-NET funded research projects
- ICSSR journal publications
- International conference papers (AAAI, ACL, EMNLP workshops)

---

*Built for Indian Arts & Science PhD Scholars · Powered by SentenceTransformers + FAISS + BERTopic*
