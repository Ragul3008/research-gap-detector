---
title: Research Gap Detector
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# 🔬 AI Research Novelty & Gap Detector (v3.0)
### For Arts, Science & Engineering PhD Scholars in India

A production-ready, RAG-powered AI platform that helps Indian PhD scholars evaluate the originality of their proposed research topics, discover literature gaps across multiple dimensions, assess plagiarism risks, and review hallucination-free AI advisory reports.

---

## 📋 Table of Contents
1. [What is this Project?](#what-is-this-project)
2. [Why was this Project Developed?](#why-was-this-project-developed)
3. [What Has Been Done? (Technical Core & Features)](#what-has-been-done-technical-core--features)
4. [Architecture](#architecture)
5. [Project Structure](#project-structure)
6. [Setup Instructions](#setup-instructions)
7. [Running the System](#running-the-system)
8. [API Reference](#api-reference)
9. [Evaluation](#evaluation)
10. [Deployment](#deployment)
11. [Scalability](#scalability)
12. [Research Contribution](#research-contribution)

---

## What is this Project?

The **AI Research Novelty & Gap Detector** is a specialized academic analytics platform designed for PhD scholars in India. It works as a virtual research co-advisor, taking a scholar's proposed research title, abstract, keywords, and domain, and evaluating them against a corpus of existing academic papers. 

The system provides:
* **Novelty Evaluation**: Measures the semantic uniqueness of a topic on a multi-dimensional scale (High, Medium, Low novelty).
* **Taxonomy-Driven Gap Detection**: Scans the literature for omissions in geographic regions, population subgroups, research methodologies, theories, themes, and timelines.
* **Plagiarism & Similarity Auditing**: Detects Jaccard word-overlap, exact multi-word phrases, and dense semantic duplicate risks before academic submission.
* **Explainable, Grounded AI Reports**: Leverages Google Gemini LLM (with a robust template fallback) to compose an advisory report, verified by a custom Hallucination Detection Engine to ensure all claims are fully backed by the retrieved literature.

---

## Why was this Project Developed?

Developing a thesis topic is one of the most challenging phases of a researcher's journey. This project was developed to solve several structural issues in Indian academia:

1. **Accidental Topic Duplication**: Scholars frequently spend months formulating a topic, only to later discover that a highly similar study was already published. Manual literature search is slow, fragmented, and prone to human oversight.
2. **Lack of Actionable Recommendations**: General search engines (like Google Scholar or Scopus) tell scholars *what exists*, but do not compile *what is missing*. Scholars need to know how to pivot their topics (e.g., using a different research methodology, shifting regional focus, or choosing an underrepresented population).
3. **Strict Academic Integrity Standards**: Under the University Grants Commission (UGC) regulations in India, academic plagiarism is met with severe penalties. Scholars need reliable tools to check similarity and rewrite overlapping passages early in the drafting stage.
4. **Untrustworthy AI Explanations**: While general-purpose LLMs (like ChatGPT or Claude) can discuss research ideas, they regularly hallucinate references, citing studies that do not exist. This tool grounds AI responses in real, retrieved research papers and runs a validation check on every generated claim.

---

## What Has Been Done? (Technical Core & Features)

The project is fully implemented with a FastAPI backend and a Streamlit frontend. The following features and architectural components have been successfully developed:

* **Retrieval-Augmented Generation (RAG) Core**:
  * Powered by [embedding_model.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/embedding_model.py) which loads SentenceTransformers (`all-MiniLM-L6-v2` or `bge-large-en-v1.5` embeddings) to convert research texts into dense vectors.
  * Utilizes [faiss_index.py](file:///c:/Users/ragul/Desktop/research-gap-detector/retrieval/faiss_index.py) and [retriever.py](file:///c:/Users/ragul/Desktop/research-gap-detector/retrieval/retriever.py) to execute high-speed cosine-similarity queries over a vector corpus.
* **Multi-Dimensional Novelty Engine**:
  * Implemented in [novelty_engine.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/novelty_engine.py). It combines a similarity sub-score (50%), domain coverage (20%), and recency of similar studies (20%), applying a penalty if a single document is extremely close.
  * Categorizes novelty as `HIGH`, `MEDIUM`, or `LOW` and suggests specific actions (e.g., switching from qualitative to ethnographic, or narrowing geographic scope).
* **Taxonomy-Driven Research Gap Engine**:
  * Implemented in [gap_engine.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/gap_engine.py). It checks abstracts and titles of retrieved papers against pre-defined taxonomies for Region, Population, Method, Theme, Theory, and Time.
  * Extracts latent topics using **BERTopic** (with a TF-IDF fallback) and suggests 3 optimized research titles targeting the identified gaps.
* **Document Metadata Extraction**:
  * Implemented in [metadata_extractor.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/metadata_extractor.py) and [pdf_extractor.py](file:///c:/Users/ragul/Desktop/research-gap-detector/utils/pdf_extractor.py) to automatically extract titles, abstracts, keywords, and academic domains from uploaded PDF or TXT draft files.
* **Plagiarism Detection Engine**:
  * Developed in [plagiarism_detector.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/plagiarism_detector.py) to run standalone or pipeline plagiarism checks.
  * Evaluates overall risk by combining FAISS semantic similarity, sentence Jaccard similarity (word overlap with stopwords excluded), and exact n-gram phrase matches.
* **Hallucination Detection Engine**:
  * Developed in [hallucination_detector.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/hallucination_detector.py). It parses LLM-generated reports, extracts concrete claims (numerical percentages, years, quoted titles, specific domains/regions), verifies them against the retrieved papers, and scores the grounding (0-100).
* **Grounded LLM Advisory Reports**:
  * Created in [explanation_engine.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/explanation_engine.py). Connects to Google Gemini (`gemini-1.5-flash` or `gemini-1.5-pro`) using strict instructions to prevent hallucinated references. Integrates a template-based advisory report generator as an offline fallback.
* **Interactive Frontend & API**:
  * The Streamlit dashboard ([app.py](file:///c:/Users/ragul/Desktop/research-gap-detector/frontend/app.py) & [app_hf.py](file:///c:/Users/ragul/Desktop/research-gap-detector/frontend/app_hf.py)) offers responsive gauge widgets, Plotly similarity bar charts, domain and publication year distributions, and downloadable text reports.
  * The FastAPI service ([main.py](file:///c:/Users/ragul/Desktop/research-gap-detector/api/main.py)) exposes endpoints with complete OpenAPI validation, process-timing headers, and file upload capabilities.
* **Docker Containerization**:
  * Setup files [Dockerfile](file:///c:/Users/ragul/Desktop/research-gap-detector/Dockerfile) and [docker-compose.yml](file:///c:/Users/ragul/Desktop/research-gap-detector/docker-compose.yml) package both the API backend and Streamlit frontend.

---

## Architecture

```
User Query (Title + Abstract + Keywords) or File Upload (.pdf / .txt)
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                         │
│                                                            │
│  ┌───────────────────────┐    ┌─────────────────────────┐  │
│  │ Metadata Extraction   │───▶│  Embedding Model        │  │
│  │ (Gemini/Heuristics)   │    │  (bge-large-en-v1.5)    │  │
│  └───────────────────────┘    └──────────┬──────────────┘  │
│                                          │ query vector     │
│                               ┌──────────▼──────────────┐  │
│                               │   FAISS Index           │  │
│                               │   (cosine similarity)   │  │
│                               └──────────┬──────────────┘  │
│                                          │ top-k papers    │
│           ┌──────────────────────────────▼────────────────┐│
│           │                Analysis Engines               ││
│           │  ┌──────────────┐  ┌───────────────────────┐  ││
│           │  │ Novelty      │  │ Plagiarism Detector   │  ││
│           │  │ Engine       │  │ (Sentence / n-grams)  │  ││
│           │  └──────────────┘  └───────────────────────┘  ││
│           │  ┌──────────────┐  ┌───────────────────────┐  ││
│           │  │ Gap Engine   │  │ Explanation Engine    │  ││
│           │  │ (BERTopic)   │  │ (Gemini / Templates)  │  ││
│           │  └──────────────┘  └──────────┬────────────┘  ││
│           │                               │ text report   ││
│           │                    ┌──────────▼────────────┐  ││
│           │                    │ Hallucination Checker │  ││
│           │                    │ (Fact-verification)   │  ││
│           │                    └───────────────────────┘  ││
│           └───────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────┘
         │
         ▼
  Streamlit Dashboard
  (Gauges · Similarity Bars · Gap Metrics · Title Pivots · Plagiarism Flags)
```

---

## Project Structure

* [config/](file:///c:/Users/ragul/Desktop/research-gap-detector/config)
  * [settings.py](file:///c:/Users/ragul/Desktop/research-gap-detector/config/settings.py): Central configuration file housing paths, thresholds, models, and LLM flags.
* [data/](file:///c:/Users/ragul/Desktop/research-gap-detector/data)
  * `raw/papers.csv`: Seed/input corpus.
  * `processed/papers_processed.csv`: Cleaned dataset.
  * `embeddings/`: Contains `faiss_index.bin`, `metadata.pkl`, and embeddings arrays.
* [models/](file:///c:/Users/ragul/Desktop/research-gap-detector/models)
  * [embedding_model.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/embedding_model.py): Singleton wrapper for the sentence embedding transformers.
  * [similarity_engine.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/similarity_engine.py): Rank calculations, duplicate tracking, and corpus metrics.
  * [novelty_engine.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/novelty_engine.py): Multi-factor novelty scoring and direction pivoting suggestions.
  * [gap_engine.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/gap_engine.py): Predefined taxonomies, keyword scans, and BERTopic topic extraction.
  * [explanation_engine.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/explanation_engine.py): Prepares grounded advisor reports via Gemini.
  * [hallucination_detector.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/hallucination_detector.py): Parses generated text to check alignment against retrieved references.
  * [plagiarism_detector.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/plagiarism_detector.py): Computes similarity percentages and flags overlapping segments.
  * [metadata_extractor.py](file:///c:/Users/ragul/Desktop/research-gap-detector/models/metadata_extractor.py): Extracts title/abstract/keywords from raw paper text.
* [retrieval/](file:///c:/Users/ragul/Desktop/research-gap-detector/retrieval)
  * [faiss_index.py](file:///c:/Users/ragul/Desktop/research-gap-detector/retrieval/faiss_index.py): Index build, search, save, and load operations.
  * [retriever.py](file:///c:/Users/ragul/Desktop/research-gap-detector/retrieval/retriever.py): High-level retrieve orchestrator.
* [api/](file:///c:/Users/ragul/Desktop/research-gap-detector/api)
  * [main.py](file:///c:/Users/ragul/Desktop/research-gap-detector/api/main.py): FastAPI app implementing analysis, metadata, plagiarism, and metrics endpoints.
* [frontend/](file:///c:/Users/ragul/Desktop/research-gap-detector/frontend)
  * [app.py](file:///c:/Users/ragul/Desktop/research-gap-detector/frontend/app.py): Complete streamlit workspace with metrics gauges, interactive charts, and PDF ingestion.
  * [app_hf.py](file:///c:/Users/ragul/Desktop/research-gap-detector/frontend/app_hf.py): Streamlit app structured for Hugging Face Spaces.
* [utils/](file:///c:/Users/ragul/Desktop/research-gap-detector/utils)
  * [preprocessing.py](file:///c:/Users/ragul/Desktop/research-gap-detector/utils/preprocessing.py): Queries cleanup and token-cleaning operations.
  * [pdf_extractor.py](file:///c:/Users/ragul/Desktop/research-gap-detector/utils/pdf_extractor.py): Utility reading uploaded PDF files.
* [scripts/](file:///c:/Users/ragul/Desktop/research-gap-detector/scripts)
  * [seed_data.py](file:///c:/Users/ragul/Desktop/research-gap-detector/scripts/seed_data.py): Seeds a realistic test dataset of 1,500 research papers.
  * [build_index.py](file:///c:/Users/ragul/Desktop/research-gap-detector/scripts/build_index.py): Compiles and writes the FAISS vector index database.
* [experiments/](file:///c:/Users/ragul/Desktop/research-gap-detector/experiments)
  * [evaluation.ipynb](file:///c:/Users/ragul/Desktop/research-gap-detector/experiments/evaluation.ipynb): Jupyter evaluation framework comparing system scores against expert assessments.

---

## Setup Instructions

### Prerequisites
* Python 3.10+
* 4 GB RAM minimum (8 GB recommended for downloading embedding models)
* Internet connection (first run downloads the embedding model, ~1.3 GB for BGE-Large, ~350 MB for MiniLM)

### Step 1: Clone & Install

```bash
git clone <your-repo-url>
cd research-gap-detector

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Dataset

```bash
# Generate the built-in synthetic dataset of 1,500 papers
python scripts/seed_data.py

# (Optional) Use your own dataset:
# Replace/populate the CSV file at data/raw/papers.csv with columns:
# title, abstract, keywords, year, domain
```

### Step 3: Build the FAISS Vector Index

```bash
# This embeds the dataset papers and compiles the FAISS index files
python scripts/build_index.py
```

### Step 4: Configure API Credentials (Optional)

Create a `.env` file in the project root to enable Gemini-based report generation and metadata parsing:
```bash
# .env
GEMINI_API_KEY=AIzaSy...               # Get a key from https://aistudio.google.com
EMBEDDING_MODEL=all-MiniLM-L6-v2       # Use BAAI/bge-large-en-v1.5 for maximum quality
```

---

## Running the System

### Start the FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Interactive API documentation: http://localhost:8000/docs
```

### Start the Streamlit Frontend (Legacy)
```bash
streamlit run frontend/app.py
# Opens legacy dashboard at http://localhost:8501
```

### Start the Next.js Web Frontend (v3.1)
The Next.js React dashboard is located in the `web/` folder:
```bash
cd web
npm install
npm run dev
# Opens the Next.js scholarly dashboard at http://localhost:3000
```
Configure `NEXT_PUBLIC_API_URL` in `web/.env.local` to override the default API base url if your FastAPI server runs on a different port or host.

### Quick Test via cURL
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
Executes the full research pipeline (Retrieval, Novelty Scoring, Gap analysis, LLM explanation, Hallucination checks, and Plagiarism auditing).

* **Request Payload**:
  ```json
  {
    "title": "string (required, min_length=5)",
    "abstract": "string (optional)",
    "keywords": "string (optional)",
    "domain": "string (optional)",
    "top_k": 10,
    "check_plagiarism": true,
    "full_text": "string (optional)"
  }
  ```

### `POST /api/plagiarism`
A standalone endpoint to evaluate plagiarism risk.
* **Request Payload**:
  ```json
  {
    "title": "string (required)",
    "abstract": "string (optional)",
    "keywords": "string (optional)",
    "top_k": 10
  }
  ```

### `POST /api/extract-metadata`
Accepts an uploaded PDF or TXT research file, extracts raw text, and generates structured fields.
* **Payload**: Form-data file upload (`file: UploadFile`).

### `GET /api/health`
Returns system component states, active model name, corpus length, and database readiness.

### `GET /api/stats`
Exposes corpus analytics, including document volume distributions by publication domain and year.

---

## Evaluation

Validate and inspect model configurations using the [evaluation.ipynb](file:///c:/Users/ragul/Desktop/research-gap-detector/experiments/evaluation.ipynb) notebook:
```bash
jupyter notebook experiments/evaluation.ipynb
```
The evaluation module benchmarks:
1. **Novelty Classification Accuracy**: Replicability of system categories against human expert assessments.
2. **Correlation Metrics**: Pearson and Spearman rank correlation between computed metrics and manual evaluations.
3. **Statistical Errors**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
4. **Precision/Recall**: Precision@K evaluations for gap listings and semantic retrievals.

---

## Deployment

### Containerized Deployment (Docker)
Build and run the entire stack in isolated containers:
```bash
# Build and start both API and Streamlit UI
docker-compose up --build

# API:      http://localhost:8000
# Frontend: http://localhost:8501
```

### Production Deployment (Manual)
Run the application using production servers:
```bash
# API deployment with Gunicorn (4 workers)
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 120

# Frontend headless launch
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
```

---

## Scalability

The FAISS index scales dynamically based on corpus size:
* **Small Corpora (<10,000 papers)**: Uses exact matching (`FlatIP`). Search latency is under 100ms.
* **Medium Corpora (10,000 - 500,000 papers)**: Uses index partitioning (`IVFFlat` by setting `USE_IVF=True`). Search latency is under 200ms.
* **Large Corpora (>500,000 papers)**: Recommended configuration utilizes HNSW (Hierarchical Navigable Small World) or IVF-PQ index designs. Search latency is under 300ms.

---

## Research Contribution

The system introduces several scientific and practical implementations:
1. **Multi-dimensional Novelty Formulas**: Computes a single, normalized percentage score combining semantic distances, temporal publication distributions, and thematic clustering.
2. **Multi-Dimensional Taxonomy Mapping**: Dynamically tracks research variables (methods, regions, demographics) to identify structural omissions in literature.
3. **Dual Plagiarism Checking**: Evaluates both dense semantic duplication (FAISS distance) and verbatim sentence matches.
4. **Grounded Fact Checking**: A dedicated validator verifies LLM assertions against academic citations to ensure reliable information delivery.

---

*Built for Indian Arts, Science & Engineering PhD Scholars · Powered by SentenceTransformers + FAISS + BERTopic + Gemini LLM*
