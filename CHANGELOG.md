# Changelog

All notable changes to this project will be documented in this file.

## [3.1.0] - 2026-07-15

### Added
- **Hybrid Retrieval System**: Fuses FAISS dense vector search with BM25 sparse search using Reciprocal Rank Fusion (RRF, $k=60$) over title, abstract, and keywords.
- **Cross-Encoder Re-ranker**: Integrates `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranking over fused candidates to prioritize the top papers.
- **Auditable Gap size formula**: Implemented $GapSize(x) = 1.0 - AvgCoverage(x)$ with dynamic recency weights. All calculations are returned in the API logs for auditable verification.
- **Literature Review Drafting Engine**: Grouping papers according to BERTopic topic identifiers, generating synthesised reviews with inline citation tags via Gemini.
- **Self-Correction Grounding Loop**: Integrates paragraph verification against `detect_hallucinations` with up to 2 correction retries.
- **PostgreSQL Job Store**: Backed async review creation with a multi-worker safe PostgreSQL relational database to prevent race conditions across Gunicorn worker threads.
- **Streamlit Tabbed UI**: Upgraded Streamlit frontend to support the Research Novelty Analyzer and the Literature Review Drafting engine under two tabs, with async polling progress bars.
