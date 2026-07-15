export interface AnalyzeRequest {
  title: string;
  abstract?: string;
  keywords?: string;
  domain?: string | null;
  top_k?: number;
  num_authors?: number;
  check_plagiarism?: boolean;
  full_text?: string;
}

export interface NoveltySubScores {
  similarity_sub: number;
  coverage_sub: number;
  recency_sub: number;
}

export interface NoveltyDetails {
  label: "HIGH" | "MEDIUM" | "LOW";
  percentage: number;
  raw_score: number;
  color: string;
  description: string;
  sub_scores: NoveltySubScores;
  suggestions: string[];
}

export interface SimilarityStats {
  max_sim: number;
  mean_sim: number;
  median_sim: number;
  std_sim: number;
}

export interface DomainDistribution {
  [domainName: string]: number;
}

export interface YearDistribution {
  [year: string]: number;
}

export interface SimilarityData {
  stats: SimilarityStats;
  domain_dist: DomainDistribution;
  year_dist: YearDistribution;
  duplicate_risk: boolean;
  duplicates: any[];
}

export interface PaperMetadata {
  id: string;
  title: string;
  abstract?: string;
  keywords?: string;
  year?: number;
  domain?: string;
  region?: string;
  method?: string;
  theme?: string;
  similarity_pct?: number;
  citation_tag?: string;
}

export interface GapDimensionLog {
  item: string;
  gap_size: number;
  avg_coverage: number;
  matched_papers: string[];
}

export interface AuditableFormula {
  formula: string;
  dimensions_log: GapDimensionLog[];
}

export interface GapDimensions {
  regional_gaps: string[];
  population_gaps: string[];
  methodological_gaps: string[];
  thematic_gaps: string[];
  theoretical_gaps: string[];
  temporal_gaps: string[];
}

export interface GapsDetails {
  covered_topics: string[];
  bertopic_topics: string[];
  gap_dimensions: GapDimensions;
  gap_statements: string[];
  auditable_formulas: {
    [dimensionName: string]: AuditableFormula;
  };
}

export interface HallucinationFlag {
  claim: string;
  issue: string;
  severity: "HIGH" | "MEDIUM" | "LOW";
}

export interface HallucinationDetails {
  grounding_score: number;
  label: string;
  color: string;
  hallucination_count: number;
  total_claims: number;
  flagged_claims: HallucinationFlag[];
  warnings: string[];
  summary: string;
}

export interface SentenceMatch {
  user_sentence: string;
  paper_sentence: string;
  paper_title: string;
  paper_year: number;
  jaccard: number;
}

export interface PhraseMatch {
  phrase: string;
  paper_title: string;
  word_count: number;
}

export interface PlagiarismResult {
  plagiarism_score: number;
  originality_score: number;
  risk_level: "HIGH" | "MODERATE" | "LOW";
  matched_papers: any[];
  sentence_matches: SentenceMatch[];
  phrase_matches: PhraseMatch[];
}

export interface AnalyzeResponse {
  status: "success";
  input: {
    title: string;
    abstract?: string;
    keywords?: string;
    domain?: string;
    full_text?: string;
  };
  novelty: NoveltyDetails;
  similarity: SimilarityData;
  similar_papers: PaperMetadata[];
  gaps: GapsDetails;
  title_suggestions: string[];
  author_suggestions: { name: string; reason: string }[];
  explanation: string;
  hallucination: HallucinationDetails;
  plagiarism?: PlagiarismResult;
}

export interface LiteratureReviewRequest {
  title: string;
  abstract?: string;
  keywords?: string;
  domain?: string | null;
  top_k?: number;
  section_preference?: string;
  run_async?: boolean;
}

export interface ReviewSectionPaper {
  citation_tag: string;
  title: string;
  year: string;
  domain: string;
}

export interface ReviewSection {
  title: string;
  paragraph: string;
  papers: ReviewSectionPaper[];
}

export interface LiteratureReviewResult {
  sections: ReviewSection[];
  combined_markdown: string;
  citations: string[];
}

export interface LiteratureReviewJobStatus {
  job_id: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";
  created_at: string;
  updated_at: string;
  result?: LiteratureReviewResult;
  error?: string;
}
