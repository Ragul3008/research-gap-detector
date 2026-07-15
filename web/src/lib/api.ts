import { 
  AnalyzeRequest, 
  AnalyzeResponse, 
  LiteratureReviewRequest, 
  LiteratureReviewJobStatus
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export async function postAnalyze(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || "Failed to run analysis pipeline");
  }
  return res.json();
}

export async function extractMetadata(file: File): Promise<{ status: string; filename: string; metadata: any; full_text: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/extract-metadata`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || "Failed to extract metadata");
  }
  return res.json();
}

export async function postLiteratureReview(req: LiteratureReviewRequest): Promise<{ job_id: string; status: string; message: string }> {
  const res = await fetch(`${API_BASE}/api/literature-review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(errorText || "Failed to trigger literature review");
  }
  return res.json();
}

export async function getLiteratureReviewStatus(jobId: string): Promise<LiteratureReviewJobStatus> {
  const res = await fetch(`${API_BASE}/api/literature-review/status/${jobId}`);
  if (!res.ok) {
    throw new Error("Failed to fetch literature review status");
  }
  return res.json();
}
