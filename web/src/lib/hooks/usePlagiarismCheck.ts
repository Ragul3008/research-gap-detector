import { useMutation } from "@tanstack/react-query";
import { PlagiarismResult } from "../types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

interface PlagiarismCheckPayload {
  title: string;
  abstract?: string;
  keywords?: string;
  top_k?: number;
}

async function postPlagiarism(req: PlagiarismCheckPayload): Promise<{ status: string; plagiarism: PlagiarismResult }> {
  const res = await fetch(`${API_BASE}/api/plagiarism`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    throw new Error("Failed to check plagiarism");
  }
  return res.json();
}

export function usePlagiarismCheck() {
  return useMutation<{ status: string; plagiarism: PlagiarismResult }, Error, PlagiarismCheckPayload>({
    mutationFn: postPlagiarism,
  });
}
