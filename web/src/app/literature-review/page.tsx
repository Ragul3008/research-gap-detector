"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { FormField } from "@/components/forms/FormField";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { useLiteratureReview } from "@/lib/hooks/useLiteratureReview";
import { AlertCircle } from "lucide-react";

const DOMAIN_OPTIONS = [
  "",
  "Sociology", "History", "Political Science", "Economics",
  "Psychology", "Environmental Science", "Literature", "Philosophy",
  "Anthropology", "Education", "Geography", "Commerce", "Linguistics",
  "Botany", "Zoology", "Chemistry", "Physics", "Mathematics", "Statistics",
  "Computer Science and Engineering", "Electronics and Communication Engineering",
  "Electrical Engineering", "Mechanical Engineering", "Civil Engineering"
];

export default function LiteratureReviewSetupPage() {
  const router = useRouter();
  const [title, setTitle] = useState("");
  const [domain, setDomain] = useState("");
  const [abstract, setAbstract] = useState("");
  const [keywords, setKeywords] = useState("");
  const [sectionPreference, setSectionPreference] = useState("thematic");
  const [error, setError] = useState<string | null>(null);

  const { mutate: startReview, isPending } = useLiteratureReview();

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim()) {
      setError("Title is required");
      return;
    }

    setError(null);
    startReview(
      {
        title: title.trim(),
        abstract: abstract.trim() || undefined,
        keywords: keywords.trim() || undefined,
        domain: domain || null,
        top_k: 8,
        section_preference: sectionPreference,
        run_async: true,
      },
      {
        onSuccess: (data) => {
          router.push(`/literature-review/${data.job_id}`);
        },
        onError: (err: any) => {
          setError(err.message || "Failed to trigger literature review compilation.");
        },
      }
    );
  };

  return (
    <div className="space-y-8 font-sans max-w-3xl mx-auto">
      {/* Header */}
      <div className="border-b border-[#CBD5E1] pb-6">
        <h1 className="text-4xl font-serif font-bold text-[#1A2536]">
          Literature Review Compiler
        </h1>
        <p className="text-sm text-gray-600 mt-2">
          Submit your proposal parameters to draft structured, citation-grounded literature review paragraphs in the background.
        </p>
      </div>

      {error && (
        <div className="bg-[#FFF5F5] border-l-4 border-[#C92A2A] p-4 text-xs text-[#C92A2A] rounded flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <Card className="p-6">
        <h2 className="text-xl font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2 mb-4">
          📝 Configuration & Constraints
        </h2>
        <form onSubmit={handleFormSubmit} className="space-y-4">
          <FormField label="Proposed Research Title" required hint="Minimum 5 characters.">
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., Deep Learning for Medical Image Segmentation in Indian Rural Hospitals"
              className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans"
              required
              minLength={5}
            />
          </FormField>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <FormField label="Subject Domain">
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans"
              >
                {DOMAIN_OPTIONS.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt || "Select Subject Domain"}
                  </option>
                ))}
              </select>
            </FormField>

            <FormField label="Synthesis Organization">
              <select
                value={sectionPreference}
                onChange={(e) => setSectionPreference(e.target.value)}
                className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans"
              >
                <option value="thematic">Thematic Sections (BERTopic clusters)</option>
                <option value="chronological">Chronological Synthesis</option>
                <option value="methodological">Methodological Frameworks</option>
              </select>
            </FormField>
          </div>

          <div className="grid grid-cols-1 gap-4">
            <FormField label="Keywords (comma-separated)">
              <input
                type="text"
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
                placeholder="e.g., medical imaging, deep learning, rural health"
                className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans"
              />
            </FormField>
          </div>

          <FormField label="Abstract / Abstract Scope (recommended)">
            <textarea
              value={abstract}
              onChange={(e) => setAbstract(e.target.value)}
              rows={5}
              placeholder="Provide thesis abstract or key research goals to align contextual semantic search..."
              className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans"
            />
          </FormField>

          <Button type="submit" disabled={isPending} className="w-full">
            {isPending ? "⏳ Starting Relational Review Task..." : "✍️ Generate Grounded Literature Review"}
          </Button>
        </form>
      </Card>
    </div>
  );
}
