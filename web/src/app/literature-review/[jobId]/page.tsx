"use client";

import React, { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useLiteratureReviewStatus } from "@/lib/hooks/useLiteratureReviewStatus";
import { ReviewProgress } from "@/components/literature-review/ReviewProgress";
import { ThematicSection } from "@/components/literature-review/ThematicSection";
import { CitationList } from "@/components/literature-review/CitationList";
import { GapSummarySection } from "@/components/literature-review/GapSummarySection";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { ArrowLeft, BookOpen, Download, FileText } from "lucide-react";

export default function LiteratureReviewReaderPage() {
  const router = useRouter();
  const params = useParams();
  const jobId = params?.jobId as string;

  const [activeCitation, setActiveCitation] = useState<string | null>(null);

  // Poll status checks every 2.5s
  const { data: job, error } = useLiteratureReviewStatus(jobId, true);

  const handleCitationClick = (tag: string) => {
    // If tag is "[Paper 1]", convert/clean to "Paper 1" or keep format matching activeCitation comparison
    setActiveCitation(tag);
  };

  const handleDownloadMarkdown = () => {
    if (!job?.result?.combined_markdown) return;
    const element = document.createElement("a");
    const file = new Blob([job.result.combined_markdown], { type: "text/markdown" });
    element.href = URL.createObjectURL(file);
    element.download = `literature_review_${jobId.substring(0, 8)}.md`;
    document.body.appendChild(element);
    element.click();
    element.remove();
  };

  const isCompleted = job?.status === "COMPLETED";
  const isFailed = job?.status === "FAILED";
  const isProgressing = job && (job.status === "PENDING" || job.status === "RUNNING");

  return (
    <div className="space-y-8 font-sans">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center bg-[#E9EEF5] border border-[#CBD5E1] p-4 rounded gap-4 no-print">
        <div className="space-y-1 text-left">
          <div className="flex items-center space-x-2 text-xs font-mono text-gray-500 uppercase font-bold">
            <BookOpen className="w-3.5 h-3.5" />
            <span>Job ID: {jobId}</span>
          </div>
          <h2 className="text-lg font-serif font-bold text-[#1A2536]">
            {isCompleted ? "Grounded Literature Review Draft" : "Review Compilation Queue"}
          </h2>
        </div>
        <div className="flex items-center space-x-2 shrink-0">
          <Button variant="outline" onClick={() => router.push("/literature-review")} className="flex items-center space-x-1 py-2">
            <ArrowLeft className="w-4 h-4" />
            <span>Setup New Review</span>
          </Button>
          {isCompleted && (
            <Button variant="outline" onClick={handleDownloadMarkdown} className="flex items-center space-x-1 py-2">
              <Download className="w-4 h-4" />
              <span>Download Draft (MD)</span>
            </Button>
          )}
        </div>
      </div>

      {error && (
        <Card className="max-w-md mx-auto p-8 text-center space-y-4 my-12">
          <XCircle className="w-12 h-12 text-[#C92A2A] mx-auto" />
          <h2 className="text-lg font-serif font-bold text-[#1A2536]">Connection Error</h2>
          <p className="text-xs text-gray-500">
            Failed to connect to the background task checker. Check if your API backend is online.
          </p>
        </Card>
      )}

      {/* Progress Checker */}
      {isProgressing && <ReviewProgress status={job.status} />}
      {isFailed && <ReviewProgress status="FAILED" errorMsg={job.error} />}

      {/* Renders Compiled Paragraphs & Margins */}
      {isCompleted && job.result && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-start">
          {/* Synthesised paragraphs list */}
          <div className="lg:col-span-2 space-y-6">
            {job.result.sections
              ?.filter((s) => s.title !== "Research Gap & Novelty Analysis")
              .map((section, idx) => (
                <Card key={idx} className="p-6">
                  <ThematicSection
                    title={section.title}
                    paragraph={section.paragraph}
                    papers={section.papers}
                    onCitationClick={handleCitationClick}
                    activeCitation={activeCitation}
                  />
                </Card>
              ))}

            {/* Closing Research Gaps block */}
            {job.result.sections
              ?.filter((s) => s.title === "Research Gap & Novelty Analysis")
              .map((section, idx) => (
                <GapSummarySection key={idx} paragraph={section.paragraph} />
              ))}
          </div>

          {/* Dynamic Annotation Margin Bibliographies list */}
          <div className="space-y-6 lg:sticky lg:top-20">
            <CitationList 
              citations={job.result.citations} 
              activeCitation={activeCitation}
            />
            
            <Card className="p-4 space-y-2 text-left bg-gray-50 border-dashed border-2">
              <span className="text-[10px] uppercase font-mono font-bold tracking-wide text-gray-400">Interaction Tip</span>
              <p className="text-[10px] text-gray-600 leading-relaxed font-sans">
                Click any citation tag (e.g. <span className="bg-[#E9EEF5] border border-[#CBD5E1] text-[#2563EB] px-1 py-0.2 rounded font-mono font-bold text-[9px]">[Paper 1]</span>) in the review paragraphs to highlight the corresponding paper references details in the bibliography list.
              </p>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}

// Simple fallback import helper for type checker
function XCircle(props: any) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <circle cx="12" cy="12" r="10" />
      <path d="m15 9-6 6" />
      <path d="m9 9 6 6" />
    </svg>
  );
}
