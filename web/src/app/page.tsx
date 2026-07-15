"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { FileUploadField } from "@/components/forms/FileUploadField";
import { AnalysisForm } from "@/components/forms/AnalysisForm";
import { Card } from "@/components/ui/Card";
import { useAnalyze } from "@/lib/hooks/useAnalyze";
import { AlertCircle, BookOpen } from "lucide-react";

export default function LandingPage() {
  const router = useRouter();
  const [extractedValues, setExtractedValues] = useState<{ title?: string; abstract?: string; keywords?: string; domain?: string } | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);

  const { mutate: runAnalysis, isPending } = useAnalyze();

  const handleExtracted = (meta: any) => {
    setExtractedValues(meta);
  };

  const handleFormSubmit = (values: any) => {
    setError(null);
    runAnalysis(
      {
        title: values.title,
        abstract: values.abstract,
        keywords: values.keywords,
        domain: values.domain || null,
        top_k: values.topK,
        num_authors: values.numAuthors,
        check_plagiarism: values.checkPlagiarism,
      },
      {
        onSuccess: (data) => {
          // Generate client-side UUID, cache results, and route to dashboard
          const analysisId = Math.random().toString(36).substring(2, 15);
          localStorage.setItem(`analysis_${analysisId}`, JSON.stringify(data));
          router.push(`/analysis/${analysisId}`);
        },
        onError: (err: any) => {
          setError(err.message || "An error occurred during the analysis run.");
        },
      }
    );
  };

  return (
    <div className="space-y-8 font-sans">
      {/* Header */}
      <div className="border-b border-[#CBD5E1] pb-6">
        <h1 className="text-4xl font-serif font-bold text-[#1A2536]">
          Research Novelty & Gap Analyzer
        </h1>
        <p className="text-sm text-gray-600 mt-2 max-w-3xl">
          Evaluate the scientific novelty of your proposed thesis, identify literature coverage gap dimensions, 
          detect potential plagiarism risks, and inspect grounded advisory reports.
        </p>
      </div>

      {error && (
        <div className="bg-[#FFF5F5] border-l-4 border-[#C92A2A] p-4 text-xs text-[#C92A2A] rounded flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Config Form */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="p-6">
            <h2 className="text-xl font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2 mb-4">
              📝 Research Specifications
            </h2>
            <AnalysisForm
              initialValues={extractedValues}
              onSubmit={handleFormSubmit}
              loading={isPending}
            />
          </Card>
        </div>

        {/* Ingestion & Guidelines Panel */}
        <div className="space-y-6">
          {/* PDF Extraction */}
          <Card className="p-6 space-y-4">
            <h3 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2">
              📁 PDF / Text Metadata Ingestion
            </h3>
            <p className="text-xs text-gray-600 leading-relaxed">
              Upload your draft research proposal. The backend extracts text contents and pre-fills your spec form for review.
            </p>
            <FileUploadField
              onExtracted={handleExtracted}
              onError={(msg) => setError(msg)}
            />
          </Card>

          {/* Quick Guidelines */}
          <Card className="p-6 space-y-3 bg-[#E9EEF5] border-[#CBD5E1]">
            <h3 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#CBD5E1] pb-2 flex items-center space-x-2">
              <BookOpen className="w-4 h-4 text-[#B25E00]" />
              <span>Academic Guidelines</span>
            </h3>
            <ul className="text-xs text-gray-700 space-y-2 list-disc list-inside leading-relaxed">
              <li>Analyzes dense text semantics via hybrid RRF search.</li>
              <li>Cross-validates claimed gaps against 6 taxonomy vectors.</li>
              <li>Reduces confidence scores on outlying semantic topics.</li>
              <li>Highlights hallucination assertions using grounding checks.</li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  );
}
