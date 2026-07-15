"use client";

import React, { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import { NoveltyGauge } from "@/components/results/NoveltyGauge";
import { GapRadarChart } from "@/components/results/GapRadarChart";
import { GapDataTable } from "@/components/results/GapDataTable";
import { PlagiarismPanel } from "@/components/results/PlagiarismPanel";
import { AdvisoryReport } from "@/components/results/AdvisoryReport";
import { SuggestedTitles } from "@/components/results/SuggestedTitles";
import { SuggestedAuthors } from "@/components/results/SuggestedAuthors";
import { SuggestedPapers } from "@/components/results/SuggestedPapers";
import { ReportDownloadButton } from "@/components/results/ReportDownloadButton";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { ArrowLeft, BookOpen, Clock, FileWarning } from "lucide-react";
import { AnalyzeResponse } from "@/lib/types";

export default function AnalysisResultsPage() {
  const router = useRouter();
  const params = useParams();
  const id = params?.id as string;

  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (id) {
      const cached = localStorage.getItem(`analysis_${id}`);
      if (cached) {
        try {
          setResult(JSON.parse(cached));
        } catch (e) {
          console.error("Failed to parse cached analysis results");
        }
      }
      setLoading(false);
    }
  }, [id]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-[#1A2536]" />
        <p className="text-sm font-mono text-gray-500 animate-pulse">Loading analysis metadata...</p>
      </div>
    );
  }

  if (!result) {
    return (
      <Card className="max-w-md mx-auto p-8 text-center space-y-4 my-12">
        <FileWarning className="w-12 h-12 text-[#C92A2A] mx-auto animate-bounce" />
        <h2 className="text-lg font-serif font-bold text-[#1A2536]">Report Not Found</h2>
        <p className="text-xs text-gray-500">
          The requested novelty assessment report is unavailable or has expired from cache.
        </p>
        <Button onClick={() => router.push("/")} className="w-full">
          Start New Topic Analysis
        </Button>
      </Card>
    );
  }

  // Map values for Radar Chart
  const getAverageGapSize = (dimKey: string) => {
    const formulaObj = result.gaps?.auditable_formulas?.[dimKey];
    if (!formulaObj || !formulaObj.dimensions_log) return 0;
    const logs = formulaObj.dimensions_log;
    const avg = logs.reduce((sum, item) => sum + item.gap_size, 0) / logs.length;
    return Math.round(avg * 100);
  };

  const radarData = [
    { subject: "Region", value: getAverageGapSize("regional_gaps") },
    { subject: "Population", value: getAverageGapSize("population_gaps") },
    { subject: "Method", value: getAverageGapSize("methodological_gaps") },
    { subject: "Theme", value: getAverageGapSize("thematic_gaps") },
    { subject: "Theory", value: getAverageGapSize("theoretical_gaps") },
    { subject: "Time Period", value: getAverageGapSize("temporal_gaps") },
  ];

  return (
    <div className="space-y-8 font-sans">
      {/* Header Panel */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center bg-[#E9EEF5] border border-[#CBD5E1] p-4 rounded gap-4">
        <div className="space-y-1">
          <div className="flex items-center space-x-2 text-xs font-mono text-gray-500 uppercase font-bold">
            <Clock className="w-3.5 h-3.5" />
            <span>Analysis Report ID: {id}</span>
          </div>
          <h2 className="text-lg font-serif font-bold text-[#1A2536] max-w-2xl truncate">
            {result.input?.title}
          </h2>
        </div>
        <div className="flex items-center space-x-2 shrink-0">
          <Button variant="outline" onClick={() => router.push("/")} className="flex items-center space-x-1 py-2">
            <ArrowLeft className="w-4 h-4" />
            <span>New Analysis</span>
          </Button>
          <ReportDownloadButton result={result} />
        </div>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4 flex flex-col justify-between">
          <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Novelty Category</span>
          <span className="text-xl font-bold text-[#1A2536] mt-2 block font-mono">{result.novelty?.percentage}% ({result.novelty?.label})</span>
        </Card>
        <Card className="p-4 flex flex-col justify-between">
          <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Advisory Grounding</span>
          <span className="text-xl font-bold text-[#B25E00] mt-2 block font-mono">{result.hallucination?.grounding_score ?? 100}/100</span>
        </Card>
        <Card className="p-4 flex flex-col justify-between">
          <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Top Sim Metric</span>
          <span className="text-xl font-bold text-[#2563EB] mt-2 block font-mono">
            {result.similarity?.stats?.max_sim ? (result.similarity.stats.max_sim * 100).toFixed(1) : 0}%
          </span>
        </Card>
        <Card className="p-4 flex flex-col justify-between">
          <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Index Checked</span>
          <span className="text-xl font-bold text-gray-700 mt-2 block font-mono">
            {result.similar_papers?.length || 0} papers
          </span>
        </Card>
      </div>

      {/* Gauge and Plagiarism Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <NoveltyGauge
          percentage={result.novelty?.percentage}
          label={result.novelty?.label}
          description={result.novelty?.description}
          subScores={result.novelty?.sub_scores}
        />
        <PlagiarismPanel result={result.plagiarism} />
      </div>

      {/* Radar Chart and Accessible Data Table Fallback */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card className="p-6 flex flex-col items-center">
          <h3 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2 w-full text-left mb-4">
            📊 Gap Radar Map
          </h3>
          <GapRadarChart data={radarData} />
        </Card>
        <Card className="p-6">
          <h3 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2 w-full text-left mb-4">
            📋 Accessible Gap Metrics Table
          </h3>
          <GapDataTable data={radarData} gaps={result.gaps} />
        </Card>
      </div>

      {/* Advisory Report */}
      <AdvisoryReport
        explanation={result.explanation}
        hallucination={result.hallucination}
      />

      {/* Pivoted Titles & Suggested Authors */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <SuggestedTitles titles={result.title_suggestions} />
        <SuggestedAuthors authors={result.author_suggestions} />
      </div>

      {/* Suggested Papers & Advisory Recommendations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <SuggestedPapers papers={result.similar_papers} />
        <Card className="p-6 text-left space-y-4">
          <h3 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2">
            💡 Advisory Recommendations
          </h3>
          <p className="text-xs text-gray-600">
            Actionable strategies compiled to bypass identified literature gaps and strengthen originality:
          </p>
          <ul className="text-xs text-gray-700 space-y-2.5 list-disc list-inside leading-relaxed font-sans">
            {result.novelty?.suggestions?.map((s, idx) => (
              <li key={idx}>{s}</li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
