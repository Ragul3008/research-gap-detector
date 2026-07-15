import React from "react";
import { Card } from "@/components/ui/Card";

interface GapSummarySectionProps {
  paragraph: string;
}

export function GapSummarySection({ paragraph }: GapSummarySectionProps) {
  return (
    <Card className="p-6 space-y-4 text-left border-l-4 border-[#B25E00]">
      <h3 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2">
        🔍 Research Gap & Novelty Analysis
      </h3>
      <div className="text-xs text-gray-700 leading-relaxed bg-gray-50 p-4 rounded border border-[#CBD5E1] whitespace-pre-line font-sans">
        {paragraph}
      </div>
    </Card>
  );
}
