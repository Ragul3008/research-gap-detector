import React from "react";
import { HallucinationDetails } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { ShieldCheck, AlertTriangle } from "lucide-react";

interface AdvisoryReportProps {
  explanation: string;
  hallucination: HallucinationDetails;
}

export function AdvisoryReport({ explanation, hallucination }: AdvisoryReportProps) {
  const { grounding_score, label, flagged_claims } = hallucination;

  return (
    <Card className="p-6 space-y-4 text-left">
      <div className="border-b border-[#E5E7EB] pb-2 flex justify-between items-center">
        <h3 className="text-lg font-serif font-bold text-[#1A2536]">
          📖 Grounded AI Advisory Report
        </h3>
        <Badge variant={grounding_score >= 80 ? "success" : "warning"}>
          Grounding: {grounding_score}/100 ({label})
        </Badge>
      </div>

      {/* Hallucinations Grounding Banner */}
      {flagged_claims?.length > 0 ? (
        <div className="bg-[#FFF5F5] border border-red-200 rounded p-4 space-y-2 text-xs">
          <h4 className="font-bold text-[#C92A2A] flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4" />
            <span>Academic Grounding Alert: Contradictions Flagged ({flagged_claims.length})</span>
          </h4>
          <p className="text-red-700 leading-relaxed">
            The advisory report contains claims that contradict the coverage statistics computed from the reference corpus. 
            Review these items before submitting your proposal:
          </p>
          <div className="space-y-2 mt-2">
            {flagged_claims.map((fc, i) => (
              <div key={i} className="bg-white p-3 rounded border border-red-200 shadow-sm font-sans">
                <p className="font-semibold text-gray-800 italic">“{fc.claim}”</p>
                <p className="text-[#C92A2A] mt-1 font-bold">⚠️ Issue: {fc.issue}</p>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="bg-emerald-50 border border-emerald-200 rounded p-3 text-xs text-emerald-800 flex items-center space-x-2">
          <ShieldCheck className="w-4 h-4 text-emerald-600 flex-shrink-0" />
          <span><b>Verified Grounded:</b> All semantic gap claims in this report are aligned with the retrieved literature.</span>
        </div>
      )}

      {/* Report Paragraphs */}
      <div className="prose prose-sm max-w-none text-xs text-gray-700 leading-relaxed bg-gray-50 p-4 rounded border border-[#CBD5E1] whitespace-pre-line font-sans">
        {explanation}
      </div>
    </Card>
  );
}
