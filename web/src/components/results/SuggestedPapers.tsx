import React from "react";
import { Card } from "@/components/ui/Card";
import { PaperMetadata } from "@/lib/types";
import { BookOpen } from "lucide-react";

interface SuggestedPapersProps {
  papers: PaperMetadata[];
}

export function SuggestedPapers({ papers }: SuggestedPapersProps) {
  return (
    <Card className="p-6 space-y-4 text-left">
      <h3 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2 flex items-center space-x-2">
        <BookOpen className="w-5 h-5 text-[#2563EB]" />
        <span>Related Publications & Literature</span>
      </h3>
      <p className="text-xs text-gray-600 font-sans">
        Prominent historical research papers retrieved and re-ranked from the reference corpus based on semantic overlap:
      </p>
      <div className="space-y-3 font-sans max-h-96 overflow-y-auto pr-2">
        {papers?.map((paper, i) => (
          <div 
            key={paper.id || i} 
            className="p-3 bg-gray-50 border border-[#CBD5E1] rounded flex flex-col gap-1 hover:border-[#2563EB] transition duration-150"
          >
            <div className="flex justify-between items-start gap-2">
              <span className="text-xs font-bold text-[#1A2536]">
                <a
                  href={`https://scholar.google.com/scholar?q=${encodeURIComponent(paper.title)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[#1A2536] hover:text-[#2563EB] hover:underline"
                  title="Click to search paper on Google Scholar"
                >
                  {paper.title}
                </a>
              </span>
              {paper.similarity_pct !== undefined && (
                <span className="text-[10px] bg-[#E9EEF5] text-[#2563EB] px-1.5 py-0.5 rounded font-mono font-bold shrink-0">
                  {paper.similarity_pct.toFixed(1)}% Sim
                </span>
              )}
            </div>
            <div className="flex flex-wrap gap-2 text-[10px] text-gray-500 font-mono mt-1">
              <span>📅 Year: {paper.year || "N/A"}</span>
              <span>•</span>
              <span>📁 Domain: {paper.domain || "N/A"}</span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
