import React from "react";
import { ReviewSectionPaper } from "@/lib/types";

interface ThematicSectionProps {
  title: string;
  paragraph: string;
  papers: ReviewSectionPaper[];
  onCitationClick?: (tag: string) => void;
  activeCitation?: string | null;
}

export function ThematicSection({ title, paragraph, papers, onCitationClick, activeCitation }: ThematicSectionProps) {
  // Regex to match citation tags [Paper X]
  const regex = /(\[Paper \d+\])/g;
  const parts = paragraph.split(regex);

  return (
    <div className="space-y-3 text-left">
      <h3 className="text-xl font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-1.5">
        {title}
      </h3>
      <div className="text-xs text-gray-700 leading-relaxed bg-gray-50 p-4 rounded border border-[#CBD5E1] font-sans">
        {parts.map((part, index) => {
          if (regex.test(part)) {
            const isHighlighted = activeCitation === part || activeCitation === part.slice(1, -1);
            return (
              <span
                key={index}
                onClick={() => onCitationClick && onCitationClick(part)}
                className={`border px-1.5 py-0.5 rounded font-mono font-bold text-[10px] mx-0.5 cursor-pointer select-none transition ${
                  isHighlighted 
                    ? "bg-[#FFF9DB] border-[#B25E00] text-[#B25E00]" 
                    : "bg-[#E9EEF5] border-[#CBD5E1] hover:border-[#2563EB] text-[#2563EB]"
                }`}
                title="Click to view reference details"
              >
                {part}
              </span>
            );
          }
          return <span key={index}>{part}</span>;
        })}
      </div>
      
      {/* Short bibliography checklist for this section */}
      {papers?.length > 0 && (
        <div className="flex flex-wrap gap-2 pt-1 font-mono text-[9px] text-gray-500">
          {papers.map((p) => {
            const isHighlighted = activeCitation === p.citation_tag || activeCitation === `[${p.citation_tag}]`;
            return (
              <span 
                key={p.citation_tag} 
                className={`border px-2 py-0.5 rounded shadow-sm transition ${
                  isHighlighted ? "bg-[#FFF9DB] border-[#B25E00] text-[#B25E00] font-bold" : "bg-white border-gray-200"
                }`}
              >
                📄 {p.citation_tag}: {p.title} ({p.year})
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
