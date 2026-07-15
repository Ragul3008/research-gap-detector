import React from "react";
import { Card } from "@/components/ui/Card";

interface CitationListProps {
  citations: string[];
  activeCitation?: string | null;
}

export function CitationList({ citations, activeCitation }: CitationListProps) {
  return (
    <Card className="p-6 space-y-4 text-left">
      <h4 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2">
        📚 References & Bibliography
      </h4>
      <div className="space-y-3 text-xs text-gray-600 font-sans">
        {citations.map((cite, idx) => {
          const paperTag = `[Paper ${idx}]`;
          const isHighlighted = activeCitation === paperTag || activeCitation === `Paper ${idx}`;
          return (
            <p 
              key={idx} 
              className={`pl-4 -indent-4 transition duration-200 p-1.5 rounded ${
                isHighlighted ? "bg-[#FFF9DB] border-l-2 border-[#B25E00] text-[#1A2536] font-semibold" : ""
              }`}
            >
              {cite}
            </p>
          );
        })}
      </div>
    </Card>
  );
}
