import React from "react";
import { Card } from "@/components/ui/Card";
import { ChevronRight } from "lucide-react";

interface SuggestedTitlesProps {
  titles: string[];
}

export function SuggestedTitles({ titles }: SuggestedTitlesProps) {
  return (
    <Card className="p-6 space-y-4 text-left">
      <h3 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2">
        ✍️ Suggested Pivoted Titles
      </h3>
      <p className="text-xs text-gray-600">
        Pivoted titles incorporating the uncovered gap taxonomies to maximize contribution value:
      </p>
      <div className="space-y-2">
        {titles?.map((title, i) => (
          <div 
            key={i} 
            className="p-3 bg-[#E9EEF5] rounded border border-[#CBD5E1] flex items-center space-x-2 text-xs font-semibold text-[#1A2536] font-sans"
          >
            <ChevronRight className="w-4 h-4 text-[#B25E00] flex-shrink-0" />
            <span>{title}</span>
          </div>
        ))}
      </div>
    </Card>
  );
}
