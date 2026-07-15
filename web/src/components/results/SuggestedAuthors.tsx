import React from "react";
import { Card } from "@/components/ui/Card";
import { Users, UserCheck } from "lucide-react";

interface SuggestedAuthorsProps {
  authors: { name: string; reason: string }[];
}

export function SuggestedAuthors({ authors }: SuggestedAuthorsProps) {
  return (
    <Card className="p-6 space-y-4 text-left">
      <h3 className="text-lg font-serif font-bold text-[#1A2536] border-b border-[#E5E7EB] pb-2 flex items-center space-x-2">
        <Users className="w-5 h-5 text-[#B25E00]" />
        <span>Suggested Researchers & Authors</span>
      </h3>
      <p className="text-xs text-gray-600 font-sans">
        Key real-world prominent academic authors whose publications are highly relevant to your research direction:
      </p>
      <div className="space-y-3 font-sans max-h-96 overflow-y-auto pr-2">
        {authors?.map((author, i) => (
          <div 
            key={i} 
            className="p-3 bg-gray-50 border border-[#CBD5E1] rounded flex flex-col gap-1 hover:border-[#2563EB] transition duration-150"
          >
            <span className="text-xs font-bold text-[#1A2536] flex items-center space-x-1.5">
              <UserCheck className="w-3.5 h-3.5 text-[#2563EB] shrink-0" />
              <a
                href={`https://scholar.google.com/scholar?q=${encodeURIComponent(author.name)}`}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline text-[#2563EB] cursor-pointer"
                title="Click to search publications on Google Scholar"
              >
                {author.name}
              </a>
            </span>
            <span className="text-[11px] text-gray-600 leading-relaxed pl-5">
              {author.reason}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}
