import React from "react";

interface GapDataTableProps {
  data: { subject: string; value: number }[];
  gaps: {
    gap_dimensions: {
      regional_gaps: string[];
      population_gaps: string[];
      methodological_gaps: string[];
      thematic_gaps: string[];
      theoretical_gaps: string[];
      temporal_gaps: string[];
    };
  };
}

export function GapDataTable({ data, gaps }: GapDataTableProps) {
  const getTaxonomyCount = (subject: string): number => {
    switch (subject) {
      case "Region":
        return gaps?.gap_dimensions?.regional_gaps?.length || 0;
      case "Population":
        return gaps?.gap_dimensions?.population_gaps?.length || 0;
      case "Method":
        return gaps?.gap_dimensions?.methodological_gaps?.length || 0;
      case "Theme":
        return gaps?.gap_dimensions?.thematic_gaps?.length || 0;
      case "Theory":
        return gaps?.gap_dimensions?.theoretical_gaps?.length || 0;
      case "Time Period":
        return gaps?.gap_dimensions?.temporal_gaps?.length || 0;
      default:
        return 0;
    }
  };

  const getTaxonomyItems = (subject: string): string[] => {
    switch (subject) {
      case "Region":
        return gaps?.gap_dimensions?.regional_gaps || [];
      case "Population":
        return gaps?.gap_dimensions?.population_gaps || [];
      case "Method":
        return gaps?.gap_dimensions?.methodological_gaps || [];
      case "Theme":
        return gaps?.gap_dimensions?.thematic_gaps || [];
      case "Theory":
        return gaps?.gap_dimensions?.theoretical_gaps || [];
      case "Time Period":
        return gaps?.gap_dimensions?.temporal_gaps || [];
      default:
        return [];
    }
  };

  return (
    <div className="w-full overflow-x-auto">
      <table className="min-w-full text-xs text-left font-sans border-collapse">
        <thead>
          <tr className="bg-[#E9EEF5] text-[#1A2536] font-bold border-b border-[#CBD5E1]">
            <th className="px-4 py-3 font-semibold">Dimension</th>
            <th className="px-4 py-3 font-semibold">Calculated Gap Size</th>
            <th className="px-4 py-3 font-semibold">Flagged Taxonomy Omissions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#E5E7EB]">
          {data.map((row) => {
            const count = getTaxonomyCount(row.subject);
            const items = getTaxonomyItems(row.subject);
            return (
              <tr key={row.subject} className="hover:bg-gray-50/50">
                <td className="px-4 py-3 font-bold text-[#1A2536]">{row.subject}</td>
                <td className="px-4 py-3 font-mono font-bold text-sm">
                  <div className="flex items-center space-x-2">
                    <span>{row.value}%</span>
                    <div className="w-20 bg-gray-200 h-1.5 rounded-full overflow-hidden">
                      <div 
                        className="bg-[#2563EB] h-full" 
                        style={{ width: `${row.value}%` }} 
                      />
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3 text-gray-600">
                  {count > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {items.map((it) => (
                        <span key={it} className="bg-gray-100 text-gray-700 px-1.5 py-0.5 rounded text-[10px] border border-gray-200 font-mono">
                          {it}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="text-gray-400 italic">No omissions flagged</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
