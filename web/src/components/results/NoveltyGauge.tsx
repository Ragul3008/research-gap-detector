import React, { useState } from "react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { ChevronDown, ChevronUp } from "lucide-react";

interface NoveltyGaugeProps {
  percentage: number;
  label: "HIGH" | "MEDIUM" | "LOW";
  description: string;
  subScores: {
    similarity_sub: number;
    coverage_sub: number;
    recency_sub: number;
  };
}

export function NoveltyGauge({ percentage, label, description, subScores }: NoveltyGaugeProps) {
  const [showMath, setShowMath] = useState(false);

  // SVG Gauge calculations
  const radius = 50;
  const strokeWidth = 8;
  const normalizedRadius = radius - strokeWidth / 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <Card className="p-6 space-y-4">
      <div className="flex justify-between items-center border-b border-[#E5E7EB] pb-2">
        <h3 className="text-lg font-serif font-bold text-[#1A2536]">Novelty Assessment</h3>
        <Badge variant={label === "HIGH" ? "success" : label === "MEDIUM" ? "warning" : "destructive"}>
          {label} NOVELTY
        </Badge>
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-around py-4 gap-6">
        {/* Radial SVG Progress dial */}
        <div className="relative flex items-center justify-center">
          <svg height={radius * 2} width={radius * 2} className="transform -rotate-90">
            <circle
              stroke="#CBD5E1"
              fill="transparent"
              strokeWidth={strokeWidth}
              r={normalizedRadius}
              cx={radius}
              cy={radius}
            />
            <circle
              stroke={label === "HIGH" ? "#2563EB" : label === "MEDIUM" ? "#B25E00" : "#C92A2A"}
              fill="transparent"
              strokeWidth={strokeWidth}
              strokeDasharray={circumference + " " + circumference}
              style={{ strokeDashoffset }}
              strokeLinecap="round"
              r={normalizedRadius}
              cx={radius}
              cy={radius}
            />
          </svg>
          <div className="absolute text-center">
            <span className="text-2xl font-mono font-bold text-[#1A2536]">{percentage}%</span>
          </div>
        </div>

        <div className="flex-1 space-y-2 text-left">
          <p className="text-xs text-gray-600 leading-relaxed">{description}</p>
        </div>
      </div>

      {/* Math Audit Panel */}
      <div className="border-t border-[#E5E7EB] pt-2">
        <button
          onClick={() => setShowMath(!showMath)}
          className="flex justify-between items-center w-full text-xs text-gray-500 hover:text-gray-800 font-bold focus:outline-none cursor-pointer"
        >
          <span>📊 Vetting Mechanics Formulas</span>
          {showMath ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showMath && (
          <div className="mt-3 p-3 bg-gray-50 border border-[#CBD5E1] rounded text-[11px] font-sans space-y-2">
            <p className="font-semibold text-gray-700">Formula: Novelty = 0.5 * SimSub + 0.3 * CoverSub + 0.2 * RecencySub</p>
            <div className="grid grid-cols-3 gap-2 text-center mt-2">
              <div className="p-2 bg-white border border-[#CBD5E1] rounded">
                <span className="block text-gray-500 font-mono">Similarity (50%)</span>
                <span className="font-mono font-bold text-[#1A2536]">{Math.round((subScores?.similarity_sub ?? 0) * 100)}%</span>
              </div>
              <div className="p-2 bg-white border border-[#CBD5E1] rounded">
                <span className="block text-gray-500 font-mono">Coverage (30%)</span>
                <span className="font-mono font-bold text-[#1A2536]">{Math.round((subScores?.coverage_sub ?? 0) * 100)}%</span>
              </div>
              <div className="p-2 bg-white border border-[#CBD5E1] rounded">
                <span className="block text-gray-500 font-mono">Recency (20%)</span>
                <span className="font-mono font-bold text-[#1A2536]">{Math.round((subScores?.recency_sub ?? 0) * 100)}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
