import React from "react";
import { PlagiarismResult } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { BookCheck } from "lucide-react";

interface PlagiarismPanelProps {
  result?: PlagiarismResult;
}

export function PlagiarismPanel({ result }: PlagiarismPanelProps) {
  if (!result) {
    return (
      <Card className="p-6 text-center text-gray-500 italic text-xs">
        Plagiarism scanning was not run or has no matches.
      </Card>
    );
  }

  const { plagiarism_score, risk_level, sentence_matches, phrase_matches } = result;

  return (
    <Card className="p-6 space-y-4">
      <div className="flex justify-between items-center border-b border-[#E5E7EB] pb-2">
        <h3 className="text-lg font-serif font-bold text-[#1A2536] flex items-center space-x-2">
          <BookCheck className="w-5 h-5 text-[#2563EB]" />
          <span>Plagiarism & Citation Scan</span>
        </h3>
        <Badge variant={risk_level === "HIGH" ? "destructive" : risk_level === "MODERATE" ? "warning" : "success"}>
          {risk_level} RISK ({plagiarism_score.toFixed(1)}%)
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-left">
        {/* Sentence Level Matches */}
        <div className="space-y-3">
          <h4 className="font-bold text-gray-800 uppercase tracking-wider">Flagged Sentence Overlaps</h4>
          {sentence_matches?.length > 0 ? (
            <div className="space-y-2 max-h-60 overflow-y-auto pr-2">
              {sentence_matches.map((sm, i) => (
                <div key={i} className="p-3 bg-[#FFF5F5] border border-red-200 rounded text-gray-700">
                  <p className="font-semibold text-gray-800">Draft: "{sm.user_sentence}"</p>
                  <p className="text-gray-500 mt-1">Matched: "{sm.paper_sentence}"</p>
                  <div className="flex justify-between items-center mt-2 text-[10px] text-gray-400">
                    <span className="truncate max-w-[200px]">Source: {sm.paper_title} ({sm.paper_year})</span>
                    <span className="bg-red-100 text-red-700 px-1 py-0.2 rounded font-mono shrink-0">
                      Jaccard: {Math.round(sm.jaccard * 100)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 italic">No sentence-level plagiarism overlaps found.</p>
          )}
        </div>

        {/* Phrase Level Matches */}
        <div className="space-y-3">
          <h4 className="font-bold text-gray-800 uppercase tracking-wider">Flagged Exact Phrases</h4>
          {phrase_matches?.length > 0 ? (
            <div className="space-y-2 max-h-60 overflow-y-auto pr-2">
              {phrase_matches.map((pm, i) => (
                <div key={i} className="p-2.5 bg-[#FFF9DB] border border-yellow-200 rounded flex justify-between items-center">
                  <div>
                    <span className="font-mono text-gray-800 font-bold">"{pm.phrase}"</span>
                    <p className="text-[10px] text-gray-500 mt-0.5">Found in: {pm.paper_title}</p>
                  </div>
                  <span className="text-[10px] bg-[#B25E00] text-white px-2 py-0.5 rounded font-mono font-bold shrink-0">
                    {pm.word_count} words
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 italic">No exact phrase matches flagged.</p>
          )}
        </div>
      </div>
    </Card>
  );
}
