import React from "react";
import { Hourglass, Cpu, XCircle } from "lucide-react";

interface ReviewProgressProps {
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";
  errorMsg?: string;
}

export function ReviewProgress({ status, errorMsg }: ReviewProgressProps) {
  return (
    <div className="w-full space-y-6 text-center max-w-md mx-auto py-12">
      {status === "PENDING" && (
        <div className="space-y-4">
          <Hourglass className="w-10 h-10 text-gray-400 animate-spin mx-auto" />
          <h3 className="text-lg font-serif font-bold text-[#1A2536]">Literature Review Queued</h3>
          <p className="text-xs text-gray-500">
            Waiting for worker process allocation in the PostgreSQL relational task queue...
          </p>
        </div>
      )}

      {status === "RUNNING" && (
        <div className="space-y-4">
          <Cpu className="w-10 h-10 text-[#2563EB] animate-pulse mx-auto" />
          <h3 className="text-lg font-serif font-bold text-[#1A2536]">Drafting Literature Review</h3>
          <p className="text-xs text-gray-500 leading-relaxed">
            FastAPI workers are compiling thematic BERTopic clusters, generating cohesive citations paragraphs, 
            and executing the self-correcting grounding validator checks.
          </p>
        </div>
      )}

      {status === "FAILED" && (
        <div className="space-y-4 bg-[#FFF5F5] border border-red-200 rounded p-6">
          <XCircle className="w-10 h-10 text-[#C92A2A] mx-auto" />
          <h3 className="text-lg font-serif font-bold text-[#C92A2A]">Process Compile Failed</h3>
          <p className="text-xs text-red-700 leading-relaxed">
            {errorMsg || "The literature review compiler failed to synthesize the corpus paragraphs."}
          </p>
        </div>
      )}

      {/* Structured visual step bar */}
      <div className="grid grid-cols-3 gap-2 text-[10px] font-mono font-bold uppercase tracking-wider text-gray-500 pt-4">
        <div className="flex flex-col items-center">
          <div className={`w-3 h-3 rounded-full mb-1 ${status !== "PENDING" ? "bg-[#1A2536]" : "bg-gray-300 animate-ping"}`} />
          <span>Queueing</span>
        </div>
        <div className="flex flex-col items-center">
          <div className={`w-3 h-3 rounded-full mb-1 ${(status === "RUNNING" || status === "COMPLETED") ? "bg-[#2563EB] animate-pulse" : "bg-gray-300"}`} />
          <span>Synthesizing</span>
        </div>
        <div className="flex flex-col items-center">
          <div className={`w-3 h-3 rounded-full mb-1 ${status === "COMPLETED" ? "bg-emerald-500" : status === "FAILED" ? "bg-[#C92A2A]" : "bg-gray-300"}`} />
          <span>Compiling</span>
        </div>
      </div>
    </div>
  );
}
