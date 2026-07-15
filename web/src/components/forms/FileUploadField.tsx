import React, { useRef } from "react";
import { Upload } from "lucide-react";
import { useExtractMetadata } from "@/lib/hooks/useExtractMetadata";

interface FileUploadFieldProps {
  onExtracted: (metadata: { title?: string; abstract?: string; keywords?: string; domain?: string }) => void;
  onError: (msg: string) => void;
}

export function FileUploadField({ onExtracted, onError }: FileUploadFieldProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { mutate: runExtract, isPending } = useExtractMetadata();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith(".pdf") && !file.name.endsWith(".txt")) {
      onError("Only PDF and TXT proposal documents are supported.");
      return;
    }

    runExtract(file, {
      onSuccess: (data) => {
        if (data.status === "success" && data.metadata) {
          onExtracted(data.metadata);
        } else {
          onError("Failed to parse document metadata.");
        }
      },
      onError: (err: any) => {
        onError(err.message || "Failed to extract metadata from PDF.");
      },
    });
  };

  return (
    <div className="space-y-2">
      <div 
        onClick={() => fileInputRef.current?.click()}
        className="border-2 border-dashed border-[#CBD5E1] hover:border-[#2563EB] rounded p-6 flex flex-col items-center justify-center cursor-pointer transition duration-150 bg-gray-50 text-center"
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".pdf,.txt"
          className="hidden"
          disabled={isPending}
        />
        <Upload className="w-6 h-6 text-gray-400 mb-2" />
        <span className="text-xs font-semibold text-gray-700">
          {isPending ? "⏳ Extracting proposal parameters..." : "Upload PDF or TXT draft file to prefill"}
        </span>
        <span className="text-[10px] text-gray-500 mt-1">
          Title, Abstract, and Keywords will be parsed for your review
        </span>
      </div>
    </div>
  );
}
