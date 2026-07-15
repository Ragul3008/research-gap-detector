import React from "react";
import { cn } from "@/lib/utils";

interface FormFieldProps {
  label: string;
  error?: string;
  hint?: string;
  required?: boolean;
  children: React.ReactNode;
}

export function FormField({ label, error, hint, required, children }: FormFieldProps) {
  return (
    <div className="space-y-1 w-full text-left">
      <div className="flex justify-between items-baseline">
        <label className="block text-xs font-bold text-gray-700 uppercase tracking-wider">
          {label} {required && <span className="text-[#C92A2A] font-mono">*</span>}
        </label>
        {error && (
          <span className="text-[10px] text-[#C92A2A] font-semibold font-sans">
            ⚠️ {error}
          </span>
        )}
      </div>
      <div className="relative">
        {children}
      </div>
      {hint && !error && (
        <p className="text-[10px] text-gray-500 font-sans leading-relaxed">
          {hint}
        </p>
      )}
    </div>
  );
}
