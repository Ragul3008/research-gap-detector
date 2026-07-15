import React from "react";
import { cn } from "@/lib/utils";
import { X, AlertCircle, CheckCircle, Info } from "lucide-react";

interface ToastProps {
  message: string;
  type?: "success" | "error" | "info";
  onClose?: () => void;
}

export function Toast({ message, type = "info", onClose }: ToastProps) {
  return (
    <div
      className={cn(
        "fixed bottom-4 right-4 z-50 flex items-center space-x-3 px-4 py-3 rounded shadow-lg border text-xs font-sans max-w-sm transition-all duration-300",
        type === "success" && "bg-emerald-50 text-emerald-800 border-emerald-200",
        type === "error" && "bg-[#FFF5F5] text-[#C92A2A] border-red-200",
        type === "info" && "bg-blue-50 text-blue-800 border-blue-200"
      )}
    >
      {type === "success" && <CheckCircle className="w-4 h-4 text-emerald-600 flex-shrink-0" />}
      {type === "error" && <AlertCircle className="w-4 h-4 text-red-600 flex-shrink-0" />}
      {type === "info" && <Info className="w-4 h-4 text-blue-600 flex-shrink-0" />}
      
      <span className="flex-grow font-semibold">{message}</span>
      {onClose && (
        <button onClick={onClose} className="hover:text-gray-600 focus:outline-none cursor-pointer">
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}
