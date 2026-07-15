import React from "react";
import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warning" | "destructive" | "info";
}

export function Badge({ className, variant = "default", children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center px-2.5 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider",
        variant === "default" && "bg-gray-100 text-gray-800 border border-gray-200",
        variant === "success" && "bg-emerald-50 text-emerald-700 border border-emerald-200",
        variant === "warning" && "bg-[#FFF9DB] text-[#B25E00] border border-yellow-200",
        variant === "destructive" && "bg-[#FFF5F5] text-[#C92A2A] border border-red-200",
        variant === "info" && "bg-blue-50 text-blue-700 border border-blue-200",
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}
