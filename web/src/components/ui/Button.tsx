import React from "react";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost";
}

export function Button({ className, variant = "primary", children, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "px-4 py-2.5 rounded text-sm font-sans font-bold transition duration-150 focus:outline-none focus:ring-2 focus:ring-[#2563EB] disabled:opacity-50 cursor-pointer shadow-sm",
        variant === "primary" && "bg-[#1A2536] hover:bg-[#2563EB] text-white",
        variant === "secondary" && "bg-[#2563EB] hover:bg-blue-700 text-white",
        variant === "outline" && "bg-white hover:bg-gray-50 border border-[#CBD5E1] text-[#1A2536]",
        variant === "ghost" && "bg-transparent hover:bg-gray-100 text-[#1A2536] shadow-none",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
