import React from "react";
import { cn } from "@/lib/utils";

export function Card({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "bg-white border border-[#CBD5E1] rounded shadow-sm p-6",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
