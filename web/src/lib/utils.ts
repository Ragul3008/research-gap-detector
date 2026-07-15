export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatScore(value: number): string {
  return value.toFixed(2);
}

export function gapSeverityLabel(gapSize: number): "HIGH" | "MEDIUM" | "LOW" {
  if (gapSize >= 0.7) return "HIGH";
  if (gapSize >= 0.4) return "MEDIUM";
  return "LOW";
}

export function cn(...classes: (string | boolean | undefined | null)[]) {
  return classes.filter(Boolean).join(" ");
}
