import React from "react";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from "recharts";

interface GapRadarChartProps {
  data: { subject: string; value: number }[];
}

export function GapRadarChart({ data }: GapRadarChartProps) {
  return (
    <div className="w-full h-80 flex flex-col items-center justify-center">
      <div className="w-full h-full">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
            <PolarGrid stroke="#D1D5DB" />
            <PolarAngleAxis dataKey="subject" tick={{ fill: "#1A2536", fontSize: 11, fontWeight: 700 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "#6B7280", fontSize: 10 }} />
            <Radar
              name="Research Gap Size"
              dataKey="value"
              stroke="#1A2536"
              fill="#2563EB"
              fillOpacity={0.2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
