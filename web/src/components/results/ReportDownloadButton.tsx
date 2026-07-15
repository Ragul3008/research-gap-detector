import React from "react";
import { Download, Printer } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface ReportDownloadButtonProps {
  result: any;
}

export function ReportDownloadButton({ result }: ReportDownloadButtonProps) {
  const handlePrint = () => {
    window.print();
  };

  const handleDownloadJson = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `novelty_report_${result.input?.title?.substring(0, 20)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="flex items-center space-x-2 no-print">
      <Button variant="outline" onClick={handlePrint} className="flex items-center space-x-1.5">
        <Printer className="w-4 h-4" />
        <span>Print Report / Save PDF</span>
      </Button>
      <Button variant="outline" onClick={handleDownloadJson} className="flex items-center space-x-1.5">
        <Download className="w-4 h-4" />
        <span>Export Data (JSON)</span>
      </Button>
    </div>
  );
}
