import React, { useState } from "react";
import { FormField } from "./FormField";
import { Button } from "@/components/ui/Button";

const DOMAIN_OPTIONS = [
  "",
  "Sociology", "History", "Political Science", "Economics",
  "Psychology", "Environmental Science", "Literature", "Philosophy",
  "Anthropology", "Education", "Geography", "Commerce", "Linguistics",
  "Botany", "Zoology", "Chemistry", "Physics", "Mathematics", "Statistics",
  "Computer Science and Engineering", "Electronics and Communication Engineering",
  "Electrical Engineering", "Mechanical Engineering", "Civil Engineering"
];

interface AnalysisFormProps {
  initialValues?: { title?: string; abstract?: string; keywords?: string; domain?: string };
  onSubmit: (values: { 
    title: string; 
    abstract: string; 
    keywords: string; 
    domain: string; 
    checkPlagiarism: boolean;
    topK: number;
    numAuthors: number;
  }) => void;
  loading: boolean;
}

export function AnalysisForm({ initialValues, onSubmit, loading }: AnalysisFormProps) {
  const [title, setTitle] = useState(initialValues?.title || "");
  const [domain, setDomain] = useState(initialValues?.domain || "");
  const [abstract, setAbstract] = useState(initialValues?.abstract || "");
  const [keywords, setKeywords] = useState(initialValues?.keywords || "");
  const [checkPlagiarism, setCheckPlagiarism] = useState(true);
  const [topK, setTopK] = useState<string | number>(10);
  const [numAuthors, setNumAuthors] = useState<string | number>(10);

  React.useEffect(() => {
    if (initialValues) {
      if (initialValues.title) setTitle(initialValues.title);
      if (initialValues.abstract) setAbstract(initialValues.abstract);
      if (initialValues.keywords) setKeywords(initialValues.keywords);
      if (initialValues.domain && DOMAIN_OPTIONS.includes(initialValues.domain)) {
        setDomain(initialValues.domain);
      }
    }
  }, [initialValues]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const finalTopK = Math.max(1, Math.min(100, typeof topK === "number" ? topK : parseInt(topK as string) || 10));
    const finalNumAuthors = Math.max(1, Math.min(100, typeof numAuthors === "number" ? numAuthors : parseInt(numAuthors as string) || 10));
    onSubmit({ title, abstract, keywords, domain, checkPlagiarism, topK: finalTopK, numAuthors: finalNumAuthors });
  };

  const wordCount = abstract.trim() ? abstract.trim().split(/\s+/).length : 0;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <FormField label="Proposed Research Title" required hint="Describe your core thesis focus clearly. Minimum 5 characters.">
        <input
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="e.g., Deep Learning for Medical Image Segmentation in Indian Rural Hospitals"
          className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans"
          required
          minLength={5}
        />
      </FormField>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <FormField label="Subject Domain" hint="Select the primary academic domain.">
          <select
            value={domain}
            onChange={(e) => setDomain(e.target.value)}
            className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans"
          >
            {DOMAIN_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt || "Select Subject Domain"}
              </option>
            ))}
          </select>
        </FormField>

        <FormField label="Keywords (comma-separated)" hint="Separate academic tags with commas.">
          <input
            type="text"
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
            placeholder="e.g., deep learning, MRI, rural India, health gaps"
            className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans"
          />
        </FormField>
      </div>

      <FormField 
        label={`Abstract / Summary (${wordCount} words)`} 
        hint="Summarize your goals, region, method, theme, and theoretical objectives."
      >
        <textarea
          value={abstract}
          onChange={(e) => setAbstract(e.target.value)}
          rows={6}
          placeholder="Describe your research questions, methodology, region of study, target population, and key theoretical objectives..."
          className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans"
        />
      </FormField>

      <div className="flex items-center space-x-2 pt-2 border-t border-[#E5E7EB]">
        <input
          type="checkbox"
          id="plag"
          checked={checkPlagiarism}
          onChange={(e) => setCheckPlagiarism(e.target.checked)}
          className="rounded border-[#CBD5E1] text-[#2563EB] focus:ring-[#2563EB] h-4 w-4"
        />
        <label htmlFor="plag" className="text-xs text-gray-700 font-semibold cursor-pointer">
          Enable phrase-level Plagiarism and Originality scanning
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-[#E5E7EB]">
        <FormField label="Related Papers to Search" hint="Select number of papers to retrieve (1 - 100)">
          <input
            type="number"
            min={1}
            max={100}
            value={topK}
            onChange={(e) => setTopK(e.target.value === "" ? "" : parseInt(e.target.value) || 0)}
            className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans font-mono"
          />
        </FormField>

        <FormField label="Authors to Suggest" hint="Select number of researchers to suggest (1 - 100)">
          <input
            type="number"
            min={1}
            max={100}
            value={numAuthors}
            onChange={(e) => setNumAuthors(e.target.value === "" ? "" : parseInt(e.target.value) || 0)}
            className="w-full text-sm px-3 py-2.5 border border-[#CBD5E1] rounded focus:outline-none focus:ring-2 focus:ring-[#2563EB] bg-white font-sans font-mono"
          />
        </FormField>
      </div>

      <Button type="submit" disabled={loading} className="w-full">
        {loading ? "⏳ Run Full Analysis Pipeline..." : "🔬 Vetting Novelty & Gap Dimensions"}
      </Button>
    </form>
  );
}
