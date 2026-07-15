import type { Metadata } from "next";
import Link from "next/link";
import ReactQueryProvider from "@/lib/queryClient";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Research Novelty & Gap Detector",
  description: "Advanced Academic Literature Analysis and Literature Review Synthesizer for PhD Scholars",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-[#F4F6F9] text-[#1A2536] font-sans">
        <ReactQueryProvider>
          {/* Header */}
          <header className="bg-[#1A2536] text-white border-b border-[#CBD5E1] sticky top-0 z-50 shadow-sm no-print">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between items-center h-16">
                <div className="flex items-center space-x-3">
                  <span className="text-xl font-serif font-bold tracking-tight text-white">
                    🔬 AI Research Novelty & Gap Detector
                  </span>
                  <span className="bg-[#B25E00] text-white text-[10px] px-2 py-0.5 rounded font-mono font-bold">
                    v3.1
                  </span>
                </div>
                <nav className="flex space-x-8 font-sans font-medium text-sm">
                  <Link href="/" className="hover:text-gray-200 text-white border-b-2 border-transparent hover:border-[#B25E00] py-2 transition duration-150">
                    🔬 Novelty & Gap Analyzer
                  </Link>
                  <Link href="/literature-review" className="hover:text-gray-200 text-white border-b-2 border-transparent hover:border-[#B25E00] py-2 transition duration-150">
                    ✍️ Literature Review
                  </Link>
                </nav>
              </div>
            </div>
          </header>

          {/* Page Content */}
          <main className="flex-grow max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </main>

          {/* Footer */}
          <footer className="bg-white border-t border-[#CBD5E1] py-6 text-center text-xs text-gray-500 font-sans no-print">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <p className="font-semibold text-gray-700">AI Research Novelty & Gap Detector</p>
              <p className="mt-1">Tailored for Arts, Science & Engineering PhD Scholars in India · Powered by Gemini LLM & FAISS</p>
            </div>
          </footer>
        </ReactQueryProvider>
      </body>
    </html>
  );
}
