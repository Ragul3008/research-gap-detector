import { useMutation } from "@tanstack/react-query";
import { postAnalyze } from "../api";
import { AnalyzeRequest, AnalyzeResponse } from "../types";

export function useAnalyze() {
  return useMutation<AnalyzeResponse, Error, AnalyzeRequest>({
    mutationFn: postAnalyze,
  });
}
