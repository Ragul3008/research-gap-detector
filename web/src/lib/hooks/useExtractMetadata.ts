import { useMutation } from "@tanstack/react-query";
import { extractMetadata } from "../api";

export function useExtractMetadata() {
  return useMutation<
    { status: string; filename: string; metadata: any; full_text: string },
    Error,
    File
  >({
    mutationFn: extractMetadata,
  });
}
