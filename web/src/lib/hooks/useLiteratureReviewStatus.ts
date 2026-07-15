import { useQuery } from "@tanstack/react-query";
import { getLiteratureReviewStatus } from "../api";
import { LiteratureReviewJobStatus } from "../types";

export function useLiteratureReviewStatus(jobId: string | null, enabled: boolean) {
  return useQuery<LiteratureReviewJobStatus, Error>({
    queryKey: ["literature-review-status", jobId],
    queryFn: () => getLiteratureReviewStatus(jobId!),
    enabled: !!jobId && enabled,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data && (data.status === "COMPLETED" || data.status === "FAILED")) {
        return false;
      }
      return 2500; // Poll every 2.5s
    },
  });
}
