import { useMutation } from "@tanstack/react-query";
import { postLiteratureReview } from "../api";
import { LiteratureReviewRequest } from "../types";

export function useLiteratureReview() {
  return useMutation<{ job_id: string; status: string; message: string }, Error, LiteratureReviewRequest>({
    mutationFn: postLiteratureReview,
  });
}
