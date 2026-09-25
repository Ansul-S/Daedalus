import { useMutation, useQueryClient } from "@tanstack/react-query";

import { getQuestionQueryKey } from "@/client/@tanstack/react-query.gen";
import { editQuestion } from "@/client/sdk.gen";
import type { QuestionDetailOut, QuestionEditIn } from "@/client/types.gen";

// What else shows a question: the lists, the topic counts, and practice, where a retired
// question leaves its room on the map and the library that coins are counted against
const SHOWN_ELSEWHERE = [
  "listQuestions",
  "listTopics",
  "practiceMap",
  "practiceProgress",
  "practiceStats",
];

/** Correct a question, or retire it and put it back. The page shows what the API returns. */
export function useEditQuestion() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, body }: { id: number; body: QuestionEditIn }) => {
      const { data } = await editQuestion({ path: { question_id: id }, body, throwOnError: true });
      return data;
    },
    onSuccess: (saved: QuestionDetailOut) => {
      queryClient.setQueryData(getQuestionQueryKey({ path: { question_id: saved.id } }), saved);
      for (const _id of SHOWN_ELSEWHERE) {
        void queryClient.invalidateQueries({ queryKey: [{ _id }] });
      }
    },
  });
}
