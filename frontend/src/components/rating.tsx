"use client";

import { type QueryClient, useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";

import { getQuestionQueryKey } from "@/client/@tanstack/react-query.gen";
import { addRating } from "@/client/sdk.gen";
import type { QuestionDetailOut, RatingIn, RatingOut } from "@/client/types.gen";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ApiError } from "@/lib/api-errors";
import { isSubmitKey } from "@/lib/keyboard";
import { cn } from "@/lib/utils";

// Your own verdict on a question or on a grade, said in words: two small buttons, the chosen
// one inked in. A poor question or an unfair grade can take a short note on why. The API
// keeps every rating, and the latest is the one that stands.

/** What the API takes (backend/app/api/ratings.py). */
export const MAX_NOTE_CHARS = 500;

export type Rated = "question" | "grade";
export type RatingValue = 1 | -1;

export const RATING_WORDS: Record<
  Rated,
  { good: string; poor: string; group: string; ask: string }
> = {
  question: {
    good: "Good question",
    poor: "Poor question",
    group: "Rate this question",
    ask: "What makes it poor?",
  },
  grade: {
    good: "Fair grade",
    poor: "Unfair grade",
    group: "Rate this grade",
    ask: "What did the grader get wrong?",
  },
};

export function ratingWord(rated: Rated, value: RatingValue): string {
  return value === 1 ? RATING_WORDS[rated].good : RATING_WORDS[rated].poor;
}

/** The pair of buttons alone, the chosen one pressed. */
export function RatingButtons({
  rated,
  value,
  onChoose,
  poorRef,
}: {
  rated: Rated;
  value: RatingValue | null;
  onChoose: (value: RatingValue) => void;
  poorRef?: React.Ref<HTMLButtonElement>;
}) {
  const words = RATING_WORDS[rated];
  return (
    <div role="group" aria-label={words.group} className="flex flex-wrap gap-2">
      {([1, -1] as const).map((choice) => (
        <Button
          key={choice}
          ref={choice === -1 ? poorRef : undefined}
          variant="outline"
          size="sm"
          aria-pressed={value === choice}
          className="aria-pressed:border-fg aria-pressed:bg-fg aria-pressed:text-ground"
          onClick={() => onChoose(choice)}
        >
          {choice === 1 ? words.good : words.poor}
        </Button>
      ))}
    </div>
  );
}

async function saveRating(body: RatingIn): Promise<RatingOut> {
  const { data } = await addRating({ body, throwOnError: true });
  return data;
}

/** Whatever shows the rated question or grade learns of the rating. */
function refresh(queryClient: QueryClient, saved: RatingOut) {
  if (saved.question_id !== null) {
    queryClient.setQueryData<QuestionDetailOut>(
      getQuestionQueryKey({ path: { question_id: saved.question_id } }),
      (question) => question && { ...question, rating: saved },
    );
    void queryClient.invalidateQueries({ queryKey: [{ _id: "listQuestions" }] });
  } else {
    for (const _id of ["listAttempts", "getAttempt"]) {
      void queryClient.invalidateQueries({ queryKey: [{ _id }] });
    }
  }
}

function problem(error: Error): string {
  if (error instanceof ApiError && error.status === null) return "the API can't be reached";
  return error.message.replace(/\.$/, "");
}

/** Rate a question good or poor, or a grade fair or unfair. A poor rating is kept at once,
 * then asks for a note. `save` sends the rating: the API unless told otherwise. */
export function Rate({
  rated,
  id,
  initial,
  save = saveRating,
  className,
}: {
  rated: Rated;
  id: number;
  initial: RatingOut | null;
  save?: (body: RatingIn) => Promise<RatingOut>;
  className?: string;
}) {
  const words = RATING_WORDS[rated];
  const queryClient = useQueryClient();
  const [rating, setRating] = useState(initial);
  const [noting, setNoting] = useState(false);
  const [note, setNote] = useState("");
  const noteField = useRef<HTMLTextAreaElement>(null);
  const poor = useRef<HTMLButtonElement>(null);
  const noteId = useId();

  const rate = useMutation({
    mutationFn: save,
    onSuccess: (saved) => {
      setRating(saved);
      refresh(queryClient, saved);
    },
  });

  useEffect(() => {
    if (noting) noteField.current?.focus();
  }, [noting]);

  function send(value: RatingValue, text: string | null, then?: () => void) {
    const target = rated === "question" ? { question_id: id } : { grade_id: id };
    rate.mutate({ ...target, value, note: text }, { onSuccess: then });
  }

  function choose(value: RatingValue) {
    if (rate.isPending) return;
    if (value === -1 && rating?.value === -1) {
      // Rated poor already: pressing it again opens the note
      setNote(rating.note ?? "");
      setNoting(true);
    } else if (value !== rating?.value) {
      setNoting(false);
      send(value, null, () => {
        if (value !== -1) return;
        setNote("");
        setNoting(true);
      });
    }
  }

  function closeNote() {
    setNoting(false);
    poor.current?.focus();
  }

  function saveNote(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (rate.isPending) return;
    const text = note.trim();
    if (text === (rating?.note ?? "")) closeNote();
    else send(-1, text || null, closeNote);
  }

  return (
    <div className={cn("grid content-start gap-2.5", className)}>
      <RatingButtons
        rated={rated}
        value={rating?.value ?? null}
        onChoose={choose}
        poorRef={poor}
      />
      {noting ? (
        <form
          onSubmit={saveNote}
          onKeyDown={(event) => {
            if (event.key !== "Escape") return;
            event.preventDefault();
            closeNote();
          }}
          className="grid max-w-[30rem] gap-2"
        >
          <label htmlFor={noteId} className="type-label text-fg-2">
            {words.ask} <span className="text-fg-3">· optional</span>
          </label>
          <Textarea
            ref={noteField}
            id={noteId}
            value={note}
            maxLength={MAX_NOTE_CHARS}
            rows={2}
            onChange={(event) => setNote(event.target.value)}
            onKeyDown={(event) => {
              if (!isSubmitKey(event)) return;
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }}
            className="min-h-0 px-3 py-2 text-small"
          />
          <div className="flex flex-wrap gap-2">
            <Button type="submit" size="sm">
              Save note
            </Button>
            <Button type="button" variant="ghost" size="sm" onClick={closeNote}>
              {rating?.note ? "Cancel" : "No note"}
            </Button>
          </div>
        </form>
      ) : (
        rating?.value === -1 &&
        rating.note && (
          <p className="max-w-[34rem] text-small text-fg-2">
            <span className="type-label mr-2">Your note</span>“{rating.note}”
          </p>
        )
      )}
      {rate.isError && (
        <p role="alert" className="text-small text-thread">
          The rating wasn&apos;t saved: {problem(rate.error)}.
        </p>
      )}
    </div>
  );
}
