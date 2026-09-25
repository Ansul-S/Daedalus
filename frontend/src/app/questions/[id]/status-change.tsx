"use client";

import { useEffect, useRef, useState } from "react";

import type { QuestionDetailOut } from "@/client/types.gen";
import { Field } from "@/components/field";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, fieldErrors } from "@/lib/api-errors";
import { MAX_REASON_CHARS, questionNumber } from "@/lib/questions";
import { cn } from "@/lib/utils";

import { useEditQuestion } from "./save";

export type StatusAction = "retire" | "restore";

function refusal(error: Error): string {
  if (error instanceof ApiError && error.status === null) {
    return "The API can't be reached, so nothing changed. Start it with make api, then try again.";
  }
  return `Nothing changed: ${error.message.replace(/\.$/, "")}.`;
}

/** Take a question out of practice, or put it back, with the reason kept beside the change. */
export function StatusChange({
  question,
  action,
  onClose,
}: {
  question: QuestionDetailOut;
  action: StatusAction;
  onClose: (changed: boolean) => void;
}) {
  const save = useEditQuestion();
  const [reason, setReason] = useState("");
  const field = useRef<HTMLTextAreaElement>(null);
  const retire = action === "retire";
  const name = questionNumber(question.id);

  useEffect(() => {
    field.current?.focus();
  }, []);

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (save.isPending) return;
    save.mutate(
      {
        id: question.id,
        body: { status: retire ? "retired" : "accepted", reason: reason.trim() || null },
      },
      { onSuccess: () => onClose(true) },
    );
  }

  return (
    <form
      aria-label={retire ? `Retire ${name}` : `Put ${name} back`}
      onSubmit={submit}
      onKeyDown={(event) => {
        if (event.key !== "Escape") return;
        event.preventDefault();
        onClose(false);
      }}
      className={cn(
        "grid max-w-[40rem] gap-3.5 border border-l-[3px] border-line-2 px-4 py-4",
        retire && "border-l-thread",
      )}
    >
      <p className="text-small">
        {retire
          ? "Retiring it takes it out of practice and out of the duplicate check. The question and its answers are kept, and it can be put back."
          : "Putting it back returns it to practice, where its schedule left off."}
      </p>
      <Field
        label={
          <>
            Why <span className="text-fg-3">· optional, kept with the change</span>
          </>
        }
        error={fieldErrors(save.error).get("reason")}
      >
        {(control) => (
          <Textarea
            {...control}
            ref={field}
            maxLength={MAX_REASON_CHARS}
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            className="min-h-12 py-2.5 text-small"
          />
        )}
      </Field>
      {save.error && (
        <p role="alert" className="text-small text-thread">
          {refusal(save.error)}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        <Button
          type="submit"
          size="sm"
          variant={retire ? "destructive" : "default"}
          disabled={save.isPending}
        >
          {retire ? `Retire ${name}` : `Put ${name} back`}
        </Button>
        <Button type="button" variant="ghost" size="sm" onClick={() => onClose(false)}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
