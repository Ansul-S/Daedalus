"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";

import { deletePractice } from "@/client/sdk.gen";
import type { DeletedOut } from "@/client/types.gen";
import { useSignInFirst } from "@/components/sign-in";
import { Button } from "@/components/ui/button";
import { clearDrafts } from "@/lib/drafts";
import { plural, problem } from "@/lib/library";
import { signIn } from "@/lib/session";
import { SIGN_IN } from "@/lib/sign-in";

import { DeleteButton } from "./delete-button";

// Deleting a visitor's practice: their answers and grades, the review history and schedule,
// their ratings, and the drafts in this browser. Asked twice, since it can't be undone.

async function remove(): Promise<DeletedOut> {
  const { data } = await deletePractice({ throwOnError: true });
  return data;
}

/** The practice that went, e.g. "3 answers, 3 grades, 2 reviews, 2 scheduled questions and
 * 1 rating"; null when there was none. */
export function practiceDeleted({ attempts, grades, reviews, cards, ratings }: DeletedOut) {
  if (attempts + cards + ratings === 0) return null;
  const parts = [
    plural(attempts, "answer"),
    plural(grades, "grade"),
    plural(reviews, "review"),
    plural(cards, "scheduled question"),
  ];
  return `${parts.join(", ")} and ${plural(ratings, "rating")}`;
}

function deletedLine(deleted: DeletedOut): string {
  const went = practiceDeleted(deleted);
  return went ? `Deleted ${went}.` : "There was no practice to delete.";
}

export function DeletePractice() {
  const signInFirst = useSignInFirst();
  const queryClient = useQueryClient();
  const removal = useMutation({
    mutationFn: remove,
    onSuccess: () => {
      clearDrafts();
      // Practice shows all over the app: the header's XP, the dashboard, ratings in the bank
      void queryClient.invalidateQueries();
    },
  });

  if (signInFirst === undefined) return null;
  if (signInFirst) {
    return (
      <>
        <p>
          Nothing is kept for a visitor who never signs in. To delete an account or its practice,
          sign in with it first.
        </p>
        {SIGN_IN && (
          <div>
            <Button size="sm" onClick={() => void signIn()}>
              Sign in with GitHub
            </Button>
          </div>
        )}
      </>
    );
  }

  return (
    <>
      <p>
        Deleting removes every answer you have given and its grades, your review schedule and
        its history, your ratings, and the drafts in this browser.{" "}
        {SIGN_IN ? "The library and your account stay." : "The library stays."} So does the
        count of grades used today: the free tiers are shared, so deleting doesn&apos;t give the
        day&apos;s grades back.
      </p>
      <DeleteButton
        label="Delete my practice"
        deleting={removal.isPending}
        onDelete={() => removal.mutateAsync()}
      />
      <div className="empty:hidden">
        {removal.isSuccess && <p role="status">{deletedLine(removal.data)}</p>}
        {removal.isError && (
          <p role="alert" className="text-thread">
            {problem(removal.error)}
          </p>
        )}
      </div>
    </>
  );
}
