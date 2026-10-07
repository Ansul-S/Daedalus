"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { deletePractice } from "@/client/sdk.gen";
import type { DeletedOut } from "@/client/types.gen";
import { useSignInFirst } from "@/components/sign-in";
import { Button } from "@/components/ui/button";
import { clearDrafts } from "@/lib/drafts";
import { plural, problem } from "@/lib/library";
import { signIn } from "@/lib/session";
import { SIGN_IN } from "@/lib/sign-in";

// Deleting a visitor's practice: their answers and grades, the review history and schedule,
// their ratings, and the drafts in this browser. Asked twice, since it can't be undone.

async function remove(): Promise<DeletedOut> {
  const { data } = await deletePractice({ throwOnError: true });
  return data;
}

function deletedLine({ attempts, grades, reviews, cards, ratings }: DeletedOut): string {
  if (attempts + cards + ratings === 0) return "There was no practice to delete.";
  const parts = [
    plural(attempts, "answer"),
    plural(grades, "grade"),
    plural(reviews, "review"),
    plural(cards, "scheduled question"),
  ];
  return `Deleted ${parts.join(", ")} and ${plural(ratings, "rating")}.`;
}

export function DeletePractice() {
  const signInFirst = useSignInFirst();
  const queryClient = useQueryClient();
  const [asking, setAsking] = useState(false);
  const removal = useMutation({
    mutationFn: remove,
    onSuccess: () => {
      clearDrafts();
      setAsking(false);
      // Practice shows all over the app: the header's XP, the dashboard, ratings in the bank
      void queryClient.invalidateQueries();
    },
  });

  if (signInFirst === undefined) return null;
  if (signInFirst) {
    return (
      <>
        <p>
          Nothing is kept for a visitor who never signs in. To delete the practice of an account,
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
        its history, your ratings, and the drafts in this browser. The library and your sign-in
        stay. So does the count of grades used today: the free tiers are shared, so deleting
        doesn&apos;t give the day&apos;s grades back.
      </p>
      {asking ? (
        <div className="grid gap-3 border border-l-[3px] border-line-2 border-l-thread px-[18px] py-3.5">
          <p className="font-medium">This can&apos;t be undone.</p>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="destructive"
              size="sm"
              disabled={removal.isPending}
              onClick={() => removal.mutate()}
            >
              {removal.isPending ? "Deleting…" : "Delete it"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={removal.isPending}
              onClick={() => setAsking(false)}
            >
              Keep it
            </Button>
          </div>
        </div>
      ) : (
        <div>
          <Button variant="destructive" size="sm" onClick={() => setAsking(true)}>
            Delete my practice
          </Button>
        </div>
      )}
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
