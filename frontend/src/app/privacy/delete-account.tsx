"use client";

import { useMutation } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";

import { deleteAccount } from "@/client/sdk.gen";
import type { DeletedOut } from "@/client/types.gen";
import { useSignedIn } from "@/components/sign-in";
import { clearDrafts } from "@/lib/drafts";
import { problem } from "@/lib/library";
import { signOut } from "@/lib/session";

import { DeleteButton } from "./delete-button";
import { practiceDeleted } from "./delete-practice";

// Deleting a signed-in visitor's account: their practice, the drafts in this browser, and their
// sign-in (backend/app/api/account.py). Signing out then starts the page over, so what was
// deleted is kept in this tab for the page that comes back to say.

const DELETED = "daedalus:account-deleted";

function deletedLine(deleted: DeletedOut): string {
  const went = practiceDeleted(deleted);
  return `Deleted your account${went ? `, with ${went}` : ""}. You are signed out.`;
}

async function remove(): Promise<void> {
  const { data } = await deleteAccount({ throwOnError: true });
  clearDrafts();
  try {
    sessionStorage.setItem(DELETED, deletedLine(data));
  } catch {
    // storage unavailable: the page comes back signed out, without saying so
  }
  await signOut();
}

// Read once, then forgotten, so that it is said on the first page after the deletion only
let said: string | null | undefined;

function deletedBefore(): string | null {
  if (said === undefined) {
    try {
      said = sessionStorage.getItem(DELETED);
      sessionStorage.removeItem(DELETED);
    } catch {
      said = null;
    }
  }
  return said;
}

function unchanging(): () => void {
  return () => {};
}

/** What the deletion of an account took, on the page that comes back once it is deleted. */
export function AccountDeleted() {
  const line = useSyncExternalStore(unchanging, deletedBefore, () => null);
  return line ? <p role="status">{line}</p> : null;
}

export function DeleteAccount() {
  const signedIn = useSignedIn();
  const removal = useMutation({ mutationFn: remove });

  if (!signedIn) return null;
  return (
    <>
      <p>
        Deleting your account deletes your practice as above, and your sign-in with it: your
        GitHub id, your name and the link to your avatar, and your sessions in every browser,
        so you are signed out everywhere. Today&apos;s grades still count toward the total for
        the day, but no longer as yours. Signing in again starts a new account.
      </p>
      {removal.isSuccess ? (
        // The page starts over, signed out, and says what was deleted.
        <p role="status">Deleted. Signing out…</p>
      ) : (
        <DeleteButton
          label="Delete my account"
          deleting={removal.isPending}
          onDelete={() => removal.mutateAsync()}
        />
      )}
      {removal.isError && (
        <p role="alert" className="text-thread">
          {problem(removal.error)}
        </p>
      )}
    </>
  );
}
