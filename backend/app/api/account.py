"""A signed-in visitor's account, and deleting it.

Deleting an account deletes its practice, as `DELETE /practice` does, the user that owned it
(`app.api.users`), and Better Auth's record of the sign-in: the GitHub id, name and avatar
link, the link to the GitHub account, and the sessions in every browser
(frontend/src/lib/auth.ts). It all goes in one transaction. Signing in again starts a new
account.

The grades the daily limits count stay, no longer anyone's (`app.grading.limits`): deleting an
account gives none of the day's grades back to the count in all.
"""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import delete

from app.api.practice import DeletedOut, SessionDep, remove_practice
from app.api.users import SIGNED_IN, UserDep
from app.db.models import AUTH_USER, User

router = APIRouter(tags=["account"])


@router.delete("/account")
async def delete_account(session: SessionDep, user_id: UserDep) -> DeletedOut:
    """Delete your account: your practice, as `DELETE /practice` deletes it, and your sign-in,
    which signs you out in every browser. Answers with the practice that was deleted."""
    user = await session.get(User, user_id)
    if user is None or user.provider != SIGNED_IN:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "only a signed-in visitor has an account to delete; delete the practice instead",
        )
    # Better Auth's user first, with its sessions and GitHub account (ON DELETE CASCADE): a
    # request whose token names this account and that would add the user again waits for this
    # deletion, then finds the account gone.
    await session.execute(delete(AUTH_USER).where(AUTH_USER.c.id == user.subject))
    deleted = await remove_practice(session, user_id)
    await session.execute(delete(User).where(User.id == user_id))
    await session.commit()
    return deleted
