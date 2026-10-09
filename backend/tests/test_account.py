"""Deleting an account: its practice, its user and Better Auth's record of the sign-in go
together, and a token given before can't bring any of it back."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, insert, select
from test_grading_api import ANSWER, add_question, grading, use_grader
from test_limits import limited
from test_sign_in import BetterAuth, bearer, set_up, users

from app.api.users import SIGNED_IN, user_id_for
from app.db.models import AUTH_ACCOUNT, AUTH_SESSION, AUTH_USER, LOCAL_USER, GradeRequest, User


async def sign_in(sessions, subject: str) -> None:
    """A session and a GitHub account for a user Better Auth keeps, as signing in leaves them."""
    now = datetime.now(UTC)
    async with sessions() as session, session.begin():
        await session.execute(
            insert(AUTH_SESSION).values(
                id=f"session-{subject}",
                expiresAt=now + timedelta(days=7),
                token=f"token-{subject}",
                updatedAt=now,
                userId=subject,
            )
        )
        await session.execute(
            insert(AUTH_ACCOUNT).values(
                id=f"account-{subject}",
                accountId=f"github-{subject}",
                providerId="github",
                userId=subject,
                updatedAt=now,
            )
        )


@pytest.fixture
def better_auth(client, settings, sessions) -> BetterAuth:
    return set_up(settings, sessions)


def signed_in(sessions) -> dict[str, list[str]]:
    """Whose sign-in Better Auth keeps: its users, and whose its sessions and accounts are."""

    async def read():
        async with sessions() as session:
            return {
                table.name: sorted(await session.scalars(select(column)))
                for table, column in (
                    (AUTH_USER, AUTH_USER.c.id),
                    (AUTH_SESSION, AUTH_SESSION.c.userId),
                    (AUTH_ACCOUNT, AUTH_ACCOUNT.c.userId),
                )
            }

    return asyncio.run(read())


def counted(sessions) -> list[str | None]:
    """Whose grades the daily limits count, oldest first: a user's subject, or None."""

    async def read():
        async with sessions() as session:
            rows = select(User.subject).select_from(GradeRequest).outerjoin(User)
            return list(await session.scalars(rows.order_by(GradeRequest.id)))

    return asyncio.run(read())


def test_deleting_an_account_takes_its_practice_and_sign_in_and_nobody_else_s(
    client, sessions, corpus, better_auth
) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    mine, theirs = better_auth.token("ba-1"), better_auth.token("ba-2")
    for subject, token in (("ba-1", mine), ("ba-2", theirs)):
        asyncio.run(sign_in(sessions, subject))
        answer = {"answer": ANSWER}
        client.post(f"/questions/{question_id}/attempts", json=answer, headers=bearer(token))
    client.post("/ratings", json={"question_id": question_id, "value": 1}, headers=bearer(mine))
    client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})

    deleted = client.delete("/account", headers=bearer(mine))

    assert deleted.status_code == 200
    assert deleted.json() == {
        "attempts": 1,
        "grades": 1,
        "reviews": 1,
        "cards": 1,
        "ratings": 1,
        "interviews": 0,
    }
    assert signed_in(sessions) == {
        "auth_user": ["ba-2"],
        "auth_session": ["ba-2"],
        "auth_account": ["ba-2"],
    }
    assert users(sessions) == [(*LOCAL_USER, None), (SIGNED_IN, "ba-2", "Ada Lovelace")]
    # Everyone else's practice stays, and the deleted account's grade still counts, as nobody's.
    assert client.get("/practice/progress", headers=bearer(theirs)).json()["answers"] == 1
    assert client.get("/practice/progress").json()["answers"] == 1
    assert counted(sessions) == [None, "ba-2", "local"]


def test_a_token_given_before_the_account_was_deleted_is_refused(
    client, sessions, better_auth
) -> None:
    token = better_auth.token()
    assert client.get("/practice/progress", headers=bearer(token)).status_code == 200
    assert client.delete("/account", headers=bearer(token)).status_code == 200

    progress = client.get("/practice/progress", headers=bearer(token))
    again = client.delete("/account", headers=bearer(token))

    assert (progress.status_code, progress.json()["detail"]) == (
        401,
        "this account has been deleted",
    )
    assert again.status_code == 401
    # Nobody was added back.
    assert users(sessions) == [(*LOCAL_USER, None)]


def test_a_token_that_comes_while_its_account_is_deleted_waits_and_adds_nobody(
    sessions, better_auth
) -> None:
    better_auth.token("ba-1")

    async def scenario() -> tuple[bool, int]:
        async with sessions() as deleting, sessions() as asking:
            await deleting.execute(delete(AUTH_USER).where(AUTH_USER.c.id == "ba-1"))
            adding = asyncio.create_task(user_id_for(asking, SIGNED_IN, "ba-1", "Ada Lovelace"))
            await asyncio.sleep(0.5)
            waited = not adding.done()
            await deleting.commit()
            with pytest.raises(HTTPException) as refused:
                await adding
            return waited, refused.value.status_code

    # Read without the lock, the account would still have been there, and the user added.
    assert asyncio.run(scenario()) == (True, 401)
    assert users(sessions) == [(*LOCAL_USER, None)]


def test_the_built_in_user_has_no_account_to_delete(client, sessions, corpus) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    client.post(f"/questions/{question_id}/attempts", json={"answer": ANSWER})

    response = client.delete("/account")

    assert response.status_code == 409
    assert client.get("/practice/progress").json()["answers"] == 1
    assert users(sessions) == [(*LOCAL_USER, None)]


def test_deleting_an_account_gives_back_none_of_the_day_s_grades_in_all(
    client, sessions, corpus, settings, better_auth
) -> None:
    limited(settings, in_all=1)
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    first = better_auth.token("ba-1")
    answered = client.post(
        f"/questions/{question_id}/attempts", json={"answer": ANSWER}, headers=bearer(first)
    )
    client.delete("/account", headers=bearer(first))

    # Signed in again, Better Auth makes a new account.
    again = client.post(
        f"/questions/{question_id}/attempts",
        json={"answer": ANSWER},
        headers=bearer(better_auth.token("ba-2")),
    )

    assert answered.status_code == 201
    assert again.status_code == 429
