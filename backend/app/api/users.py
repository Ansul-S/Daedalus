"""Who a request comes from.

Practice belongs to a user: their answers and grades, their review schedule, what practice has
earned them, and their ratings. The library is shared by everyone. Locally nobody signs in:
the built-in user owns all the practice, so the app works as it always has. In production
nobody is the built-in user, so a per-person route answers 401 to a request from nobody signed
in, while the library can still be read.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models import LOCAL_USER, User
from app.db.session import get_session

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


async def local_user(session: AsyncSession) -> int:
    """The built-in user's id. The migration that brought in users added it; should it have
    gone since, it is added again, and kept whatever the request goes on to do."""
    provider, subject = LOCAL_USER
    found = select(User.id).where(User.provider == provider, User.subject == subject)
    user_id = await session.scalar(found)
    if user_id is None:
        await session.execute(
            insert(User).values(provider=provider, subject=subject).on_conflict_do_nothing()
        )
        await session.commit()
        user_id = (await session.execute(found)).scalar_one()
    return user_id


async def user_or_none(session: SessionDep, settings: SettingsDep) -> int | None:
    """The user a request comes from, or None when nobody is signed in."""
    if settings.environment == "production":
        return None
    return await local_user(session)


async def current_user(user_id: Annotated[int | None, Depends(user_or_none)]) -> int:
    """The user a per-person request comes from; 401 when nobody is signed in."""
    if user_id is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "sign in to practise",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


# A per-person route takes the user; a library route that shows a user's own ratings takes
# the user when there is one.
UserDep = Annotated[int, Depends(current_user)]
MaybeUserDep = Annotated[int | None, Depends(user_or_none)]
