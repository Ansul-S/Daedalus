"""Who a request comes from.

Practice belongs to a user: their answers and grades, their review schedule, what practice has
earned them, and their ratings. The library is shared by everyone.

A visitor signs in with GitHub through Better Auth, which runs in the frontend. The frontend
then sends a short-lived token with each request (`Authorization: Bearer`), signed with a key
whose public half Better Auth publishes at `BETTER_AUTH_URL/auth/jwks`. The API checks the
token against that key and the token's issuer, audience and expiry, and knows the user by the
token's subject; it never sees the sign-in itself. A token that doesn't pass is refused with
401, whatever else the request holds.

Without a token, locally the built-in user owns all the practice, so the app works as it
always has without signing in. In production nobody is the built-in user: a per-person route
answers 401 to a request from nobody signed in, while the library can still be read.
"""

from dataclasses import dataclass, field
from time import monotonic
from typing import Annotated

import httpx
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models import LOCAL_USER, User
from app.db.session import get_session

SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

# The users Better Auth signs in, as `users.provider`
SIGNED_IN = "better-auth"
# Where Better Auth publishes its public keys, under its address
JWKS_PATH = "/auth/jwks"
# Better Auth signs with Ed25519, its default
ALGORITHMS = ["EdDSA"]
# Seconds the two servers' clocks may disagree by
LEEWAY = 10
# A token signed with a key the API doesn't know sends it to fetch the keys again, at most
# once a minute: a new key is picked up within a minute, and made-up keys cost one request.
REFETCH_AFTER = 60

bearer = HTTPBearer(
    auto_error=False,
    description="The token Better Auth gives a signed-in visitor. Without it, the built-in "
    "user locally, and nobody in production.",
)


@dataclass
class Keys:
    """Better Auth's public keys, fetched from its JWKS address and kept for the process."""

    url: str
    keys: dict[str, jwt.PyJWK] = field(default_factory=dict)
    fetched_at: float | None = None
    # How the keys are fetched: over the network unless a test gives its own
    transport: httpx.AsyncBaseTransport | None = None

    async def get(self, key_id: str) -> jwt.PyJWK | None:
        stale = self.fetched_at is None or monotonic() - self.fetched_at > REFETCH_AFTER
        if key_id not in self.keys and stale:
            await self.fetch()
        return self.keys.get(key_id)

    async def fetch(self) -> None:
        try:
            async with httpx.AsyncClient(timeout=10, transport=self.transport) as client:
                response = await client.get(self.url)
                response.raise_for_status()
                found = jwt.PyJWKSet.from_dict(response.json())
        except (httpx.HTTPError, jwt.PyJWKSetError, ValueError) as exc:
            # Not the visitor's fault: saying 401 would sign them out.
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "sign-in can't be checked right now; try again in a moment",
            ) from exc
        self.keys = {key.key_id: key for key in found.keys if key.key_id}
        self.fetched_at = monotonic()


def issuer(settings: Settings) -> str | None:
    """Better Auth's address, which its tokens name as their issuer and audience."""
    return settings.better_auth_url.rstrip("/") if settings.better_auth_url else None


def get_keys(request: Request, settings: SettingsDep) -> Keys | None:
    """Better Auth's keys, kept on the app for as long as its address stays the same; None
    when sign-in is not set up."""
    address = issuer(settings)
    if address is None:
        return None
    url = address + JWKS_PATH
    keys: Keys | None = getattr(request.app.state, "keys", None)
    if keys is None or keys.url != url:
        keys = request.app.state.keys = Keys(url)
    return keys


def refused(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


async def user_id_for(
    session: AsyncSession, provider: str, subject: str, name: str | None = None
) -> int:
    """A user's id, adding the user the first time they are seen, in a transaction of its
    own: they are kept whatever the request goes on to do."""
    found = select(User.id).where(User.provider == provider, User.subject == subject)
    user_id = await session.scalar(found)
    if user_id is None:
        await session.execute(
            insert(User)
            .values(provider=provider, subject=subject, name=name)
            .on_conflict_do_nothing()
        )
        await session.commit()
        user_id = (await session.execute(found)).scalar_one()
    return user_id


async def local_user(session: AsyncSession) -> int:
    """The built-in user's id. The migration that brought in users added it; should it have
    gone since, it is added again."""
    return await user_id_for(session, *LOCAL_USER)


async def signed_in_user(
    session: AsyncSession, settings: Settings, keys: Keys | None, token: str
) -> int:
    """The user a Better Auth token names, once the token has passed every check."""
    address = issuer(settings)
    if keys is None or address is None:
        raise refused("sign-in is not set up here")
    try:
        key_id = jwt.get_unverified_header(token).get("kid")
        key = await keys.get(key_id) if isinstance(key_id, str) else None
        if key is None:
            raise jwt.InvalidTokenError("signed with a key Better Auth doesn't publish")
        claims = jwt.decode(
            token,
            key.key,
            algorithms=ALGORITHMS,
            issuer=address,
            audience=address,
            leeway=LEEWAY,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.InvalidTokenError as exc:
        raise refused("the sign-in has expired or is not valid; sign in again") from exc
    name = claims.get("name")
    return await user_id_for(
        session, SIGNED_IN, str(claims["sub"]), name if isinstance(name, str) else None
    )


async def user_or_none(
    session: SessionDep,
    settings: SettingsDep,
    keys: Annotated[Keys | None, Depends(get_keys)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> int | None:
    """The user a request comes from, or None when nobody is signed in."""
    if credentials is not None:
        return await signed_in_user(session, settings, keys, credentials.credentials)
    if settings.environment == "production":
        return None
    return await local_user(session)


async def current_user(user_id: Annotated[int | None, Depends(user_or_none)]) -> int:
    """The user a per-person request comes from; 401 when nobody is signed in."""
    if user_id is None:
        raise refused("sign in to practise")
    return user_id


# A per-person route takes the user; a library route that shows a user's own ratings takes
# the user when there is one.
UserDep = Annotated[int, Depends(current_user)]
MaybeUserDep = Annotated[int | None, Depends(user_or_none)]
