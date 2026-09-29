"""Signing in: the API trusts a Better Auth token only once it has checked it against the keys
Better Auth publishes, and knows the visitor by its subject. Tokens here are signed with a key
made for the test, served as Better Auth serves its own."""

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sqlalchemy import select
from test_grading_api import ANSWER, add_question, grading, use_grader

from app.api.users import SIGNED_IN, Keys, get_keys
from app.db.models import LOCAL_USER, User
from app.main import app

ADDRESS = "http://localhost:3000"
JWKS = f"{ADDRESS}/auth/jwks"


def public_jwk(private: Ed25519PrivateKey, key_id: str) -> dict:
    """A public key as Better Auth's /auth/jwks lists it."""
    found = json.loads(jwt.algorithms.OKPAlgorithm.to_jwk(private.public_key()))
    return found | {"kid": key_id, "alg": "EdDSA"}


class BetterAuth:
    """Stands in for the frontend's Better Auth: it signs tokens, and serves its public keys
    at the JWKS address, counting how often they are fetched."""

    def __init__(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.jwks = {"keys": [public_jwk(self.key, "k1")]}
        self.fetches = 0
        self.down = False

    def serve(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == JWKS
        self.fetches += 1
        if self.down:
            return httpx.Response(502)
        return httpx.Response(200, json=self.jwks)

    def token(
        self,
        subject: str = "ba-user-1",
        *,
        key: Ed25519PrivateKey | None = None,
        key_id: str | None = "k1",
        lasts: int = 900,
        **claims,
    ) -> str:
        now = int(time.time())
        payload = {
            "sub": subject,
            "name": "Ada Lovelace",
            "iss": ADDRESS,
            "aud": ADDRESS,
            "iat": now,
            "exp": now + lasts,
        } | claims
        headers = {"kid": key_id} if key_id else {}
        return jwt.encode(
            {k: v for k, v in payload.items() if v is not None},
            key or self.key,
            algorithm="EdDSA",
            headers=headers,
        )


@pytest.fixture
def better_auth(client, settings) -> BetterAuth:
    """Sign-in set up: the API trusts the stand-in's keys, fetched as it would fetch them."""
    stand_in = BetterAuth()
    settings.better_auth_url = ADDRESS
    keys = Keys(JWKS, transport=httpx.MockTransport(stand_in.serve))
    # The client fixture clears every override when the test ends.
    app.dependency_overrides[get_keys] = lambda: keys
    return stand_in


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def users(sessions) -> list[tuple[str, str, str | None]]:
    async def read():
        async with sessions() as session:
            rows = await session.execute(
                select(User.provider, User.subject, User.name).order_by(User.id)
            )
            return list(rows.tuples())

    return asyncio.run(read())


def test_a_token_names_a_user_of_its_own(client, sessions, corpus, better_auth) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    use_grader(grading(corpus.scaling))
    token = better_auth.token()

    answered = client.post(
        f"/questions/{question_id}/attempts", json={"answer": ANSWER}, headers=bearer(token)
    )
    theirs = client.get("/practice/progress", headers=bearer(token)).json()
    mine = client.get("/practice/progress").json()

    assert answered.status_code == 201
    assert (theirs["answers"], mine["answers"]) == (1, 0)
    # Added once, with the name the token gives, and known by its subject from then on
    assert users(sessions) == [(*LOCAL_USER, None), (SIGNED_IN, "ba-user-1", "Ada Lovelace")]
    # The keys were fetched for the first token and kept.
    assert better_auth.fetches == 1


def test_a_signed_in_visitor_practises_in_production(client, settings, better_auth) -> None:
    settings.environment = "production"

    assert client.get("/practice/progress").status_code == 401
    assert client.get("/practice/progress", headers=bearer(better_auth.token())).status_code == 200


def other_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda auth: auth.token(lasts=-60), id="expired"),
        pytest.param(lambda auth: auth.token(iss="https://elsewhere.example"), id="other issuer"),
        pytest.param(lambda auth: auth.token(aud="https://elsewhere.example"), id="other audience"),
        pytest.param(lambda auth: auth.token(key=other_key()), id="signed with another key"),
        pytest.param(lambda auth: auth.token(key=other_key(), key_id="k9"), id="unknown key"),
        pytest.param(lambda auth: auth.token(key_id=None), id="no key id"),
        pytest.param(lambda auth: auth.token(sub=None), id="no subject"),
        pytest.param(lambda auth: auth.token(exp=None), id="no expiry"),
        pytest.param(
            lambda auth: jwt.encode(
                {"sub": "x", "iss": ADDRESS, "aud": ADDRESS, "exp": time.time() + 900},
                "s" * 32,
                algorithm="HS256",
                headers={"kid": "k1"},
            ),
            id="shared secret, not Ed25519",
        ),
        pytest.param(lambda auth: "not-a-token", id="not a token"),
    ],
)
def test_a_token_that_fails_a_check_is_refused_even_locally(
    client, sessions, better_auth, make
) -> None:
    response = client.get("/practice/progress", headers=bearer(make(better_auth)))

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert "sign in again" in response.json()["detail"]
    # Never the built-in user's practice instead, and nobody added
    assert users(sessions) == [(*LOCAL_USER, None)]


def test_a_token_is_refused_while_sign_in_is_not_set_up(client, sessions) -> None:
    token = BetterAuth().token()

    response = client.get("/practice/progress", headers=bearer(token))

    assert (response.status_code, response.json()["detail"]) == (401, "sign-in is not set up here")


def test_the_library_is_read_with_or_without_a_token(client, better_auth, settings) -> None:
    settings.environment = "production"

    assert client.get("/questions").status_code == 200
    assert client.get("/questions", headers=bearer(better_auth.token())).status_code == 200
    assert client.get("/questions", headers=bearer("not-a-token")).status_code == 401


def test_a_key_better_auth_added_is_fetched_but_made_up_ones_not_each_time(
    client, better_auth
) -> None:
    first = client.get("/practice/progress", headers=bearer(better_auth.token()))
    # Better Auth adds a key; the API has never seen it.
    added = Ed25519PrivateKey.generate()
    better_auth.jwks["keys"].append(public_jwk(added, "k2"))
    made_up = [better_auth.token(key=other_key(), key_id=f"x{n}") for n in range(3)]

    refused = [client.get("/practice/progress", headers=bearer(token)) for token in made_up]
    # A minute after the last look, the new key is fetched and accepted.
    app.dependency_overrides[get_keys]().fetched_at -= 61
    second = client.get(
        "/practice/progress", headers=bearer(better_auth.token(key=added, key_id="k2"))
    )

    assert first.status_code == 200 and second.status_code == 200
    assert [response.status_code for response in refused] == [401] * 3
    # For the first token, and for the new key a minute on. The made-up keys came while the
    # keys had just been fetched, and cost nothing.
    assert better_auth.fetches == 2


def test_keys_that_cannot_be_fetched_are_not_the_visitor_s_fault(client, better_auth) -> None:
    better_auth.down = True

    response = client.get("/practice/progress", headers=bearer(better_auth.token()))

    # 503, not 401: a 401 would tell the frontend the visitor had been signed out.
    assert response.status_code == 503


def test_the_keys_are_kept_while_better_auth_s_address_stays_the_same(settings) -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    assert get_keys(request, settings) is None
    settings.better_auth_url = f"{ADDRESS}/"
    first = get_keys(request, settings)
    again = get_keys(request, settings)
    settings.better_auth_url = "https://daedalus.example"
    moved = get_keys(request, settings)

    assert first is again and first is not None and first.url == JWKS
    assert moved is not None and moved.url == "https://daedalus.example/auth/jwks"


def test_questions_are_not_edited_in_production(client, sessions, corpus, settings) -> None:
    question_id = asyncio.run(add_question(sessions, corpus.scaling))
    settings.environment = "production"

    response = client.patch(f"/questions/{question_id}", json={"status": "retired"})

    assert response.status_code == 403
    assert client.get(f"/questions/{question_id}").json()["status"] == "accepted"
