import asyncio

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import FunctionModel

from scripts.check_setup import ping_chat_model


def failing_model(status_code: int) -> FunctionModel:
    def respond(messages, info):
        raise ModelHTTPError(status_code=status_code, model_name="test", body={"error": "x"})

    return FunctionModel(respond)


def test_busy_provider_is_a_warning() -> None:
    for status_code in (429, 503):
        check = asyncio.run(ping_chat_model(failing_model(status_code)))
        assert check.status == "warn", status_code


def test_rejected_key_is_a_failure() -> None:
    check = asyncio.run(ping_chat_model(failing_model(401)))
    assert check.status == "fail"
    assert check.detail.startswith("HTTP 401")
