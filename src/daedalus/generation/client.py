"""Chat completion via the local Ollama server.

Ollama exposes an HTTP endpoint, so the standard library covers this and no HTTP
dependency is needed. This mirrors `daedalus.embedding`, which talks to the same
server on the same terms, and reuses its base-URL resolution rather than
introducing a second way to configure the same thing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from daedalus.embedding import ollama_url

#: Seconds to wait for one completion. Generation is bounded by memory
#: bandwidth on this hardware and a cold model load costs seconds, so the
#: timeout is generous; exceeding it means the server is wedged, not slow.
DEFAULT_TIMEOUT = 300.0


class GenerationError(RuntimeError):
    """Raised when the chat service cannot be reached or answers badly."""


def chat(
    messages: list[dict[str, str]],
    model: str,
    schema: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    keep_alive: str = "30m",
    think: bool = False,
    url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Return the assistant's message content for one non-streamed exchange.

    Raises GenerationError if the server is unreachable, returns a non-JSON
    body, or returns a body without message content. A schema, when given, is
    passed as Ollama's structured-output format so the model is constrained
    rather than merely asked.
    """
    endpoint = f"{url or ollama_url()}/api/chat"
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "keep_alive": keep_alive,
    }
    if schema is not None:
        body["format"] = schema
    if options is not None:
        body["options"] = options

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        raise GenerationError(
            f"could not reach Ollama at {endpoint}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise GenerationError(f"Ollama returned a non-JSON body: {error}") from error

    message = payload.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise GenerationError(f"no message content in the response from {model}")
    return str(message["content"])
