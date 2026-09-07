"""Tests for the Ollama chat client.

Exercised against a stub HTTP server on a loopback port rather than a live
model, so the suite stays deterministic and needs no Ollama running.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from daedalus.generation.client import DEFAULT_TIMEOUT, GenerationError, chat


class _Stub(BaseHTTPRequestHandler):
    """Records the last request body and replies with a canned response."""

    status = 200
    body = json.dumps({"message": {"content": '{"question": "ok"}'}})
    received: dict[str, Any] = {}

    def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        _Stub.received = json.loads(self.rfile.read(length))
        payload = _Stub.body.encode()
        self.send_response(_Stub.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr logging."""


@pytest.fixture
def server() -> Iterator[str]:
    _Stub.status = 200
    _Stub.body = json.dumps({"message": {"content": '{"question": "ok"}'}})
    _Stub.received = {}
    httpd = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_content_of_the_reply_is_returned(server: str) -> None:
    assert chat([{"role": "user", "content": "hi"}], "qwen3:8b", url=server) == (
        '{"question": "ok"}'
    )


def test_the_request_carries_the_protocol_settings(server: str) -> None:
    chat(
        [{"role": "user", "content": "hi"}],
        "qwen3:8b",
        schema={"type": "object"},
        options={"temperature": 0.3, "seed": 7},
        keep_alive="30m",
        think=False,
        url=server,
    )

    assert _Stub.received["model"] == "qwen3:8b"
    assert _Stub.received["stream"] is False
    assert _Stub.received["think"] is False
    assert _Stub.received["keep_alive"] == "30m"
    assert _Stub.received["format"] == {"type": "object"}
    assert _Stub.received["options"] == {"temperature": 0.3, "seed": 7}


def test_the_schema_and_options_are_omitted_when_not_given(server: str) -> None:
    chat([{"role": "user", "content": "hi"}], "qwen3:8b", url=server)

    assert "format" not in _Stub.received
    assert "options" not in _Stub.received


def test_an_unreachable_server_raises_generation_error() -> None:
    with pytest.raises(GenerationError, match="could not reach Ollama"):
        chat(
            [{"role": "user", "content": "hi"}],
            "qwen3:8b",
            url="http://127.0.0.1:1",
            timeout=1.0,
        )


def test_a_non_json_body_raises_generation_error(server: str) -> None:
    _Stub.body = "not json"

    with pytest.raises(GenerationError, match="non-JSON body"):
        chat([{"role": "user", "content": "hi"}], "qwen3:8b", url=server)


@pytest.mark.parametrize(
    "body",
    [
        json.dumps({}),
        json.dumps({"message": {}}),
        json.dumps({"message": {"content": 42}}),
        json.dumps({"message": "text"}),
    ],
)
def test_a_reply_without_message_content_raises(server: str, body: str) -> None:
    _Stub.body = body

    with pytest.raises(GenerationError, match="no message content"):
        chat([{"role": "user", "content": "hi"}], "qwen3:8b", url=server)


def test_the_default_timeout_allows_for_a_cold_model_load() -> None:
    assert DEFAULT_TIMEOUT >= 60.0
