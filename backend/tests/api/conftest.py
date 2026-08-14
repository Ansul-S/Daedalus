"""
Fixtures for the API tests.

The client here talks to the real routes, the real pipeline, and a real
SQLite file in a temporary directory. Only the embedder is substituted —
everything else is the code that runs in production, which is the point:
these tests are what prove the stages actually connect.

TestClient runs background tasks before returning a response, so an upload
is fully ingested by the time the call returns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from daedalus.api.dependencies import get_embedder
from daedalus.api.main import app
from daedalus.embeddings import FakeEmbedder

MARKDOWN = """# Attention

The dot product is scaled by the square root of d_k before the softmax,
because for large values of d_k the dot products grow large in magnitude.

## Normalization

Layer normalization stabilizes training by rescaling activations across
the feature dimension rather than across the batch.

## Convolution

Convolution kernels slide across an image to extract local features such
as edges and textures, which pool into higher level structure.
"""


@pytest.fixture
def api(data_dir: Path) -> Iterator[TestClient]:
    """A client serving with the model-free embedder."""

    # A lambda, not the class itself: FastAPI introspects a dependency's
    # signature, so passing FakeEmbedder would turn its __init__ parameters
    # into request fields and every request body would fail validation.
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def document(tmp_path: Path) -> Path:
    """A small markdown file to upload."""

    path = tmp_path / "attention.md"
    path.write_text(MARKDOWN, encoding="utf-8")

    return path


@pytest.fixture
def upload(api: TestClient) -> Callable[..., httpx.Response]:
    """POST a file to the upload endpoint."""

    def post(path: Path, filename: str | None = None) -> httpx.Response:
        with path.open("rb") as handle:
            return api.post(
                "/documents",
                files={"file": (filename or path.name, handle, "application/octet-stream")},
            )

    return post
