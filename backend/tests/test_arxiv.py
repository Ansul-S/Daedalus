import asyncio
import time

import httpx
import pytest

from app.ingest import arxiv
from app.ingest.arxiv import ArxivClient, ArxivError, parse_arxiv_id, parse_oai_record

RECORD = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <GetRecord><record><metadata>
    <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
      <id>1706.03762</id>
      <version version="v1"><date>Mon, 12 Jun 2017 17:57:34 GMT</date></version>
      <version version="v10"><date>Wed, 02 Aug 2023 00:41:18 GMT</date></version>
      <version version="v2"><date>Mon, 19 Jun 2017 16:49:45 GMT</date></version>
      <title>Attention Is All
        You Need</title>
      <authors>Ashish Vaswani, Noam Shazeer</authors>
      <categories>cs.CL cs.LG</categories>
      <license>http://creativecommons.org/licenses/by/4.0/</license>
      <abstract>  The dominant sequence transduction models...  </abstract>
    </arXivRaw>
  </metadata></record></GetRecord>
</OAI-PMH>"""

NOT_FOUND = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <error code="idDoesNotExist">Value of the identifier argument is unknown</error>
</OAI-PMH>"""

PAPER_HTML = "<article class='ltx_document'>" + "<div class='ltx_para'>x</div>" * 5 + "</article>"


@pytest.fixture(autouse=True)
def no_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arxiv, "REQUEST_INTERVAL", 0.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1706.03762", ("1706.03762", None)),
        (" arXiv:1706.03762v7 ", ("1706.03762", "v7")),
        ("https://arxiv.org/abs/1706.03762v7", ("1706.03762", "v7")),
        ("https://arxiv.org/pdf/1706.03762.pdf", ("1706.03762", None)),
        ("arxiv.org/html/2510.10824v1/", ("2510.10824", "v1")),
        ("http://export.arxiv.org/abs/hep-th/9901001v2", ("hep-th/9901001", "v2")),
        ("math.GT/0309136", ("math.GT/0309136", None)),
    ],
)
def test_ids_and_urls_are_accepted(value: str, expected: tuple[str, str | None]) -> None:
    assert parse_arxiv_id(value) == expected


@pytest.mark.parametrize("value", ["attention", "1706.037", "../etc/passwd", "1706.03762?x=1"])
def test_anything_else_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="not an arXiv ID"):
        parse_arxiv_id(value)


def test_metadata_is_read_from_the_oai_record() -> None:
    metadata = parse_oai_record(RECORD, "1706.03762")

    assert metadata.title == "Attention Is All You Need"
    assert metadata.latest_version == "v10"
    assert metadata.authors == "Ashish Vaswani, Noam Shazeer"
    assert metadata.categories == ["cs.CL", "cs.LG"]
    assert metadata.license == "http://creativecommons.org/licenses/by/4.0/"
    assert metadata.abstract == "The dominant sequence transduction models..."
    assert metadata.abs_url == "https://arxiv.org/abs/1706.03762v10"


def test_an_unknown_paper_is_reported() -> None:
    with pytest.raises(ArxivError, match="no record"):
        parse_oai_record(NOT_FOUND, "0000.00000")


def client_for(tmp_path, responses: dict[str, httpx.Response], seen: list[str]) -> ArxivClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return responses.get(str(request.url), httpx.Response(404))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ArxivClient(tmp_path, http=http)


def test_html_comes_from_arxiv_and_is_cached(tmp_path) -> None:
    seen: list[str] = []
    url = "https://arxiv.org/html/1706.03762v7"
    responses = {url: httpx.Response(200, text=PAPER_HTML)}

    async def run() -> list[tuple[str, str] | None]:
        async with client_for(tmp_path, responses, seen) as client:
            return [await client.html("1706.03762", "v7") for _ in range(2)]

    assert asyncio.run(run()) == [(url, PAPER_HTML), (url, PAPER_HTML)]
    assert seen == [url]


def test_a_paper_without_html_falls_through_both_sources_once(tmp_path) -> None:
    seen: list[str] = []
    # ar5iv answers papers it does not have with the arXiv abstract page.
    responses = {
        "https://ar5iv.labs.arxiv.org/html/2510.10824": httpx.Response(200, text="<html>abs</html>")
    }

    async def run() -> list[tuple[str, str] | None]:
        async with client_for(tmp_path, responses, seen) as client:
            return [await client.html("2510.10824", "v1") for _ in range(2)]

    assert asyncio.run(run()) == [None, None]
    assert seen == [
        "https://arxiv.org/html/2510.10824v1",
        "https://ar5iv.labs.arxiv.org/html/2510.10824",
    ]


def test_pdf_downloads_are_checked(tmp_path) -> None:
    good = "https://arxiv.org/pdf/2510.10824v1"
    bad = "https://arxiv.org/pdf/2510.10824v2"
    responses = {
        good: httpx.Response(200, content=b"%PDF-1.5 ..."),
        bad: httpx.Response(200, content=b"<html>captcha</html>"),
    }

    async def run() -> None:
        async with client_for(tmp_path, responses, []) as client:
            url, path = await client.pdf("2510.10824", "v1")
            assert url == good and path.read_bytes().startswith(b"%PDF")
            with pytest.raises(ArxivError):
                await client.pdf("2510.10824", "v2")

    asyncio.run(run())


def test_requests_are_spaced_out(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arxiv, "REQUEST_INTERVAL", 0.3)

    async def run() -> float:
        async with client_for(tmp_path, {}, []) as client:
            started = time.monotonic()
            await client.html("1706.03762", "v7")  # two requests: arxiv.org, then ar5iv
            return time.monotonic() - started

    assert asyncio.run(run()) >= 0.25
