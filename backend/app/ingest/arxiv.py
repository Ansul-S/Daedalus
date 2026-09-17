"""arXiv: ID parsing, metadata and license (OAI-PMH), and downloads of the HTML or PDF.

arXiv asks clients to send at most one request every 3 seconds; every request made through
`ArxivClient` waits its turn. Downloads are cached under the data directory.
"""

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Self
from xml.etree import ElementTree

import httpx

REQUEST_INTERVAL = 3.0
USER_AGENT = "Daedalus/0.1 (study tool; +https://github.com/Ansul-S/Daedalus)"
OAI_URL = "https://oaipmh.arxiv.org/oai"
HTML_URLS = (
    "https://arxiv.org/html/{id}{version}",
    "https://ar5iv.labs.arxiv.org/html/{id}",
)
PDF_URL = "https://arxiv.org/pdf/{id}{version}"

_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "raw": "http://arxiv.org/OAI/arXivRaw/"}
_ID = re.compile(r"(?P<id>\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[A-Z]{2})?/\d{7})(?P<version>v\d+)?")
_URL_PREFIX = re.compile(
    r"(?:https?://)?(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/|arxiv:", re.IGNORECASE
)
# A LaTeXML page with real content, not an error page or a redirect to the abstract
_MIN_PARAGRAPHS = 5


class ArxivError(Exception):
    pass


def parse_arxiv_id(value: str) -> tuple[str, str | None]:
    """Accept "1706.03762", "arXiv:1706.03762v7", "hep-th/9901001" or an arxiv.org
    abs/pdf/html URL. Returns the ID and the version, if one was given."""
    text = _URL_PREFIX.sub("", value.strip(), count=1).removesuffix(".pdf").rstrip("/")
    match = _ID.fullmatch(text)
    if match is None:
        raise ValueError(f"not an arXiv ID: {value!r}")
    return match["id"], match["version"]


@dataclass
class ArxivMetadata:
    arxiv_id: str
    latest_version: str
    title: str
    authors: str
    abstract: str
    categories: list[str]
    license: str | None

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}{self.latest_version}"


def parse_oai_record(xml: str, arxiv_id: str) -> ArxivMetadata:
    root = ElementTree.fromstring(xml)
    error = root.find("oai:error", _NS)
    if error is not None:
        raise ArxivError(f"arXiv has no record for {arxiv_id} ({error.get('code')})")
    record = root.find(".//raw:arXivRaw", _NS)
    if record is None:
        raise ArxivError(f"unexpected OAI-PMH response for {arxiv_id}")

    def field(name: str) -> str:
        return " ".join((record.findtext(f"raw:{name}", "", _NS)).split())

    versions = [element.get("version", "") for element in record.findall("raw:version", _NS)]
    versions = [version for version in versions if re.fullmatch(r"v\d+", version)]
    if not versions:
        raise ArxivError(f"no versions listed for {arxiv_id}")
    return ArxivMetadata(
        arxiv_id=arxiv_id,
        latest_version=max(versions, key=lambda version: int(version[1:])),
        title=field("title"),
        authors=field("authors"),
        abstract=field("abstract"),
        categories=field("categories").split(),
        license=field("license") or None,
    )


def looks_like_paper(html: str) -> bool:
    return "ltx_document" in html and html.count("ltx_para") >= _MIN_PARAGRAPHS


class ArxivClient:
    """Rate-limited, caching arXiv client. Use one instance per process."""

    def __init__(self, cache_dir: Path, http: httpx.AsyncClient | None = None) -> None:
        self.cache_dir = cache_dir
        self._http = http or httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True
        )
        self._lock = asyncio.Lock()
        self._next_request = 0.0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    async def _get(self, url: str, **params: str) -> httpx.Response:
        async with self._lock:
            delay = self._next_request - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                return await self._http.get(url, params=params or None)
            finally:
                self._next_request = time.monotonic() + REQUEST_INTERVAL

    def _paper_dir(self, arxiv_id: str) -> Path:
        path = self.cache_dir / arxiv_id.replace("/", "_")
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def metadata(self, arxiv_id: str) -> ArxivMetadata:
        """Always fetched fresh, since a paper can gain versions."""
        response = await self._get(
            OAI_URL,
            verb="GetRecord",
            identifier=f"oai:arXiv.org:{arxiv_id}",
            metadataPrefix="arXivRaw",
        )
        response.raise_for_status()
        metadata = parse_oai_record(response.text, arxiv_id)
        cache = self._paper_dir(arxiv_id) / "metadata.json"
        cache.write_text(json.dumps(asdict(metadata), indent=2))
        return metadata

    async def html(self, arxiv_id: str, version: str) -> tuple[str, str] | None:
        """The paper as LaTeXML HTML from arxiv.org, else from ar5iv; None if neither has it.
        Returns (url, html)."""
        paper_dir = self._paper_dir(arxiv_id)
        # Remembers that this version has no HTML; delete the file to check again.
        missing = paper_dir / f"{version}.no-html"
        if missing.exists():
            return None
        for index, template in enumerate(HTML_URLS):
            url = template.format(id=arxiv_id, version=version)
            cache = paper_dir / f"{version if index == 0 else 'ar5iv'}.html"
            if cache.exists():
                return url, cache.read_text(encoding="utf-8")
            response = await self._get(url)
            if response.status_code == 200 and looks_like_paper(response.text):
                cache.write_text(response.text, encoding="utf-8")
                return url, response.text
        missing.touch()
        return None

    async def pdf(self, arxiv_id: str, version: str) -> tuple[str, Path]:
        url = PDF_URL.format(id=arxiv_id, version=version)
        cache = self._paper_dir(arxiv_id) / f"{version}.pdf"
        if not cache.exists():
            response = await self._get(url)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ArxivError(f"{url} did not return a PDF")
            cache.write_bytes(response.content)
        return url, cache
