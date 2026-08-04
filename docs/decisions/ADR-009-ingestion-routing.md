# ADR-009: Per-Page Routing Between Text Extraction, OCR, and Vision

**Status:** Accepted
**Date:** 2026-08-04

---

## Context

Daedalus ingests four kinds of study material: arXiv papers, course-note PDFs, Jupyter notebooks, and standalone images. Three different extraction mechanisms are available, with very different cost and fidelity:

| Mechanism | Speed | Fidelity | Deterministic |
|---|---|---|---|
| `pymupdf4llm` text extraction | Milliseconds | Exact | Yes |
| OCR (`ocrmypdf` / `paddleocr`) | Seconds per page | Lossy on math and layout | Yes |
| Vision model (`qwen2.5vl:7b`) | Tens of seconds per image | Good on diagrams, tables, layout | **No** |

Something must decide which runs for a given input. Measuring the current corpus made the shape of the problem concrete:

- All six PDFs carry clean text layers, between 1,770 and 4,006 characters per page. None benefits from OCR, and running it on them would be strictly worse — slower and lossier than reading the text directly.
- The three images are not scanned text. They are rendered infographics whose meaning lives in spatial layout, arrows, mathematical notation, and tables. OCR flattens all of that into an unordered bag of strings.

A single global mechanism cannot serve both cases well.

## Decision

Route **per page**, not per document, using a deterministic probe.

**PDF** — extract the text layer for each page and measure its length.

- `>= 100` characters on the page → `pymupdf4llm`
- `< 100` characters → OCR that page

The threshold is empirical. Pages with a real text layer in this corpus produce 1,770–4,006 characters; a scanned page produces essentially zero. Anything between 20 and 1,000 would separate them, and 100 sits comfortably in the gap.

Per-page rather than per-document routing handles the common real case of a mostly-digital PDF containing a few scanned or photographed pages.

**Standalone image** (`.png`, `.jpg`, `.jpeg`) — vision model. The prompt asks for structured Markdown: headings preserved, tables as Markdown tables, mathematical notation as LaTeX, and diagram topology described in prose ("the cache-miss branch flows into the embedding model, then the vector database").

**Jupyter notebook** (`.ipynb`) — `nbformat`. Markdown cells and code cell sources are indexed; **cell outputs are stripped**, retaining only short `text/plain` results and error messages. Outputs account for 512,105 characters in the current corpus — 44% of raw volume — and consist almost entirely of tensor dumps, progress bars, and warnings that match no query a student would ask.

**Markdown** (`.md`) — read directly.

This requires adding image extensions to `constants.SUPPORTED_EXTENSIONS`, which currently lists only `.pdf`, `.md`, and `.ipynb`.

## Alternatives Considered

**Always OCR everything.** Rejected as strictly worse for the 100% of PDF pages here that have text layers: slower, and it discards known-correct characters in favour of guessed ones. It would also mangle every equation in the arXiv papers.

**Always use the vision model.** Highest fidelity on diagrams, but tens of seconds per page across a corpus of hundreds of pages, and it introduces nondeterminism where none is needed. Reading an exact text layer through a probabilistic model is an unforced error.

**Ask the user per document.** Pushes an implementation detail onto someone uploading their lecture notes. It also does not scale to bulk uploads and cannot handle mixed-mode documents.

**Detect scanned pages by image-area heuristics** (page is one large raster). More brittle than measuring extracted text directly, and it fails on pages that are genuinely image-heavy but still carry a text layer.

## Consequences

**Positive**

- The fast, exact path handles the overwhelming majority of content. OCR and the vision model run only where they add something.
- Mixed-mode documents work without special handling.
- The routing decision is a single measurable number, so it is easy to test and easy to explain.
- Each chunk can record which mechanism produced it, enabling retrieval metrics sliced by extraction path — which is how the OCR and vision paths get held accountable rather than assumed to work.

**Negative**

- **Vision output is nondeterministic.** The same image parsed twice yields different text and therefore different character offsets. Since evaluation labels anchor to offsets in parsed text, the parsed output for the evaluation corpus **must be frozen and committed**. See `EVALUATION_ENGINE.md`.
- Three code paths to maintain and test instead of one.
- OCR and vision are slow enough that ingestion cannot be synchronous, reinforcing the background-task decision in ADR-008. A page requiring vision can take 30+ seconds.
- The 100-character threshold is a heuristic. A page containing only a figure caption and a page number could fall below it and be sent to OCR unnecessarily. The cost is wasted time, not wrong output, so it fails safe.
- Keeping OCR means carrying `paddleocr`, `paddlepaddle`, and `opencv` — over 1 GB — for a path no document in the current corpus exercises. This is accepted deliberately: scanned and photographed notes are a real use case for the target user, and the path is validated with rendered synthetic scans (see `EVALUATION_ENGINE.md`) until genuine scanned material is added.

## Revisit If

The corpus gains no genuinely scanned documents over time, in which case the OCR dependency should be dropped and the sub-threshold branch routed to the vision model instead.
