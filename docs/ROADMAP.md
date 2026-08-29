# Daedalus — Roadmap

Direction, not a work queue. **Do not implement phases automatically.** Each
phase starts when I say it starts.

**Current phase:** <!-- TODO: set this, e.g. "Phase 1 — PDF ingestion only" -->

Keep this line accurate. It's the fastest way for a new session to know where
things stand.

---

## Phase 0 — Problem definition

Target users, interview scenarios, supported content types, expected outputs,
constraints, success criteria.

## Phase 1 — Document ingestion

PDF, Markdown, Jupyter notebook, image/OCR. One format at a time, each with
tests, before moving to the next.

## Phase 2 — Canonical representation

A normalized representation all parsers produce, preserving the
heading → concept → explanation → code → output structure.

## Phase 3 — Storage

Document and chunk storage. Data model discussion before schema.

## Phase 4 — Retrieval

Lexical retrieval, then semantic retrieval, then hybrid *only if the measured
failure modes justify it*. Evaluate each before adding the next.

## Phase 5 — Question generation

Grounded interview questions with references back to source chunks.

## Phase 6 — Interview session

Select topic and difficulty, receive questions, submit answers, receive
feedback and follow-ups.

## Phase 7 — Evaluation harness

Proper harness for the metrics in `docs/PROJECT.md`.

## Phase 8 — API

Only if required.

## Phase 9 — UI

Simple interface.

## Phase 10 — Deployment

Only after the core system is stable.

---

## Notes / decisions log

Record decisions here as they're made, so a future session doesn't relitigate
them.

<!--
Format:
### YYYY-MM-DD — <decision>
Chose X over Y because Z. Alternatives considered: ... Limitation: ...
-->
