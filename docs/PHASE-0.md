# Phase 0 — Problem Definition

Settled scope for the first milestone. Decisions here are binding until changed
deliberately; measured figures state the hardware and date they came from.

## Target user

A single learner preparing for AI/ML interviews from study material they already
own — initially the author. The repository also serves as a capstone shown
during interviews, but where the two purposes conflict, usefulness to the
learner wins.

Priority order when trade-offs arise:

1. Correctness and usefulness to the learner
2. Measurable retrieval and generation quality
3. Engineering quality
4. Portfolio impressiveness

## Content types

The corpus in `corpus/` is an initial test dataset, not the boundary of the
product. The system targets arbitrary study material across arbitrary topics.

Measured contents, 2026-08-29 — 12 files, 7.3 MB:

| Group | Files | Notes |
|---|---|---|
| `arXiv/` | 3 PDFs | Attention, BERT, RAG — 50 pages total |
| `course notes/` | 3 PDFs | Detection and segmentation topics — 32 pages total |
| `notebooks/` | 3 `.ipynb` | 734 cells, ~102,000 tokens of source |
| `Images/` | 3 files | Standalone diagrams, portrait orientation |

All six PDFs carry extractable text layers (919–3195 text-drawing operators
each); none are scanned, so OCR is not required for the current corpus. The
notebooks are markdown-dominant — 510 of 734 cells — and contain no image
outputs and no cell attachments.

The corpus contains no `.md` and no standalone `.py` files. Support for those
formats remains in scope for the product; they are simply absent from the
initial dataset.

**Phase 1 parser priority: `.ipynb`.** The notebooks hold the largest share of
prose theory while also carrying code-cell context.

Later parsers, in no fixed order: PDF, Markdown, `.py`, images.

## Success criteria

Measured against a notebook the learner has already studied.

- At least 9 of 10 generated questions grounded in the source material, each
  traceable to retrieved chunks sufficient to justify it.
- At least 8 of 10 interview-relevant under a defined rubric rewarding
  conceptual understanding, reasoning, application, and technical depth over
  recall.
- At least 8 of 10 matching their assigned difficulty level under a defined
  rubric.
- The supporting chunk present in the top-K retrieved results, so retrieval can
  be reported as Recall@K.
- Meaningful topic coverage without repeated near-duplicate questions.

Failure condition: an unsupported-question rate above 10% across the evaluation
benchmark. That is the point at which the system is not trustworthy for
interview preparation.

### Constraints on these criteria

Four qualifications, recorded so the numbers are not overstated later.

**Recall@K cannot be measured on generated questions alone.** A question
generated from a retrieved chunk has that chunk as its answer by construction,
so retrieving it back measures only that the index round-trips. Reportable
Recall@K requires queries whose relevant chunks were labelled independently of
generation.

**Ten questions cannot support a 90% threshold.** At N=10, an observed 9/10 has
a 95% Wilson interval of 59.6%–98.2%. A single session is a smoke test. Reported
figures need N=150 (interval width 10 points) or N=300 (7 points).

**The rubrics do not exist and the judge is not ground truth.** Groundedness,
interview-relevance, and difficulty each need a written rubric, and an automated
judge scoring generated output is an estimator with its own biases. Judge
agreement must be measured against human labels rather than assumed.

**Difficulty agreement is not difficulty validity.** A generator and a judge
concurring that a question is hard measures shared prior, not difficulty.
Validity requires learner performance data.

## Constraints

**Time.** Two weeks from 2026-08-29, extendable but not open-ended. No external
deadline.

**Budget.** Zero. No paid API usage. All inference runs locally.

**Hardware.** MacBook Air, Apple M4, 10 cores (4P/6E), 16 GB unified memory,
186 GB free. Fanless — sustained inference throttles measurably.

**Privacy.** The current corpus is course material and public papers; no
personal data. Local-only inference is a consequence of the zero budget rather
than a privacy requirement, but it does mean uploaded material never leaves the
machine.

**Corpus in git.** `corpus/` stays untracked. It is third-party material and
7.3 MB of binaries.

## Model configuration

Binding defaults for local inference. Figures measured 2026-08-29 on the machine
above, generating ten labelled questions from a real notebook chunk.

| Role | Model | Setting |
|---|---|---|
| Generation | `qwen3:8b` | `think: false`, `keep_alive: "30m"`, capped `num_predict`, JSON-schema output |
| Judging | `qwen3:8b` | as above; thinking on/off to be compared against human labels |
| Smoke testing | `llama3.2:3b` | pipeline debugging only, where output quality is irrelevant |
| Embeddings | `BAAI/bge-m3` | dense; multi-vector available |
| Embedding baseline | `all-MiniLM-L6-v2` | ablation comparison |

Rationale, measured:

- **Thinking off.** Identical task took 42.0s with thinking enabled versus 14.6s
  disabled — 2.9× — because the model emitted 1,958 characters of internal
  monologue before answering.
- **`keep_alive`.** Cold-start `load_duration` measured at 3.2s for the 8B and
  10.7s for the 4B. Without a keep-alive, every call pays it.
- **Output caps and JSON.** Generation is bounded by memory bandwidth at roughly
  19 tok/s cold and 13.1 tok/s sustained, so output length dominates wall time.
- **8B over 4B.** `qwen3:4b` generates at 33.6 tok/s against the 8B's 19.1, but
  on an identical prompt emitted 1,029 tokens versus 274, finishing in 34.9s
  against 14.6s. A smaller model without output discipline is slower.
- **Thermal throttling.** Five consecutive runs measured 15.5, 13.9, 13.6, 12.2,
  13.1 tok/s — a 15.4% decline. Long benchmark runs should assume sustained
  rather than first-run rates.

Projected from those measured rates: a full benchmark cycle at N=300 with three
rubrics and three repeats costs about 15.2 hours at one item per call, falling
to about 3.8 hours with items batched five per call and repeats reduced to one.
Question generation itself is not a bottleneck — roughly 12 minutes for 300
questions.

## Scope for this milestone

In scope: notebook ingestion, chunking, local embedding and indexing, lexical
and semantic retrieval, a labelled query set, measured Recall@K, and grounded
question generation with references to source chunks.

Deferred, with reasons recorded in `docs/FEATURES.md`: MCQ, MSQ and NAT question
modes, adaptive difficulty, scoring, interview mode, PDF and image parsers,
page-image retrieval, API, and UI.

Deferring these is a consequence of the two-week constraint and the priority
order above. A measured retrieval spine is worth more than a broader system with
nothing measured.

## Largest open dependency

Every success criterion above depends on a human-labelled reference set that
does not exist: questions paired with their supporting chunks and with
groundedness, relevance, and difficulty labels assigned by hand. Estimated at
roughly ten hours of attention for 300 items, and not delegable without
destroying the independence that makes it useful.

It needs its own phase with hours attached, not a task inside another phase.
