# Daedalus

An interview-preparation system that generates AI/ML interview questions
grounded in your own study material, and evaluates your answers against that
same material.

Not a general chatbot. An examiner that has read your notes.

---

## The problem

People preparing for machine-learning interviews study from material they have
already collected — course PDFs, personal notes, Jupyter notebooks, screenshots
of diagrams. Generic question banks do not match that material. Generic chatbots
will happily ask about things you never studied and produce answers your sources
never supported.

Daedalus generates questions from your documents, keeps a reference from every
question back to the passages that justify it, and never presents outside
knowledge as though it came from your material.

---

## Status

**Milestone 1 complete.** One measured path runs end to end: a Jupyter notebook
goes in, and grounded interview questions come out, with retrieval quality,
question quality, and the reliability of the automated judge all measured rather
than asserted.

| Area | State |
|---|---|
| Jupyter notebook parsing | working, tested |
| Canonical document representation | working, tested |
| PostgreSQL storage with pgvector | working, tested |
| Local embedding via Ollama | working, tested |
| Vector, lexical, and pooled retrieval | working, tested |
| Hand-labelled reference set | complete — 50 queries, 1,422 judgements |
| Retrieval quality | **measured** — `bge-m3` NDCG@10 0.6127 |
| Question generation | working — 228 questions from 300 sections |
| Question quality | **measured** — 684 hand labels, four bars missed |
| LLM judge | built and **measured against the human labels** |
| Interview session, answer evaluation | not built — deferred, see below |

### The headline result is a negative one

Four quality thresholds were fixed in a protocol committed **before** any
question was generated. All four were missed:

| criterion | threshold | measured |
|---|---|---|
| grounded in the cited material | >= 90% | **57.5%** |
| unsupported questions | > 10% fails | **25.9%** |
| interview-relevant | >= 80% | **11.8%** |
| matches its assigned difficulty | >= 80% | **46.1%** |

Nothing was tuned in response. No threshold moved, no prompt was revised, no
question was regenerated. The protocol says a failure is a completed outcome,
and this is that outcome.

The relevance number is the substantive one, and it is not a scatter of bad
questions: **136 of 228 sit at grade 1** — on topic and sensible, but answerable
by restating a sentence. The generator asks about the right subject and tests
recall rather than understanding.

### The judge does not work here either, and that was measured before it was used

Two local models scored all 228 questions on all three rubrics. Neither can
substitute for the human labels:

| rubric | judge | raw agreement | weighted kappa |
|---|---|---|---|
| groundedness | `qwen3:8b` | 0.6009 | 0.1201 |
| groundedness | `llama3.2` | 0.5381 | 0.0559 |
| relevance | `qwen3:8b` | 0.1184 | **−0.0013** |
| relevance | `llama3.2` | 0.5825 | 0.1069 |
| difficulty | `qwen3:8b` | 0.6129 | **0.3190** |
| difficulty | `llama3.2` | 0.5741 | 0.2809 |

`qwen3:8b` answered "2" on 219 of 228 groundedness questions and 227 of 228
relevance questions — near-identical behaviour — and scored 60.1% raw agreement
on one and 11.8% on the other. The difference is entirely the human's label
distribution, not the judge. A rater that ignored the input and always answered
"2" would score 57.5% on groundedness; the judge beat that by six questions.

That is why no automated quality score appears anywhere in this project.

Full numbers, confidence intervals, confusion matrices and limitations are in
[`results/README.md`](results/README.md), and every judgement is frozen in
`results/labels_20260910T120516Z.json`.

---

---

## How it works

```
notebook (.ipynb)
      ↓  parse — faithful to the format, nothing grouped or merged
parsed notebook
      ↓  canonicalise — format-independent segments with heading context
document
      ↓  store — PostgreSQL
chunks
      ↓  embed — bge-m3 via Ollama, 1024 dimensions
embeddings
      ↓
   ┌──┴───────────────┬──────────────────┐
vector search    lexical search    random sample
   └──┬───────────────┴──────────────────┘
      ↓  pool — union, de-duplicated, order scrambled
candidates
      ↓  human judgement — 0 / 1 / 2
reference set
      ↓  measure — NDCG, recall, precision, reciprocal rank, bootstrap intervals
retrieval quality
      ↓
300 sections drawn by hash from a frozen selection
      ↓  generate — qwen3:8b, one question per section, no retries
      ↓  validate — seven deterministic checks; failures rejected, not repaired
228 questions + 72 recorded rejections
      ↓  label — three blinded human passes, one rubric each
684 human labels
      ↓  judge — two local models, then measure agreement against the human
judge agreement, and the decision not to trust it
```

### Parsing keeps the structure

Notebooks are read cell by cell. Each cell records the heading path it sits
under — `Section 2 > 2.1 The Core Question` — so the relationship between a
heading, its explanation, the code, and the code's output survives ingestion.

Cells are deliberately **not** grouped into larger units at parse time.
Grouping is a chunking decision, and chunking should be measurable against the
reference set rather than fixed by the parser.

### Chunks are segments

Measured across the corpus, the median chunk is 418 characters, roughly 104
tokens, and only 1.5% exceed 512 tokens. The material was already written in
well-sized sections, so one segment is one chunk.

Each chunk carries its heading path, its kind (`prose`, `code`, `output`), tags
identifying pedagogical scaffolding, and — for outputs — a link to the code that
produced it.

### Retrieval pools three sources

Vector search finds semantically similar passages. PostgreSQL full-text search
finds literal term matches, including function names that embeddings miss.
A random sample finds what both miss.

Measured on this corpus, **only 12% of pooled candidates were found by more than
one retriever**. They genuinely disagree, which is why a reference set built from
one of them alone would produce an inflated recall figure.

### The reference set is judged by hand

Retrieval cannot be called good or bad without ground truth, so every pooled
candidate was graded against a fixed policy — 0 not relevant, 1 partially
relevant, 2 fully relevant. No grade was produced by a model, a score, or a
heuristic. Pools hold 16 to 25 candidates per query.

| | queries | judgements | 0 | 1 | 2 |
|---|---|---|---|---|---|
| Harvested from the corpus | 25 | 533 | 63.2% | 17.4% | 19.3% |
| Written independently | 25 | 531 | 69.1% | 21.7% | 9.2% |
| **Total** | **50** | **1,064** | 66.2% | 19.5% | 14.3% |

The pool was later expanded to **1,422 judgements** so that all four retrieval
variants compared have 100% top-10 coverage and no reported number depends on
how unjudged candidates are treated.

The two halves exist to be compared. Harvested queries are taken verbatim from
the material's own concept-check cells, so they share its vocabulary. Authored
queries were written separately, asking the same material about itself in
different words.

The split already shows in the labels. Harvested queries average **4.1**
fully-relevant chunks each; authored queries average **2.0**. The material
answers its own concept checks in dedicated instructor-answer cells that each
serve five to eight questions at once, while an independently phrased question
is usually answered in one or two places. The authored half therefore carries a
sparser relevance signal, and should be the stricter test of ranking.

Three authored queries have no fully-relevant chunk at all. Each asks why the
material does one thing rather than another, in a case where it does the thing
without ever explaining why — leaving the implementation as the best available
evidence, which the policy grades a 1. They are kept: a query the corpus cannot
fully answer is real information about the corpus.

---

## Prerequisites

| Component | Version used |
|---|---|
| Python | 3.11 |
| uv | any recent |
| PostgreSQL | 18.3 |
| pgvector | 0.8.6 |
| Ollama | 0.33.2 |
| bge-m3 | 1024 dimensions |
| qwen3:8b | generation and judging |
| llama3.2 | second judge |

Everything runs locally. There are no paid services and no API keys.

---

## Setup

```bash
# 1. PostgreSQL and the vector extension
brew install postgresql@18 pgvector
brew services start postgresql@18

# 2. A database with the extension enabled
createdb -p 5434 daedalus
psql -p 5434 -d daedalus -c "CREATE EXTENSION vector;"

# 3. Schema — every migration, in order
for m in migrations/*.sql; do
  psql -p 5434 -d daedalus --single-transaction -v ON_ERROR_STOP=1 -f "$m"
done

# 4. Models — embedding, generation, and the second judge
ollama pull bge-m3
ollama pull qwen3:8b
ollama pull llama3.2

# 5. Python dependencies
uv sync

# 6. Connection string
export DAEDALUS_DATABASE_URL="postgresql:///daedalus?host=/tmp&port=5434"
```

The connection string is **required** and has no default. A default would let
the application connect silently to whichever PostgreSQL happened to answer,
which is a real failure mode on machines with more than one installed.

Note it connects over the Unix socket (`host=/tmp`) rather than TCP: no
password, and immune to another process holding the same port.

`docs/PGVECTOR.md` covers all of this in depth, including diagnosing a machine
with several PostgreSQL installations.

---

## Usage

```bash
# Ingest notebooks — files or directories
uv run daedalus ingest corpus/notebooks

# Embed anything not yet embedded
uv run daedalus embed

# What is stored
uv run daedalus status

# Reference set
uv run daedalus query add "How does the retriever narrow candidates?" --source authored
uv run daedalus query list
uv run daedalus label

# Generate the benchmark questions from the frozen selection
uv run python scripts/generate_questions.py

# Label generated questions — one rubric per pass, each in its own shuffled order
uv run daedalus label-questions groundedness
uv run daedalus label-questions relevance
uv run daedalus label-questions difficulty

# Judge sampled question pairs as duplicates
uv run daedalus label-pairs
```

The three rubric passes are deliberately separate runs in different orders.
Judging several rubrics in one sitting invites a halo, where a question judged
ungrounded is then judged irrelevant on the strength of that first impression
rather than on the scale. Requested difficulty is also recoverable from selection
rank with 100% accuracy, so presenting questions in rank order would hand the
labeller the answer the difficulty rubric exists to assign independently.

Ingestion skips documents whose content is unchanged, because document
identity is a hash of content — so re-running it does not discard embeddings
that cost minutes to produce. `--force` overrides.

Ingestion and embedding are separate commands because their costs differ by
three orders of magnitude: 925 chunks ingest in 0.36 s and embed in 115 s.

---

## Data model

Three tables, plus two for the reference set.

**`documents`** — one row per source file. The primary key is a hash of the
content, so re-ingesting unchanged material is a no-op and editing a file
produces a new document rather than overwriting the old one.

**`chunks`** — one row per retrievable unit, with `heading_path` and `tags` as
PostgreSQL arrays and a self-referencing foreign key linking each output to the
code that produced it.

**`embeddings`** — a separate table rather than a column, keyed by
`(chunk_id, model)`. This is the one decision worth explaining: a column would
allow exactly one embedding per chunk, and two planned experiments need several
at once — comparing embedding models of different dimensions, and comparing
precisions. The `vector` column is deliberately dimensionless for the same
reason, with a `CHECK (vector_dims(embedding) = dim)` supplying the integrity a
typed column would.

**`questions`**, **`question_sources`**, **`question_rejections`** — generated
questions, the chunks each cites, and the responses that failed validation. A
rejection stores its raw body, so a failure can be diagnosed later without asking
the model again.

**`question_labels`**, **`judge_scores`** — human labels and automated ones, kept
in separate tables on purpose. The human scales are enforced by database CHECK
constraints; the judge's deliberately are not, because an answer off its own
scale is a finding about the judge and rejecting the write would discard the
evidence.

**`question_embeddings`**, **`duplicate_labels`** — question vectors under a
distinct model identity, and the hand-judged pairs behind the duplicate check.
The vectors are kept out of `embeddings` so the production chunk embeddings that
every retrieval measurement reads are never touched.

**`queries`** and **`judgements`** — the reference set. Judgements carry no
foreign key into documents or chunks, deliberately: storing a document deletes
and reinserts its rows, so a cascade would destroy hours of human labelling to
save milliseconds of parsing. An `orphaned_judgements` view reports judgements
whose chunk no longer exists.

---

## Development

```bash
uv run pytest -q          # 485 tests
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

Tests that need PostgreSQL create and drop their own throwaway database, and
**skip with a message** when no server is reachable — so a checkout without
PostgreSQL still gets a green run on everything else.

Database code is tested against a real database rather than mocks. Mocking the
client would assert that a function was called with a string; it would verify
nothing about whether the SQL is correct, and the SQL is the interesting part.
No test contacts a live language model: the embedder is injected, so the storage
layer is driven with a stub.

---

## Layout

```
src/daedalus/
  document.py            canonical types shared by every layer
  embedding.py           Ollama client
  cli.py                 command line interface
  ingestion/
    notebook.py          .ipynb parsing, faithful to the format
    canonical.py         notebook -> canonical document
  storage/
    database.py          connection handling
    documents.py         documents and chunks
    embeddings.py        embeddings and backfill
    queries.py           reference set
  retrieval/
    search.py            vector, lexical, random, and pooling
  evaluation/
    metrics.py           NDCG, recall, precision, RR — validated against trec_eval
    harness.py           scoring and bootstrap intervals
    agreement.py         judge-versus-human agreement, weighted kappa
  generation/
    selection.py         the frozen 300-section sample
    prompt.py            the generation contract, versioned
    validation.py        seven deterministic checks
    runner.py            one attempt per section, no retries
  judging/
    prompt.py            the judging contract, versioned
    runner.py            one rubric per call
  labelling.py           the human labelling loops
  duplicates.py          similarity bands and the duplicate safety net
migrations/              numbered SQL, applied in order
tests/
docs/
corpus/                  study material (not tracked)
```

---

## Documentation

- **`docs/PROJECT.md`** — problem definition, scope, and the grounding model
- **`docs/ROADMAP.md`** — phases, current position, and the decisions log
- **`docs/PHASE-0.md`** — target user, content types, success criteria, constraints
- **`docs/FEATURES.md`** — the intended end-state product, most of it deferred
- **`docs/PGVECTOR.md`** — a complete pgvector methodology, reproducible from scratch
- **`docs/LABELLING.md`** — the fixed relevance grading policy
- **`docs/HYBRID-PROTOCOL.md`** — the pre-registered hybrid retrieval experiment
- **`docs/PHASE-6-PROTOCOL.md`** — the generation protocol, committed before generating
- **`docs/PHASE-6-RUBRICS.md`** — the three labelling rubrics, committed before labelling
- **`results/README.md`** — every measured result, with its limitations
- **`Instructions.md`** — step-by-step procedure for building the reference set

---

## What was not built, and why

Milestone 1 was cut to a measured retrieval-and-generation spine under a
two-week constraint. The cuts were made at the start and recorded in
`docs/FEATURES.md`, not discovered at the end:

- the interview session — topic selection, answering, feedback, follow-ups
- answer evaluation against the source material
- adaptive difficulty and scoring
- MCQ, MSQ and numeric answer modes
- parsers for PDF, Markdown, `.py` and images
- page-image retrieval for figures and diagrams
- any API, UI or deployment

The choice was a complete measured path over a broader unmeasured system. What
exists is small, and every claim about it has a number behind it.

### What the results actually license

The generator does not meet the bar in `docs/PHASE-0.md`, and this system is not
currently trustworthy for interview preparation — the failure condition in that
document, an unsupported-question rate above 10%, is met at 25.9%.

Three defects are identified and measured rather than merely suspected:

- **25.0% of questions leak prompt scaffolding.** 57 of 228 refer to `chunk 170`
  or "the SEED chunk" — things a candidate cannot see. The class is strongly
  associated with unusable relevance, and it does *not* explain the relevance
  failure: had those 57 behaved like the rest, grade 2 would land near 12.9%
  rather than 11.8%.
- **Half of the groundedness failures are citation failures, not missing
  material** — 49 of 97, where the supporting text exists in the section but was
  not cited. That is a retrieval-and-citation problem with a different fix from a
  generation problem.
- **15.0% of attempted sections produced a quotation that appears nowhere in the
  corpus.** This is the rate at which one field of one contract was fabricated,
  on one corpus, at one model size. It is not a general hallucination rate.

Fixing any of these means a new contract committed in advance and a fresh run
reported beside the current one — not a revision of `p6-v2` after seeing its
numbers.

---

## Design decisions worth knowing

**No approximate vector index.** pgvector offers HNSW and IVFFlat. Neither is
used, because at 925 chunks exact search takes milliseconds and an approximate
index would trade away recall for a speedup too small to notice. The index will
be added when measurement shows it is needed, not before.

**Embeddings run locally.** Using `sentence-transformers` would pull in PyTorch,
around 2.5 GB, on a 16 GB machine already running a language model. Ollama
serves the same model over HTTP and the standard library can make an HTTP
request, so the project has no machine-learning dependency at all.

**Candidates are presented in scrambled order.** The labelling interface orders
candidates by a hash of the query and chunk id, never by any retriever's
ranking, and never shows which retriever found a chunk. A judgement influenced
by the retriever would be measuring itself.

**Every reported number comes from code that ran.** Figures in this README —
chunk counts, timings, the 12% pooling overlap — were measured on this corpus,
on the machine described above. Nothing here is an estimate presented as a
result.
