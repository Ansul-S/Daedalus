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

**Under construction.** The ingestion, storage, and retrieval spine works and is
tested end to end. Question generation and answer evaluation are not built yet.

| Area | State |
|---|---|
| Jupyter notebook parsing | working, tested |
| Canonical document representation | working, tested |
| PostgreSQL storage with pgvector | working, tested |
| Local embedding via Ollama | working, tested |
| Vector, lexical, and pooled retrieval | working, tested |
| Command line interface | working, tested |
| Reference set and labelling tool | working, tested — annotation not yet done |
| Retrieval quality | **not yet measured** |
| Question generation | not started |
| Answer evaluation | not started |

**No retrieval quality figure has been measured.** Search returns plausible
results, but "plausible" is not a number. Building the labelled reference set
that would allow Recall@K to be reported honestly is the current phase.

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
      ↓  (next) measure Recall@K, then generate questions
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

---

## Prerequisites

| Component | Version used |
|---|---|
| Python | 3.11 |
| uv | any recent |
| PostgreSQL | 18.3 |
| pgvector | 0.8.6 |
| Ollama | 0.32.3 |
| bge-m3 | 1024 dimensions |

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

# 3. Schema
psql -p 5434 -d daedalus --single-transaction -v ON_ERROR_STOP=1 -f migrations/001_initial.sql
psql -p 5434 -d daedalus --single-transaction -v ON_ERROR_STOP=1 -f migrations/002_reference_set.sql

# 4. The embedding model
ollama pull bge-m3

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
```

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

**`queries`** and **`judgements`** — the reference set. Judgements carry no
foreign key into documents or chunks, deliberately: storing a document deletes
and reinserts its rows, so a cascade would destroy hours of human labelling to
save milliseconds of parsing. An `orphaned_judgements` view reports judgements
whose chunk no longer exists.

---

## Development

```bash
uv run pytest -q          # 135 tests
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
- **`Instructions.md`** — step-by-step procedure for building the reference set

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
