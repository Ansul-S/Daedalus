# pgvector: A Complete Methodology

How semantic search over documents was built on PostgreSQL and pgvector — the
concepts, the setup, the schema, the code, the failures, and the verification
steps at each stage.

This is written to be reproducible. Someone with the prerequisites below should
be able to work through it from top to bottom and end with a working pipeline,
without needing anything else. It is also written to be reusable: the reasoning
behind each decision is given, so the method can be adapted rather than copied
blindly.

---

## Table of contents

1. [The problem, and what pgvector solves](#1-the-problem-and-what-pgvector-solves)
2. [Why PostgreSQL rather than a dedicated vector database](#2-why-postgresql-rather-than-a-dedicated-vector-database)
3. [Prerequisites](#3-prerequisites)
4. [Stage 1 — Survey the machine before installing anything](#4-stage-1--survey-the-machine-before-installing-anything)
5. [Stage 2 — Install pgvector](#5-stage-2--install-pgvector)
6. [Stage 3 — Configure the server](#6-stage-3--configure-the-server)
7. [Stage 4 — Create the database and enable the extension](#7-stage-4--create-the-database-and-enable-the-extension)
8. [Stage 5 — Verify pgvector works](#8-stage-5--verify-pgvector-works)
9. [Stage 6 — Design the schema](#9-stage-6--design-the-schema)
10. [Stage 7 — Apply the migration](#10-stage-7--apply-the-migration)
11. [Stage 8 — Embeddings](#11-stage-8--embeddings)
12. [Stage 9 — The Python layer](#12-stage-9--the-python-layer)
13. [Stage 10 — Similarity search](#13-stage-10--similarity-search)
14. [Stage 11 — The pipeline end to end](#14-stage-11--the-pipeline-end-to-end)
15. [Indexing, and why we deliberately have none](#15-indexing-and-why-we-deliberately-have-none)
16. [Quantization](#16-quantization)
17. [Testing strategy](#17-testing-strategy)
18. [Troubleshooting](#18-troubleshooting)
19. [Reuse checklist](#19-reuse-checklist)

---

## 1. The problem, and what pgvector solves

### Why ordinary search is not enough

A conventional database index answers questions about *equality* and *ordering*.
`WHERE title = 'BERT'` is fast because a B-tree can find that exact string.
Full-text search goes further, matching word stems, so a search for "running"
finds "run" — but it is still fundamentally about which **words** appear.

Ask "how does BERT answer questions from a document?" and word matching fails in
a specific way: the passage that answers it may never use the word "answer". It
might say "the model predicts a start and end span over the passage". A human
recognises those as the same idea. A word index does not.

### What an embedding is

An **embedding** is a list of numbers — a *vector* — produced by a model that
has been trained so that texts with similar meanings get vectors that are close
together, and texts with unrelated meanings get vectors that are far apart.

The model used here, `bge-m3`, produces **1024 numbers** for any input text.
That list of 1024 numbers is a point in 1024-dimensional space. The important
property is entirely relational: the individual numbers mean nothing on their
own, but the *distance* between two vectors approximates how related their texts
are.

So semantic search becomes a geometry problem:

1. Embed every chunk of your documents once, and store the vectors.
2. When a query arrives, embed the query text the same way.
3. Return the stored chunks whose vectors are nearest to the query's vector.

### What pgvector adds to PostgreSQL

PostgreSQL has no native concept of a vector. It could store 1024 numbers in a
`double precision[]` array, but it would have no idea how to measure distance
between two of them, and no way to sort by that distance.

pgvector is an **extension** — a compiled library plus SQL definitions that
PostgreSQL loads on request — which adds:

- **New column types.** `vector` for 32-bit floats, `halfvec` for 16-bit, and
  support for `bit` used as a binary vector.
- **Distance operators.** `<->` (L2 / Euclidean), `<=>` (cosine),
  `<#>` (negative inner product), plus `<+>` (L1) in recent versions.
- **Functions.** `vector_dims()`, `l2_normalize()`, and others.
- **Approximate-nearest-neighbour index types**, HNSW and IVFFlat, for when
  exact search becomes too slow.

Crucially, these are ordinary PostgreSQL types and operators. A vector column
sits in a normal table beside normal columns, participates in normal joins, and
is filtered by a normal `WHERE` clause. That property is the entire argument for
this approach.

---

## 2. Why PostgreSQL rather than a dedicated vector database

Be precise about this, because the wrong justification collapses under one
follow-up question.

### The wrong reason: performance

At small corpus sizes, PostgreSQL with pgvector is not chosen for speed. This
project's corpus is a few hundred to a few thousand chunks. Comparing a query
vector against 1000 stored vectors is roughly a million floating-point
multiplications — a trivial amount of work, measured in **milliseconds**. A
specialised vector database would not be meaningfully faster, and any
approximate index would only trade away accuracy for a speedup you cannot
perceive.

If someone claims they chose a vector store "for performance" at this scale,
they have not measured it.

### The real reasons

**One store, no synchronisation.** Documents, chunks, metadata, and vectors live
in the same database. With a separate vector store you hold identifiers in one
system and content in another, and you own the problem of keeping them
consistent — deletions especially. Here, deleting a document removes its chunks
and their embeddings through a foreign key, in one statement.

**Metadata filtering is just SQL.** "Search only chunks that are not tagged as
answer keys" is a `WHERE` clause. In many vector stores this requires either a
pre-filter the engine may not support efficiently, or fetching more results than
you need and filtering afterwards.

**Lexical and semantic search in one query.** PostgreSQL has built-in full-text
search (`tsvector`). Hybrid retrieval — combining keyword matching with vector
similarity — becomes one query against one system rather than two systems and a
merge step.

**Transactions.** Writing a document, its chunks, and their embeddings either
all succeeds or all rolls back.

**Operational familiarity.** PostgreSQL is a skill that transfers. Backup,
restore, monitoring, and access control are solved problems with abundant
documentation.

### The honest costs

- A server to run and configure, versus an embedded file-based store.
- ANN indexes require a fixed vector dimension, which constrains schema design
  (see [§15](#15-indexing-and-why-we-deliberately-have-none)).
- At genuinely large scale — tens of millions of vectors — purpose-built engines
  have real advantages in memory layout and index construction time.

---

## 3. Prerequisites

| Component | Version used here | Purpose |
|---|---|---|
| macOS with Homebrew | — | package management |
| PostgreSQL | 18.3 | the database server |
| pgvector | 0.8.6 | vector types and operators |
| Python | 3.11 | application language |
| uv | — | Python dependency management |
| psycopg | 3.3.4 | PostgreSQL client for Python |
| Ollama | 0.32.3 | local embedding model server |
| bge-m3 | 1024 dimensions | the embedding model |

Nothing here requires a paid service or a network connection after the initial
downloads. Everything runs locally.

On Linux the package names differ (`postgresql-16-pgvector` or similar) but
every subsequent step is identical. On Windows, use the Docker route described
in [§18](#18-troubleshooting).

---

## 4. Stage 1 — Survey the machine before installing anything

**This stage exists because skipping it cost an hour.** Development machines
frequently accumulate more than one PostgreSQL installation — one from Homebrew,
one from the EnterpriseDB graphical installer, one inside Docker — and they
compete for the same port. Symptoms appear much later and are baffling: tables
that seem to vanish, an extension that "isn't installed" despite installing it.

Find out what exists:

```bash
# Which client binaries are on PATH, and in what order?
which -a psql

# What did the package manager install?
brew list --versions | grep -iE "postgres|pgvector"

# Is anything already listening on the default port?
lsof -nP -iTCP:5432 -sTCP:LISTEN

# Is a server answering on the Unix socket?
pg_isready

# Are there containers holding a port?
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
```

On the machine this project was built on, that survey found **three** PostgreSQL
installations:

| Installation | State | pgvector |
|---|---|---|
| EnterpriseDB `/Library/PostgreSQL/18` | running, holding port 5432 | no |
| Homebrew `postgresql@18` | installed, failing to start | yes |
| Homebrew `postgresql@16` | installed, stopped | no |

plus a Docker container `postgres:18` mapped to port 5433.

The Homebrew server was failing to start *because* the EnterpriseDB server
already held 5432, and pgvector was only visible to the Homebrew server. The
server that was running could not use pgvector; the server that could use
pgvector was not running.

**Two facts worth internalising:**

- An extension installed by Homebrew lands in Homebrew's directory tree
  (`/opt/homebrew/share/postgresql@18/extension/`). Another PostgreSQL
  installation reads a *different* directory and will not see it.
- `pg_isready` with no arguments checks the **Unix socket**, while
  `lsof -iTCP` checks **TCP**. A server can be listening on one and not the
  other, so the two commands can disagree. Run both.

Check where an extension's files actually landed:

```bash
ls /opt/homebrew/share/postgresql@18/extension/vector*
```

If that lists files, the Homebrew server can use pgvector. If you are running a
different PostgreSQL, it cannot.

---

## 5. Stage 2 — Install pgvector

```bash
brew install pgvector
```

This compiles or downloads a prebuilt binary of the extension and places three
kinds of file where PostgreSQL looks for them:

- `vector.so` — the compiled C library implementing the types and operators.
- `vector.control` — metadata telling PostgreSQL the extension exists and what
  version it is.
- `vector--*.sql` — scripts that create the SQL-level objects, plus upgrade
  scripts between versions.

Installing the package does **not** make vectors usable yet. It makes the
extension *available* to be enabled. Enabling happens per database, in
[§7](#7-stage-4--create-the-database-and-enable-the-extension).

**Verify:**

```bash
ls /opt/homebrew/share/postgresql@18/extension/vector.control
```

---

## 6. Stage 3 — Configure the server

Three things need to be right: which port, who may connect, and as which role.

### 6.1 Vocabulary

These four words are constantly confused, and the confusion causes most setup
failures:

- **Server** (also **cluster**) — one running `postgres` process tree serving
  one data directory. A machine can host several, on different ports.
- **Database** — a named container *inside* a cluster. One cluster holds many.
  `postgres` is the default administrative one.
- **Role** — a user account. Roles are cluster-wide, not per database.
- **Extension** — added functionality, enabled **per database**. Enabling it in
  one database does nothing for another. This is the single most common
  pgvector mistake.

### 6.2 Choosing a port

If the default port is taken, do not fight over it — move. Ports are free.

```bash
grep -n "^#*port" /opt/homebrew/var/postgresql@18/postgresql.conf
```

`postgresql.conf` lives in the data directory and is read at startup. A leading
`#` means the line is commented out and the built-in default (5432) applies.
Set an explicit value:

```bash
sed -i '' 's/^#*port = .*/port = 5434/' /opt/homebrew/var/postgresql@18/postgresql.conf
grep -n "^port" /opt/homebrew/var/postgresql@18/postgresql.conf
```

(On Linux, `sed -i` without the `''`.)

Then restart and confirm **exactly one** process holds the port:

```bash
brew services restart postgresql@18
sleep 3
lsof -nP -iTCP:5434 -sTCP:LISTEN
```

If a Docker container also appears in that output, the port is contested and
TCP connections will be non-deterministic. Choose another port.

**Verify the server started and why it did not, if it did not:**

```bash
tail -20 /opt/homebrew/var/log/postgresql@18.log
```

### 6.3 Authentication

`pg_hba.conf` — "host-based authentication" — controls who may connect, from
where, to which database, and how. It is read top to bottom, and the **first
matching line wins**.

```bash
grep -v '^#' /opt/homebrew/var/postgresql@18/pg_hba.conf | grep -v '^$'
```

A typical starting state:

```
local   all   all                        scram-sha-256
host    all   all   127.0.0.1/32         scram-sha-256
host    all   all   ::1/128              scram-sha-256
```

Reading the columns: connection **type**, **database**, **user**, **address**,
**method**.

- `local` means the Unix domain socket — a file on disk, reachable only from
  this machine.
- `host` means TCP.
- `scram-sha-256` demands a password. `trust` accepts any connection claiming a
  valid role, with no password.

For a local development database, a good configuration is **`trust` on the local
socket, password required over TCP**:

```bash
cp /opt/homebrew/var/postgresql@18/pg_hba.conf \
   /opt/homebrew/var/postgresql@18/pg_hba.conf.backup

# edit the 'local all all' line to read: local   all   all   trust
```

Then tell the running server to re-read it. This does **not** require a restart:

```bash
pg_ctl -D /opt/homebrew/var/postgresql@18 reload
```

`reload` sends SIGHUP, which re-reads configuration files. `restart` stops and
starts the server and drops all connections. Authentication changes need only a
reload; the `port` setting is marked "change requires restart" and needs the
heavier operation.

> **Security note.** `trust` on the local socket means any user account on this
> machine can connect as any role, including superuser. That is acceptable for a
> single-user development laptop and is Homebrew's own default. It is not
> acceptable on a shared or production machine.

### 6.4 Roles

Homebrew's normal installation creates a superuser role named after your
operating-system account, which is why `psql` usually needs no `-U` flag. A
cluster initialised differently — for instance with
`initdb --username=postgres` — will not have that role, producing:

```
FATAL:  role "yourname" does not exist
```

Find out which roles do exist by connecting as one that does:

```bash
psql -p 5434 -U postgres -d postgres -c "SELECT rolname, rolsuper FROM pg_roles WHERE rolcanlogin;"
```

Then create your own:

```bash
psql -p 5434 -U postgres -d postgres -c \
  "CREATE ROLE yourname LOGIN SUPERUSER CREATEDB;"
```

- `LOGIN` makes it an account that can connect, rather than a group.
- `SUPERUSER` bypasses all permission checks — appropriate for a local
  development role, not for an application role in production.
- `CREATEDB` allows creating databases, which the test suite needs.

**Verify:**

```bash
psql -p 5434 -d postgres -c "SELECT current_user;"
```

No `-U`, no password prompt.

---

## 7. Stage 4 — Create the database and enable the extension

```bash
createdb -p 5434 daedalus
```

`createdb` is a thin command-line wrapper around `CREATE DATABASE`.

Now enable pgvector **inside that database**:

```bash
psql -p 5434 -d daedalus -c "CREATE EXTENSION vector;"
```

This runs the extension's SQL script, creating the `vector` type, the distance
operators, the index access methods, and the supporting functions — all inside
`daedalus` only.

**Verify:**

```bash
psql -p 5434 -d daedalus -c "\dx"
```

Expect `plpgsql` and `vector`, with a version number. If `vector` is missing but
the command reported success, you almost certainly enabled it in a different
database — re-check the `-d` argument.

---

## 8. Stage 5 — Verify pgvector works

Before building anything, prove the extension functions. This takes ten seconds
and rules out a whole class of later confusion.

```bash
psql -p 5434 -d daedalus <<'SQL'
CREATE TABLE demo (id int, embedding vector(3));
INSERT INTO demo VALUES (1, '[1,0,0]'), (2, '[0,1,0]'), (3, '[0.9,0.1,0]');
SELECT id, embedding <-> '[1,0,0]' AS l2 FROM demo ORDER BY 2;
DROP TABLE demo;
SQL
```

Expected output:

```
 id |         l2
----+---------------------
  1 |                   0
  3 | 0.14142137441313676
  2 |  1.4142135623730951
```

Read that carefully, because it demonstrates the whole idea:

- Row 1 is the query vector itself, so its distance is exactly 0.
- Row 3 is nearly the same direction, so its distance is small.
- Row 2 is perpendicular, so its distance is large.

Three-dimensional vectors were used deliberately: the geometry is easy to check
by hand. The same operators behave identically at 1024 dimensions.

Note the literal syntax: a vector is written as a **string** in square brackets,
`'[1,0,0]'`, and PostgreSQL casts it to the `vector` type.

---

## 9. Stage 6 — Design the schema

Three tables: `documents`, `chunks`, `embeddings`.

The complete migration is reproduced here, then explained piece by piece.

```sql
-- Initial schema: documents, their chunks, and embeddings of those chunks.

CREATE TABLE documents (
    doc_id        text PRIMARY KEY,
    source_path   text NOT NULL,
    source_format text NOT NULL,
    title         text,
    ingested_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_id         text NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    ordinal        integer NOT NULL,
    kind           text NOT NULL CHECK (kind IN ('prose', 'code', 'output')),
    text           text NOT NULL,
    heading_path   text[] NOT NULL DEFAULT '{}',
    tags           text[] NOT NULL DEFAULT '{}',
    locator        text NOT NULL,
    parent_ordinal integer,

    UNIQUE (doc_id, ordinal),

    -- An output chunk points at the code chunk in the same document that
    -- produced it. Enforced, so an orphaned output cannot be inserted.
    FOREIGN KEY (doc_id, parent_ordinal)
        REFERENCES chunks (doc_id, ordinal) ON DELETE CASCADE,

    -- Only outputs may have a parent.
    CHECK (parent_ordinal IS NULL OR kind = 'output')
);

CREATE TABLE embeddings (
    chunk_id   bigint NOT NULL REFERENCES chunks (id) ON DELETE CASCADE,
    model      text NOT NULL,
    dim        integer NOT NULL,
    embedding  vector NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (chunk_id, model),

    -- The recorded dimension must match the stored vector.
    CHECK (vector_dims(embedding) = dim)
);
```

### 9.1 `documents`

One row per ingested source file.

**`doc_id text PRIMARY KEY`** — the identifier is a hash of the document's
content, not a random number and not the file path. Two consequences follow, and
both are deliberate. Re-ingesting an unchanged file produces the same id, so it
collides with the existing row instead of silently duplicating. Editing the file
produces a *different* id, so the new version does not overwrite the old one.

Deriving identity from content is a design choice with a trade-off: moving or
renaming a file does not create a new document (good), but you also cannot use
the id to find the file (hence `source_path` as a separate column).

**`ingested_at timestamptz NOT NULL DEFAULT now()`** — `timestamptz` stores an
absolute instant, converting to and from the session's timezone on the way in
and out. `timestamp` without the zone stores a wall-clock reading with no
indication of where the clock was, which is almost never what you want.
`DEFAULT now()` means the application never sets it.

### 9.2 `chunks`

One row per retrievable unit of a document.

**`id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY`** — the SQL-standard
auto-incrementing key, which replaces the older PostgreSQL-specific `serial`
type. `GENERATED ALWAYS` means an `INSERT` cannot supply its own value, so the
sequence is the only source of ids.

**`doc_id ... REFERENCES documents (doc_id) ON DELETE CASCADE`** — a foreign key.
It guarantees no chunk can reference a document that does not exist, and
`ON DELETE CASCADE` means deleting a document automatically deletes its chunks.
That turns "re-ingest this file" into a single `DELETE` followed by inserts,
with no orphan cleanup.

**`kind text NOT NULL CHECK (kind IN ('prose', 'code', 'output'))`** — a
constrained set of values. PostgreSQL also offers `CREATE TYPE ... AS ENUM`.
The trade-off: an enum is a genuine type with slightly tighter integrity, but
adding a value to it is a schema-level operation with more ceremony, whereas a
`CHECK` constraint is dropped and recreated in one statement. At three values,
the flexibility is worth more than the type strictness.

**`text[]` for `heading_path` and `tags`** — real PostgreSQL arrays, not
comma-joined strings. `heading_path` preserves order, which matters because it
is a hierarchy: `{'Section 2', '2.1 The Core Question'}`. `tags` becomes
directly queryable:

```sql
WHERE NOT ('instructor-answers' = ANY(tags))
```

Storing these as a delimited string would force `LIKE '%…%'` matching, which is
both slower and wrong whenever a value contains the delimiter.

**`UNIQUE (doc_id, ordinal)`** does two jobs. It enforces that positions do not
repeat within a document, and — because PostgreSQL implements a unique
constraint with a B-tree index — it *creates the index* required by the
self-referencing foreign key below. This is why no separate index on `doc_id` is
needed: one already exists with `doc_id` as its leading column.

**The self-referencing composite foreign key:**

```sql
FOREIGN KEY (doc_id, parent_ordinal) REFERENCES chunks (doc_id, ordinal)
```

An output chunk records which code chunk produced it. This makes "every output
points at a real code chunk in the same document" a guarantee enforced by the
database rather than a property the application hopes it maintains.

One consequence must be understood: **inserts must occur in ordinal order**,
because a parent row has to exist before a child can reference it. Any future
attempt to sort, batch, or parallelise these inserts differently will fail.

**`CHECK (parent_ordinal IS NULL OR kind = 'output')`** — only outputs may have
a parent. Prose claiming a parent is meaningless, so the database refuses it.

### 9.3 `embeddings`

This is where pgvector enters, and it has the two most consequential design
decisions in the schema.

**Decision 1: a separate table, not a column on `chunks`.**

A `vector` column on `chunks` would be simpler, and it would be wrong for this
project, because it allows exactly one embedding per chunk. Two planned
experiments require several at once:

- comparing `bge-m3` (1024 dimensions) against a smaller model such as
  `all-MiniLM-L6-v2` (384 dimensions) on the same chunks;
- comparing full precision against reduced precision for the same model.

With `PRIMARY KEY (chunk_id, model)`, each chunk may hold one embedding **per
model**, and comparing two models is a join rather than a re-ingest.

**Decision 2: `vector` with no declared dimension.**

Writing `vector(1024)` would let PostgreSQL enforce the width. But a
`vector(1024)` column physically cannot store a 384-dimensional vector, which
would make the model comparison above impossible without a second table.

Integrity is recovered another way — a `dim` column recording the width, plus:

```sql
CHECK (vector_dims(embedding) = dim)
```

`vector_dims()` is a pgvector function returning a vector's length. The
constraint makes it impossible to record a width that disagrees with the stored
data, which is the specific corruption worth preventing: an embedding stored
against the wrong model's label.

**The cost, stated plainly:** pgvector's approximate indexes (HNSW, IVFFlat)
require a fixed dimension, so this column can never carry one. See
[§15](#15-indexing-and-why-we-deliberately-have-none) for why that is currently
the right trade.

---

## 10. Stage 7 — Apply the migration

Keep schema changes in numbered files under `migrations/`, applied in order. At
this size a migration *tool* is unnecessary ceremony; the files themselves are
the value, because they record exactly what was run.

```bash
psql -p 5434 -d daedalus --single-transaction -v ON_ERROR_STOP=1 -f migrations/001_initial.sql
```

Both flags matter:

- **`--single-transaction`** wraps the whole file in one transaction. Without
  it, a failure halfway leaves the database in a partly-migrated state that has
  to be unpicked by hand. With it, any failure rolls everything back.
- **`-v ON_ERROR_STOP=1`** makes `psql` stop at the first error. The default is
  to print the error and carry on, which produces a long output where the one
  failure scrolls past unnoticed.

**Test the migration on a throwaway database first.** It costs seconds:

```bash
createdb -p 5434 schema_check
psql -p 5434 -d schema_check -c "CREATE EXTENSION vector;"
psql -p 5434 -d schema_check --single-transaction -v ON_ERROR_STOP=1 -f migrations/001_initial.sql
dropdb -p 5434 schema_check
```

**Verify the applied structure:**

```bash
psql -p 5434 -d daedalus -c "\d chunks"
```

`\d` prints columns, indexes, check constraints, foreign keys, and what
references the table. Confirm the two indexes (`chunks_pkey` and
`chunks_doc_id_ordinal_key`), both check constraints, and both foreign keys
including the self-reference.

**Verify the constraints actually fire.** A constraint you have not seen reject
something is a constraint you are only assuming exists:

```sql
INSERT INTO documents VALUES ('d1','/x.ipynb','notebook','T');
INSERT INTO chunks (doc_id,ordinal,kind,text,locator) VALUES ('d1',0,'code','x=1','cell:0');

-- must succeed: output pointing at ordinal 0
INSERT INTO chunks (doc_id,ordinal,kind,text,locator,parent_ordinal)
VALUES ('d1',1,'output','1','cell:0',0);

-- must fail: parent does not exist
INSERT INTO chunks (doc_id,ordinal,kind,text,locator,parent_ordinal)
VALUES ('d1',2,'output','?','cell:9',99);

-- must fail: prose may not have a parent
INSERT INTO chunks (doc_id,ordinal,kind,text,locator,parent_ordinal)
VALUES ('d1',3,'prose','p','cell:3',0);

-- must fail: declared dimension disagrees with the vector
INSERT INTO embeddings (chunk_id,model,dim,embedding)
SELECT id,'bge-m3',1024,'[1,2,3]'::vector FROM chunks LIMIT 1;
```

The last three produced, respectively,
`violates foreign key constraint "chunks_doc_id_parent_ordinal_fkey"`,
`violates check constraint "chunks_check"`, and
`violates check constraint "embeddings_check"`.

---

## 11. Stage 8 — Embeddings

### 11.1 Choosing where embeddings come from

The model can be served by a hosted API or run locally. This project runs
`bge-m3` locally through Ollama. The reasoning:

- No cost and no network dependency.
- No Python machine-learning stack. Using `sentence-transformers` would pull in
  PyTorch — around 2.5 GB — whereas Ollama is already installed and speaks HTTP.
- The standard library can make an HTTP request, so **no new dependency at all**.

The cost of the Ollama route: less control over pooling and normalisation, and
no access to `bge-m3`'s multi-vector output mode.

```bash
ollama pull bge-m3
```

### 11.2 Confirm the dimension by measurement

Never take a model's dimension from documentation or memory. Measure it, because
this number goes into the schema and into every stored row:

```bash
curl -s http://localhost:11434/api/embed \
  -d '{"model":"bge-m3","input":"hello"}' \
  | python3 -c "import json,sys; print('dimensions:', len(json.load(sys.stdin)['embeddings'][0]))"
```

This returned `1024`.

### 11.3 The embedding client

`src/daedalus/embedding.py`, in full:

```python
"""Embedding generation via a local Ollama server."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "bge-m3"
OLLAMA_URL_ENV = "DAEDALUS_OLLAMA_URL"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0


class EmbeddingError(RuntimeError):
    """Raised when the embedding service cannot be reached or returns badly."""


def ollama_url() -> str:
    """Return the Ollama base URL, honouring the environment override."""
    return os.environ.get(OLLAMA_URL_ENV, "").strip() or DEFAULT_OLLAMA_URL


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[list[float]]:
    """Embed a batch of texts, returning one vector per input in order."""
    if not texts:
        return []

    endpoint = f"{url or ollama_url()}/api/embed"
    payload = json.dumps({"model": model, "input": texts}).encode()
    request = urllib.request.Request(
        endpoint, data=payload, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        message = f"could not reach Ollama at {endpoint}: {error}"
        raise EmbeddingError(message) from error
    except json.JSONDecodeError as error:
        raise EmbeddingError(f"Ollama returned a non-JSON body: {error}") from error

    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbeddingError(
            f"expected {len(texts)} vectors from {model}, got {_describe(vectors)}"
        )
    return [[float(value) for value in vector] for vector in vectors]
```

**The count check is the most important line in the file.** If the service
returns fewer vectors than inputs, every subsequent chunk-to-vector pairing
shifts by one. The database would fill with valid-looking embeddings attached to
the wrong chunks, no error would be raised, and search results would be subtly
wrong forever. Verifying `len(vectors) == len(texts)` costs nothing and closes
that failure mode.

**On defaults:** this module *has* a default URL while the database connection
string (below) does not. The distinction is deliberate and worth carrying into
other projects — **a wrong Ollama URL fails loudly**, with a connection error on
the first call. **A wrong database URL fails silently**, connecting successfully
to the wrong server. Defaults are safe where failure is loud.

### 11.4 Batching

Requests carry a list of inputs rather than one, amortising HTTP and model
startup overhead across many texts. Batch size is a tuning knob
(`DEFAULT_BATCH_SIZE = 32`); larger batches are faster but hold more memory.

---

## 12. Stage 9 — The Python layer

### 12.1 psycopg fundamentals

Four things to understand before reading the modules.

**Parameters are not string formatting.** This is the single most important
habit:

```python
cursor.execute("SELECT 1 FROM documents WHERE doc_id = %s", (doc_id,))
```

The `%s` is *not* Python's `%` operator. psycopg sends the query text and the
parameter values to PostgreSQL separately, so a value can never be interpreted
as SQL. The vulnerable version — never write it — is:

```python
cursor.execute(f"SELECT 1 FROM documents WHERE doc_id = '{doc_id}'")  # WRONG
```

Note also that the parameters argument is a **tuple**; a single parameter needs
the trailing comma, `(doc_id,)`.

**Identifiers cannot be parameters.** Table and database *names* are not values,
so `%s` does not work for them. Use `psycopg.sql`:

```python
from psycopg import sql

connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
```

**Python lists become PostgreSQL arrays; tuples become composite types.** This
catches people out. Our segments hold tuples for immutability, so they must be
converted:

```python
list(segment.heading_path)  # -> text[]
```

Passing the tuple directly would attempt to build a composite value and fail.

**The connection is a context manager.** `with psycopg.connect(...)` **commits**
when the block exits normally and **rolls back** if an exception escapes. That
is what makes a partly-written document impossible.

### 12.2 Connection configuration

`src/daedalus/storage/database.py`:

```python
DATABASE_URL_ENV = "DAEDALUS_DATABASE_URL"


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when the connection string is absent from the environment."""


def database_url() -> str:
    url = os.environ.get(DATABASE_URL_ENV, "").strip()
    if not url:
        raise DatabaseNotConfiguredError(
            f"{DATABASE_URL_ENV} is not set. Example: "
            f"postgresql:///daedalus?host=/tmp&port=5434"
        )
    return url


@contextmanager
def connect(url: str | None = None) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with psycopg.connect(url or database_url()) as connection:
        yield connection
```

Set it in your shell profile:

```bash
export DAEDALUS_DATABASE_URL="postgresql:///daedalus?host=/tmp&port=5434"
```

**Read that connection string carefully.** The empty space between `///` and
`daedalus` is where a host would normally go; leaving it empty and passing
`host=/tmp` tells libpq to connect over the **Unix socket** in `/tmp` rather
than over TCP. Two benefits: no password is involved, since `pg_hba.conf` grants
`trust` on local connections; and it is immune to another process holding the
same TCP port.

**The variable is required, with no fallback.** A default would reintroduce
exactly the silent-wrong-server failure that this setup already suffered once.
The error names the variable and shows a valid example, so it is self-fixing.

The optional `url` argument exists so tests can target a throwaway database
without mutating the environment.

### 12.3 Writing documents and chunks

`src/daedalus/storage/documents.py`:

```python
_INSERT_DOCUMENT = """
INSERT INTO documents (doc_id, source_path, source_format, title)
VALUES (%s, %s, %s, %s)
"""

_INSERT_CHUNK = """
INSERT INTO chunks
    (doc_id, ordinal, kind, text, heading_path, tags, locator, parent_ordinal)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


def store_document(connection, document) -> int:
    rows = [
        (
            document.doc_id,
            segment.ordinal,
            segment.kind.value,
            segment.text,
            list(segment.heading_path),
            list(segment.tags),
            segment.locator,
            segment.parent_ordinal,
        )
        for segment in document.segments
    ]

    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM documents WHERE doc_id = %s", (document.doc_id,))
        cursor.execute(_INSERT_DOCUMENT, (...))
        cursor.executemany(_INSERT_CHUNK, rows)

    return len(rows)
```

**Delete-then-insert rather than `ON CONFLICT`.** Because the identifier is a
content hash, a row with the same id has the same content — so replacing is
equivalent to skipping, and additionally repairs any partial state left by an
interrupted run. Both statements execute inside the caller's transaction, so the
replacement is atomic: there is no moment when the document exists without its
chunks.

**`executemany` with rows in segment order.** The order is a correctness
requirement, not a performance detail — the self-referencing foreign key means a
parent must be inserted before its children.

**`delete_document` touches only `documents`.** Chunks and embeddings disappear
through `ON DELETE CASCADE`. The schema does work the application would
otherwise have to do correctly every time.

### 12.4 Storing vectors

`src/daedalus/storage/embeddings.py` holds the pgvector-specific parts.

**Rendering a vector for insertion:**

```python
def _vector_literal(vector: Sequence[float]) -> str:
    """Render a vector in the text form pgvector accepts."""
    return f"[{','.join(repr(float(value)) for value in vector)}]"
```

pgvector accepts a vector as the text `[1.0,2.0,3.0]`, cast to the type. The
`pgvector` Python package provides a proper binary adapter, but this achieves
the same result with no added dependency. `repr()` on a Python float
round-trips exactly, so no precision is lost.

**The insert:**

```sql
INSERT INTO embeddings (chunk_id, model, dim, embedding)
VALUES (%s, %s, %s, %s::vector)
ON CONFLICT (chunk_id, model) DO UPDATE
SET embedding = EXCLUDED.embedding, dim = EXCLUDED.dim, created_at = now()
```

- `%s::vector` casts the text literal to the vector type.
- `ON CONFLICT (chunk_id, model) DO UPDATE` is PostgreSQL's upsert. When a row
  already exists for that chunk and model, it is updated rather than raising a
  duplicate-key error. `EXCLUDED` refers to the row that *would* have been
  inserted. This makes re-embedding idempotent.

**Finding work to do — the anti-join:**

```sql
SELECT c.id, c.heading_path, c.text
FROM chunks c
LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = %s
WHERE e.chunk_id IS NULL
ORDER BY c.id
LIMIT %s
```

This is the standard "rows in A with no match in B" pattern: `LEFT JOIN` keeps
every chunk whether or not a matching embedding exists, filling missing columns
with `NULL`; `WHERE e.chunk_id IS NULL` then keeps exactly the unmatched ones.

**The model condition belongs in `ON`, not `WHERE`.** This is subtle and
important. In the `ON` clause it participates in the join, so a chunk embedded
with `bge-m3` still appears as missing for `all-MiniLM-L6-v2` — which is what
makes per-model backfill work. Moved to `WHERE`, it would filter *after* the
join and silently break that behaviour.

**Adding context before embedding:**

```python
def embedding_input(heading_path: Sequence[str], text: str) -> str:
    if not heading_path:
        return text
    return f"{' > '.join(heading_path)}\n\n{text}"
```

A chunk of a few tokens — a short code output, say — carries almost no meaning
on its own. Prepending its heading path gives the model context. Only the
**embedded** text is affected; the stored text is untouched, so embedding with
and without this prefix can be compared later without re-chunking anything.

**The backfill loop:**

```python
def backfill_embeddings(connection, embed, model, batch_size=DEFAULT_BATCH_SIZE) -> int:
    total = 0
    while True:
        batch = chunks_missing_embeddings(connection, model, batch_size)
        if not batch:
            return total

        inputs = [embedding_input(heading, text) for _, heading, text in batch]
        vectors = embed(inputs)
        total += store_embeddings(
            connection,
            model,
            list(zip([row[0] for row in batch], vectors, strict=True)),
        )
        connection.commit()
```

Three decisions here.

**The embedder is injected, not imported.** `embed` is a parameter, so the
storage layer contains no knowledge of Ollama, HTTP, or model names. This is not
abstraction for its own sake — it is what lets the tests drive real SQL against
a real database with a fake embedder, requiring no network.

**Each batch commits.** Unlike document storage, this is a long-running job.
Committing per batch means an interrupted run keeps its completed work, and the
anti-join makes the next run resume precisely where it stopped.

**`zip(..., strict=True)`** raises if the two sequences differ in length. It is
a second line of defence behind the count check in the embedding client, guarding
the same misalignment failure.

---

## 13. Stage 10 — Similarity search

### 13.1 The distance operators

| Operator | Distance | Use when |
|---|---|---|
| `<->` | L2 (Euclidean) | vectors are not normalised and magnitude matters |
| `<=>` | Cosine | you care about direction, not magnitude — the usual choice for text |
| `<#>` | Negative inner product | vectors are normalised and you want the fastest option |
| `<+>` | L1 (Manhattan) | rarely, for specific metrics |

**Use `<=>` for text embeddings.** Cosine distance measures the angle between
vectors and ignores their length, which is what you want when comparing meaning:
a long passage and a short one about the same topic should be close.

`<=>` returns **0 for identical direction and 2 for opposite direction**, so
smaller is more similar and `ORDER BY` ascending gives the best matches first.

Choosing the wrong operator does not raise an error — it silently returns a
worse ranking, which is exactly the kind of bug that survives to production.

### 13.2 The search query

```sql
SELECT c.kind,
       array_to_string(c.heading_path, ' > ') AS heading,
       left(c.text, 68)                        AS preview,
       round((e.embedding <=> %s::vector)::numeric, 4) AS distance
FROM embeddings e
JOIN chunks c ON c.id = e.chunk_id
WHERE NOT ('instructor-answers' = ANY(c.tags))
ORDER BY e.embedding <=> %s::vector
LIMIT 5
```

Line by line:

- **`JOIN chunks c ON c.id = e.chunk_id`** — vectors live in `embeddings`, text
  lives in `chunks`. The join brings them together.
- **`array_to_string(c.heading_path, ' > ')`** — renders the array as a readable
  breadcrumb.
- **`e.embedding <=> %s::vector`** — the query vector arrives as a text literal
  and is cast; the operator computes cosine distance per row.
- **`WHERE NOT ('instructor-answers' = ANY(c.tags))`** — `ANY` tests membership
  in an array. **This is the argument for keeping vectors in a relational
  database**: excluding a category of content from search is one clause.
- **`ORDER BY … LIMIT 5`** — nearest first, top five.

Note the query vector appears twice — once for display, once for ordering. Both
are passed as parameters.

### 13.3 Searching from Python

```python
from daedalus.storage.database import connect
from daedalus.embedding import embed_texts
from daedalus.storage.embeddings import _vector_literal

query = "How does BERT answer questions from a document?"
vec = _vector_literal(embed_texts([query])[0])

with connect() as conn, conn.cursor() as cur:
    cur.execute(
        """
        SELECT c.kind, array_to_string(c.heading_path,' > '),
               left(c.text, 68), round((e.embedding <=> %s::vector)::numeric, 4)
        FROM embeddings e JOIN chunks c ON c.id = e.chunk_id
        WHERE NOT ('instructor-answers' = ANY(c.tags))
        ORDER BY e.embedding <=> %s::vector
        LIMIT 5
    """,
        (vec, vec),
    )
    for row in cur.fetchall():
        print(row)
```

**The query must be embedded with the same model as the stored chunks.**
Vectors from different models are not comparable — the numbers are in unrelated
spaces. Nothing will error; results will simply be meaningless. This is a strong
argument for the `model` column: it lets you assert that query and corpus agree.

### 13.4 Measured results

Against 352 chunks of one document, for the query above:

```
0.3045  [prose] Section 6: Asking Questions Over PDF Chunks and Rank
0.3076  [code]  Section 6: Asking Questions Over PDF Chunks and Rank
0.3076  [code]  6.2 Ask Multiple Questions from the Same PDF
0.3077  [prose] 6.2 Ask Multiple Questions from the Same PDF
0.3121  [code]  Section 7: Failure Case Analysis and Limitations
```

The results are topically relevant. But note the distances: the top five span
**0.008**. That is a very flat ranking — the embeddings are not discriminating
strongly between these chunks. Three of the five are code chunks beginning with
the same comment banner, which may be homogenising them.

This is recorded rather than fixed, because there is not yet a labelled
evaluation set to say what good looks like. It is the correct posture: observe
the signal, resist the urge to tune against a single anecdote.

---

## 14. Stage 11 — The pipeline end to end

```python
from pathlib import Path

from daedalus.ingestion.notebook import parse_notebook
from daedalus.ingestion.canonical import notebook_to_document
from daedalus.storage.database import connect
from daedalus.storage.documents import store_document
from daedalus.storage.embeddings import backfill_embeddings
from daedalus.embedding import DEFAULT_MODEL, embed_texts

path = Path("corpus/notebooks/Bert_QnA_Complete.ipynb")

# 1. Parse the source file into a format-faithful representation.
parsed = parse_notebook(path)

# 2. Convert to the canonical, format-independent representation.
document = notebook_to_document(parsed)

# 3. Store the document and its chunks.
with connect() as conn:
    store_document(conn, document)
    conn.commit()

    # 4. Embed everything that lacks an embedding for this model.
    backfill_embeddings(conn, embed_texts, DEFAULT_MODEL, batch_size=16)
```

Five stages, each independently testable: **parse → canonicalise → store →
embed → search**.

**Measured on this corpus:**

| Stage | Result |
|---|---|
| Ingest 352 chunks | 0.02 s |
| Embed 352 chunks | 32.7 s (0.09 s/chunk) |
| Stored dimension | 1024, uniform |

Embedding dominates by three orders of magnitude, which is why it is a separate
step: re-running ingestion is free, re-running embedding is not.

**Verify:**

```sql
SELECT (SELECT count(*) FROM documents)  AS documents,
       (SELECT count(*) FROM chunks)     AS chunks,
       (SELECT count(*) FROM embeddings) AS embeddings,
       (SELECT DISTINCT dim FROM embeddings) AS dim;
```

`chunks` and `embeddings` should match once backfill completes, and `dim` should
be a single value.

---

## 15. Indexing, and why we deliberately have none

pgvector offers two approximate-nearest-neighbour index types:

```sql
-- HNSW: better recall and query speed, slower to build, more memory
CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);

-- IVFFlat: faster to build, smaller, needs a list count and training data
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

**Neither is used here, on purpose.**

Both are *approximate*. They trade recall for speed: some genuinely nearest
neighbours will be missed. That trade is worth making when exact search is too
slow — and at a few hundred to a few thousand vectors, exact search takes
milliseconds. Adding an index would introduce recall loss too small to observe
in exchange for a speedup too small to notice.

There is also a hard constraint: **ANN indexes require a fixed-dimension
column**, and `embedding` is deliberately dimensionless so several models can
coexist. Adding an index means either splitting per model or adding a typed
column.

The defensible position is measurement-driven: *"I did not add an index because
I measured that exact search was fast enough at my corpus size."* That is a
better answer than an index nobody asked for.

**When to revisit:** when exact query latency becomes noticeable — typically
somewhere in the 10,000 to 100,000 vector range, depending on dimension and
hardware. Measure first:

```sql
EXPLAIN ANALYZE
SELECT chunk_id FROM embeddings ORDER BY embedding <=> '[...]'::vector LIMIT 10;
```

A `Seq Scan` with acceptable timing means no index is needed. Note the operator
class must match the operator you query with — `vector_cosine_ops` for `<=>`,
`vector_l2_ops` for `<->`. A mismatch means the index is silently unused.

---

## 16. Quantization

Quantization means storing vectors at lower precision to save space, accepting
some accuracy loss. pgvector supports it natively, which makes it a small
experiment rather than a project:

| Type | Precision | Relative size |
|---|---|---|
| `vector` | 32-bit float | 1× |
| `halfvec` | 16-bit float | 0.5× |
| `bit` + Hamming distance | 1 bit per dimension | 0.03× |

```sql
-- half precision
ALTER TABLE embeddings ADD COLUMN embedding_half halfvec;
UPDATE embeddings SET embedding_half = embedding::halfvec;

-- binary, searched with Hamming distance
SELECT chunk_id FROM embeddings
ORDER BY binary_quantize(embedding)::bit(1024) <~> binary_quantize('[...]'::vector)
LIMIT 10;
```

The usual production pattern is **binary for a fast first pass, full precision
for rescoring**: retrieve a few hundred candidates with the cheap distance, then
re-rank those candidates exactly. That recovers most of the accuracy at a
fraction of the memory.

Storing the same vectors at three precisions and measuring recall against
storage is a genuine, self-contained experiment — and the schema already permits
it, because `embeddings` is a separate table keyed by model.

---

## 17. Testing strategy

### The choice: a real database, not mocks

Mocking the database client would assert that a function was called with a
string. It would verify nothing about whether the SQL is correct — and the SQL
*is* the interesting part here: the composite foreign key, the check
constraints, the cascade, the anti-join.

So the tests run against a real, throwaway PostgreSQL database.

**The honest cost:** the tests acquire an environmental dependency and are no
longer hermetic. They remain deterministic — same input, same output — but they
need a server. The mitigation is to skip them with a clear message rather than
fail when no server is reachable.

### The fixtures

```python
def _url_for(database: str) -> str:
    """Return the configured connection string pointed at another database."""
    parts = conninfo_to_dict(os.environ[DATABASE_URL_ENV])
    parts["dbname"] = database
    return make_conninfo(**parts)


@pytest.fixture(scope="session")
def database_url():
    if not os.environ.get(DATABASE_URL_ENV, "").strip():
        pytest.skip(f"{DATABASE_URL_ENV} is not set")

    try:
        admin = psycopg.connect(_url_for("postgres"), autocommit=True)
    except psycopg.OperationalError as error:
        pytest.skip(f"no PostgreSQL server reachable: {error}")

    name = sql.Identifier(TEST_DATABASE)
    with admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(name))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(name))

        url = _url_for(TEST_DATABASE)
        with psycopg.connect(url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            for migration in sorted(MIGRATIONS.glob("*.sql")):
                connection.execute(migration.read_text())

        yield url

        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(name))


@pytest.fixture
def connection(database_url: str):
    with psycopg.connect(database_url) as conn:
        conn.execute("TRUNCATE documents, chunks, embeddings RESTART IDENTITY CASCADE")
        conn.commit()
        yield conn
```

Points worth carrying elsewhere:

- **Derive the test URL from the configured one** by swapping `dbname`. Nothing
  is hardcoded, so changing the port later does not break the tests.
- **`autocommit=True` for the admin connection.** `CREATE DATABASE` cannot run
  inside a transaction block.
- **`sql.Identifier` for the database name**, since identifiers cannot be
  parameters.
- **Session scope for the schema, function scope for the data.** Creating and
  migrating a database is slow and happens once; `TRUNCATE ... RESTART IDENTITY
  CASCADE` between tests is fast and resets identity sequences so ids are
  predictable.
- **Migrations are globbed and sorted**, so future migration files are picked up
  with no change to the fixture.

### What to test

- Every field round-trips, arrays included.
- Each constraint rejects what it should — assert on the specific exception, for
  example `psycopg.errors.ForeignKeyViolation`.
- Idempotence: storing twice does not duplicate; backfilling twice embeds zero
  the second time.
- Cascades: deleting a document removes chunks and embeddings.
- The per-model behaviour of the anti-join.

---

## 18. Troubleshooting

Every entry below was encountered while building this.

### `could not bind IPv4 address "127.0.0.1": Address already in use`

Another PostgreSQL already holds the port.

```bash
lsof -nP -iTCP:5432 -sTCP:LISTEN
ps aux | grep [p]ostgres
```

Change your server's port in `postgresql.conf` and restart, or stop the other
server. Changing the port requires a **restart**, not a reload.

### `brew services list` shows `error`

The service is failing to start. The reason is in the log, not in the service
listing:

```bash
tail -20 /opt/homebrew/var/log/postgresql@18.log
```

### `fe_sendauth: no password supplied`

`pg_hba.conf` requires a password for this connection type. Either supply one,
or set `trust` for local connections on a development machine, then:

```bash
pg_ctl -D /path/to/data reload
```

### `FATAL: password authentication failed for user "..."`

The role exists but the password is wrong — or the `pg_hba.conf` edit did not
take effect. Two frequent causes: editing the wrong line (the `local` line
governs socket connections, the `host` lines govern TCP), or forgetting to
reload.

Confirm what the server is actually using:

```bash
grep -v '^#' /path/to/data/pg_hba.conf | grep -v '^$'
```

### `FATAL: role "yourname" does not exist`

Authentication is now working — this is progress. The cluster was initialised
with a different superuser name. Connect as one that exists and create yours:

```bash
psql -p 5434 -U postgres -d postgres -c "CREATE ROLE yourname LOGIN SUPERUSER CREATEDB;"
```

### `ERROR: type "vector" does not exist`

The extension is not enabled **in this database**. Extensions are per database:

```bash
psql -p 5434 -d yourdb -c "CREATE EXTENSION vector;"
psql -p 5434 -d yourdb -c "\dx"
```

If `CREATE EXTENSION` itself fails with `could not open extension control file`,
the extension is not installed for *this* PostgreSQL installation — check
[§4](#4-stage-1--survey-the-machine-before-installing-anything); you are
probably running a different server from the one the package installed into.

### Two servers answering on one port

```bash
lsof -nP -iTCP:5433 -sTCP:LISTEN
```

If both a Docker proxy and a `postgres` process appear, TCP connections are
non-deterministic. Move one of them, or connect over the Unix socket
(`host=/tmp`), which Docker does not touch.

### `expected N vectors from model, got M`

The embedding service returned a different number of vectors than inputs. Check
that the model is pulled (`ollama list`) and that the server is running. Never
"fix" this by ignoring it — the misalignment silently corrupts every downstream
row.

### `new row violates check constraint "embeddings_check"`

`dim` disagrees with the actual vector width. Usually means a model was swapped
without updating what is recorded — which is precisely the error the constraint
exists to catch.

### Search returns irrelevant results

Work through in order: were query and chunks embedded with the **same model**?
Is the operator right — `<=>` for normalised text embeddings, not `<->`? Is the
ordering ascending? Is the query being embedded at all, rather than compared as
text?

### Terminal mangles pasted multi-line commands

Long pastes with `\` continuations can drop characters, producing commands that
partly execute. Paste one line at a time, or write the sequence to a script file
and run it. Always check `git log` and `git status` after a paste-driven
sequence.

---

## 19. Reuse checklist

To apply this methodology to another project:

**Setup**

1. Survey the machine for existing PostgreSQL installations and port conflicts.
2. Install PostgreSQL and pgvector *from the same package source*.
3. Pick an uncontested port; set it in `postgresql.conf`; restart; verify with
   `lsof` that exactly one process holds it.
4. Set `pg_hba.conf` to `trust` on the local socket, passwords over TCP; reload.
5. Create your role if the cluster's superuser is not your OS account.
6. `createdb`, then `CREATE EXTENSION vector` **in that database**.
7. Run the three-dimensional hello-world to prove the extension works.

**Schema**

8. Decide the unit of retrieval — what one row of your chunk table represents.
9. Choose whether embeddings are a column or a table. A table if you will ever
   compare models or precisions; a column otherwise.
10. Decide whether the vector column carries a fixed dimension. Fixed enables
    ANN indexes; dimensionless allows several models to coexist.
11. Add constraints that make invalid states impossible, then **watch each one
    reject something** before trusting it.
12. Write the schema as a numbered migration; test it on a throwaway database;
    apply with `--single-transaction -v ON_ERROR_STOP=1`.

**Embeddings**

13. Choose the model; **measure its dimension**, do not assume it.
14. Validate that the count of returned vectors equals the count of inputs.
15. Embed as a separate step from ingestion, so re-embedding never means
    re-parsing.
16. Batch requests; commit per batch so long runs are resumable.
17. Use an anti-join to find work, with the model condition in `ON`, not
    `WHERE`.

**Application**

18. Require the connection string from the environment; never default it.
19. Connect over the Unix socket for local development.
20. Always use `%s` parameters; `psycopg.sql.Identifier` for identifiers.
21. Convert tuples to lists before passing them as PostgreSQL arrays.
22. Inject the embedder so the storage layer stays free of the model.

**Search**

23. Use `<=>` (cosine) for text embeddings.
24. Confirm the query is embedded with the same model as the corpus.
25. Do metadata filtering in the `WHERE` clause — the reason the data is in a
    relational database.

**Discipline**

26. Do not add an ANN index until measurement shows exact search is too slow.
27. Test against a real database, not mocks; skip cleanly when none is present.
28. Verify at every stage, and record measured numbers rather than expected ones.
