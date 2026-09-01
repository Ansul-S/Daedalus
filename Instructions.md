# Labelling Instructions

How to build the Daedalus reference set: a hand-graded collection of query and
chunk pairs used to measure whether the retrieval system actually finds the
right material.

These instructions assume no prior knowledge of the project. Follow them in
order. `docs/LABELLING.md` is the authoritative statement of the grading policy;
this document is the procedure for applying it.

---

## Part 1 — What you are doing, and why it matters

Daedalus searches a collection of study material and returns the passages most
likely to answer a question. To know whether it is any good at that, there has
to be an independent answer to "which passages *actually* answer this question?"

That answer cannot come from the search system, because then the system would be
marking its own homework: a passage it never finds could never be recorded as
relevant, and it would appear to find everything. So a human reads the passages
and decides.

That human is you. Your judgements become the standard every future measurement
is compared against.

**Two consequences follow, and they govern everything below.**

First, **consistency matters more than any single judgement.** You will make
roughly 1,100 judgements over several sessions. If your standard drifts — stricter
on Tuesday than on Monday — the resulting numbers measure your mood as much as
the software. A slightly-wrong rule applied consistently is far better than a
perfect rule applied erratically.

Second, **you must ignore how a passage was found.** The interface deliberately
does not tell you which search method surfaced a passage, and presents them in a
scrambled order rather than best-first. Do not try to infer it. Judge only the
question and the passage in front of you.

---

## Part 2 — Vocabulary

Five terms are used throughout.

- **Corpus** — the study material being searched. Currently three Jupyter
  notebooks about question answering, retrieval, and BERT.
- **Chunk** — one piece of that material: a passage of prose, a block of code,
  or the output a block of code produced. The corpus holds 925 of them.
- **Query** — a question someone might ask the system.
- **Candidate** — a chunk offered to you for judging against a particular query.
- **Judgement** — your grade of 0, 1, or 2 for one candidate against one query.

---

## Part 3 — Before you start

### Step 3.1 — Start the database

```bash
brew services start postgresql@18
sleep 3
pg_isready -p 5434
```

Expect `accepting connections`. If it says no server is running, see
Troubleshooting at the end.

### Step 3.2 — Start the embedding service

```bash
ollama serve &
sleep 2
curl -s http://localhost:11434/api/version
```

Expect a version number in JSON.

### Step 3.3 — Set the connection string

```bash
export DAEDALUS_DATABASE_URL="postgresql:///daedalus?host=/tmp&port=5434"
```

If this is missing, every command fails with a message naming the variable. Put
the line in `~/.zshrc` so you never think about it again.

### Step 3.4 — Confirm the corpus is loaded

```bash
cd ~/Desktop/Daedalus
uv run daedalus status
```

Expect:

```
documents: 3
chunks:    925
embeddings: 925 with bge-m3 (1024 dimensions)
```

If chunks and embeddings do not match, or either is zero:

```bash
uv run daedalus ingest corpus/notebooks
uv run daedalus embed
```

Ingestion takes under a second. Embedding takes roughly two minutes and only
processes what is missing, so it is safe to re-run.

**Do not begin labelling until these numbers are right.** Labelling against a
partly-loaded corpus produces a reference set that cannot measure the parts that
were absent.

---

## Part 4 — Writing queries

You need queries before you can label. Aim for **around 50**.

### Step 4.1 — Understand the two kinds

Every query is recorded as either `harvested` or `authored`.

**Harvested** questions are taken from the corpus itself. The notebooks contain
cells headed "Concept Check" and "Thought Experiment" holding real questions
written by the material's author. There are 27 such cells.

**Authored** questions are ones you write yourself, phrased the way you would
actually ask them.

**Why both.** Harvested questions share vocabulary with the material, which
quietly flatters keyword matching — the words are already there to be matched.
Your own phrasing is the harder and more realistic test. Recording which is
which lets the two groups be compared later, and any difference between them is
itself a finding.

Aim for a rough balance. Twenty-five of each is a reasonable target.

### Step 4.2 — Find harvestable questions

```bash
psql -p 5434 -d daedalus -c "
SELECT left(text, 300) FROM chunks
WHERE 'concept-check' = ANY(tags)
ORDER BY doc_id, ordinal;"
```

Read through and pick questions that are self-contained — one that says "explain
the diagram above" is useless without the diagram and should be skipped.

### Step 4.3 — Write good queries

A good query for this purpose:

- **Is answerable from the corpus.** A question about Kubernetes has no answer in
  notebooks about BERT, and labelling it teaches nothing.
- **Is specific enough to judge.** "Tell me about embeddings" makes every
  passage arguably relevant. "Why are embeddings normalised before cosine
  similarity?" has a defensible answer.
- **Is phrased as a real question**, not as keywords. `retriever reader
  pipeline` is a search box query; "How does the retriever narrow down passages
  before the reader runs?" is what you would actually ask.
- **Is not a near-duplicate** of one you already added. Two queries differing by
  a word waste an hour of labelling on nearly the same candidates.

Aim for a spread of difficulty: some questions answered by one obvious passage,
some needing information assembled from several.

### Step 4.4 — Add them

```bash
uv run daedalus query add "How does the retriever narrow down passages before the reader runs?" --source authored
uv run daedalus query add "What is the difference between extractive and generative answering?" --source harvested
```

Quote the whole question. Duplicated text is refused with `query already
present, not added` — that is a safeguard, not an error.

Check your list at any time:

```bash
uv run daedalus query list
```

Each line shows the id, source, judgement count, and text.

---

## Part 5 — The labelling session

### Step 5.1 — Start

```bash
uv run daedalus label
```

The policy summary prints, then the first candidate.

### Step 5.2 — Read the screen

```
==============================================================================
QUERY: Why does a retriever help before running a reader model?
[3/21]  output  0ed93313f216978b:284
SECTION: Section 13: Combine FAISS Retriever with BERT Reader
------------------------------------------------------------------------------
CONTEXT — the code that produced this output. NOT judged.
------------------------------------------------------------------------------
results = faiss_index.search(query_embedding, top_k=5)
answer = reader(question, results)
------------------------------------------------------------------------------
CANDIDATE — judge this:
------------------------------------------------------------------------------
Retrieved 5 chunks in 0.03s
Answer: the reader predicts a span
------------------------------------------------------------------------------
0 not relevant  1 partially answers  2 fully answers   s skip   f full text   ? policy   q quit >
```

Line by line:

- **QUERY** — the question. Re-read it for every candidate. It is easy to drift
  into judging "is this passage interesting" instead of "does it answer *this*".
- **`[3/21]`** — third of twenty-one candidates for this query.
- **`output`** — the kind of chunk: `prose`, `code`, or `output`.
- **`0ed93313f216978b:284`** — which document and position. You do not need it,
  but it identifies the chunk exactly if you want to inspect it later.
- **SECTION** — where it sits in the document's headings.
- **CONTEXT** — appears only for output chunks. Explained in Part 7.
- **CANDIDATE** — the thing you are grading.

### Step 5.3 — Grade it

Press one key. There is no Enter.

| key | meaning |
|---|---|
| `0` | not relevant |
| `1` | partially answers |
| `2` | fully answers |
| `s` | skip — record nothing, ask again later |
| `f` | show the full text of a truncated chunk |
| `?` | reprint the policy |
| `q` | quit |

Each grade is saved to the database immediately. You can close the terminal at
any moment and lose nothing.

---

## Part 6 — How to decide the grade

The single question is:

> **Does this chunk help answer the query, and how completely?**

### Grade 2 — fully answers

The chunk contains enough information to answer the question directly. It does
not have to be well written or phrased as an answer. Ask yourself: *if I had
only this passage, could I give a complete answer?* If yes, it is a 2.

### Grade 1 — partially answers

The chunk contributes without being sufficient. Typical cases:

- it gives one part of a multi-part answer;
- it explains a related mechanism but not the whole concept;
- it is useful evidence but needs something else to complete the answer;
- it demonstrates part of the answer without explaining the reasoning.

### Grade 0 — not relevant

The chunk does not help. It may be about a related topic, or share words with
the question, and still be a 0. Sharing vocabulary is not relevance.

### The test to apply when unsure

Imagine writing the answer using only this chunk.

- Could you write the whole answer? **2**
- Could you write part of it, or would it be genuine supporting evidence? **1**
- Would it not help you write any of it? **0**

### Things that must not affect your grade

- How the chunk was found. You are not told, and must not guess.
- Its position in the list. Position 1 is not more likely to be relevant than
  position 19 — the order is scrambled deliberately.
- Whether it is code or prose.
- Whether it looks impressive or sophisticated.
- Whether you already gave several 0s in a row and feel "due" a 2. Each
  judgement is independent.

---

## Part 7 — Edge cases

### Truncated chunks

Chunks longer than 2,000 characters are cut, and the cut is marked:

```
[TRUNCATED — 830 more characters. Press f to read all before judging.]
```

**Press `f` before grading whenever the hidden part could change your answer.**
The policy is to judge the complete content. Grading a truncated preview as
though it were the whole chunk is the most likely way to introduce error.

If the visible part already settles it — clearly irrelevant, or already
sufficient for a 2 — you may grade without expanding.

### Output chunks

An output chunk is what a piece of code printed. Alone it is often
uninterpretable: `0.847` means nothing without knowing what was computed.

So when the candidate is an output, the code that produced it appears above,
marked `CONTEXT ... NOT judged`.

**Use the context to understand the output. Then grade the output.**

The distinction matters and is easy to get wrong. If the *code* fully answers
the question but the *output* is just `Done in 0.03s`, the grade is **0** — the
output contributes nothing toward answering, and the code is not what you are
grading. The output does not inherit its parent's relevance.

Conversely, an output showing a retrieved passage and the answer extracted from
it may well be a 1 or 2 in its own right.

### Code chunks

Judge code by the same standard as prose. Do not downgrade it for being code and
do not promote it for looking sophisticated.

- **2** — the implementation contains enough to answer the question directly.
- **1** — it demonstrates part of the answer or provides useful evidence.
- **0** — it does not meaningfully help.

**A function signature alone is not a 2.** If a chunk is only:

```python
def retrieve_then_answer(query, retriever, reader, ...):
```

with no meaningful body, it does not answer "how does retrieval work" merely
because its name matches the question. Press `f` first — the body may be there
and simply truncated. If there genuinely is no body, grade what is present.

### When you genuinely cannot decide

Press `s`. It records nothing and offers the candidate again in a later session.

Use it when the candidate is corrupted, unintelligible, or you need to think.
Do **not** use it to avoid hard calls generally — if you skip everything
difficult, the reference set only contains easy cases and the measurement is
worthless. Aim to skip rarely.

**A skipped candidate is not a 0.** It is absent from the set entirely.

### Correcting a mistake

There is no undo during a session. A normal session never offers a candidate you
have already graded, so correcting one takes `--regrade`, which names a single
query and offers its whole pool again:

```bash
uv run daedalus label --regrade 6
```

Every candidate for that query comes back, including the ones already graded, and
the earlier grade is not shown — the second reading has to stand on its own. A new
grade replaces the old one rather than adding to it, so the query's judgement count
does not change. The `--per-query` ceiling is ignored, so a query that is already
full can still be revisited.

If you fumble a key, note the `doc_id:ordinal` shown on screen and re-grade that
query later. Note that there is no way to target one candidate: you page through
the whole pool to reach it.

An unrecognised key does not record anything and moves to the next candidate, the
same as `s`. Nothing is lost — an ungraded candidate is offered again in a later
session — but the screen gives no sign that it happened.

### A query that turns out to be bad

If every candidate for a query is a 0, the query is probably unanswerable from
this corpus. Finish it, then note it. It is not a failure — a query with no
relevant chunks is legitimate data — but too many of them mean your queries are
drifting away from the material.

---

## Part 8 — Pacing

There are roughly 1,100 judgements. At 20 seconds each that is about six hours.

- **Work in sessions of 45–60 minutes.** Accuracy falls off well before you
  notice it, and a tired hour of labelling is worse than no hour.
- **Stop when you start pattern-matching** on chunk appearance rather than
  reading. That is the failure mode: it feels fast and productive and quietly
  destroys the set's value.
- **Re-read the policy with `?` at the start of each session.** Drift is
  gradual, and re-anchoring costs ten seconds.

Check progress at any time:

```bash
uv run daedalus query list
```

---

## Part 9 — Finishing

You are done when most queries show a judgement count near 25.

Check the distribution:

```bash
psql -p 5434 -d daedalus -c "
SELECT grade, count(*) FROM judgements GROUP BY grade ORDER BY grade;"
```

Expect many more 0s than 2s — most candidates in a pool of about 21 are not
relevant, which is normal and correct. If you have almost no 0s, you are
probably grading too generously. If you have almost no 2s, the queries may be
too hard or the standard too strict.

Also check no query was left barely started:

```bash
uv run daedalus query list
```

---

## Part 10 — Troubleshooting

**`DAEDALUS_DATABASE_URL is not set`** — run the export from Step 3.3.

**`could not reach Ollama`** — run `ollama serve &` and wait a moment.

**`connection refused` from the database** — run
`brew services start postgresql@18`, then `pg_isready -p 5434`.

**`nothing to label`** — every query already has its full quota of judgements, or
you have not added any queries. Check `uv run daedalus query list`.

**Very few candidates offered for a query** — the query's words may barely appear
in the corpus. That is legitimate; a small pool is real information about the
query.

**The terminal looks scrambled after quitting** — run `reset`. The interface puts
the terminal into single-keypress mode and restores it on exit, but an abrupt
kill can leave it confused.

---

## Part 11 — The rules in one place

1. Judge only whether the chunk helps answer the query.
2. Ignore how it was found, where it appears in the list, and whether it is code.
3. 2 = sufficient to answer. 1 = contributes. 0 = does not help.
4. Press `f` on truncated chunks whenever the hidden part might matter.
5. For outputs, use the context to interpret — but grade the output.
6. `s` skips and records nothing. A skip is not a 0.
7. Apply the same standard on the last day as on the first.
