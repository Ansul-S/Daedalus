# Relevance Labelling Policy

The fixed grading standard for the Daedalus reference set. It governs every
judgement in the set, and it does not change during annotation.

Consistency matters more than any individual judgement here. The set is roughly
1,100 judgements made over several sessions; a standard that drifts halfway
through produces a reference set that cannot be trusted, and every number
derived from it inherits that.

Changes to this policy, to the grading scale, or to the pooling methodology
require an explicit decision recorded in `docs/ROADMAP.md`. They are not made
in passing.

## The question being answered

Each candidate chunk is judged **only by how useful its actual content is for
answering the query**:

> Does this chunk help answer the query, and how completely does it do so?

## What must not influence a judgement

- Which retriever surfaced the chunk.
- Whether it came from vector search, lexical search, or random sampling.
- The chunk's position in the candidate list.
- Any retriever ranking or score.
- Whether the chunk is code or prose.
- Whether the chunk looks technically sophisticated.
- Whether the chunk seems likely to be useful from its metadata alone.

The reference set exists to evaluate the retrieval system, so the judgement has
to be independent of the retriever. The reasoning to avoid is:

> "Vector search ranked this highly, so it is probably relevant."

The reasoning to use is:

> "Ignoring how this chunk was retrieved, and looking only at the query and the
> chunk, would this chunk help answer the question?"

This is why the labelling interface presents candidates in an order derived from
a hash of the query and chunk id rather than in any retriever's ranking, and why
it does not display which retrievers found a chunk.

## The grades

### 0 — Not relevant

The chunk does not help answer the question. It may discuss a related topic or
share terminology, but it does not provide information useful for answering the
specific query.

### 1 — Partially relevant

The chunk contributes to the answer without fully answering the question:

- it provides one part of a multi-part answer;
- it explains a related mechanism but not the complete concept;
- it provides useful evidence but needs other information to complete the answer;
- it demonstrates part of the answer without explaining the reasoning.

### 2 — Fully relevant

The chunk contains enough information to answer the question directly. The
wording need not be perfect; the grade is 2 when the chunk's content is
sufficient to construct a direct and complete answer.

## Code and prose are judged the same way

A code chunk is not downgraded for being code, and is not promoted for looking
sophisticated.

- **2** when the implementation contains enough information to answer the
  question directly.
- **1** when it demonstrates part of the answer or provides useful evidence.
- **0** when it does not meaningfully help.

For the query *"How does the retriever work before the reader model?"*, a code
chunk showing the retriever selecting chunks and passing them to the reader is
relevant. A chunk containing only a signature such as

```python
def retrieve_then_answer(query, retriever, reader, ...):
```

with no meaningful body or explanation does not earn a 2 merely because the
function name matches the question.

## Grades are human judgements

No grade in the reference set is produced by a retriever, an embedding, a
similarity score, or a language model. The set is the ground truth those systems
are measured against; deriving any part of it from them would make the
measurement circular and worthless.

The only way a grade enters the database is a keystroke from the person
labelling.

## What the annotator is shown

### The complete chunk, never a preview alone

The default display shows up to 2,000 characters. That figure was chosen
against the corpus: median chunk length is about 418 characters and p90 about
1,188, so most chunks appear whole, while the rare very large chunk does not
fill the terminal.

When a chunk is longer, the display is marked `[TRUNCATED — N more characters]`
and `f` reveals the whole thing.

**A judgement is made on the complete content.** The preview is never treated as
the chunk. If a chunk is truncated and the hidden part could change the grade,
press `f` before grading.

### Parent context for output chunks

An output chunk is frequently meaningless read alone — 196 of the corpus's 925
chunks are outputs, and their median length is around 36 tokens. Excluding them
would leave a fifth of the corpus unmeasured, so they are judged with help
instead.

When the candidate is an output chunk and the code that produced it exists, that
code is displayed above the candidate, labelled:

```
CONTEXT — the code that produced this output. NOT judged.
```

followed by

```
CANDIDATE — judge this:
```

**The grade belongs to the output chunk, not to the parent code.** Context exists
only to make the candidate interpretable — to show what the numbers or text in
the output actually represent.

An output chunk does not inherit a grade from its parent. If the parent code
fully answers the query but the output itself contributes nothing toward
answering it, the output is graded 0. The same 0/1/2 standard applies:

- **0** — the candidate output does not help answer the question.
- **1** — the candidate output contributes useful information but does not fully
  answer it.
- **2** — the candidate output contains enough information to answer directly.

## Session mechanics

- Every judgement is committed as it is made. Nothing is buffered.
- Candidates already judged for a query are skipped, so a session can be
  stopped at any point and resumed without repeating work.
- `s` skips a candidate. It records nothing, and the candidate is offered again
  in a later session. It is the correct action when a candidate cannot be judged
  because of ambiguity, corruption, or missing context.
- Re-judging a candidate replaces the earlier grade.

## The grading scale is 0, 1, 2 — and only that

There is no fourth grade and no "undecided" value. A candidate that cannot be
judged is skipped with `s`, not recorded under some other label. Adding a grade
would change what every existing judgement means relative to the others, so the
scale is fixed for the life of the reference set.

## Unjudged is not zero

A candidate that was never judged — skipped with `s`, or left unreached because
`--per-query` capped the query — is **absent** from the reference set. It is not
a grade 0 and it is not a grade of any kind.

Metrics computed from this set must not count an unjudged candidate as either
relevant or irrelevant. Treating absence as irrelevance inflates precision and
understates recall, and it would do so silently.

Where a metric requires a closed-world assumption in order to be computable at
all, that assumption must be stated explicitly alongside the number, never
applied quietly.

## Pooling parameters

Fixed for the reference set:

| parameter | value |
|---|---|
| `--vector-k` | 10 |
| `--lexical-k` | 10 |
| `--random-k` | 5 |
| `--per-query` | 25 |

Changing any of these changes what the reference set covers, so a set built
under different values is a different set. They are not adjusted mid-annotation.
