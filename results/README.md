# Measured results

Frozen benchmark output. Each file here is the artifact of one experiment that
was run once, against a stated reference set, and is not regenerated casually —
re-running produces a new file rather than overwriting an old one.

Exploratory runs do not belong here.

---

## `retrieval_four_way_2026-09-06.json`

Four retrieval variants scored against the 1,422-judgement reference set.

### What was compared

| variant | embeddings | input embedded |
|---|---|---|
| `bge-m3 (production)` | `bge-m3`, 1024d | heading path + chunk text |
| `bge-m3-noheading` | `bge-m3-noheading`, 1024d | chunk text alone |
| `all-minilm` | `all-minilm`, 384d | heading path + chunk text |
| `lexical` | none | PostgreSQL full-text search |

All four scored by identical code — `daedalus.evaluation.harness` over
`daedalus.evaluation.metrics` — against the same 50 queries and the same
judgements, at K = 1, 3, 5, 7, 10, with graded 0/1/2 NDCG under linear gain and
binary precision, recall and reciprocal rank at thresholds >= 1 and = 2.

### How unjudged candidates were treated

**Every variant's top 10 was fully judged: 0.00 unjudged per query, all four.**
The `zero` policy was declared before the run, but with no unjudged results to
apply it to, `zero` and `skip` are provably identical here and no reported
number depends on the choice.

That is not luck. The judged pool was originally built from `bge-m3` and lexical
search only, leaving the two challengers at 76.60% and 48.60% coverage. 358
candidates from both challengers were pooled and hand-labelled on 2026-09-06
specifically to remove that bias before any score was computed. See
`reference-set/README.md`.

### Headline means, 95% bootstrap CI

| metric | bge-m3 | bge-m3-noheading | lexical | all-minilm |
|---|---|---|---|---|
| NDCG@10 | 0.6127 | 0.6237 | 0.4893 | 0.4748 |
| NDCG@5 | 0.5850 | 0.5903 | 0.4473 | 0.4407 |
| MRR (>=1) | 0.8118 | 0.8204 | 0.7037 | 0.6977 |
| Recall@10 (>=1) | 0.5832 | 0.5831 | 0.4812 | 0.4767 |
| Recall@10 (=2) | 0.7230 | 0.7587 | 0.5588 | 0.4802 |
| P@5 (>=1) | 0.5960 | 0.5880 | 0.4640 | 0.4600 |

`n = 50` for every metric except recall at threshold 2, which is **`n = 47`**:
q41, q45 and q51 have no grade-2 chunk anywhere in their pool, so recall is
undefined for them and they are excluded rather than counted as zero.

Intervals are a percentile bootstrap over queries, 10,000 resamples, seed
20260906, so any interval here can be reproduced exactly.

### What is separable, and what is not

Differences are paired bootstraps — both retrievers scored on the same resampled
queries, so the query-to-query variation they share cancels out. Comparing two
independent intervals would not do this and would call real differences
inseparable.

**Separable on NDCG@10 and on recall at both thresholds:**

- `bge-m3` over `all-minilm`: +0.1379 [0.070, 0.205]
- `bge-m3` over `lexical`: +0.1234 [0.059, 0.188]
- `bge-m3-noheading` over `all-minilm`: +0.1489 [0.083, 0.217]
- `bge-m3-noheading` over `lexical`: +0.1344 [0.066, 0.208]

**Not separable — recorded as null results:**

- **`bge-m3` vs `bge-m3-noheading`: −0.0110 [−0.050, +0.025] on NDCG@10, and
  exactly 0.0000 [−0.043, +0.042] on recall@10 at threshold 1.** The heading
  prefix does not measurably help or hurt on this corpus. The production
  configuration keeps the prefix; this is a null result, not a reason to change
  it, and not evidence that the prefix is useless in general.
- **`all-minilm` vs `lexical`: −0.0145 [−0.093, +0.067] on NDCG@10.** MiniLM is
  not shown to be worse than lexical search, or better. They are
  indistinguishable at this sample size.

Two tiers, then: the `bge-m3` pair above the `lexical`/`all-minilm` pair, with
no separable difference inside either tier.

### Reading the intervals honestly

Six pairs are compared per metric at 95% confidence each, so the family-wise
error rate is well above 5%. The four differences above clear zero on every
headline metric at both thresholds, with lower bounds of 0.05 to 0.17. Marginal
intervals do not: `bge-m3` over `lexical` on MRR at threshold 1 is
[−0.007, +0.219] and does not clear zero, while the same pair clears it
comfortably on NDCG. A single marginal interval here is not evidence.

Precision falls as K rises for every variant, which is expected when relevant
chunks concentrate near the top of a ranking, and is not a defect.

### Provenance

- Reference set: 50 queries, 1,422 judgements, checksum
  `4920b9e2b936d238ae61f4eb96b87ae1`. Corpus: 3 notebooks, 925 chunks.
- Production `bge-m3` vectors unchanged throughout, checksum
  `809bdb448d6a187f7d0cd16cbe48db9a`.
- `vector_search` gained a deterministic `(doc_id, ordinal)` tie-break before
  the run. Verified inert on this data: the recorded top-10 reproduced with the
  same set **and** the same order for all 50 queries.
- The two challenger embedding sets reproduce the rankings their pooled
  candidates were drawn from exactly — same set and same order, 50/50 each — so
  the retrievers scored are the retrievers that built the pool.
- Mixed 1024d and 384d vectors verified to coexist without interference, using
  two models given deliberately opposite preference orders. A cross-dimension
  query raises rather than returning a plausible wrong ranking.
- The metric implementation was cross-checked against `trec_eval` 9.0.8 via
  `pytrec_eval` 0.5; see the decisions log in `docs/ROADMAP.md`.

### What was not done

No parameter was tuned. No threshold or K was chosen after seeing a number. No
retriever was modified in response to its score. Hybrid retrieval was not built.

Reproduce with:

```bash
uv run python scripts/evaluate_retrieval.py --unjudged zero
```

---

## `hybrid_20260907T071245Z.json`

Reciprocal rank fusion of `bge-m3` production vector search and lexical search,
scored against the same 1,422-judgement reference set as the four-way benchmark.

### Pre-registered before the run

The full method — fusion rule, constants, baseline, primary metric, and the
decision rule for reading the interval — was fixed in `docs/HYBRID-PROTOCOL.md`
and committed as `59e5918` on 2026-09-06T10:29Z. This run is stamped
2026-09-07T07:12Z, so the pre-registration precedes the result in the history
and can be checked rather than taken on trust.

The constants are module-level in `scripts/evaluate_hybrid.py` with no
command-line override, and
`tests/test_hybrid_fusion.py::test_the_frozen_constants_are_what_the_protocol_declares`
fails if any of them drifts from what the protocol declares.

| parameter | value |
|---|---|
| fusion | reciprocal rank fusion, unweighted |
| `k` | 60 |
| input depth | 10 from each retriever |
| output depth | 10 |
| baseline | `bge-m3 (production)` |
| primary outcome | NDCG@10, single metric |
| interval | paired bootstrap, 10,000 resamples, seed 20260906 |

### How unjudged candidates were treated

**`zero` policy, declared in the protocol before the run. It made no
difference:** all three variants returned 0.00 unjudged candidates per query in
the top 10, across all 50 queries. `zero` and `skip` are provably identical
here, so no number below depends on the choice.

The fused ranking draws only from two retrievers whose top 10 was already fully
judged, so fusion could not introduce an unjudged chunk at this depth.

### Primary outcome

**NDCG@10, hybrid minus `bge-m3` production: −0.0183, 95% CI
[−0.0525, +0.0184], n = 50.**

The interval includes zero. By the decision rule fixed in the protocol, this is
a **null result**: hybrid retrieval is not shown to be better than production
`bge-m3`, and is not shown to be worse.

### Means, 95% bootstrap CI

| metric | hybrid (RRF k=60) | bge-m3 (production) | lexical |
|---|---|---|---|
| NDCG@10 | 0.5944 | 0.6127 | 0.4893 |
| NDCG@5 | 0.5402 | 0.5850 | 0.4473 |
| NDCG@1 | 0.6000 | 0.6300 | 0.4700 |
| MRR (>=1) | 0.7888 | 0.8118 | 0.7037 |
| MRR (=2) | 0.6524 | 0.6393 | 0.5229 |
| Recall@10 (>=1) | 0.5864 | 0.5832 | 0.4812 |
| Recall@10 (=2) | 0.7164 | 0.7230 | 0.5588 |
| P@5 (>=1) | 0.5520 | 0.5960 | 0.4640 |

The `bge-m3` and `lexical` columns reproduce the four-way benchmark exactly —
NDCG@10 of 0.6127 and 0.4893 — which is a consistency check on the harness
across two independent runs, not a second measurement.

`n = 50` throughout except recall at threshold 2, which is `n = 47`: q41, q45
and q51 have no grade-2 chunk in their pool, so recall is undefined and they are
excluded rather than counted as zero.

### Secondary comparisons — descriptive only, no decision weight

The protocol names one primary outcome. Everything below was computed and is
recorded for completeness; none of it carries decision weight, and none of it
may be substituted for the primary metric after the fact.

Hybrid minus `bge-m3` production:

| metric | difference | 95% CI | separable |
|---|---|---|---|
| NDCG@5 | −0.0449 | [−0.0942, +0.0047] | no |
| NDCG@1 | −0.0300 | [−0.1500, +0.0900] | no |
| MRR (>=1) | −0.0229 | [−0.0989, +0.0546] | no |
| MRR (=2) | +0.0131 | [−0.0756, +0.1027] | no |
| Recall@10 (>=1) | +0.0032 | [−0.0443, +0.0489] | no |
| Recall@10 (=2) | −0.0066 | [−0.0681, +0.0567] | no |

Not one of the six separates from zero. The primary outcome is not a lone null
surrounded by signal.

Hybrid minus `lexical` separates on five of six — NDCG@5 +0.0928
[+0.0463, +0.1433], recall@10 at threshold 2 +0.1576 [+0.0691, +0.2534] — which
says the fusion inherits the vector retriever's advantage over lexical search,
not that the fusion adds anything to the vector retriever.

Six pairs are compared per metric at 95% confidence each, so the family-wise
error rate is well above 5%.

### What was not done

The protocol forbids changing anything in response to this result, and nothing
was changed. `k` was not adjusted, weights were not introduced, depth was not
altered, the primary metric was not switched, and no second fusion variant was
run. The result is reported as it came out.

Production retrieval stays `bge-m3` vector search. An inseparable difference is
not a reason to add a second retriever, a fusion step, and a constant to the
production path.

This is a null result on this corpus at this sample size. It is not evidence
that hybrid retrieval is useless in general, and the earlier complementarity
diagnostic — an oracle union recall of 0.7958 against 0.5832 for vector alone —
still stands as an upper bound that some other fusion might approach. That bound
was never an expected gain, and RRF at these settings did not approach it.

### Provenance

- Reference set: 50 queries, 1,422 judgements, checksum
  `4920b9e2b936d238ae61f4eb96b87ae1`, verified unchanged before the run.
- Production `bge-m3` vectors unchanged, checksum
  `809bdb448d6a187f7d0cd16cbe48db9a`.
- 232 tests passing, `ruff check`, `ruff format --check` and `mypy src/` clean at
  the time of the run.
- Scored by the same `daedalus.evaluation.harness` and
  `daedalus.evaluation.metrics` as the four-way benchmark, unmodified.

Reproduce with:

```bash
uv run python scripts/evaluate_hybrid.py --unjudged zero
```

---

## `questions_20260907T140310Z.json`

The Phase 6 question generation run: one question attempted for each of the 300
sections in the frozen draw. Protocol `docs/PHASE-6-PROTOCOL.md`, contract
`p6-v2`, `params_hash 257d63813c72d12de6f79c5f9b623240`.

This artifact records **generation**, not quality. No question in it has been
judged. The groundedness, interview-relevance and difficulty rates the phase
exists to produce come from the human-labelled benchmark, which is separate work
and is not reported here.

### Outcome

| outcome | sections |
|---|---|
| accepted | **228** |
| rejected | **72** |
| transport failure | 0 |
| total | 300 |

All 300 selected sections carry exactly one outcome; none was attempted twice.
Every stored row is `p6-v2` — no output from the withdrawn `p6-v1` contract
contributes (see protocol section 18).

Coverage context, always reported together: 300 of 337 eligible sections
selected (89.0%), 337 of 408 sections eligible (82.6%), 300 of 408 generated
from (73.5%).

Accepted questions by requested type and difficulty:

| type | easy | medium | hard | total |
|---|---|---|---|---|
| conceptual | 34 | 39 | 33 | 106 |
| explanation | 19 | 20 | 20 | 59 |
| code_reasoning | 10 | 14 | 10 | 34 |
| comparison | 13 | 8 | 8 | 29 |

### Every rejection was the quote check

All 72 rejections were `quote_not_found`. There were **no** schema violations,
no type or difficulty mismatches, no unknown citations, and no transport
failures across 295 model calls. The model conformed to every part of the
contract except reproducing its quotation verbatim.

Rejection rate by requested type and difficulty:

| type | rate | difficulty | rate |
|---|---|---|---|
| code_reasoning | 16/50 = 32.0% | easy | 24/100 = 24.0% |
| explanation | 23/82 = 28.0% | medium | 19/100 = 19.0% |
| comparison | 11/40 = 27.5% | hard | 29/100 = 29.0% |
| conceptual | 22/128 = 17.2% | | |

These are observed counts on 300 sections of one corpus, not established
effects. No significance test was run and none is claimed.

### What the 72 rejections were — corrected diagnostic

Every rejection stores the exact response that produced it, so the 72 were
classified by string matching against their own section, with no further model
calls.

**A first pass of this analysis was wrong and was discarded.** It classified
matches by longest contiguous common substring, which misread quotes where the
model had removed spaces inside LaTeX — `$$\text{Question}\rightarrow` against a
source reading `$$ \text{Question} \rightarrow` is the same text, and was
counted as absent from the material. The classifier was rebuilt around
whitespace-insensitive matching and an overall similarity ratio. The figures
below are from the corrected version.

| classification | n | share of rejections |
|---|---|---|
| not traceable to the material | 45 | 62.5% |
| code-fence artifact | 11 | 15.3% |
| whitespace only | 8 | 11.1% |
| reworded from the material | 6 | 8.3% |
| markdown-marker artifact | 1 | 1.4% |
| quoted an uncited chunk | 1 | 1.4% |

Twelve rejections (16.7%) trace to ingestion artifacts — stray markdown fences
and heading markers the notebook parser left embedded in prose. Eight more
(11.1%) differ from their source by whitespace alone.

### The untraceable quotes

The 45 quotes in the first row were checked against **every chunk in the corpus**,
not only their own section:

```
found verbatim elsewhere in the corpus:  0
found nowhere in the corpus at all:      45
similarity to best-matching chunk:       median 0.198, max 0.481
```

They are not misattributions or quotes lifted from a neighbouring section. The
model produced plausible prose in the register of the material and presented it
as a verbatim quotation.

**Stated precisely: on this corpus, with `qwen3:8b` under the `p6-v2` contract,
45 of 300 attempted sections (15.0%) yielded a `grounding_quote` that appears
nowhere in the 925 chunks.**

That figure is the rate at which one field of one contract was fabricated, on
one corpus, at one model size, with one prompt. It is **not** a general
hallucination rate for the model, and it says nothing about how often the
*questions* are ungrounded — a question can be sound while its stated quotation
is invented, and the reverse. Nothing here should be quoted as a hallucination
measurement.

What it does support is narrower and firmer: the verbatim-quote rule is not
ceremony. It removed 45 questions whose stated evidence did not exist, and a
fabricated quotation is precisely the failure a judge reading the question alone
would be least likely to catch.

The classification thresholds (similarity 0.90 and 0.55) are chosen, not
measured. The 45-of-45 corpus-wide absence is a string-matching fact and does
not depend on them; the split between "reworded" and "not traceable" does.

### Provenance

- Contract `p6-v2` throughout, one question per section, no regeneration and no
  repair. A rejected section was never retried.
- Rejections carry their raw response, which is what made this diagnostic
  possible without calling the model again.
- Phase 5 store unchanged: 1,422 judgements, checksum
  `4920b9e2b936d238ae61f4eb96b87ae1`, 925 chunks, 2,775 embeddings.
- The classification script is not committed; it is a diagnostic, and its inputs
  are the stored responses rather than anything derived.

Reproduce the generation with:

```bash
uv run python scripts/generate_questions.py
```

Completed sections are skipped, so on the current store this is a no-op.

---

## Phase 6 human labelling — three passes

Labelled in the database rather than to a file: 684 rows in `question_labels`,
228 per rubric, one per accepted question, no item unlabelled on any rubric.

| pass | rubric | when | span |
|---|---|---|---|
| 1 | groundedness | 2026-09-08 | 15:34:57 – 16:27:50 |
| 2 | interview relevance | 2026-09-09 | 10:02:19 – 11:04:48 |
| 3 | difficulty | 2026-09-09 | 12:30:48 – 14:10:15 |

Each pass ran in its own shuffled order under the seeds fixed in
`docs/PHASE-6-RUBRICS.md` section 7. The order actually labelled, reconstructed
from `labelled_at`, matches the order the seed produces for all three passes.

### Corrections, recorded as section 7 step 5 requires

**Two tooling defects, both found by the labeller, both fixed before the labels
they affected were recorded.**

1. `f full text` was offered on chunks with nothing hidden, so it read as a
   broken key rather than an empty one. Four groundedness labels had already
   been recorded and were deleted at the labeller's instruction; the pass
   restarted from zero. The discarded grades were selection rank 233 grade 2,
   rank 147 grade 2, rank 9 grade 0 support, rank 67 grade 0 support. This was a
   tooling defect, not a change of judgement.

2. Pressing `?` for the rubric reminder printed it and then immediately
   repainted the question over it, so the reminder was never readable. Found at
   the start of the relevance pass, with no relevance label yet recorded, and
   fixed before that pass began. No label was affected.

**Two mis-key corrections during the difficulty pass**, both on the labeller's
explicit instruction, both relabelled from the same position in the same order:

| positions | rows deleted | 
|---|---|
| 44–48 | 5 |
| 144–161 | 18 |

The discarded grades were preserved before deletion and were not displayed back
to the labeller, so the relabelling was not anchored on them.

### The `unusable` coincidence expectation does not fully hold

Rubric 3 states that `unusable` should coincide with relevance 0. Measured
across the completed passes, 8 of the 11 `unusable` items are also relevance 0.
Three are not: one at relevance 2 and two at relevance 1.

The 57 items that are relevance 0 without being `unusable` are not exceptions.
Relevance 0 also covers trivial questions and notebook mechanics, which are
coherent enough to carry a difficulty; the expectation runs in one direction
only.

The three exceptions were audited after the pass. None was a fast decision —
10.65 s, 7.56 s and 8.39 s against a pass median of 8.83 s — and none fell in
either range that was relabelled for mis-keys. The labeller reviewed all three
and confirmed they are considered judgements, not slips.

So the expectation is an approximation rather than an invariant: a question can
be answerable in principle and still not sit anywhere meaningful on a difficulty
scale. This is recorded as a limitation of the rubric as written. The rubric was
not changed, and the three labels stand.

---

## Phase 6 results — human labels and groundedness judging

Two measurements, reported against the denominators
`docs/PHASE-6-PROTOCOL.md` fixes and kept apart throughout: **300 sections** is
the generation denominator, **228 accepted questions** the evaluation one. A
rejected section is an absent question, not an ungrounded one.

### The four pre-registered criteria, all missed

Thresholds were fixed in `docs/PHASE-6-RUBRICS.md` section 6 before any label
was recorded, and are not restated here in weakened form.

| criterion | threshold | measured | outcome |
|---|---|---|---|
| grounded, grade 2 | >= 90% | 131/228 = 57.5% | missed |
| unsupported, grade 0 | > 10% fails | 59/228 = 25.9% | fails |
| interview-relevant, grade 2 | >= 80% | 27/228 = 11.8% | missed |
| difficulty match | >= 80% | 100/217 = 46.1% | missed |

Protocol section 15 anticipates this: a null or a failure is a completed
outcome, reported as measured rather than repaired until the number improves.
Nothing in the protocol, the rubrics, the thresholds, the 228 questions or the
72 rejections was changed after these numbers were seen.

### Distributions

| grade | groundedness | relevance |
|---|---|---|
| 0 | 59 (25.9%) | 65 (28.5%) |
| 1 | 38 (16.7%) | 136 (59.6%) |
| 2 | 131 (57.5%) | 27 (11.8%) |

| difficulty | easy | medium | hard | unusable |
|---|---|---|---|---|
| count | 94 (41.2%) | 99 (43.4%) | 24 (10.5%) | 11 (4.8%) |

The relevance result is not a scatter of bad questions. The mass is at grade 1 —
136 of 228, which the scale defines as *on topic and sensible, but answerable by
restating a sentence*. The generator produces questions about the right subject
that test recall rather than understanding.

### Groundedness failure modes

Every question graded 0 or 1 carries which failure it was, and the split is
close to even.

| grade | citation failure | support failure | total |
|---|---|---|---|
| 0 | 33 | 26 | 59 |
| 1 | 16 | 22 | 38 |
| both | 49 (50.5%) | 48 (49.5%) | 97 |

Half of the groundedness failures are questions whose supporting material exists
in the section but was not cited. That is a retrieval-and-citation failure rather
than a generation failure, and the two are worth separating because they have
different fixes.

### Difficulty control

Requested difficulty against the human label, `unusable` excluded as section 5
requires, so the denominator is 217 and the 11 exclusions are stated rather than
absorbed.

| requested / labelled | easy | medium | hard |
|---|---|---|---|
| easy | 43 | 24 | 2 |
| medium | 40 | 37 | 2 |
| hard | 11 | 38 | 20 |

100 of 217 match. The generator's requested difficulty is assigned by
`(rank - 1) mod 3` and is recoverable from selection rank with 100% accuracy, so
the request is a label the generator was given, not a property it inferred. What
this measures is whether it could act on that instruction.

### A structural defect: pointers into the prompt scaffolding

**57 of 228 questions (25.0%) refer to something the candidate cannot see** —
`chunk 170`, "the SEED chunk", "this chunk". Counted by pattern over the stored
question text:

| pattern | questions |
|---|---|
| `chunk <number>` | 39 |
| "SEED chunk" | 14 |
| "this/the/given/provided/cited chunk" | 5 |
| distinct questions matching any | 57 |

101 questions contain the word "chunk" at all, but 44 of those use it as a
domain term — chunking is the subject matter — and are not defective. Only the
57 point at the scaffolding. A separate 69 questions carry "based on the material
provided" and similar, addressing the prompt's framing rather than a candidate;
only 16 of those also carry a pointer.

The class is strongly associated with relevance 0: 71.9% of the 57, against
14.0% of the other 171. Chi-square on a 2x2 of grade 0 against the rest is 70.30
on 1 degree of freedom, p = 5.1e-17, odds ratio 15.7, and 63.1% of all
relevance-0 questions carry a pointer. Groundedness and difficulty show no
comparable difference.

Two limits on that association. The labels were **not blind to the class** — the
pattern was noticed during the relevance pass and discussed during the
difficulty pass — so this is an association measured by a labeller aware of it,
not an independent confirmation. And the mechanism is a **hypothesis**: the
prompt labels chunks `--- chunk N (SEED) ---`, which the model appears to quote
back, but all 228 questions are `p6-v2` and the five `p6-v1` responses were
deleted, so there is no contrast corpus to test it against.

It does not explain the relevance failure. Had the 57 behaved like the other
171, grade 2 would land near 12.9% rather than 11.8% — arithmetic on an
assumption, not a measurement, but enough to locate the failure in the 171 clean
questions rather than in the defective ones.

### Labelling pace

Sub-2-second decisions are reported rather than silently accepted, as in Phase 4.

| pass | gaps | median | fastest | sub-2s | sub-5s |
|---|---|---|---|---|---|
| groundedness | 227 | 11.68 s | 1.539 s | 4 | 13 |
| relevance | 227 | 9.31 s | 5.701 s | 0 | 0 |
| difficulty | 227 | 8.83 s | 6.397 s | 0 | 0 |

Three of the four sub-2-second groundedness decisions were grade 2, the shortest
path through the loop, since only 0 and 1 ask a follow-up. The difficulty gaps
span two deletions and two restarts, so the largest gaps are interruptions
rather than deliberation; the sub-2s and sub-5s counts are unaffected, since a
restart can only inflate a gap.

---

## Phase 6 — automated groundedness judging

Contract `j6-v2`, params `4c1a74610c2a9b84501b33627faf90a4`, temperature 0, one
rubric per call, seed fixed per question and rubric. The judge is shown the
question and its cited chunks with the same fields withheld that the labeller
had withheld, and reads the same rubric text, less the block instructing the
labeller which key to press for a failure mode.

Two judges, chosen so that a bad result would be interpretable: `qwen3:8b`,
which is also the generator, and `llama3.2`, which is not.

### Both judges answer "2" almost regardless of the question

| grade | human | qwen3:8b | llama3.2 |
|---|---|---|---|
| 0 | 59 | 6 | 0 |
| 1 | 38 | 3 | 25 |
| 2 | 131 | 219 | 198 |
| off scale | — | 0 | 5 |

| judge | n | raw | weighted kappa | kappa 95% CI |
|---|---|---|---|---|
| qwen3:8b | 228 | 0.6009 | 0.1201 | 0.0487 – 0.1985 |
| llama3.2, off-scale excluded | 223 | 0.5381 | 0.0559 | −0.0020 – 0.1180 |
| llama3.2, off-scale as disagreement | 228 | 0.5263 | withheld | — |

Kappa is linearly weighted over the ordered scale; a chance-corrected figure is
withheld where off-scale answers are kept, because there is no expected rate for
a category the scale does not contain. Intervals are percentile bootstrap over
questions, 10,000 resamples, seed 20260909. `llama3.2`'s interval includes zero:
its agreement is not distinguishable from chance.

Raw agreement alone would have been misleading. **A rater that ignored the input
and answered "2" every time would score 131/228 = 57.5%.** `qwen3:8b` scored
60.1% — it beat a constant answer by six questions.

### The errors are one-sided

`qwen3:8b` against the human:

| | judge 0 | judge 1 | judge 2 |
|---|---|---|---|
| human 0 | 6 | 3 | 50 |
| human 1 | 0 | 0 | 38 |
| human 2 | 0 | 0 | 131 |

Every cell below the diagonal is empty. The judge was more generous than the
human on 91 of 228 questions and harsher on none. `llama3.2` is generous on
41.7% and harsher on 4.5%.

**Of the 59 questions the human graded unsupported, `qwen3:8b` identified 6 and
`llama3.2` identified none.**

### Self-preference is not supported as the explanation

The second judge was run to separate a model favouring its own output from a
model that is simply lenient. `llama3.2` did not write these questions, shares no
lineage with the generator, and is **more** generous than `qwen3:8b`, not less.
The leniency is therefore not attributable to self-preference on this evidence.

The two judges agree with each other far more than either agrees with the human —
raw 0.8924 on the 223 both scored on scale — but that is largely degeneracy: two
raters who almost always answer "2" agree by construction, and their weighted
kappa is 0.3281.

### What this supports, and what it does not

It supports a narrow, firm claim: **on this corpus, with this prompt, at
temperature 0, neither local judge can detect an ungrounded question.** Automated
groundedness scoring cannot substitute for the human labels here, and any
automated groundedness rate quoted from these models would be close to a constant.

It does not support a general claim about LLM judges, about larger models, or
about groundedness being unjudgeable. Two models at 8B and 3B on consumer
hardware, one run each, is the whole of the evidence.

### Off-scale output, kept rather than suppressed

The response schema requires a JSON string, deliberately not an enum, because
protocol section 7 step 7 records an off-scale answer as a finding about the
judge. `qwen3:8b` produced none in 228 calls. `llama3.2` produced five: three
bare `"}"` and two prose answers, both of which read as the judge reasoning about
absent support and then failing to emit a grade. On two items that is an
observation, not a pattern — but an enum would have hidden it.

Throughput, measured: `qwen3:8b` 228 calls in 847 s (3.7 s each), `llama3.2` 228
in 353 s (1.5 s each).

**Relevance and difficulty judging is still running and is deliberately not
reported here.**

---

## Phase 6 — relevance and difficulty judging

Same contract `j6-v2`, same two judges, same 228 questions. Throughput measured:
`qwen3:8b` 882 s and 851 s for the two rubrics, `llama3.2` 349 s and 339 s. No
reply was unparsable in any of the six passes.

### Agreement across the full grid

Kappa is linearly weighted on the ordinal scales and unweighted once `unusable`
is included, since a question too incoherent to place is not a fourth degree of
difficulty. Intervals are percentile bootstrap over questions, 10,000 resamples,
seed 20260909.

| rubric | judge | n | off scale | raw | weighted kappa | kappa 95% CI |
|---|---|---|---|---|---|---|
| groundedness | qwen3:8b | 228 | 0 | 0.6009 | 0.1201 | 0.0487 – 0.1985 |
| groundedness | llama3.2 | 223 | 5 | 0.5381 | 0.0559 | −0.0020 – 0.1180 |
| relevance | qwen3:8b | 228 | 0 | 0.1184 | **−0.0013** | −0.0045 – 0.0000 |
| relevance | llama3.2 | 194 | 34 | 0.5825 | 0.1069 | 0.0128 – 0.2027 |
| difficulty | qwen3:8b | 217 | 0 | 0.6129 | **0.3190** | 0.2187 – 0.4169 |
| difficulty | llama3.2 | 216 | 1 | 0.5741 | 0.2809 | 0.1741 – 0.3840 |

Difficulty with `unusable` kept as a fourth category, unweighted: `qwen3:8b`
0.5833 raw and 0.2716 kappa on 228; `llama3.2` 0.5463 and 0.2126 on 227. Keeping
off-scale answers as disagreements instead of dropping them moves `llama3.2`
relevance from 0.5825 to 0.4956 raw, and withholds kappa.

### Raw agreement is worthless on its own, and this measures how worthless

`qwen3:8b` answered "2" on 227 of 228 relevance questions, and "2" on 219 of 228
groundedness questions. Near-identical behaviour on both rubrics. Its raw
agreement was **60.1% on groundedness and 11.8% on relevance.**

The difference is entirely in the human's marginal, not in the judge. The human's
modal groundedness grade is 2, so a constant "2" looks competent; the human's
modal relevance grade is 1, so the same constant looks useless. On relevance the
judge scores exactly the human's grade-2 rate, 27/228 = 11.8%, because that is
the only way a constant "2" can be right.

Weighted kappa reports that behaviour as −0.0013, with an interval whose upper
bound is 0.0000: **worse than chance, and not distinguishable from it.**

### Neither judge will use the top or bottom of a scale

| | human | qwen3:8b | llama3.2 |
|---|---|---|---|
| relevance 0 | 65 | 1 | 2 |
| relevance 2 | 27 | 227 | 27 |
| difficulty hard | 24 | **0** | 1 |
| groundedness 0 | 59 | 6 | 0 |

`qwen3:8b` never once answered "hard" across 228 difficulty judgements;
`llama3.2` did so once. Both judges' difficulty errors are almost entirely the
collapse of hard into medium — of the human's 24 hard questions, `qwen3:8b`
called 22 medium and 2 easy.

Difficulty is nonetheless the rubric where agreement is highest, at kappa 0.319
and 0.281. Both intervals exclude zero, so the agreement is real, but it is
carried by the two categories the judges are willing to use.

### The judges do not track the generator's intent either

`docs/PHASE-6-RUBRICS.md` section 5 records that the difficulty definitions given
to the judge are the same ones given to the generator, and warns that agreement
would then partly measure a shared definition. That is testable against the
requested difficulty directly:

| rater | matches requested difficulty |
|---|---|
| human | 100/217 = 46.1% |
| qwen3:8b | 100/228 = 43.9% |
| llama3.2 | 87/226 = 38.5% |

Both judges match the generator's request **less** often than the human does, so
the low human agreement is not explained by the judges having inherited the
generator's difficulty prior. The shared definition did not produce a shared
answer.

### Off-scale output, characterised

`llama3.2` produced 34 off-scale relevance answers, all format failures rather
than refusals: `>1` thirteen times, `> 1` five, `Issue 2` five, `}` three, `>2`
twice, and one reply that echoed a question — "What does BERT stand for?" —
instead of grading one. `qwen3:8b` produced none in 684 calls across three
rubrics.

That asymmetry is itself a result: the larger judge is reliable at emitting a
value and unreliable at choosing it, and a schema enforcing an enum would have
made the two look identical.

### What the six passes support

**No automated score from either model can substitute for the human labels on
this corpus.** The strongest agreement measured anywhere in the grid is kappa
0.319, on the one rubric where both judges refuse the top category. Relevance
judging by `qwen3:8b` is worse than chance. Groundedness judging identifies 6 of
59 unsupported questions at best.

Criterion 5 of the protocol required agreement to be measured before any
automated score was quoted. It has been, and the answer is that no automated
score should be quoted. That is a completed outcome, not a blocked one.

The limits stand as stated for groundedness: two models at 8B and 3B, on
consumer hardware, one run each at temperature 0, one prompt, one corpus of three
documents. Nothing here measures whether a larger judge would do better, and
nothing here establishes difficulty validity — `docs/PHASE-0.md` already records
that validity requires learner performance data, which this project does not have.

---

## Phase 6 — structural coverage

Protocol section 14 defines structural coverage as selected sections producing at
least one question that passes the deterministic checks, over the **300 selected
sections**, and requires three further figures beside it so the denominator is
never mistaken for the corpus. Duplicate detection is excluded from the
definition.

| figure | value | |
|---|---|---|
| **structural coverage** | **228/300** | **76.0%** |
| selected of eligible | 300/337 | 89.0% |
| eligible of all sections | 337/408 | 82.6% |
| selected of all sections | 300/408 | 73.5% |

Derived from the same counts, and the most conservative statement of how much of
the corpus carries a question: **228 of 408 sections, 55.9%.** The 180 sections
without one are 71 never eligible (56 thin, 15 prose-free), 37 eligible but not
drawn, and 72 drawn but rejected.

Section 18 item 5 records why the denominator is 300 rather than 337: the
approved decision defined coverage over eligible sections, but the benchmark
generates from 300 of them, and a denominator of 337 would cap coverage at 88.7%
for a reason unrelated to generator quality. The narrowing was approved on
2026-09-07 before generation.

### The selection reproduces exactly

The 300 selected sections were recomputed from `daedalus.generation.selection`
against the current store and compared with what the run actually attempted:

- 228 selected sections hold a question, 72 hold a rejection, and the two sets
  are disjoint and sum to 300.
- **No question comes from a section outside the selection**, and no selected
  section is unaccounted for.
- Section counts recompute unchanged: 408 total, 337 eligible, 56 thin, 15
  prose-free.

So the frozen sample is reproducible from committed code and the current
database, not merely recorded.

### The 72 uncovered sections

All 72 were rejected for the same reason, `quote_not_found` — the diagnostic is
in the generation section above. Coverage is therefore limited by the verbatim
quote check rather than by schema failures, type or difficulty mismatches, or
transport errors, of which there were none.

A rejected section is an **absent** question, not an ungrounded one. The 72 are
part of the 300-section coverage denominator and take no part in any rate
computed over the 228 accepted questions.
