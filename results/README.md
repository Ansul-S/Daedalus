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
