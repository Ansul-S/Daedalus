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
