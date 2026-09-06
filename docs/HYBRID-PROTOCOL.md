# Hybrid retrieval — experiment protocol

Pre-registered. This document is committed **before** the experiment is run, so
that what was decided in advance is separable from what the result turned out to
be.

Every parameter here is either fixed by an external source or forced by the
reference set. None is estimated from the 50 benchmark queries.

Background: `results/README.md` holds the four-way benchmark this compares
against, and `docs/ROADMAP.md` records why hybrid was deferred until measured
failure modes justified it.

---

## Why run this at all

The per-query diagnostic between production `bge-m3` and lexical search found
complementarity that is near-universal rather than driven by outliers:

- No query of the 50 has identical relevant sets under the two retrievers.
- 36 of 50 queries have **both** retrievers contributing relevant chunks the
  other misses; 8 have only vector adding, 6 only lexical.
- At K=10, threshold >= 1: 112 relevant chunks found by both, 143 by vector
  alone, 100 by lexical alone.
- Oracle union recall at K=10 is 0.7958 against vector's 0.5832 — a headroom of
  +0.2127 at threshold 1, +0.1558 at threshold 2.

The oracle figure is an upper bound on what any fusion could reach, assuming
perfect selection from the union. It is not an expected gain and not a target.

---

## Fusion algorithm

**Reciprocal Rank Fusion**, Cormack, Clarke & Buettcher, SIGIR 2009.

```
score(d) = sum over retrievers r that returned d of  1 / (k + rank_r(d))
k = 60
```

Vector search ranks by cosine distance and lexical by `ts_rank`. Those scales are
not comparable, so a weighted score combination would need both a normalisation
scheme and a weight, each of which would have to be fitted on this benchmark.
RRF consumes only ranks, so neither choice arises.

Fusion is **unweighted**. Weighted RRF exists and may well do better. It has a
free parameter and there is no held-out split on which to set it, which is the
only reason it is excluded.

### A consequence of k=60 at depth 10, recorded in advance

A chunk returned by both retrievers scores at least `2/70 = 0.0286`. A chunk
returned by one scores at most `1/61 = 0.0164`. So **every chunk both retrievers
returned outranks every chunk only one returned**, whatever their ranks. At this
depth RRF orders agreed chunks first by rank sum, then singletons by rank.

Mean overlap between the two lists is 3.7 of 10, so roughly the top four
positions will be consensus chunks and ranks 5–10 singletons. About
three-quarters of each retriever's unique contribution sits at ranks 4–10, which
is where the singletons land.

Whether consensus-first ordering helps is what this experiment measures. `k` is
not being adjusted to avoid this behaviour; doing so would be tuning against the
benchmark.

---

## Candidate depth

**Exactly 10 from each retriever.** This is a constraint, not a choice. The
reference set judged each pool-building retriever to depth 10, so at depth 11 a
retriever returns chunks nobody assessed, which would score zero by default —
the bias the 358-judgement pool expansion existed to remove.

Verified on the frozen rankings before writing this: across 50 queries the union
of the two depth-10 lists holds 815 distinct candidates, union size 11 to 20 per
query, mean 16.3, and **0 of them are unjudged**. Any fusion over these lists is
therefore fully judged at every K <= 10, whatever the fusion rule. Neither
retriever returns a duplicate within its own list.

Output depth is 10, matching the frozen benchmark.

---

## Parameters, frozen

| parameter | value | source |
|---|---|---|
| RRF `k` | 60 | published default, Cormack et al. 2009 |
| input depth per retriever | 10 | forced by judged-pool depth |
| output depth | 10 | matches the frozen benchmark |
| retriever weights | none | a weight would be a fitted parameter |
| retrievers fused | `bge-m3` production, `lexical` | the two that built the pool |
| bootstrap resamples | 10,000 | matches the frozen benchmark |
| bootstrap seed | 20260906 | matches the frozen benchmark |
| unjudged policy | `zero` | declared; verified inert, coverage is complete |

`k = 60` was tuned by its authors on TREC collections. It is a fixed external
constant, not a value fitted here, which is the reason for taking a published
default rather than choosing one.

---

## Tie-breaking

Sort key: `(-rrf_score, doc_id, ordinal)`, the same convention as
`vector_search` and the challenger rankings.

Ties are expected rather than hypothetical: chunks at ranks (2,5) and (3,4)
across the two lists receive identical scores. Without a deterministic key the
fused ranking would not be reproducible.

---

## Duplicates

A chunk returned by both retrievers receives one term per retriever, summed.
That is the mechanism rather than a special case. Fusion is computed over a
mapping keyed by `(doc_id, ordinal)`, so each chunk appears once in the fused
list. `metrics.py` rejects a ranking containing a repeated reference, so a
de-duplication failure raises rather than quietly inflating a score.

---

## Metrics reported

Computed by the unmodified `daedalus.evaluation.metrics` through
`daedalus.evaluation.harness`, identically to the four-way benchmark.

- NDCG at K = 1, 3, 5, 7, 10 — graded 0/1/2, linear gain
- Precision at the same K, at thresholds >= 1 and = 2
- Recall at the same K, at thresholds >= 1 and = 2
- Reciprocal rank at thresholds >= 1 and = 2
- `unjudged@10`, expected 0.00, reported as evidence rather than assumed

Recall at threshold 2 reports n = 47. Queries 41, 45 and 51 have no grade-2
chunk in their pool, so recall is undefined for them and they are excluded
rather than counted as zero.

---

## Statistical comparison

**Primary outcome, declared here: the paired difference in NDCG@10 between the
hybrid and `bge-m3` production.** Paired bootstrap over queries, 10,000
resamples, seed 20260906, 95% percentile interval.

One primary comparison. Reporting thirty intervals and highlighting whichever
excluded zero is the failure mode this guards against.

### Decision rule, fixed in advance

| interval | reading |
|---|---|
| excludes zero, difference positive | hybrid measurably better on this benchmark |
| excludes zero, difference negative | hybrid measurably worse |
| includes zero | **null result**, recorded as such |

A null result is a null result. Not "promising", not "trending", not "worth
another configuration".

Secondary and descriptive only, reported in full but carrying no decision
weight: hybrid against lexical; every other metric and K; and the
harvested/authored split, where the diagnostic found lexical contributing 65
unique relevant chunks on harvested queries against 35 on authored.

---

## Why this is not tuning on the benchmark

1. No parameter is estimated from the 50 queries. `k` is external, depth is
   forced by the judged pool, there are no weights.
2. One variant is run, once. Not a family from which the best is reported.
3. The primary outcome and the decision rule are declared in this document,
   committed before the run.
4. The result is reported whichever way it falls, as the heading-prefix null
   result was.

**Forbidden, because each would make this tuning:** trying several values of `k`
and reporting the best; adding retriever weights; changing depth after seeing a
result; switching the primary metric or K after the fact; or adjusting anything
between running the fusion and reporting its number.

**After this document is committed, nothing changes on the basis of the hybrid
result before that result has been reported.**

---

## Limitations, stated before the number exists

- The +0.2127 oracle headroom is not an expected gain. It assumes perfect
  selection from the union; RRF has to rank, and can push a relevant chunk down
  while pulling an irrelevant one up.
- Unweighted fusion gives equal standing to a measurably weaker retriever —
  lexical scores NDCG@10 0.4893 against production's 0.6127. Dragging the
  stronger retriever down is a real risk of this protocol, and is not grounds
  for adding weights afterwards.
- 50 queries over one three-notebook corpus, with no held-out split. Untuned or
  not, the estimate is of this corpus and this query set.
- Consensus-first ordering means most of the measured complementarity sits at
  ranks 5–10, where singletons land.

---

## Reproducing

```bash
uv run python scripts/evaluate_hybrid.py --unjudged zero
```
