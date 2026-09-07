# Phase 6 — question generation protocol

Pre-registered. This document is committed **before** any question is generated,
so that what was decided in advance is separable from what the output turned out
to be. It follows the precedent set by `docs/HYBRID-PROTOCOL.md`.

Every threshold here is either forced by a measurement of the corpus or is a
reasoned choice with its rationale recorded. Where a number is a judgement call
rather than something the data determined, this document says so.

Background: `docs/PHASE-0.md` fixes the success criteria and the model
configuration, `docs/ROADMAP.md` records the phase order, and `results/README.md`
holds the Phase 5 retrieval benchmarks this phase does not touch.

---

## 1. What this phase delivers

A generator that produces grounded interview questions from the user's own
material, with references back to the source chunks, and a labelled benchmark
that measures whether it works.

Out of scope, deferred in `docs/FEATURES.md` and not built here: interview
sessions, answer submission, answer evaluation, adaptive difficulty, scoring,
MCQ/MSQ/NAT modes, follow-up questions, source correction, and any UI or API.

---

## 2. Generation unit and section eligibility

The generation and coverage unit is the **heading section**, identified by
`(doc_id, heading_path)`. Topic boundaries stay explicit; no ordinal window is
allowed to cross a heading.

### Measured corpus structure

408 sections across 3 notebooks, maximum heading depth 3.

Prose characters per section, over the 393 sections containing any prose:

| p05 | p10 | p25 | p50 | p75 | p90 | max |
|---|---|---|---|---|---|---|
| 209 | 254 | 376 | 536 | 729 | 1107 | 2686 |

### Eligibility rule

A section is **eligible** if it contains at least **300 characters of prose**.

| class | sections | disposition |
|---|---|---|
| prose >= 300 | **337** | eligible |
| 0 < prose < 300 | 56 | excluded: thin |
| prose = 0 | 15 | excluded: prose-free |

The three classes partition the 408 sections exactly.

### Why 300

The prose distribution was inspected in 50-character bands from 50 to 700 and
**has no natural gap**. There is therefore no threshold the data selects on its
own, and 300 is a reasoned choice rather than a discovered boundary. Two
independent anchors support it:

1. **Semantic floor.** At roughly 100–120 characters per English sentence, 300
   characters is two to three sentences. Below that a question can do little but
   restate its source, which is the failure mode `docs/FEATURES.md` names: "A
   question that can be answered by pattern-matching against a sentence in the
   source material has failed."
2. **Benchmark sufficiency.** The benchmark requires 300 questions at one
   question per section. A threshold of 300 leaves 337 eligible sections. The
   next candidates leave too few: 350 leaves 309, and 400 leaves 278.

300 is the loosest threshold meeting the semantic floor while retaining headroom
above the required 300 sections. Alternatives measured: 200 (380 eligible), 250
(355), 350 (309), 400 (278), 500 (216).

**Prose-free sections are excluded** because this phase evaluates grounded
natural-language interview questions. Code and output-only generation is
deliberately deferred, not judged infeasible. Note that code reasoning is *not*
excluded: 59 eligible sections contain code alongside qualifying prose.

Eligibility is frozen at commit time. The 337 eligible sections and the 71
excluded ones are fixed for the whole phase.

---

## 3. Bounded context algorithm

A section's full text is not passed to the model. Context is built
deterministically:

```
1. Seed = the longest prose chunk in the section,
   ties broken by lowest ordinal.
2. Include ALL prose chunks of the section, in ordinal order.
   Prose is never truncated and never dropped.
3. Append code and output chunks in ordinal order under a
   2,000-character budget.
   A chunk that would exceed the remaining budget is truncated at a
   character boundary and marked [truncated].
   Once the budget is exhausted, remaining chunks are omitted and
   their count is recorded.
4. Every included chunk is labelled with its (doc_id, ordinal).
   The seed chunk is explicitly marked as the seed.
```

### Why this is safe

Every oversized section in this corpus is oversized because of code and output,
never prose. The largest section is 19,959 characters — 1,025 prose, 8,839 code,
10,095 output. **The maximum prose in any eligible section is 2,686
characters.** All prose therefore always fits, and truncation can only ever
remove code or output.

Worst-case context is 2,686 + 2,000 = **4,686 characters**, roughly 1,200
tokens.

### Why a 2,000-character non-prose budget

Measured over the 337 eligible sections, the number whose code and output would
be truncated:

| budget | sections truncated | share |
|---|---|---|
| 1,500 | 39 | 11.6% |
| **2,000** | **26** | **7.7%** |
| 3,000 | 18 | 5.3% |

2,000 keeps truncation rare while holding worst-case context under 4,700
characters. Because 278 of the 337 eligible sections contain no code or output
at all, truncation can affect at most 59 sections by construction.

---

## 4. Section selection — the 300

337 sections are eligible; 300 are generated from. The selection is fixed here
in advance and is fully reproducible.

### The rule

```
section_key = doc_id || CHR(31) || array_to_string(heading_path, CHR(31))
hash        = md5(section_key || ':' || 'phase6-selection-20260907')
order       = ascending by (hash, doc_id, heading_path)
selected    = the first 300 sections in that order
rank        = position in that order, 1-based
```

The tie-break on `(doc_id, heading_path)` makes the order total even if two
hashes collide. The literal seed string `phase6-selection-20260907` is part of
the protocol and must not be changed. The construction follows the existing
precedent in `random_chunks`, which orders by `md5(chunk_id || seed)`.

The hash is independent of every section property used elsewhere in this
protocol, so the selection does not favour rich sections, code sections, or any
question type.

### The 37 unselected sections

They remain eligible and are recorded, not deleted. They are the natural holdout
if a later phase needs material this benchmark has not seen.

---

## 5. Question type mapping

Type is assigned deterministically from section properties. The model does not
choose its own type.

```
if the section contains at least one code chunk -> code_reasoning
elif prose >= 1000 characters                   -> comparison
elif prose >= 600 characters                    -> explanation
else                                            -> conceptual
```

First match wins, so the classes are disjoint.

| type | eligible (337) | selected (300) |
|---|---|---|
| conceptual | 141 | 128 |
| explanation | 95 | 82 |
| code_reasoning | 59 | 50 |
| comparison | 42 | 40 |

### Four types, not eight

`docs/PROJECT.md` lists eight question types. Four are deferred because the
corpus cannot ground them:

- **system design**, **scenario-based**, **debugging** — the median eligible
  section carries 536 characters of prose. A system-design or scenario question
  built from that would take its scenario from the model's own knowledge, which
  `docs/PROJECT.md` classifies as *external* and forbids presenting as
  document-derived.
- **applied** — the same objection at lower severity. Only 48 eligible sections
  reach 1,000 characters of prose.

This is a scope limit for Phase 6 driven by measured corpus capacity. It is not
a revision of the product goal in `docs/PROJECT.md`.

### A stated weakness of the comparison gate

`prose >= 1000` is a proxy for *capacity to support a comparison*, not a test of
whether the section actually discusses two comparable things. A long section
about a single concept will be asked for a comparison it cannot ground.

The deterministic checks in section 9 cannot catch this; it falls to the
interview-relevance rubric. If comparison questions fail that rubric at a higher
rate than other types, **that is a finding to report, not a reason to retune the
gate.** No post-hoc adjustment (section 16).

---

## 6. Difficulty protocol

Difficulty is **requested explicitly** at generation time and is one of `easy`,
`medium`, `hard`.

### Assignment

```
difficulty = ['easy', 'medium', 'hard'][(rank - 1) mod 3]
```

where `rank` is the section's position in the selection order from section 4.
This yields exactly **100 easy, 100 medium, 100 hard**.

Difficulty is assigned independently of every section property. It is
deliberately *not* matched to section richness: matching would confound the
difficulty instruction with the material, making it impossible to tell whether a
question was judged hard because it was generated hard or because its source was
dense.

Verified non-degenerate across the selection — every type-by-difficulty cell is
populated:

| type | easy | medium | hard |
|---|---|---|---|
| conceptual | 41 | 46 | 41 |
| explanation | 29 | 23 | 30 |
| code_reasoning | 13 | 20 | 17 |
| comparison | 17 | 11 | 12 |

### A risk recorded in advance

Assigning `hard` to a thin section may push the model to import external
knowledge to meet the instruction, which would hurt groundedness. This is a
predicted interaction, not a defect to be designed away. It will be measured:
difficulty agreement and groundedness will both be reported stratified by
section prose length.

### What difficulty can and cannot show

`docs/PHASE-0.md` states that "a generator and a judge concurring that a question
is hard measures shared prior, not difficulty" and that validity requires learner
performance data.

- **Measured here:** difficulty *control* — whether the requested level changes
  the output — and *agreement* between the human labeller and the judge.
- **Not measured, and not claimed:** difficulty *validity*.

The requested difficulty is **hidden from both evaluators**. Neither the human
labeller nor the LLM judge sees it when assigning a difficulty label.

---

## 7. Generation contract

Model configuration is binding from `docs/PHASE-0.md`:

| setting | value |
|---|---|
| model | `qwen3:8b` |
| `think` | `false` |
| `keep_alive` | `"30m"` |
| `num_predict` | 400 |
| `temperature` | 0.3 |
| `seed` | first 8 hex digits of the section hash, as an integer |
| endpoint | `/api/chat` with a JSON schema in `format` |

Reached over `urllib`, exactly as `src/daedalus/embedding.py` does. **No new
dependency.**

`temperature` and a per-section `seed` are set so a run reproduces exactly.
Sampling diversity is not needed: each question comes from a different context,
so diversity is supplied by the corpus rather than by the sampler. A low
temperature also improves schema adherence. Both values are frozen here.

### Response schema

```json
{
  "question":        "string",
  "question_type":   "conceptual | explanation | comparison | code_reasoning",
  "difficulty":      "easy | medium | hard",
  "cited_ordinals":  [integer],
  "grounding_quote": "string"
}
```

`question_type` and `difficulty` are echoed back by the model and checked
against what was requested. A mismatch is a rejection, not a correction.

---

## 8. Grounding and citation rules

1. Every entry in `cited_ordinals` must be a chunk that was in the supplied
   context.
2. `cited_ordinals` must be non-empty and **must include the seed ordinal**.
3. `grounding_quote` must appear **verbatim in a cited chunk** after whitespace
   normalisation (runs of whitespace collapsed to one space, ends stripped).

### What the quote check is and is not

The verbatim check is deterministic and machine-verifiable, which makes it worth
having: it establishes that the model was reading the supplied material rather
than generating from memory.

**It is not proof of groundedness.** A model can quote a passage accurately and
still ask a question that passage does not support. Groundedness is measured by
the human rubric in section 12; the quote check is a rejection filter that runs
first and costs nothing.

---

## 9. Deterministic validity checks

A generated question is **valid** if and only if all of the following hold. Each
is mechanical; no judge is involved.

1. The response parses and conforms to the schema.
2. `question` is non-empty after stripping.
3. `question_type` equals the requested type.
4. `difficulty` equals the requested difficulty.
5. Every cited ordinal was in the supplied context.
6. The seed ordinal is among the cited ordinals.
7. `grounding_quote` matches verbatim in a cited chunk after whitespace
   normalisation.

A question failing any check is **rejected and recorded with its failure
reason**. It is not repaired, regenerated, or re-prompted. Rejection counts by
reason are part of the reported result.

Duplicate detection is **not** part of validity. It is reported separately
(section 11).

---

## 10. Storage schema

Migration `004_questions.sql`.

```
questions
  id, doc_id, heading_path, seed_ordinal, selection_rank,
  requested_type, requested_difficulty,
  text, grounding_quote, model, prompt_version, params_hash,
  generated_at

question_sources
  question_id, doc_id, ordinal, role ('seed' | 'context')

question_labels        -- human labels only
  question_id, rubric, value, labelled_at

judge_scores           -- automated judge output only
  question_id, rubric, value, judge_model, judge_version, scored_at
```

Following the precedent established in `002_reference_set.sql` and
`003_candidates.sql`, `question_sources` is keyed by `(doc_id, ordinal)` with
**no foreign key into `chunks`**: storing a document deletes and reinserts its
rows, so a foreign key would cascade and destroy generated work and human labels
on every re-ingest.

Human labels and judge output are kept in **separate tables** so that a judge run
can never overwrite, contaminate, or be mistaken for a human label.

---

## 11. Duplicate detection and its validation

Diversity is structural first: one question per section, and no section
generated from twice, so questions cannot share a context by construction.

Similarity is a **safety net measured after the fact**, not a filter that the
design depends on and not part of structural coverage.

Procedure:

1. Embed all accepted questions with `bge-m3`, stored under a **distinct
   embedding identity**. The production `bge-m3` chunk embeddings are not touched.
2. Compute pairwise cosine similarity over all accepted questions.
3. Sample pairs **stratified across similarity bands**, including the high-
   similarity tail.
4. The user labels each sampled pair duplicate / not duplicate.
5. Report where a threshold would sit, and how well any threshold separates the
   labelled pairs — **including the outcome that no threshold separates them
   cleanly**, if that is what the data shows.

No threshold is chosen in advance, and no threshold may be selected by looking at
which value produces the most flattering duplicate rate.

---

## 12. The three rubrics

Written, worked through with examples, and **committed before any labelling
begins**. Each defines its levels, gives a worked example per level, and states
its edge cases.

1. **Groundedness** — is the question answerable from the cited material, and is
   nothing in it supplied from outside the documents?
2. **Interview relevance** — is this a question an interviewer would actually
   ask, and does it distinguish understanding from recall?
3. **Difficulty** — easy, medium, or hard, assigned without sight of the
   requested level.

`docs/PHASE-0.md` records that these rubrics do not exist. Writing them is Phase
6 work and gates everything downstream of it.

---

## 13. Benchmark and labelling procedure

In order. No step may be reordered to see a result sooner.

1. Freeze and commit this protocol.
2. Write and commit the three rubrics.
3. Generate one question for each of the 300 selected sections.
4. Apply the deterministic checks. Record every rejection and its reason.
5. The user labels the accepted questions against the three rubrics.
6. Run the LLM judge over the same questions.
7. **Measure judge agreement against the human labels.** Report raw agreement
   and a chance-corrected statistic.
8. Only after step 7 may any automated score be reported, and only alongside its
   measured agreement.

### The annotation boundary

The boundary established in Phase 4 carries over without change. **The user
assigns every rubric grade.** The assistant may verify provenance, report
distribution and pace statistics, display question and chunk text, diagnose
tooling, and transcribe a correction the user has already decided. It never
assigns a grade.

---

## 14. What will be reported

Every figure states how rejected and unlabelled questions were treated.

**Structural coverage** — selected sections producing at least one question that
passes the deterministic checks, divided by **300 selected sections**. Duplicate
detection is excluded from this definition.

Reported alongside it, so the denominator is never mistaken for the whole corpus:

- **300 of 337** eligible sections selected — 89.0%
- **337 of 408** sections eligible — 82.6% (56 thin, 15 prose-free)
- **300 of 408** sections generated from — 73.5%

The third is the product of the first two and is the most conservative statement
of how much of the corpus this benchmark touches. All three are reported
together; structural coverage is never quoted without them.

**Quality rates** — groundedness, interview relevance, and difficulty agreement,
each from the human labels, each as a rate with a confidence interval.

**Judge agreement** — raw and chance-corrected, reported before any automated
score.

**Rejection breakdown** — counts by failure reason.

**Duplicate analysis** — as described in section 11.

At N=300 a rate near 0.9 carries a 95% interval roughly 7 points wide. This is
recorded so the interval is expected rather than discovered.

---

## 15. Exit criteria

Phase 6 is complete when all of the following are true:

1. The generator is implemented, tested, and passes lint, format and type checks.
2. This protocol and the three rubrics were committed before generation.
3. 300 questions generated; rejections recorded by reason.
4. The user's labels are complete for the accepted set.
5. Judge agreement measured and reported before any automated score is quoted.
6. Groundedness, interview relevance and difficulty agreement reported as rates
   with intervals.
7. Structural coverage reported against the frozen denominators.
8. Duplicate threshold validated, or reported as unvalidatable.
9. Results frozen in `results/`; decisions recorded in `docs/ROADMAP.md`.

A null or a failure is a completed outcome. If the generator does not meet the
`docs/PHASE-0.md` criteria, that is reported as measured, not repaired until the
number improves.

---

## 16. What is frozen

Fixed at commit and not adjustable after seeing any output:

- the 300-character eligibility threshold and the 337/56/15 partition
- the seed rule, the all-prose rule, and the 2,000-character non-prose budget
- the selection seed `phase6-selection-20260907` and the 300 sections it picks
- the type mapping and the difficulty assignment
- the generation contract, model configuration, temperature and seeding
- the validity checks and the rule that failures are rejected, not repaired
- the reporting definitions in section 14

**No post-hoc tuning.** Thresholds are not moved, prompts are not revised,
questions are not regenerated, and the type or difficulty mapping is not adjusted
in response to a measured result. A poor rate is reported as a poor rate. If a
change is genuinely warranted, it requires a new protocol committed in advance
and a fresh run recorded separately.

---

## 17. Contradictions with existing documents

Recorded explicitly rather than resolved silently.

**1. `docs/PROJECT.md` shows Retrieval preceding question generation. This phase
bypasses retrieval entirely.** Generation is driven by a deterministic partition
of the corpus, not by a query. This is deliberate: `docs/PHASE-0.md` establishes
that "a question generated from a retrieved chunk has that chunk as its answer by
construction," so routing generation through retrieval would measure nothing
while adding retrieval error to generation error. The documented pipeline is
wrong for this phase, and this is the notice of that.

**2. `docs/PROJECT.md` lists eight question types; this phase delivers four.** A
scope limit driven by measured corpus capacity, not a revision of the product
goal. Section 5 gives the reasoning per type.

**3. `docs/PHASE-0.md` states criteria as "at least 9 of 10".** They are reported
here as rates with confidence intervals at N=300, following that document's own
constraint that "ten questions cannot support a 90% threshold." The criteria are
unchanged; only their presentation is honest about sample size.

**4. `docs/FEATURES.md` lists difficulty labelling as an unresolved open
question.** This protocol settles it **for Phase 6 only** — requested explicitly,
hidden from evaluators, judged independently, validity not claimed. The general
open question remains open.

**5. Coverage denominator.** The approved decision defined coverage over
*eligible* sections; the approved benchmark generates from 300 of the 337. A
denominator of 337 would cap coverage at 88.7% for a reason unrelated to
generator quality. Structural coverage is therefore reported over the **300
selected** sections, with 300/337, 337/408 and 300/408 always reported beside it
(section 14). This narrowing was surfaced for approval and approved on
2026-09-07; it is recorded here because it changes the meaning of a figure the
project will report.

**6. `docs/PHASE-0.md` requires the supporting chunk to appear in the top-K
retrieved results so retrieval can be reported as Recall@K.** No Recall@K is
reported in this phase, for the reason that document itself gives. Retrieval
quality is already measured in `results/`, against the independently labelled
Phase 4 reference set.
