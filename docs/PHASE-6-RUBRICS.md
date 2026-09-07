# Phase 6 — labelling rubrics

Frozen before labelling begins, as `docs/PHASE-6-PROTOCOL.md` section 12
requires. Committing these first is what separates a rubric from a
rationalisation of whatever the labels turned out to be.

Three rubrics, applied to the **228 accepted questions** of the 300-section
benchmark. Rejected sections have no question and are not labelled; they are
already accounted for in `results/README.md`.

Every worked example below is **invented for illustration**. None is drawn from
the benchmark: using a real item would pre-assign its grade and contaminate the
set this document exists to measure.

---

## 1. The annotation boundary

Unchanged from Phase 4. **The user assigns every grade.**

The assistant may verify provenance, report distribution and pace statistics,
display question and chunk text, diagnose tooling, audit a completed rubric
pass, and transcribe a correction the user has already decided. It never assigns
a grade, and never suggests one for a specific item.

---

## 2. What the labeller sees

| shown | withheld |
|---|---|
| the question text | the requested question type |
| the cited chunks, in full, with their ordinals | the requested difficulty |
| | the selection rank |
| | the grounding quote |
| | any earlier label for the same item |

**Only the cited chunks are shown.** The phase's claim is that questions carry
references back to source chunks, so groundedness is exactly the question of
whether those references suffice. A question that is answerable from its section
but cites the wrong chunk scores low, and that is intended rather than a
side effect.

The grounding quote is withheld because it has already been verified verbatim by
the deterministic check. It carries no information the cited chunk does not, and
showing a sentence pre-selected as "the evidence" would anchor the groundedness
judgement.

### A blinding defect that must be fixed before labelling

Requested difficulty is assigned by `['easy','medium','hard'][(rank - 1) mod 3]`.
Measured on the 228 accepted questions, **difficulty is recoverable from
selection rank with 100% accuracy**. Presented in rank order the sequence reads
`e m h e e m h e e m h e ...` — visibly periodic.

Labelling in rank order, or showing the rank, would therefore leak the very
value the difficulty rubric is supposed to assign independently, and the
resulting agreement figure would be worthless.

**Presentation order is shuffled**, by:

```
order_key = md5(doc_id || CHR(31) || array_to_string(heading_path, CHR(31))
                || ':' || 'phase6-labelling-20260907')
```

ascending, tie-broken on `(doc_id, heading_path)`. The seed differs from the
selection seed so the two orders are unrelated. The rank is not displayed.

---

## 3. Rubric 1 — Groundedness

**The question to answer:** could someone holding only the cited chunks answer
this question correctly and completely?

| grade | meaning |
|---|---|
| **2** | **Supported.** Everything needed is in the cited material. |
| **1** | **Partially supported.** The subject is in the material and the material contributes, but answering fully needs knowledge the cited chunks do not contain. |
| **0** | **Unsupported.** The question needs knowledge absent from the cited chunks, misstates what they say, or asks about something not there. |

### Criteria

Judge against the cited chunks **only**. Not the rest of the section, not the
rest of the corpus, and not what you personally know about the topic. The
hardest part of this rubric is setting aside your own knowledge: a question can
look perfectly answerable because *you* can answer it.

A question is still grade 2 if it is trivial. Triviality is a relevance
judgement, not a groundedness one, and the two axes are kept apart deliberately.

### Edge cases

- **False premise.** The question asserts something the material contradicts, or
  assumes a mechanism the material does not describe. Grade 0, even if the
  question reads well.
- **Term used but not needed.** The question mentions something the chunks do
  not define, but answering does not require defining it. Still 2.
- **"Why" against a "what".** The material states that something is done; the
  question asks why. If the reason is not in the cited chunks, this is 1 at
  best, and 0 if the material gives no basis at all.
- **Answerable only in part.** Two-part questions where the material covers one
  part are 1, not 2.
- **Code questions.** If the cited chunks include the code, reasoning about what
  that code does is grounded. Reasoning about library behaviour the code calls
  but does not show is not.

### Illustrative examples

Material: *"Chunk overlap helps preserve information near chunk boundaries.
Without overlap, an answer may be split between two chunks."*

- Grade 2 — "Why can chunking without overlap cause an answer to be missed?"
- Grade 1 — "How would you choose an overlap size for a production system?"
  (the material motivates overlap but gives no basis for sizing it)
- Grade 0 — "What overlap ratio does the BERT paper recommend?" (not in the
  material at all)

---

## 4. Rubric 2 — Interview relevance

**The question to answer:** would a competent AI/ML interviewer ask this, and
does it separate understanding from recall?

| grade | meaning |
|---|---|
| **2** | **Interview-quality.** Probes understanding, reasoning, comparison or application. |
| **1** | **Weak but usable.** On topic and sensible, but answerable by restating a sentence. Tests recall more than understanding. |
| **0** | **Not usable.** Trivial, ambiguous, malformed, or about notebook mechanics rather than an AI/ML idea. |

### Criteria

`docs/FEATURES.md` sets the bar: *"A question that can be answered by
pattern-matching against a sentence in the source material has failed."* A
question meeting only that bar is 1, not 2. Grade 0 is reserved for questions
that should not be asked at all.

Judge the question, not its answer's difficulty, and not whether the material
supports it — that is rubric 1. A well-formed interview question resting on
absent material is relevance 2 and groundedness 0. Recording both is the point.

### Edge cases

- **Notebook plumbing.** "What does the `max_answer_length` argument default
  to?" is 0 — it tests recall of a specific notebook, not an ML idea.
- **Too broad.** "Explain RAG." is 1: legitimate, but it invites a recall dump
  rather than discriminating between candidates.
- **Leading questions.** A question containing its own answer is 0.
- **Multi-part sprawl.** Three questions bolted together are 1 at best; an
  interviewer would ask one.
- **Code reasoning.** "Why a nested loop rather than two independent argmaxes?"
  is 2 — it requires understanding the constraint. "What does line 4 do?" is 1.

### Illustrative examples

- Grade 2 — "Why does averaging over bootstrap samples reduce variance more than
  it reduces bias?"
- Grade 1 — "What does bagging stand for?"
- Grade 0 — "What is the variable name used for the classifier in this cell?"

---

## 5. Rubric 3 — Difficulty

**The question to answer:** for a candidate preparing for an AI/ML interview who
has studied this material, how hard is this question?

| grade | meaning |
|---|---|
| **easy** | Answerable by someone who has read the material once. A single fact or a direct explanation. |
| **medium** | Requires connecting two ideas, or explaining a mechanism rather than naming it. |
| **hard** | Requires reasoning about consequences, trade-offs or edge cases that the material supports but does not state outright. |
| **unusable** | The question is too incoherent for difficulty to mean anything. Should coincide with relevance 0. |

### An honest caveat about this rubric

These definitions are **deliberately the same** as the ones given to the
generator at generation time. That makes the resulting agreement figure partly a
measure of a shared definition rather than of independent judgement, and it is
recorded here so the number is not read as more than it is.

`docs/PHASE-0.md` already states the stronger limitation: *"A generator and a
judge concurring that a question is hard measures shared prior, not
difficulty. Validity requires learner performance data."* Nothing in this phase
establishes difficulty validity, and nothing in the write-up will claim it.

### Edge cases

- Judge difficulty **for the candidate**, not for you. You wrote the reference
  set and know this corpus better than any candidate will.
- Do not let groundedness bleed in. A question resting on absent material is not
  "hard" — it is ungrounded, and rubric 1 already records that.
- Length is not difficulty. A long recall question is easy.

---

## 6. How labels map to the Phase 0 criteria

Stated in advance so the mapping is not chosen after seeing the labels.

| criterion | measured as | threshold |
|---|---|---|
| grounded | share of labelled questions at groundedness **2** | >= 90% |
| unsupported rate (failure condition) | share at groundedness **0** | > 10% fails |
| interview-relevant | share at relevance **2** | >= 80% |
| difficulty match | share where labelled difficulty equals requested | >= 80% |

Groundedness **2 + 1** is reported alongside the headline, never in place of it.
Every rate is reported with a confidence interval and with its denominator
stated.

Note that these rates are over the **228 accepted questions**, not over 300. The
72 rejected sections produced no question to label. Reporting a groundedness
rate over 300 would silently treat a rejection as an ungrounded question, which
it is not — it is an absent one. Both denominators appear in the write-up.

---

## 7. Procedure

1. Rubrics committed. This document, before any label is recorded.
2. Questions presented in the shuffled order of section 2, with the rank,
   requested type, requested difficulty and grounding quote withheld.
3. The user labels. `labelled_at` is recorded per item.
4. Pace is audited afterwards as in Phase 4 — sub-2-second judgements are
   reported, not silently accepted.
5. A mis-keyed grade is corrected only on the user's explicit instruction, and
   the correction is recorded in the write-up.
6. Only once human labelling is complete does the LLM judge run, and judge
   agreement is measured against these labels before any automated score is
   quoted.

### One decision left open

All three rubrics in a single pass per question, or difficulty in a separate
pass with its own shuffle?

- **Single pass** is roughly ten hours and keeps the material fresh in mind, but
  risks a halo effect: an item judged ungrounded is easily judged irrelevant and
  easy too.
- **Split pass** protects difficulty — the weakest of the three measurements —
  at the cost of a second run through 228 items.

Whichever is chosen is recorded here before labelling starts.

---

## 8. What is frozen

The three scales, their criteria, their edge cases, the blinding rules, the
shuffle seed `phase6-labelling-20260907`, and the mapping in section 6.

A rubric may not be revised after labelling begins. If labelling reveals that a
scale is unworkable, the fix is to stop, amend this document with the reason,
and restart that rubric from the beginning — not to reinterpret grades already
assigned.
