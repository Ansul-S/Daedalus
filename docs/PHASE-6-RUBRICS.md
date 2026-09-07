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

**The grounding quote is never shown, in any pass, and most emphatically not
during the groundedness pass.** It has already been verified verbatim by the
deterministic check, so it carries no information the cited chunk does not. What
it would carry is a sentence pre-selected by the generator as "the evidence",
and reading that before judging groundedness would anchor the judgement on the
model's own choice of support rather than on the material. A labelling tool that
displays it is defective and must not be used.

### A blinding defect that must be fixed before labelling

Requested difficulty is assigned by `['easy','medium','hard'][(rank - 1) mod 3]`.
Measured on the 228 accepted questions, **difficulty is recoverable from
selection rank with 100% accuracy**. Presented in rank order the sequence reads
`e m h e e m h e e m h e ...` — visibly periodic.

Labelling in rank order, or showing the rank, would therefore leak the very
value the difficulty rubric is supposed to assign independently, and the
resulting agreement figure would be worthless.

**Presentation order is shuffled**, and each of the three passes in section 7
uses a *different* shuffle:

```
order_key = md5(doc_id || CHR(31) || array_to_string(heading_path, CHR(31))
                || ':' || <seed>)

pass 1, groundedness:  seed 'phase6-groundedness-20260907'
pass 2, relevance:     seed 'phase6-relevance-20260907'
pass 3, difficulty:    seed 'phase6-difficulty-20260907'
```

ascending, tie-broken on `(doc_id, heading_path)`. All three seeds differ from
the selection seed, so no order is related to the draw or to any other pass. The
rank is never displayed.

Three different shuffles matter: reusing an order would let a later pass be
anchored by the remembered sequence of an earlier one, which is part of what
separating the passes exists to prevent.

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

### Two distinct failure modes, both reducing the grade

A question can fail groundedness in two different ways, and the grade alone does
not say which. Both reduce the grade identically; the labeller records **which
mode applies** alongside the grade, so the two are separable in the write-up.

| mode | meaning |
|---|---|
| **support failure** | The material needed to answer does not exist in this section at all. The question reaches outside the corpus. |
| **citation failure** | The material needed *is* in the section, but not in the chunks the question cited. The question is answerable; its references are wrong. |

Both are real defects and neither is excused. A question whose citations do not
support it has not delivered "references back to source chunks", which is the
phase's stated deliverable — so a citation failure is not a lesser sin scored on
a softer scale. It is recorded separately because the two imply different fixes:
support failure points at the generator inventing material, citation failure at
it mis-attributing material it did read.

Recording the mode is only required where the grade is **0 or 1**. A grade 2
question has neither failure.

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

- **`unusable` is excluded from the agreement denominator.** A question too
  incoherent to carry a difficulty is not evidence for or against the
  generator's difficulty control, and counting it as a mismatch would blame the
  difficulty instruction for a defect in the question. The count of `unusable`
  items is reported separately and always alongside the agreement figure, so the
  denominator is never quietly reduced without the reader seeing by how much.
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
| difficulty match | share where labelled difficulty equals requested, **over items not labelled `unusable`** | >= 80% |

Groundedness **2 + 1** is reported alongside the headline, never in place of it.
Grade 1 does not count toward the >= 90% criterion; only grade 2 does. Every
rate is reported with a confidence interval and with its denominator stated.

Groundedness failure modes — support failure against citation failure, from
section 3 — are reported as a breakdown of the grade 0 and grade 1 items. They
do not change any rate.

The difficulty agreement denominator is `228 minus the unusable count`, and both
numbers are printed together. No other rate excludes anything.

### The thresholds do not move

The >= 90% groundedness bar, the >= 80% relevance and difficulty bars, and the
> 10% unsupported failure condition are fixed here and **do not change in
response to the measured result**. If the generator comes in below a bar, that is
reported as a failure against a pre-registered criterion. It is not answered by
rereading the scale, by promoting grade 1 into the numerator, or by lowering the
threshold to what was achieved.

### Two denominators, kept apart

**Generation is reported over 300 sections. Evaluation is reported over 228
questions.** They are different measurements of different things and are never
combined into one rate.

- **300** — the selected sections. Generation outcomes live here: 228 accepted,
  72 rejected. This is where structural coverage is reported.
- **228** — the accepted questions, and the only items that carry a label. Every
  groundedness, relevance and difficulty rate uses this denominator.

The 72 rejected sections produced no question to label. Reporting a groundedness
rate over 300 would silently treat a rejection as an ungrounded question, which
it is not — it is an absent one. Any figure quoting one denominator states which
it is using.

---

## 7. Procedure

1. Rubrics committed. This document, before any label is recorded.
2. Questions presented in that pass's own shuffled order from section 2, with
   the rank, requested type, requested difficulty, grounding quote and all
   earlier passes' labels withheld.
3. The user labels. `labelled_at` is recorded per item.
4. Pace is audited afterwards as in Phase 4 — sub-2-second judgements are
   reported, not silently accepted.
5. A mis-keyed grade is corrected only on the user's explicit instruction, and
   the correction is recorded in the write-up.
6. Only once human labelling is complete does the LLM judge run, and judge
   agreement is measured against these labels before any automated score is
   quoted.
7. Judge output is stored exactly as the model emitted it. The human scales
   above are enforced by the database; the judge's are deliberately not. A judge
   answer off its own scale is a finding about the judge — one of the things
   this phase sets out to measure — and rejecting the write would discard the
   evidence. Off-scale judge output is reported, never silently coerced onto the
   scale or dropped.

### Three independent passes, decided before labelling

Labelling runs as **three separate passes**, one rubric each, every pass in its
own shuffled order:

| pass | rubric | shuffle seed |
|---|---|---|
| 1 | groundedness | `phase6-groundedness-20260907` |
| 2 | interview relevance | `phase6-relevance-20260907` |
| 3 | difficulty | `phase6-difficulty-20260907` |

Judging several rubrics in one sitting invites a halo effect: an item judged
ungrounded is easily judged irrelevant, and easy, on the strength of the first
impression rather than the scale. Since the three axes are deliberately
independent — a well-formed question resting on absent material is relevance 2
and groundedness 0 — a halo would erase exactly the distinction the rubrics were
built to record.

The cost is three runs through 228 items rather than one. That is accepted.

Each pass must be complete before the next begins, and **no pass displays the
labels from any earlier pass.**

---

## 8. What is frozen

The three scales, their criteria, their edge cases, the two groundedness failure
modes, the blinding rules, the three shuffle seeds
`phase6-groundedness-20260907`, `phase6-relevance-20260907` and
`phase6-difficulty-20260907`, the three-pass structure, the thresholds, and the
mapping and denominators in section 6.

A rubric may not be revised after labelling begins. If labelling reveals that a
scale is unworkable, the fix is to stop, amend this document with the reason,
and restart that rubric from the beginning — not to reinterpret grades already
assigned.
