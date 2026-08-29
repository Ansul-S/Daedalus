# Daedalus — Target Feature Set

The intended end-state product. This describes what Daedalus should eventually
do, not what has been built, approved, or scheduled. Nothing here is settled
architecture, and no item here is assigned to a phase — see `docs/ROADMAP.md`
for sequencing.

Scope note: the corpus on disk is an initial test dataset, not the boundary of
the product. The system targets arbitrary study material across arbitrary
topics and domains.

## Priorities

Retrieval is the highest priority. The system must surface the correct and most
relevant chunks from the available material; everything downstream — question
quality, evaluation, follow-ups — is bounded by how well retrieval performs.

## Grounding and correctness

Questions and evaluations are grounded in the user's material, and where
practical keep a reference back to the source chunks they came from.

The system should not be wholly captive to the retrieved material. Where source
material is wrong, internally inconsistent, or outdated, the system should be
able to recognise that and supply the correct answer rather than repeating the
error.

This sits in tension with the grounding model in `docs/PROJECT.md`, which
separates supported, inferred, and external content and requires that external
knowledge never be presented as coming from the uploaded documents. The
mechanism by which a source is judged wrong — and how that judgement is
surfaced to the user and distinguished from a hallucination — is unresolved.

## Question generation

Generated questions must be interview-quality: technically relevant, meaningful,
and able to distinguish genuine understanding from recall. A question that can
be answered by pattern-matching against a sentence in the source material has
failed.

The question types in `docs/PROJECT.md` — conceptual, explanation, comparison,
applied, debugging, system design, code reasoning, scenario-based — describe the
intent of a question. The modes below describe how the user responds. They are
separate axes.

## Difficulty adaptation

Questions carry one of three difficulty levels: easy, medium, hard.

Difficulty of the next question depends on the quality and correctness of the
answer to the previous one. A session opens at easy and moves upward on correct
answers: easy to medium, medium to hard, and beyond a first correct hard answer
the system continues to challenge within the hard band.

The same logic applies downward. A student who struggles with a hard question
drops to medium; a student who then struggles at medium drops to easy.
Adaptation is continuous rather than a fixed sequence.

The goal is a difficulty curve that tracks actual understanding.

## Scoring

Points are awarded per question, weighted by difficulty: hard questions award
the most, medium a moderate amount, easy the least. The award reflects both the
difficulty of the question and the quality of the student's answer. Exact point
values are undecided.

## Question modes

**MCQ — multiple choice.** One correct answer selected from several options.

**MSQ — multiple select.** Several correct answers selectable from the options
offered.

**NAT — numerical answer type.** A numerical or integer answer, entered directly
rather than selected.

**Interview mode.** The final and most demanding mode, described below.

## Interview mode

Interview mode simulates a real technical interview rather than a quiz.

The system asks an interview-quality question and the student types their own
answer; there are no options to select from. The system evaluates that answer
and rates it, taking account of correctness, depth, reasoning, clarity, and
completeness, rather than marking it simply right or wrong.

The system cross-questions. Follow-ups are generated from what the student
actually said, not drawn from a fixed list. Vague, incomplete, incorrect, or
evasive answers are challenged. The exchange continues adaptively in the way a
real interviewer would probe a weak answer.

The target is that the mode feels as close as possible to sitting in front of a
real AI/ML interviewer.

## Open questions

These are unresolved and need decisions before the relevant work starts.

- **Source correction versus grounding.** How the system establishes that source
  material is wrong, how that is presented, and how it is kept distinct from
  hallucination. See the grounding section above.
- **Selection modes versus the question philosophy.** MCQ and MSQ test
  recognition and are open to elimination strategies. `docs/PROJECT.md` states
  that not all questions should be simple recall. Whether, and how, the
  selection modes can meet the interview-quality bar is undecided.
- **NAT source material.** Conceptual ML material yields few well-posed
  numerical questions. Where NAT questions come from, and whether the mode earns
  its place, is unresolved.
- **Adaptation and scoring depend on judged answers.** Routing between
  difficulty levels and awarding points both require the answer evaluator to be
  reliable. `docs/PROJECT.md` treats the LLM judge as an estimator with its own
  biases, not ground truth. A misjudged answer does not only misreport — it
  sends the next question in the wrong direction, and the error compounds across
  a session. The reliability the adaptive loop requires has not been established.
- **Difficulty labelling.** Difficulty is assumed to be a property a question
  can be assigned. How it is assigned, and whether the assignment is consistent,
  is undecided.
- **Success criteria.** No measurable criterion has been defined for any feature
  here.
