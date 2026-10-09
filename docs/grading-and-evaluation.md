# Grading and evaluation

How an answer is graded against its sources, and how the grader and search are measured. Every
measurement runs from a command in the repository, and the results are in the
[design notes](design.md#phase-5-measurements).

**Contents:** [Grading answers](#grading-answers) ·
[Checking the grader against your own grades](#checking-the-grader-against-your-own-grades) ·
[The grader's regression suite](#the-graders-regression-suite) ·
[Measuring search](#measuring-search)

## Grading answers

Every answer is graded against the passages its question was written from. The practice page
sends it; from the command line:

```sh
curl -X POST localhost:8000/questions/31/attempts -H 'Content-Type: application/json' \
  -d '{"answer": "An RNN carries a hidden state from one step to the next, so each word is read in the light of the ones before it."}'
```

The grader labels what it sees, and the score is worked out from the labels:

| Part of a grade | What it says |
|---|---|
| key points | each one covered, partial or missing, with the words of the answer that show it |
| claims | each claim supported or contradicted by a passage, which is cited and linked the way search results are, or unverified when no passage addresses it |
| score | the key points' weighted coverage (covered 1, partial 0.5), less 0.15 for each contradicted claim, never below 0 |
| clarity | 1 to 5 for how clearly the answer is written, apart from what it says |
| feedback | strengths, gaps, errors, a model answer drawn from the passages, and a follow-up question |

- **An unverified claim costs nothing.** The passages don't cover it, which doesn't make it
  wrong.
- **Qwen 3.8 on Groq grades,** in about 2 s. The free tier holds it to 1,000 written tokens a
  minute, so a second answer within the same minute waits its turn. Without Groq the local
  `qwen3.5:9b` grades (over a minute an answer), then Gemini; gpt-oss, which writes the
  questions, never grades the answers to them. Answers are sent to Groq.
- **An answer is kept whatever happens to its grading.** When no model can grade it, it gets
  a failed grade that says why, and `POST /attempts/{id}/grades` grades it again. Every grade
  is kept.
- **The answer is untrusted text.** Instructions inside it, such as a request for full
  marks, are ignored.

### Checking the grader against your own grades

`make calibrate` measures how far the grader agrees with grades given by hand. It needs
`GROQ_API_KEY`.

```sh
make calibrate ARGS=template   # write data/calibration/answers.toml, listing every accepted question
make calibrate                 # grade the answers not graded yet, then report
make calibrate ARGS=report     # report on the grades already made, without grading
```

- **In the file,** add each answer under its question, with one label per key point, how many
  of its claims contradict the sources and, optionally, your own score out of 10. The file's
  header shows the format. An answer keeps counting after its question is retired.
- **The report** gives Spearman's ρ between the grader's scores and the scores your labels
  give (trusted from 0.7), the same against your own scores, Cohen's κ on the key-point
  labels, where the two disagree, and the largest differences.
- **Grading uses Groq's Qwen alone,** about one answer a minute: a grade from a fallback model
  would measure that model instead. When Groq says the day's allowance is spent, grading
  stops and the next run carries on.
- **Grades are kept** in `data/calibration/grades.jsonl`, per answer and prompt version, so a
  second run grades only what is new. Both files are personal practice data and stay out of
  the repository.

### The grader's regression suite

`make eval-grader` checks that the grader still agrees with the hand grades, and fails when it
no longer does. Each answer is a [DeepEval](https://deepeval.com) test case, judged by custom
metrics worked out from the labels: no model judges anything, and DeepEval sends nothing
anywhere.

```sh
make eval-grader                        # the calibration answers, from their saved grades
make eval-grader LIVE=1                 # grade what has no saved grade first, then check
make eval-grader SET=stand-ins          # the stand-in answers kept with the tests
make eval-grader SET=stand-ins LIVE=1   # the stand-ins, graded afresh by the real grader
```

- **Each answer** is checked for key-point agreement (at least half its key points labelled
  as yours), score gap (within 0.25 of your own score out of 10, or of your labels scored when
  you gave none), a contradiction caught (where you counted one) and an injection held (an
  answer that tries to talk the grader round scores no more than you gave it).
- **A run passes** when every answer has a grade, Spearman's ρ is at least 0.90 and Cohen's κ
  at least 0.75, every injection held, and at least 90% of the answers each other check
  applies to pass it. ρ and κ count from 30 graded answers: on fewer, such as the 12
  stand-ins, one answer can move ρ by a tenth, so they are reported but don't decide the run.
- **Replayed, it spends nothing.** The grades saved under the current prompt version are
  scored by today's rules, so a change to the scoring shows at once. An answer without a
  saved grade fails with "grade it first": after a prompt change, `LIVE=1` grades the answers
  again, on Groq's Qwen alone, as `make calibrate` does.
- **The stand-ins** in `backend/tests/fixtures/grader_suite/` are answers written for the
  repository, with their hand grades and the output the grader would send back for each.
  Replayed through the same code as a live grade, they check everything between the model and
  the score without anyone's practice data, and `make test` runs them. Live, the real grader
  grades them afresh (about 16K tokens, which the report counts) and nothing is saved.
- **The report** is printed, and saved to `data/reports/grader-<date>.md` for the calibration
  answers.

## Measuring search

`make eval-retrieval` searches for every question with saved passages and scores how near the
top those passages come: hit@1, hit@5, Recall@5 and MRR@5, in vector, full-text and hybrid
mode. It needs the local embedding model (`make ollama`) for the vector and hybrid modes, asks
no other model anything, and takes a few seconds.

- **Three sets, scored apart:** the questions practice serves; those plus the questions turned
  down for reasons that leave them about their passage; and hand-written questions with
  passages labelled by hand, from `data/eval/hand-questions.toml`.
- **Strict and lenient.** A question's own passages are its answers. Another passage often
  answers it too; those confirmed by hand in `data/eval/also-answers.toml` (`verdict = "yes"`)
  count in the lenient score.
- **Fusion variants.** The report also tries how much of the full-text ranking hybrid search
  should use, tuned on the turned-down questions and judged on the others, and says whether the
  current setting should change.
- **The report** is printed and saved to `data/reports/retrieval-<date>.md`. Labels that point
  to passages a re-ingestion has replaced are listed, since those questions score lower until
  they are labelled again.
