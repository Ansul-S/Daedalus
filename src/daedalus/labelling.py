"""The two labelling loops: reference-set judgements and question rubrics.

Both are interactive, both run to hundreds of decisions, and both are governed
by the same rule — the person labelling assigns every grade, and the tool's job
is to withhold anything that would bias one.

They live together because they share that machinery: single-keypress input,
truncated display with a full-text escape, per-decision commits so a session can
be stopped and resumed, and a policy reminder available at any point. They are
kept out of `cli.py` because argument parsing and these loops have nothing to do
with each other, and the file was carrying both.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from typing import cast

import psycopg

from daedalus.embedding import embed_texts
from daedalus.generation.selection import KEY_SEPARATOR
from daedalus.retrieval.search import Candidate, Chunk, pool_candidates
from daedalus.storage.database import connect
from daedalus.storage.documents import chunks_at, parent_text
from daedalus.storage.queries import (
    Query,
    candidate_refs,
    grade_totals,
    judged_pairs,
    list_queries,
    record_judgement,
)
from daedalus.storage.questions import (
    StoredQuestion,
    label_totals,
    labelled_question_ids,
    list_questions,
    record_label,
)

Connection = psycopg.Connection[tuple[object, ...]]

GRADE_KEYS = {"0": 0, "1": 1, "2": 2}

#: Characters of a chunk shown before it is truncated. Chosen against the
#: corpus: median chunk length is about 418 characters and p90 about 1,188, so
#: this shows most chunks whole while capping the rare very large one. The full
#: text is always available with the "f" key, because the policy is to judge the
#: complete content and never the preview alone.
PREVIEW_LIMIT = 2000

#: One-line meaning of each grade, shown on every prompt.
GRADE_MEANINGS = (
    "0 not relevant",
    "1 partially answers",
    "2 fully answers",
)

#: The three labelling passes of docs/PHASE-6-RUBRICS.md. Each has its own
#: shuffle seed so no pass is presented in the order of any other, and none in
#: the order of the draw -- requested difficulty is recoverable from selection
#: rank with certainty, so rank order would leak it outright.
LABEL_PASSES = {
    "groundedness": {
        "seed": "phase6-groundedness-20260907",
        "keys": {"0": "0", "1": "1", "2": "2"},
        "legend": ("0 unsupported", "1 partial", "2 supported"),
    },
    "relevance": {
        "seed": "phase6-relevance-20260907",
        "keys": {"0": "0", "1": "1", "2": "2"},
        "legend": ("0 not usable", "1 weak", "2 interview-quality"),
    },
    "difficulty": {
        "seed": "phase6-difficulty-20260907",
        "keys": {"e": "easy", "m": "medium", "h": "hard", "u": "unusable"},
        "legend": ("e easy", "m medium", "h hard", "u unusable"),
    },
}

#: Groundedness grades that require a failure mode, and the keys that record it.
FAILURE_MODE_KEYS = {"s": "support", "c": "citation"}

#: Shown once at the start of a session and again on demand with "?".
POLICY_REMINDER = """\
Judge only how useful this chunk's content is for answering the query.

  0  not relevant       related topic or shared words, but does not help answer
  1  partially answers  contributes part of the answer, or evidence for it
  2  fully answers      content is sufficient to answer the query directly

Code and prose are judged by the same standard. Do not downgrade a chunk for
being code, and do not promote it for looking sophisticated. A signature with no
meaningful body does not earn a 2 because its name matches the query.

Ignore how the chunk was retrieved. The question is only whether the chunk
helps answer the query.

A truncated chunk is marked TRUNCATED; press f to read all of it before
judging. When an output chunk is shown, the code that produced it appears above
as context only — the grade belongs to the output, not to that code.

s skips without recording anything; the candidate returns in a later session.

Full policy: docs/LABELLING.md
"""


def read_key(prompt: str) -> str:
    """Read a single keypress, falling back to a line when stdin is not a tty.

    A single keypress matters here: a labelling session runs to a thousand or
    more judgements, and requiring Enter on each doubles the effort. The line
    fallback keeps the loop testable and usable through a pipe.
    """
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        return sys.stdin.readline().strip()[:1]

    import termios
    import tty

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
    print(key)
    return key


def show_candidate(
    query_text: str,
    candidate: Candidate,
    position: int,
    total: int,
    full: bool = False,
    context: str | None = None,
) -> None:
    """Print one candidate for judging.

    Which retrievers surfaced the candidate is deliberately not shown. The
    reference set measures those retrievers, so a judgement influenced by them
    would be measuring itself.

    An output chunk is often meaningless read alone, so the code that produced
    it is shown above as context. It is labelled as context and excluded from
    the judgement: the grade belongs to the candidate.
    """
    chunk = candidate.chunk
    heading = " > ".join(chunk.heading_path) or "(no heading)"

    print("\n" + "=" * 78)
    print(f"QUERY: {query_text}")
    print(f"[{position}/{total}]  {chunk.kind}  {chunk.doc_id}:{chunk.ordinal}")
    print(f"SECTION: {heading}")

    if context is not None:
        print("-" * 78)
        print("CONTEXT — the code that produced this output. NOT judged.")
        print("-" * 78)
        print(_body(context, full))

    print("-" * 78)
    if context is not None:
        print("CANDIDATE — judge this:")
        print("-" * 78)
    print(_body(chunk.text, full))
    print("-" * 78)


def _body(text: str, full: bool) -> str:
    """Render chunk text, marking truncation explicitly when it applies."""
    body = text.strip()
    if full or len(body) <= PREVIEW_LIMIT:
        return body
    hidden = len(body) - PREVIEW_LIMIT
    return (
        f"{body[:PREVIEW_LIMIT]}\n"
        f"[TRUNCATED — {hidden} more characters. Press f to read all before judging.]"
    )


def pending_candidates(
    connection: Connection,
    query: Query,
    args: argparse.Namespace,
    done: set[tuple[str, int]],
) -> list[Candidate]:
    """Return the candidates for one query that still need judging.

    The pool is the live draw from the vector, lexical and random retrievers
    unioned with whatever is recorded in the candidates table. The recorded half
    matters because a candidate surfaced by a retriever that no longer runs --
    or that never ran here, such as an ablation -- would otherwise be
    unreachable, and its absence would be silently read as irrelevance.

    Sources are not carried onto recorded-only candidates. Nothing in the
    labelling interface may reveal which retriever found a chunk.
    """
    pooled = pool_candidates(
        connection,
        query.text,
        lambda texts: embed_texts(texts, model=args.model),
        args.model,
        vector_k=args.vector_k,
        lexical_k=args.lexical_k,
        random_k=args.random_k,
    )
    pending = [c for c in pooled if (c.chunk.doc_id, c.chunk.ordinal) not in done]

    seen = {(c.chunk.doc_id, c.chunk.ordinal) for c in pooled}
    extra = sorted(candidate_refs(connection, query.query_id) - seen - done)
    for doc_id, ordinal, kind, text, heading_path in chunks_at(connection, extra):
        pending.append(
            Candidate(
                chunk=Chunk(
                    chunk_id=0,
                    doc_id=doc_id,
                    ordinal=ordinal,
                    kind=kind,
                    text=text,
                    heading_path=tuple(heading_path),
                ),
                sources=frozenset(),
            )
        )
    return pending


RUBRIC_REMINDERS = {
    "groundedness": """\
Could someone holding ONLY the cited chunks answer this question correctly and
completely?

  0  unsupported  needs knowledge absent from the cited chunks, or misstates them
  1  partial      the material contributes, but answering fully needs more
  2  supported    everything needed is in the cited material

Judge against the cited chunks only -- not the rest of the section, not the rest
of the corpus, and not what you know about the topic. A trivial question is
still 2; triviality is a relevance judgement.

At 0 or 1, record which failure it is:
  s  support   the material needed does not exist in this section at all
  c  citation  the material exists in the section but was not cited
""",
    "relevance": """\
Would a competent AI/ML interviewer ask this, and does it separate understanding
from recall?

  0  not usable          trivial, ambiguous, malformed, or notebook mechanics
  1  weak but usable     answerable by restating a sentence
  2  interview-quality   probes understanding, reasoning or application

Judge the question, not whether the material supports it. A well-formed question
resting on absent material is relevance 2 and groundedness 0.
""",
    "difficulty": """\
For a candidate preparing for an AI/ML interview who has studied this material,
how hard is this question?

  e  easy      answerable by someone who has read the material once
  m  medium    requires connecting two ideas, or explaining a mechanism
  h  hard      requires reasoning about consequences, trade-offs or edge cases
  u  unusable  too incoherent for difficulty to mean anything

Judge for the candidate, not for you. Length is not difficulty.
""",
}


def labelling_order(
    questions: Sequence[StoredQuestion], seed: str
) -> list[StoredQuestion]:
    """Return questions in one pass's presentation order.

    Ordered by a hash of the section key and the pass seed, tie-broken on the
    section itself. Every pass uses a different seed so that no pass is
    presented in the order of another, and none in the order of the draw.
    """

    def key(question: StoredQuestion) -> tuple[str, str, tuple[str, ...]]:
        section = KEY_SEPARATOR.join((question.doc_id, *question.heading_path))
        digest = hashlib.md5(f"{section}:{seed}".encode()).hexdigest()
        return (digest, question.doc_id, question.heading_path)

    return sorted(questions, key=key)


def show_question(
    question: StoredQuestion,
    cited: Sequence[tuple[str, int, str, str, list[str]]],
    position: int,
    total: int,
    full: bool = False,
) -> None:
    """Print one question and the chunks it cites, and nothing else.

    What is withheld is the point of this function. The requested type, the
    requested difficulty, the selection rank, the grounding quote and any label
    from an earlier pass are all deliberately absent: each would anchor the
    judgement on something other than the question and its material.
    """
    print("\n" + "=" * 78)
    print(f"[{position}/{total}]")
    print("-" * 78)
    print("QUESTION")
    print("-" * 78)
    print(question.text.strip())
    print("-" * 78)
    print("CITED MATERIAL — judge against this only")
    print("-" * 78)
    for _doc_id, ordinal, kind, text, _heading in cited:
        print(f"\n[chunk {ordinal}] {kind}")
        print(_body(text, full))
    print("-" * 78)


def has_truncation(
    cited: Sequence[tuple[str, int, str, str, list[str]]],
) -> bool:
    """Whether any cited chunk is long enough that the display cuts it short.

    The prompt offers "f full text" only when there is hidden text to reveal.
    Offering it unconditionally advertises an action that does nothing on a
    short chunk, which reads as a broken key rather than as an empty one, and
    invites the labeller to wonder whether they are seeing the whole chunk.
    """
    return any(len(text.strip()) > PREVIEW_LIMIT for _d, _o, _k, text, _h in cited)


def cmd_label_questions(args: argparse.Namespace) -> int:
    """Label generated questions for one rubric, one pass at a time.

    One rubric per run, in that pass's own shuffled order, skipping questions
    already labelled for it so a session can be stopped and resumed. Each label
    is committed as it is made.

    The passes are deliberately separate. Judging several rubrics in one sitting
    invites a halo in which an item judged ungrounded is then judged irrelevant
    and easy on the strength of the first impression rather than the scale.
    """
    rubric = args.rubric
    pass_spec = LABEL_PASSES[rubric]
    keys = cast("dict[str, str]", pass_spec["keys"])
    legend = cast("tuple[str, ...]", pass_spec["legend"])
    seed = cast("str", pass_spec["seed"])

    with connect() as connection:
        done = labelled_question_ids(connection, rubric)
        pending = [
            question
            for question in labelling_order(list_questions(connection), seed)
            if question.question_id not in done
        ]

        if not pending:
            print(f"nothing left to label for {rubric}")
            return 0

        print(RUBRIC_REMINDERS[rubric])
        print(f"{len(done)} already labelled, {len(pending)} remaining\n")

        recorded = 0
        for position, question in enumerate(pending, start=len(done) + 1):
            if recorded >= args.limit:
                break
            cited = chunks_at(
                connection,
                [(question.doc_id, ordinal) for ordinal in question.cited_ordinals],
            )
            full = False
            redraw = True
            while True:
                # Only repaint when the display itself has changed. Repainting
                # after every key pushed the rubric off the screen the moment
                # "?" printed it, which reads as a dead key rather than as one
                # whose output was overwritten.
                if redraw:
                    show_question(
                        question, cited, position, len(done) + len(pending), full
                    )
                    redraw = False
                truncated = has_truncation(cited)
                prompt = (
                    "  ".join(legend)
                    + ("   f full text" if truncated and not full else "")
                    + "   ? rubric   q quit > "
                )
                key = read_key(prompt)

                if key == "q":
                    print("\nstopped")
                    return 0
                if key == "f":
                    full = True
                    redraw = True
                    continue
                if key == "?":
                    print(RUBRIC_REMINDERS[rubric])
                    continue
                if key not in keys:
                    print(f"  unrecognised key {key!r}")
                    continue

                value = keys[key]
                mode = None
                if rubric == "groundedness" and value in ("0", "1"):
                    mode = _read_failure_mode()
                    if mode is None:
                        continue

                record_label(connection, question.question_id, rubric, value, mode)
                connection.commit()
                recorded += 1
                break

        totals = label_totals(connection, rubric)
        print(f"\nrecorded {recorded} this session")
        print(f"{rubric} totals: {dict(sorted(totals.items()))}")
    return 0


def _read_failure_mode() -> str | None:
    """Read which kind of groundedness failure applies, or None to re-judge."""
    while True:
        key = read_key(
            "    which failure?  s support (not in this section)   "
            "c citation (in section, not cited)   r re-judge > "
        )
        if key == "r":
            return None
        if key in FAILURE_MODE_KEYS:
            return FAILURE_MODE_KEYS[key]
        print(f"    unrecognised key {key!r}")


def cmd_label(args: argparse.Namespace) -> int:
    """Judge pooled candidates for each query, one at a time.

    A query is offered when it has a candidate without a judgement, not when its
    judgement count is below some number. Those differ once the pool holds
    candidates from more than the three original retrievers: counting would stop
    offering a query that still had unjudged material in it.

    ``--per-query`` caps how many judgements one query may receive in a single
    session. It is a stopping rule for the person labelling, not a definition of
    which candidates are eligible.

    Candidates already judged are skipped, so a session can be stopped and
    resumed. Each judgement is committed as it is made, for the same reason.

    ``--regrade`` names one query and offers its whole pool again, including
    candidates already graded, so a judgement made in error can be replaced.
    The earlier grade is not shown: the second reading has to stand on its own.
    """
    with connect() as connection:
        if args.regrade is not None:
            selected = [
                q for q in list_queries(connection) if q.query_id == args.regrade
            ]
            if not selected:
                print(f"no query with id {args.regrade}")
                return 1
        else:
            selected = list_queries(connection)

        work: list[tuple[Query, list[Candidate]]] = []
        for query in selected:
            done: set[tuple[str, int]] = set()
            if args.regrade is None:
                done = judged_pairs(connection, query.query_id)
            pending = pending_candidates(connection, query, args, done)
            if pending:
                work.append((query, pending))

        if not work:
            print("nothing to label")
            return 0

        print(POLICY_REMINDER)

        for query, pending in work:
            recorded = 0
            for position, candidate in enumerate(pending, start=1):
                # The cap is a stopping rule for a normal session. A regrade has
                # to offer the whole pool: stopping halfway would leave a query
                # part regraded against two different readings.
                if args.regrade is None and recorded >= args.per_query:
                    break
                context = parent_text(
                    connection, candidate.chunk.doc_id, candidate.chunk.ordinal
                )
                full = False
                while True:
                    show_candidate(
                        query.text,
                        candidate,
                        position,
                        len(pending),
                        full,
                        context,
                    )
                    prompt = (
                        "  ".join(GRADE_MEANINGS)
                        + "   s skip   f full text   ? policy   q quit > "
                    )
                    key = read_key(prompt)

                    if key == "q":
                        print("\nstopped")
                        return 0
                    if key == "?":
                        print("\n" + POLICY_REMINDER)
                        continue
                    if key == "f":
                        full = True
                        continue
                    if not key:
                        print("\nstopped: input ended")
                        return 0
                    # An unrecognised key must not advance the candidate. Doing
                    # so is indistinguishable from a deliberate skip, so a burst
                    # of stray input would walk a whole pool leaving no trace.
                    if key in GRADE_KEYS or key == "s":
                        break
                    print(f"\n{key!r} is not one of 0, 1, 2, s, f, ? or q.")

                if key not in GRADE_KEYS:
                    continue

                record_judgement(
                    connection,
                    query.query_id,
                    candidate.chunk.doc_id,
                    candidate.chunk.ordinal,
                    GRADE_KEYS[key],
                )
                connection.commit()
                recorded += 1

        totals = grade_totals(connection)

    print(
        "\njudgements by grade: "
        + ", ".join(
            f"{grade}={totals.get(grade, 0)}" for grade in sorted(GRADE_KEYS.values())
        )
    )
    return 0
