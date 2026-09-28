"""The retrieval evaluation: scores, query sets, label files, fusion variants and the report."""

import asyncio
from datetime import date

import pytest
from sqlalchemy import func, update

from app.db.models import Chunk, Question, QuestionSource
from app.evaluation.retrieval import (
    ASKED,
    HAND,
    PASSAGE,
    Fusion,
    LabelFileError,
    Query,
    also_answers,
    first_relevant,
    fuse,
    generated_queries,
    hand_queries,
    judge_fusion,
    moved,
    score,
    unsearchable,
)
from app.retrieval.search import CANDIDATES, keyword_ranking, search, vector_ranking
from scripts.evaluate_retrieval import measure, render


def in_session(sessions, work):
    async def scenario():
        async with sessions() as session:
            return await work(session)

    return asyncio.run(scenario())


def query(id: str, gold: set[int], *sets: str) -> Query:
    return Query(id, f"question {id}", frozenset(gold), frozenset(sets))


# Scores


def test_first_relevant_is_the_rank_of_the_first_answer() -> None:
    assert first_relevant([7, 3, 5], frozenset({5, 3})) == 2
    assert first_relevant([7, 3], frozenset({9})) is None


def test_scores_count_the_top_five_only() -> None:
    rankings = [
        [1, 2, 3],  # answer first
        [9, 8, 7, 6, 5, 2],  # answer sixth: not a hit, no reciprocal rank
        [4, 9, 8, 7, 6],  # one of two answers in the top five
    ]
    relevant = [frozenset({1}), frozenset({2}), frozenset({4, 3})]

    scores = score(rankings, relevant)

    assert (scores.questions, scores.hit1, scores.hit5) == (3, 2, 2)
    assert scores.recall5 == pytest.approx((1 + 0 + 0.5) / 3)
    assert scores.mrr5 == pytest.approx((1 + 0 + 1) / 3)


def test_no_questions_score_zero() -> None:
    assert score([], []).mrr5 == 0.0


def test_moved_counts_a_missing_answer_as_last() -> None:
    assert moved([1, 3, None, 2], [1, 2, 4, None]) == (2, 1, 1)


# Label files


def test_only_confirmed_passages_count_as_also_answering() -> None:
    text = """
[[passage]]
question = "q4"
chunk = 175
verdict = "yes"

[[passage]]
question = "q4"
chunk = 193
verdict = "partly"

[[passage]]
question = "q7"
chunk = 293
verdict = "no"
"""
    assert also_answers(text) == {"q4": frozenset({175})}


@pytest.mark.parametrize(
    "row",
    [
        'question = "q4"\nchunk = 175\nverdict = "maybe"',
        'question = "q4"\nchunk = "175"\nverdict = "yes"',
        'chunk = 175\nverdict = "yes"',
    ],
)
def test_a_malformed_passage_is_refused(row: str) -> None:
    with pytest.raises(LabelFileError, match="passage 1"):
        also_answers(f"[[passage]]\n{row}\n")


def test_hand_questions_carry_their_labels() -> None:
    text = '[[question]]\nid = "C1.1"\ntext = "Why is CNN not enough?"\ngold = [285]\n'

    [only] = hand_queries(text)

    assert only == Query("C1.1", "Why is CNN not enough?", frozenset({285}), frozenset({HAND}))


def test_hand_questions_need_unique_ids_and_labels() -> None:
    row = '[[question]]\nid = "C1.1"\ntext = "Why?"\ngold = [1]\n'
    with pytest.raises(LabelFileError, match="unique"):
        hand_queries(row + row)
    with pytest.raises(LabelFileError, match="question 1"):
        hand_queries('[[question]]\nid = "C1.1"\ntext = "Why?"\n')


def test_a_file_that_is_not_toml_is_refused() -> None:
    with pytest.raises(LabelFileError):
        also_answers("from dotenv import load")


# Query sets


def add_question(session, chunk_ids, status="accepted", failed=()):
    question = Question(
        text=f"Why {status} {len(chunk_ids)}?",
        reference_answer="Because.",
        style="why_how",
        difficulty=2,
        status=status,
        validation={"failed": list(failed)} if failed else {},
        generator_model="openai/gpt-oss-120b",
        prompt_version="generate-v3",
    )
    session.add(question)
    return question, chunk_ids


def test_questions_fall_into_sets_by_status_and_why_they_were_turned_down(sessions, corpus) -> None:
    async def work(session):
        rows = [
            add_question(session, [corpus.scaling]),
            add_question(session, [corpus.positions], status="retired"),
            add_question(session, [corpus.softmax], status="rejected", failed=["duplicate"]),
            add_question(session, [corpus.vanishing], status="rejected", failed=["answerable"]),
            add_question(session, [corpus.scaling, corpus.retriever], status="accepted"),
        ]
        await session.flush()
        for question, chunk_ids in rows:
            for position, chunk_id in enumerate(chunk_ids):
                session.add(
                    QuestionSource(question_id=question.id, chunk_id=chunk_id, position=position)
                )
        # A question with no saved passages has no answers to find
        add_question(session, [])
        await session.commit()
        return [question.id for question, _ in rows], await generated_queries(session)

    ids, (queries, left_out) = in_session(sessions, work)

    by_id = {found.id: found for found in queries}
    accepted, retired, duplicate, _, pair = ids
    assert left_out == 1
    assert set(by_id) == {f"q{accepted}", f"q{retired}", f"q{duplicate}", f"q{pair}"}
    assert by_id[f"q{accepted}"].sets == {ASKED, PASSAGE}
    assert by_id[f"q{retired}"].sets == {ASKED, PASSAGE}
    assert by_id[f"q{duplicate}"].sets == {PASSAGE}
    assert by_id[f"q{pair}"].gold == {corpus.scaling, corpus.retriever}


def test_labels_search_cannot_return_are_reported(sessions, corpus) -> None:
    async def work(session):
        await session.execute(
            update(Chunk).where(Chunk.id == corpus.vanishing).values(superseded_at=func.now())
        )
        await session.commit()
        return await unsearchable(session, [corpus.scaling, corpus.vanishing, 999_999])

    assert in_session(sessions, work) == {corpus.vanishing, 999_999}


# Fusion


def test_the_current_fusion_is_what_search_does(sessions, embedder, corpus) -> None:
    text = "why do gradients vanish in deep recurrent networks"

    async def work(session):
        vector = await vector_ranking(session, embedder.vector(text), CANDIDATES)
        keyword = await keyword_ranking(session, text, CANDIDATES)
        found = await search(session, text, limit=10, embedder=embedder)
        return fuse(vector, keyword), [hit.chunk.id for hit in found.hits]

    fused, searched = in_session(sessions, work)

    assert fused[:10] == searched


def test_fewer_full_text_candidates_leave_out_its_tail() -> None:
    vector, keyword = [1, 2], [3, 4, 5]

    assert fuse(vector, keyword, Fusion(keyword_candidates=1, keyword_weight=1.0)) == [1, 3, 2]


FULL_SHARE = Fusion(keyword_candidates=50, keyword_weight=1.0)
HALF_SHARE = Fusion(keyword_candidates=50, keyword_weight=0.5)

# Vector ranks a distractor (201) first and the answer (100) second; full-text ranks another
# distractor (300) first. At full share the two firsts tie ahead of the answer, which comes
# third; at half share full-text's first falls behind it, which comes second.
VECTOR_KNOWS = ([201, 100], [300])
# The same shape, but the answer is full-text's first (300): at half share it falls to third.
FULL_TEXT_KNOWS = ([201, 202], [300])


def rankings(**shapes):
    vector = {id: shape[0] for id, shape in shapes.items()}
    keyword = {id: shape[1] for id, shape in shapes.items()}
    return vector, keyword


def test_a_variant_that_wins_on_every_set_is_adopted() -> None:
    queries = [
        query("t1", {100}, PASSAGE),
        query("a1", {100}, ASKED, PASSAGE),
        query("h1", {201}, HAND),
    ]
    vector, keyword = rankings(t1=VECTOR_KNOWS, a1=VECTOR_KNOWS, h1=VECTOR_KNOWS)
    relevant = {found.id: found.gold for found in queries}

    verdict = judge_fusion(
        queries, vector, keyword, relevant, current=FULL_SHARE, grid=[FULL_SHARE, HALF_SHARE]
    )

    assert [value for _, value in verdict.tuning] == pytest.approx([1 / 3, 1 / 2])
    assert verdict.chosen == HALF_SHARE
    assert verdict.asked == pytest.approx((1 / 3, 1 / 2))
    assert verdict.asked_moved == (1, 0, 0)
    assert verdict.hand == pytest.approx((1.0, 1.0))
    assert verdict.adopted


def test_a_variant_that_hurts_the_hand_written_questions_is_not_adopted() -> None:
    queries = [
        query("t1", {100}, PASSAGE),
        query("a1", {100}, ASKED, PASSAGE),
        query("h1", {300}, HAND),
    ]
    vector, keyword = rankings(t1=VECTOR_KNOWS, a1=VECTOR_KNOWS, h1=FULL_TEXT_KNOWS)
    relevant = {found.id: found.gold for found in queries}

    verdict = judge_fusion(
        queries, vector, keyword, relevant, current=FULL_SHARE, grid=[FULL_SHARE, HALF_SHARE]
    )

    assert verdict.chosen == HALF_SHARE
    assert verdict.hand == pytest.approx((1 / 2, 1 / 3))
    assert not verdict.adopted


def test_ties_keep_the_current_setting() -> None:
    queries = [query("t1", {1}, PASSAGE), query("a1", {1}, ASKED, PASSAGE)]
    same = {id: [1, 2, 3] for id in ("t1", "a1")}

    verdict = judge_fusion(
        queries,
        same,
        same,
        {found.id: found.gold for found in queries},
        current=HALF_SHARE,
        grid=[FULL_SHARE, HALF_SHARE],
    )

    assert verdict.chosen == HALF_SHARE
    assert not verdict.adopted


# The report


def test_the_report_scores_each_set_apart(sessions, embedder, corpus) -> None:
    async def work(session):
        question, _ = add_question(session, [corpus.vanishing])
        question.text = "Why do gradients vanish in recurrent networks?"
        await session.flush()
        session.add(QuestionSource(question_id=question.id, chunk_id=corpus.vanishing, position=0))
        await session.commit()
        hand = [
            Query(
                "C1.1", "What does FAISS return?", frozenset({corpus.retriever}), frozenset({HAND})
            )
        ]
        also = {f"q{question.id}": frozenset({corpus.scaling})}
        return await measure(session, embedder, "fake", hand, also, [], date(2026, 9, 28))

    found = in_session(sessions, work)
    report = render(found, None)

    assert report.startswith("# Retrieval report, 2026-09-28")
    assert "## Asked: the questions practice serves (1 questions)" in report
    assert "## Hand-written: the Phase 1 questions, labelled by hand (1 questions)" in report
    assert "| Hybrid | 1 | 1 | 1.00 | 1.00 | 1 | 1 | 1.00 |" in report
    assert "1 passages confirmed to also answer 1 questions." in report


def test_without_the_embedding_model_only_full_text_is_measured(sessions, embedder, corpus) -> None:
    embedder.fail = True
    hand = [
        Query("C1.1", "What does FAISS return?", frozenset({corpus.retriever}), frozenset({HAND}))
    ]

    found = in_session(
        sessions,
        lambda session: measure(session, embedder, "fake", hand, {}, [], date(2026, 9, 28)),
    )
    report = render(found, None)

    assert found.vector is None
    assert "| Full-text |" in report
    assert "| Hybrid |" not in report and "| Vector |" not in report
    assert "Vector and hybrid search not measured" in report
