"""The question endpoints: starting a batch, reading the library, correcting and retiring a
question, browsing the topic map and having it built."""

import asyncio

import pytest
from sqlalchemy import func, select, update

from app.api.questions import MAX_TEXT_CHARS
from app.db.models import (
    Chunk,
    ChunkTags,
    ChunkTopic,
    Job,
    Question,
    QuestionSource,
    QuestionTask,
    Topic,
)
from app.questions.validation import nearest_question

SCALED = "Why are the dot products divided by the square root of the key dimension?"
GRADIENTS = "What happens to the gradient signal as it travels back through a long sequence?"
LOOKUP = "How does a FAISS lookup differ from reading every chunk in turn?"

ACCEPTED_REPORT = {
    "prompt_version": "check-v2",
    "checker_model": "qwen3.5:4b",
    "kind": "explain",
    "answerable": True,
    "missing": "nothing",
    "checker_answer": "Large dot products push the softmax where its gradients vanish.",
    "nearest_question": None,
    "nearest_similarity": None,
    "answer_agreement": 0.8123,
    "failed": [],
}


def a_question(**fields) -> Question:
    base = {
        "misconceptions": [],
        "generator_model": "groq:openai/gpt-oss-120b",
        "prompt_version": "generate-v1",
        "usage": {"requests": 1, "input_tokens": 1506, "output_tokens": 778, "attempts": 1},
    }
    return Question(**(base | fields))


@pytest.fixture
def library(sessions, embedder, corpus) -> dict[str, int]:
    """The corpus tagged and filed under topics, the way `make topics` leaves it."""
    filed = {
        "attention": (
            [corpus.scaling, corpus.positions, corpus.softmax],
            ["attention", "scaled dot-product attention", "positional encoding"],
        ),
        "retrieval": ([corpus.retriever], ["retrieval", "faiss index"]),
        "vanishing gradient": ([corpus.vanishing], ["vanishing gradient"]),
        # Left behind by an earlier run: a question still points at it, no chunk does
        "beam search": ([], ["beam search"]),
    }

    async def prepare() -> dict[str, int]:
        async with sessions() as session, session.begin():
            ids: dict[str, int] = {}
            for name, (chunk_ids, tags) in filed.items():
                topic = Topic(name=name, tags=tags, embedding=embedder.vector(name))
                session.add(topic)
                await session.flush()
                ids[name] = topic.id
                session.add_all(
                    ChunkTopic(chunk_id=chunk_id, topic_id=topic.id) for chunk_id in chunk_ids
                )
            session.add_all(
                ChunkTags(
                    chunk_id=chunk_id,
                    explains="how attention scores are scaled",
                    tags=["attention"],
                    worth_asking=True,
                    model_worth_asking=True,
                    model="ollama:qwen3.5:4b",
                    prompt_version="tags-v2",
                )
                for chunk_id in (corpus.scaling, corpus.positions, corpus.softmax, corpus.vanishing)
            )
            # Code with nothing to explain: kept as context, never asked about
            session.add(
                ChunkTags(
                    chunk_id=corpus.retriever,
                    explains="nothing",
                    tags=["retrieval"],
                    worth_asking=False,
                    model_worth_asking=False,
                    skip_reason="code only",
                    model="ollama:qwen3.5:4b",
                    prompt_version="tags-v2",
                )
            )
            return ids

    return asyncio.run(prepare())


@pytest.fixture
def questions(sessions, embedder, corpus, library) -> dict[str, int]:
    """Three questions: two accepted, one turned down as a near-duplicate."""

    async def prepare() -> dict[str, int]:
        async with sessions() as session, session.begin():
            scaled = a_question(
                text=SCALED,
                reference_answer="Their variance grows with the dimension, and large values "
                "push the softmax into a region with tiny gradients.",
                key_points=[
                    {
                        "text": "the dot products grow with the key dimension",
                        "weight": 2,
                        "evidence_quote": "We divide the dot products by the square root of "
                        "the key dimension",
                        "chunk_id": corpus.scaling,
                    },
                    {
                        "text": "a flatter distribution carries more gradient",
                        "weight": 1,
                        "evidence_quote": "Dividing logits by a temperature flattens the "
                        "distribution",
                        "chunk_id": corpus.softmax,
                    },
                ],
                misconceptions=["It rescales the vectors to unit length."],
                style="why_how",
                difficulty=3,
                topic_id=library["attention"],
                validation=ACCEPTED_REPORT,
                embedding=embedder.vector(SCALED),
            )
            gradients = a_question(
                text=GRADIENTS,
                reference_answer="It shrinks with every step, so early steps barely learn.",
                key_points=[
                    {
                        "text": "the gradient shrinks step by step",
                        "weight": 3,
                        "evidence_quote": "Gradients shrink as they flow back through many "
                        "time steps",
                        "chunk_id": corpus.vanishing,
                    }
                ],
                style="intuition",
                difficulty=2,
                topic_id=library["vanishing gradient"],
                validation=ACCEPTED_REPORT,
                embedding=embedder.vector(GRADIENTS),
            )
            session.add_all([scaled, gradients])
            await session.flush()
            lookup = a_question(
                text=LOOKUP,
                reference_answer="It compares the question embedding against an index instead.",
                key_points=[
                    {
                        "text": "the index returns the nearest chunks",
                        "weight": 2,
                        "evidence_quote": "FAISS returns the chunks closest to the question "
                        "embedding",
                        "chunk_id": corpus.retriever,
                    }
                ],
                style="compare",
                difficulty=4,
                topic_id=library["retrieval"],
                status="rejected",
                validation=ACCEPTED_REPORT
                | {
                    "nearest_question": scaled.id,
                    "nearest_similarity": 0.737,
                    "failed": ["duplicate"],
                },
                embedding=embedder.vector(LOOKUP),
            )
            session.add(lookup)
            await session.flush()
            sources = {
                scaled.id: [corpus.scaling, corpus.softmax],
                gradients.id: [corpus.vanishing],
                lookup.id: [corpus.retriever],
            }
            session.add_all(
                QuestionSource(question_id=question_id, chunk_id=chunk_id, position=position)
                for question_id, chunk_ids in sources.items()
                for position, chunk_id in enumerate(chunk_ids)
            )
            return {"scaled": scaled.id, "gradients": gradients.id, "lookup": lookup.id}

    return asyncio.run(prepare())


def supersede(sessions, chunk_id: int) -> None:
    async def replace() -> None:
        async with sessions() as session, session.begin():
            await session.execute(
                update(Chunk).where(Chunk.id == chunk_id).values(superseded_at=func.now())
            )

    asyncio.run(replace())


def stored(sessions, question_id: int) -> Question:
    async def load() -> Question:
        async with sessions() as session:
            return await session.get_one(Question, question_id)

    return asyncio.run(load())


def scaling_points(corpus) -> list[dict]:
    """Two key points for the scaling question, both quoted from its first source."""
    return [
        {
            "text": "the scale is the square root of the key dimension",
            "weight": 2,
            "evidence_quote": "We divide the dot products by the square root of the key dimension",
            "chunk_id": corpus.scaling,
        },
        {
            "text": "the scaling keeps the softmax gradients large",
            "weight": 3,
            "evidence_quote": "which keeps the softmax gradients large",
            "chunk_id": corpus.scaling,
        },
    ]


def test_a_batch_is_planned_and_queued_once(client, sessions, corpus, library) -> None:
    first = client.post("/questions/generate", json={"count": 3})
    again = client.post("/questions/generate", json={"count": 3})

    async def written():
        async with sessions() as session:
            tasks = list(
                await session.scalars(select(QuestionTask).order_by(QuestionTask.position))
            )
            jobs = await session.scalar(select(func.count()).select_from(Job))
            return tasks, jobs

    tasks, jobs = asyncio.run(written())

    assert first.status_code == 202
    body = first.json()
    assert (body["message"], body["planned"]) == ("queued", 3)
    job = body["job"]
    assert (job["kind"], job["status"], job["document_id"]) == ("generate", "queued", None)
    assert job["options"] == {"count": 3, "document_id": None, "planned": 3}
    # The plan is written down before anything runs, and leaves out the chunk that is context
    assert [task.status for task in tasks] == ["queued"] * 3
    assert corpus.retriever not in {chunk_id for task in tasks for chunk_id in task.chunk_ids}
    # A second batch would plan the same passages and pay for them twice, so it is refused
    assert again.status_code == 202
    assert (again.json()["message"], again.json()["planned"]) == ("already queued", 3)
    assert again.json()["job"]["id"] == job["id"]
    assert jobs == 1


def test_one_document_can_be_asked_about_on_its_own(client, sessions, corpus, library) -> None:
    async def paper_id() -> int:
        async with sessions() as session:
            return await session.scalar(select(Chunk.document_id).where(Chunk.id == corpus.scaling))

    document_id = asyncio.run(paper_id())
    response = client.post("/questions/generate", json={"count": 5, "document_id": document_id})

    async def planned() -> list[list[int]]:
        async with sessions() as session:
            return list(await session.scalars(select(QuestionTask.chunk_ids)))

    chunk_ids = {chunk_id for task in asyncio.run(planned()) for chunk_id in task}

    assert response.status_code == 202
    assert response.json()["job"]["options"]["document_id"] == document_id
    # Only the paper's chunks, and only the three worth asking about
    assert chunk_ids == {corpus.scaling, corpus.positions, corpus.softmax}


def test_nothing_is_queued_when_there_is_nothing_left_to_ask_about(client, sessions, corpus):
    response = client.post("/questions/generate", json={"count": 3})

    async def jobs() -> int:
        async with sessions() as session:
            return await session.scalar(select(func.count()).select_from(Job))

    assert response.status_code == 200
    assert response.json()["job"] is None
    assert response.json()["planned"] == 0
    assert "nothing left to ask about" in response.json()["message"]
    # Nothing was written down, not even an empty job
    assert asyncio.run(jobs()) == 0


def test_a_batch_needs_a_document_that_exists_and_a_local_machine(client, settings, library):
    missing = client.post("/questions/generate", json={"count": 1, "document_id": 999})
    settings.environment = "production"
    remote = client.post("/questions/generate", json={"count": 1})

    assert missing.status_code == 404
    assert remote.status_code == 403
    assert "runs locally" in remote.json()["detail"]


def test_questions_are_listed_newest_first_with_their_citations(client, questions) -> None:
    body = client.get("/questions").json()

    assert (body["total"], body["limit"], body["offset"]) == (3, 20, 0)
    assert [question["id"] for question in body["results"]] == [
        questions["lookup"],
        questions["gradients"],
        questions["scaled"],
    ]
    scaled = body["results"][-1]
    assert (scaled["text"], scaled["style"], scaled["difficulty"]) == (SCALED, "why_how", 3)
    assert (scaled["status"], scaled["topic"], scaled["source_updated"]) == (
        "accepted",
        "attention",
        False,
    )
    assert scaled["citations"] == [
        "Attention Is All You Need, § 3.2.1 Scaled Dot-Product Attention",
        "Attention Is All You Need, § Softmax Temperature",
    ]
    assert len(scaled["document_ids"]) == 1


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"status": "accepted"}, ["gradients", "scaled"]),
        ({"status": "rejected"}, ["lookup"]),
        ({"style": "intuition"}, ["gradients"]),
        ({"difficulty": 4}, ["lookup"]),
        ({"status": "accepted", "difficulty": 3}, ["scaled"]),
        ({"status": "retired"}, []),
        ({"source_updated": "false"}, ["lookup", "gradients", "scaled"]),
    ],
)
def test_every_filter_narrows_the_list(client, questions, params, expected) -> None:
    body = client.get("/questions", params=params).json()

    assert [question["id"] for question in body["results"]] == [questions[key] for key in expected]
    assert body["total"] == len(expected)


def test_questions_can_be_filtered_by_topic_and_by_source(client, sessions, corpus, questions):
    async def document_ids() -> tuple[int, int]:
        async with sessions() as session:
            rows = await session.execute(
                select(Chunk.id, Chunk.document_id).where(
                    Chunk.id.in_([corpus.scaling, corpus.vanishing])
                )
            )
            found = dict(rows.tuples().all())
            return found[corpus.scaling], found[corpus.vanishing]

    paper, notes = asyncio.run(document_ids())
    topic_id = client.get("/questions").json()["results"][-1]["topic_id"]

    by_topic = client.get("/questions", params={"topic_id": topic_id}).json()
    by_paper = client.get("/questions", params={"document_id": paper}).json()
    by_notes = client.get("/questions", params={"document_id": notes}).json()

    assert [question["id"] for question in by_topic["results"]] == [questions["scaled"]]
    assert [question["id"] for question in by_paper["results"]] == [questions["scaled"]]
    assert [question["id"] for question in by_notes["results"]] == [questions["gradients"]]


def test_the_list_pages_without_losing_the_total(client, questions) -> None:
    first = client.get("/questions", params={"limit": 2}).json()
    second = client.get("/questions", params={"limit": 2, "offset": 2}).json()

    assert (first["total"], len(first["results"])) == (3, 2)
    assert (second["total"], second["offset"], len(second["results"])) == (3, 2, 1)
    assert {question["id"] for question in first["results"] + second["results"]} == set(
        questions.values()
    )


@pytest.mark.parametrize(
    "params",
    [
        {"status": "draft"},
        {"style": "trivia"},
        {"difficulty": 0},
        {"difficulty": 6},
        {"rating": "great"},
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
    ],
)
def test_unknown_filters_are_rejected(client, params) -> None:
    assert client.get("/questions", params=params).status_code == 422


def test_a_question_comes_with_its_sources_key_points_and_report(client, corpus, questions) -> None:
    body = client.get(f"/questions/{questions['scaled']}").json()

    assert body["reference_answer"].startswith("Their variance grows")
    assert body["misconceptions"] == ["It rescales the vectors to unit length."]
    assert [(point["weight"], point["chunk_id"]) for point in body["key_points"]] == [
        (2, corpus.scaling),
        (1, corpus.softmax),
    ]
    assert body["key_points"][0]["evidence_quote"].startswith("We divide the dot products")
    # The whole report is kept, so a verdict can be read back long after the run
    assert body["validation"] == ACCEPTED_REPORT
    assert body["generator_model"] == "groq:openai/gpt-oss-120b"
    assert (body["prompt_version"], body["usage"]["input_tokens"]) == ("generate-v1", 1506)
    # Sources in the order the model was shown them, cited as search results are
    first, second = body["sources"]
    assert (first["chunk_id"], second["chunk_id"]) == (corpus.scaling, corpus.softmax)
    assert first["citation"] == "Attention Is All You Need, § 3.2.1 Scaled Dot-Product Attention"
    assert first["link"] == "https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1"
    assert first["text"].startswith("We divide the dot products")
    assert first["section"].endswith("Scaled Dot-Product Attention")
    assert [source["superseded"] for source in body["sources"]] == [False, False]
    assert client.get("/questions/999").status_code == 404


def test_a_rejected_question_says_what_turned_it_down(client, questions) -> None:
    body = client.get(f"/questions/{questions['lookup']}").json()

    assert body["status"] == "rejected"
    assert body["validation"]["failed"] == ["duplicate"]
    assert body["validation"]["nearest_question"] == questions["scaled"]
    assert body["validation"]["nearest_similarity"] == 0.737


def test_questions_are_flagged_when_a_later_ingestion_replaces_a_source(
    client, sessions, corpus, questions
) -> None:
    supersede(sessions, corpus.softmax)

    listed = {
        question["id"]: question["source_updated"]
        for question in client.get("/questions").json()["results"]
    }
    flagged = client.get("/questions", params={"source_updated": "true"}).json()
    detail = client.get(f"/questions/{questions['scaled']}").json()

    assert listed == {
        questions["scaled"]: True,
        questions["gradients"]: False,
        questions["lookup"]: False,
    }
    assert [question["id"] for question in flagged["results"]] == [questions["scaled"]]
    # The question still points at the text it was written from, and says which part moved on
    assert [source["superseded"] for source in detail["sources"]] == [False, True]
    assert detail["sources"][1]["text"].startswith("Dividing logits by a temperature")


def test_a_question_is_corrected_and_the_correction_is_kept(
    client, sessions, embedder, corpus, questions
) -> None:
    edited = "Why is the softmax input scaled by the square root of the key dimension?"
    response = client.patch(
        f"/questions/{questions['scaled']}",
        json={
            "text": edited,
            "reference_answer": "Scaling keeps the softmax gradients large.",
            "key_points": scaling_points(corpus),
            "reason": "the second key point quoted a passage about temperature",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["text"], body["reference_answer"]) == (
        edited,
        "Scaling keeps the softmax gradients large.",
    )
    assert body["key_points"] == scaling_points(corpus)
    # The checks the question first went through stay, and the correction joins them
    report = body["validation"]
    assert {key: report[key] for key in ACCEPTED_REPORT} == ACCEPTED_REPORT
    [edit] = report["edits"]
    assert edit["reason"] == "the second key point quoted a passage about temperature"
    assert set(edit["changes"]) == {"text", "reference_answer", "key_points"}
    assert edit["changes"]["text"] == {"from": SCALED, "to": edited}
    # What was replaced is kept, so an answer graded before the correction can be read again
    assert [point["chunk_id"] for point in edit["changes"]["key_points"]["from"]] == [
        corpus.scaling,
        corpus.softmax,
    ]
    assert [(quote["score"], quote["problem"]) for quote in edit["quotes"]] == [(100.0, None)] * 2
    # The new words are embedded for the duplicate check
    assert edit["embedding"] == "updated"
    assert embedder.documents == [edited]
    vector = stored(sessions, questions["scaled"]).embedding
    assert max(abs(a - b) for a, b in zip(vector, embedder.vector(edited), strict=True)) < 1e-3
    assert client.get(f"/questions/{questions['scaled']}").json() == body


def test_a_key_point_whose_quote_is_not_in_its_source_is_refused(
    client, embedder, corpus, questions
) -> None:
    right, _ = scaling_points(corpus)
    response = client.patch(
        f"/questions/{questions['scaled']}",
        json={
            "text": "Why scale the dot products at all?",
            "key_points": [
                right,
                # Words the passage does not have
                right
                | {"evidence_quote": "We multiply the dot products by the key dimension squared"},
                # Words from a passage, but not one the question was written from
                right
                | {
                    "evidence_quote": "Gradients shrink as they flow back through many time steps",
                    "chunk_id": corpus.vanishing,
                },
                # Too few words to show anything
                right | {"evidence_quote": "keeps the softmax gradients large"},
            ],
        },
    )

    assert response.status_code == 422
    problems = response.json()["detail"]
    # Each problem points at the field to fix, the way a malformed request's would
    assert [problem["loc"] for problem in problems] == [
        ["body", "key_points", 1, "evidence_quote"],
        ["body", "key_points", 2, "chunk_id"],
        ["body", "key_points", 3, "evidence_quote"],
    ]
    assert problems[0]["msg"].startswith(f"the quote is not in chunk {corpus.scaling} (closest")
    assert (problems[1]["msg"], problems[1]["input"]) == (
        f"chunk {corpus.vanishing} is not one of the sources",
        corpus.vanishing,
    )
    assert problems[2]["msg"] == "the quote is shorter than 6 words"
    # Nothing changed, not even the text sent along with the key points
    detail = client.get(f"/questions/{questions['scaled']}").json()
    assert detail["text"] == SCALED
    assert [point["chunk_id"] for point in detail["key_points"]] == [corpus.scaling, corpus.softmax]
    assert "edits" not in detail["validation"]
    assert embedder.documents == []


def test_a_new_text_goes_without_an_embedding_while_the_model_is_away(
    client, sessions, embedder, questions
) -> None:
    embedder.fail = True
    edited = "What happens to the gradient signal over a long sequence?"

    response = client.patch(f"/questions/{questions['gradients']}", json={"text": edited})

    assert response.status_code == 200
    assert response.json()["text"] == edited
    assert response.json()["validation"]["edits"][0]["embedding"] == "cleared"
    assert stored(sessions, questions["gradients"]).embedding is None

    async def nearest() -> tuple[int, float] | None:
        async with sessions() as session:
            return await nearest_question(session, embedder.vector(GRADIENTS))

    # The duplicate check passes over it, rather than compare against words it no longer has
    assert asyncio.run(nearest())[0] == questions["scaled"]


def test_a_retired_question_is_never_picked_until_it_is_put_back(client, questions) -> None:
    def picked() -> int:
        return client.get("/practice/next").json()["question"]["id"]

    # Both are new and their topics equally weak, so the tie goes to the older topic
    assert picked() == questions["scaled"]

    retired = client.patch(
        f"/questions/{questions['scaled']}",
        json={"status": "retired", "reason": "its key points are only reported results"},
    )

    assert retired.status_code == 200
    assert retired.json()["status"] == "retired"
    assert picked() == questions["gradients"]
    listed = client.get("/questions", params={"status": "retired"}).json()
    assert [question["id"] for question in listed["results"]] == [questions["scaled"]]
    client.patch(f"/questions/{questions['gradients']}", json={"status": "retired"})
    assert client.get("/practice/next").status_code == 404

    back = client.patch(f"/questions/{questions['scaled']}", json={"status": "accepted"})

    assert back.status_code == 200
    assert picked() == questions["scaled"]
    edits = back.json()["validation"]["edits"]
    assert [(edit["changes"], edit["reason"]) for edit in edits] == [
        (
            {"status": {"from": "accepted", "to": "retired"}},
            "its key points are only reported results",
        ),
        ({"status": {"from": "retired", "to": "accepted"}}, None),
    ]


def test_a_rejected_or_unknown_question_cannot_be_edited(client, questions) -> None:
    missing = client.patch("/questions/999", json={"status": "retired"})
    rejected = client.patch(f"/questions/{questions['lookup']}", json={"status": "accepted"})

    assert missing.status_code == 404
    assert rejected.status_code == 409
    assert "turned down" in rejected.json()["detail"]
    assert client.get(f"/questions/{questions['lookup']}").json()["status"] == "rejected"


POINT = {
    "text": "a point",
    "weight": 2,
    "evidence_quote": "six words copied from a source",
    "chunk_id": 1,
}


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="nothing"),
        pytest.param({"reason": "tidying up"}, id="only a reason"),
        pytest.param({"status": "rejected"}, id="rejected by hand"),
        pytest.param({"text": "   "}, id="blank text"),
        pytest.param({"text": "x" * (MAX_TEXT_CHARS + 1)}, id="text too long"),
        pytest.param({"key_points": [POINT]}, id="one key point"),
        pytest.param({"key_points": [POINT] * 5}, id="five key points"),
        pytest.param({"key_points": [POINT | {"weight": 4}] * 2}, id="weight 4"),
        pytest.param({"style": "compare"}, id="not editable"),
    ],
)
def test_an_edit_has_to_be_one_that_can_be_made(client, questions, body) -> None:
    assert client.patch(f"/questions/{questions['scaled']}", json=body).status_code == 422


def test_an_edit_that_changes_nothing_is_not_recorded(client, embedder, questions) -> None:
    response = client.patch(
        f"/questions/{questions['scaled']}", json={"text": SCALED, "status": "accepted"}
    )

    assert response.status_code == 200
    assert "edits" not in response.json()["validation"]
    assert embedder.documents == []


def test_topics_carry_their_chunk_and_question_counts(client, questions) -> None:
    body = client.get("/topics").json()
    counts = {
        topic["name"]: (topic["chunk_count"], topic["question_count"], topic["accepted_count"])
        for topic in body
    }

    # The topics with the most passages behind them first
    assert [topic["name"] for topic in body] == [
        "attention",
        "retrieval",
        "vanishing gradient",
        "beam search",
    ]
    assert counts["attention"] == (3, 1, 1)
    # The rejected question still counts towards its topic, but not as an accepted one
    assert counts["retrieval"] == (1, 1, 0)
    assert counts["vanishing gradient"] == (1, 1, 1)
    # A topic no chunk is filed under any more
    assert counts["beam search"] == (0, 0, 0)
    assert body[0]["tags"] == ["attention", "scaled dot-product attention", "positional encoding"]


def test_a_superseded_chunk_leaves_its_topic(client, sessions, corpus, library) -> None:
    supersede(sessions, corpus.positions)

    counts = {topic["name"]: topic["chunk_count"] for topic in client.get("/topics").json()}

    assert counts["attention"] == 2


def test_the_topic_map_pages(client, library) -> None:
    first = client.get("/topics", params={"limit": 2}).json()
    rest = client.get("/topics", params={"limit": 2, "offset": 2}).json()

    assert [topic["name"] for topic in first] == ["attention", "retrieval"]
    assert [topic["name"] for topic in rest] == ["vanishing gradient", "beam search"]


def test_the_topic_map_is_queued_once(client, sessions, corpus) -> None:
    async def tag_one() -> None:
        async with sessions() as session, session.begin():
            session.add(
                ChunkTags(
                    chunk_id=corpus.scaling,
                    explains="how attention scores are scaled",
                    tags=["attention"],
                    worth_asking=True,
                    model_worth_asking=True,
                    model="ollama:qwen3.5:4b",
                    prompt_version="tags-v2",
                )
            )

    async def jobs() -> int:
        async with sessions() as session:
            return await session.scalar(select(func.count()).select_from(Job))

    asyncio.run(tag_one())
    first = client.post("/topics/build")
    again = client.post("/topics/build")

    assert first.status_code == 202
    body = first.json()
    # Four of the five passages are still to be tagged
    assert (body["message"], body["untagged"]) == ("queued", 4)
    job = body["job"]
    assert (job["kind"], job["status"], job["document_id"], job["options"]) == (
        "topics",
        "queued",
        None,
        {},
    )
    # The queued build tags whatever is untagged when it runs, so a second one adds nothing.
    assert again.status_code == 202
    assert (again.json()["message"], again.json()["job"]["id"]) == ("already queued", job["id"])
    assert asyncio.run(jobs()) == 1


def test_there_is_no_topic_map_to_build_without_passages(client, sessions) -> None:
    response = client.post("/topics/build")

    async def jobs() -> int:
        async with sessions() as session:
            return await session.scalar(select(func.count()).select_from(Job))

    assert response.status_code == 200
    assert (response.json()["job"], response.json()["untagged"]) == (None, 0)
    assert "add material first" in response.json()["message"]
    assert asyncio.run(jobs()) == 0


def test_the_topic_map_is_built_on_a_local_machine_only(client, settings, corpus) -> None:
    settings.environment = "production"

    response = client.post("/topics/build")

    assert response.status_code == 403
    assert "built locally" in response.json()["detail"]
    assert client.get("/topics").status_code == 200
