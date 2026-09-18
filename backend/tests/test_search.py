import asyncio

import pytest
from sqlalchemy import func, update

from app.api.search import citation, source_link
from app.db.models import Chunk, Document
from app.llm.embeddings import EmbeddingError
from app.retrieval.search import keyword_ranking, search, vector_ranking


def in_session(sessions, query):
    """Run `query(session)` in a fresh session."""

    async def scenario():
        async with sessions() as session:
            return await query(session)

    return asyncio.run(scenario())


def ids(result) -> list[int]:
    return [hit.chunk.id for hit in result.hits]


def test_keyword_search_matches_any_word_of_a_question(sessions, corpus) -> None:
    # No chunk contains "deep", "recurrent" or "networks"; requiring all words would find nothing.
    question = "why do gradients vanish in deep recurrent networks"

    ranking = in_session(sessions, lambda session: keyword_ranking(session, question, 10))

    assert ranking == [corpus.vanishing, corpus.scaling]


def test_keyword_matches_in_a_section_title_rank_higher(sessions, corpus) -> None:
    ranking = in_session(sessions, lambda session: keyword_ranking(session, "softmax", 10))

    assert ranking == [corpus.softmax, corpus.scaling]


def test_keyword_ranking_is_divided_by_chunk_length(sessions, embedder) -> None:
    # Twenty matches among much other text rank below two matches in one sentence.
    training_log = " ".join(
        f"Step {step} logs the loss, the learning rate and the gradient norm." for step in range(20)
    )
    rows = [
        ("Training Loop", training_log),
        ("Backpropagation", "Gradients vanish when many small factors are multiplied."),
    ]

    async def query(session):
        notes = Document(source_type="pdf", title="Training Notes")
        notes.chunks = [
            Chunk(
                position=position,
                section=section,
                content_types=["text"],
                text=body,
                token_count=len(body.split()),
                embedding=embedder.vector(body),
            )
            for position, (section, body) in enumerate(rows)
        ]
        session.add(notes)
        await session.commit()
        ranking = await keyword_ranking(session, "why do gradients vanish", 10)
        return ranking, [chunk.id for chunk in notes.chunks]

    ranking, (long_chunk, short_chunk) = in_session(sessions, query)

    assert ranking == [short_chunk, long_chunk]


def test_search_leaves_out_superseded_chunks(sessions, embedder, corpus) -> None:
    """A chunk replaced by a later ingestion is kept for the questions that cite it, but no
    retriever may offer it again."""

    async def query(session):
        await session.execute(
            update(Chunk).where(Chunk.id == corpus.vanishing).values(superseded_at=func.now())
        )
        await session.commit()
        vector = await embedder.embed_query("gradients shrink over many time steps")
        return (
            await keyword_ranking(session, "why do gradients vanish", 10),
            await vector_ranking(session, vector, 10),
            await search(session, "vanishing gradients", limit=3, embedder=embedder),
        )

    keyword, vector, hybrid = in_session(sessions, query)

    assert keyword == [corpus.scaling]
    assert corpus.vanishing not in vector
    assert len(vector) == 4
    assert ids(hybrid)[0] == corpus.scaling
    assert corpus.vanishing not in ids(hybrid)


def test_vector_search_ranks_every_chunk_by_similarity(sessions, embedder, corpus) -> None:
    async def query(session):
        vector = await embedder.embed_query("square root of the key dimension")
        return await vector_ranking(session, vector, 10)

    ranking = in_session(sessions, query)

    assert ranking[0] == corpus.scaling
    assert len(ranking) == 5


def test_hybrid_search_fuses_both_rankings(sessions, embedder, corpus) -> None:
    result = in_session(
        sessions,
        lambda session: search(session, "vanishing gradients", limit=3, embedder=embedder),
    )

    assert (result.mode, result.warning) == ("hybrid", None)
    assert ids(result)[:2] == [corpus.vanishing, corpus.scaling]
    top, _, third = result.hits
    assert (top.vector_rank, top.keyword_rank, top.document.title) == (1, 1, "RNN Intuition")
    assert top.score == pytest.approx(2 / 61)
    # Found by the vectors only
    assert (third.vector_rank, third.keyword_rank) == (3, None)


def test_keyword_mode_does_not_need_the_embedding_model(sessions, corpus) -> None:
    result = in_session(
        sessions, lambda session: search(session, "softmax", mode="keyword", embedder=None)
    )

    assert (result.mode, result.warning) == ("keyword", None)
    assert ids(result) == [corpus.softmax, corpus.scaling]
    assert result.hits[0].vector_rank is None


def test_hybrid_search_falls_back_to_keywords(sessions, embedder, corpus) -> None:
    missing = in_session(sessions, lambda session: search(session, "softmax", embedder=None))
    embedder.fail = True
    failing = in_session(sessions, lambda session: search(session, "softmax", embedder=embedder))

    assert (missing.mode, failing.mode) == ("keyword", "keyword")
    assert missing.warning == (
        "semantic search needs the local embedding model; showing keyword matches only"
    )
    assert failing.warning.startswith("Ollama is not reachable")
    assert ids(missing) == ids(failing) == [corpus.softmax, corpus.scaling]


def test_vector_mode_reports_a_missing_embedding_model(sessions, corpus) -> None:
    with pytest.raises(EmbeddingError):
        in_session(
            sessions, lambda session: search(session, "softmax", mode="vector", embedder=None)
        )


@pytest.mark.parametrize(
    ("locators", "expected"),
    [
        ({"page_start": 7, "page_end": 8}, "RNN Intuition, pp. 7–8"),
        ({"page_start": 7, "page_end": 7, "section": "7 Hidden State"}, "RNN Intuition, p. 7"),
        ({"cell_start": 12, "cell_end": 14}, "RNN Intuition, cells 12–14"),
        ({"cell_start": 3}, "RNN Intuition, cell 3"),
        ({"section": "3 Model Architecture > 3.2 Attention"}, "RNN Intuition, § 3.2 Attention"),
        (
            {"section": "3 Model Architecture > 3.3 Layers · 3.4 Embeddings"},
            "RNN Intuition, § 3.3 Layers · 3.4 Embeddings",
        ),
        ({}, "RNN Intuition"),
    ],
)
def test_citations_name_the_page_cell_or_section(locators: dict, expected: str) -> None:
    assert citation(Document(title="RNN Intuition"), Chunk(**locators)) == expected


@pytest.mark.parametrize(
    ("url", "locators", "expected"),
    [
        (
            "https://arxiv.org/html/1706.03762v7",
            {"anchor": "S3.SS2", "section": "3 Model Architecture > 3.2 Attention"},
            "https://arxiv.org/html/1706.03762v7#S3.SS2",
        ),
        (
            "https://arxiv.org/pdf/2510.10824v1",
            {"page_start": 4, "page_end": 5},
            "https://arxiv.org/pdf/2510.10824v1#page=4",
        ),
        ("https://arxiv.org/html/1706.03762v7", {}, "https://arxiv.org/html/1706.03762v7"),
        # Uploaded files have no online source.
        (None, {"page_start": 4}, None),
    ],
)
def test_links_point_into_the_online_source(url, locators: dict, expected) -> None:
    assert source_link(Document(title="Paper", url=url), Chunk(**locators)) == expected
