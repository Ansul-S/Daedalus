"""Writing a question from source chunks, and making its evidence quotes hold up."""

import asyncio
import json

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from app.db.models import Chunk
from app.questions.generation import (
    GeneratedQuestion,
    Source,
    compound,
    generate_question,
    request,
    source_of,
)
from app.questions.grounding import check_quote, normalize

CHUNK = (
    "We suspect that for large values of $d_k$, the dot products grow large in magnitude, "
    "pushing the softmax function into regions where it has extremely small gradients.\n\n"
    "To counteract this effect, we scale the dot products by $1/\\sqrt{d_k}$."
)
SOURCES = [
    Source(
        chunk_id=248,
        citation="Attention Is All You Need > 3.2.1 Scaled Dot-Product Attention",
        text=CHUNK,
    )
]
GOOD_QUOTE = "the dot products grow large in magnitude, pushing the softmax function"


@pytest.mark.parametrize(
    ("quote", "problem"),
    [
        (GOOD_QUOTE, None),
        # The dollar signs around inline maths are spelling, not wording.
        ("for large values of d_k, the dot products grow large", None),
        # A line break in the source, a space in the quote.
        ("extremely small gradients. To counteract this effect, we scale", None),
        # U+2011 for the plain hyphen, and NFKC-foldable characters.
        ("we scale the dot‑products by 1/\\sqrt{d_k}", None),
        ("THE DOT PRODUCTS GROW LARGE IN MAGNITUDE, PUSHING", None),
        # The punctuation a quote ends on is spelling too: a colon, or an ellipsis saying the
        # sentence runs on, is judged by the score like the words before it.
        ("To counteract this effect, we scale the dot products by:", None),
        ("pushing the softmax function into regions where...", None),
        (
            "the dot products grow large ... extremely small gradients",
            "the quote leaves words out instead of running on",
        ),
        ("we scale the dot products", "the quote is shorter than 6 words"),
        # A mark left standing at the end is not a sixth word.
        ("we scale the dot products …", "the quote is shorter than 6 words"),
        (
            "the learning rate is decayed over the first four thousand steps",
            "the quote is not in chunk 248",
        ),
    ],
)
def test_a_quote_has_to_be_in_the_chunk_it_names(quote: str, problem: str | None) -> None:
    check = check_quote(quote, 248, CHUNK)

    assert check.grounded == (problem is None)
    if problem is not None:
        assert check.problem.startswith(problem)


def test_a_quote_keeps_an_ellipsis_the_source_writes_itself() -> None:
    """A paper that writes a sequence as (x_1, ..., x_n) is quoted faithfully with the dots
    in it. Only a quote that stops matching its chunk is one that left words out."""
    chunk = (
        "The encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a "
        "sequence of continuous representations $\\mathbf{z} = (z_1, ..., z_n)$."
    )
    quote = "maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence"

    assert check_quote(quote, 1, chunk).grounded


def test_a_quote_may_stop_at_the_colon_that_opens_a_list() -> None:
    """Refusing a colon on sight turned away quotes copied character for character up to the
    list they introduce. The quote is trimmed to be judged, and kept as it was written."""
    chunk = (
        "Keeping retrieval separate makes the output easier to debug, because we can inspect:"
        "\n\n- which chunks were retrieved\n- how each one was scored"
    )
    quote = "makes the output easier to debug, because we can inspect:"

    check = check_quote(quote, 1, chunk)

    assert (check.grounded, check.score, check.quote) == (True, 100.0, quote)


def test_a_quote_cannot_name_a_chunk_that_was_not_a_source() -> None:
    check = check_quote(GOOD_QUOTE, 999, None)

    assert check.problem == "chunk 999 is not one of the sources"


def test_spelling_is_normalized_away_but_words_are_not() -> None:
    assert normalize("**Scaled** $dot‑product$  attention") == "scaled dot-product attention"


def test_the_request_shows_every_source_in_delimiters() -> None:
    prompt = request(SOURCES, "why_how")

    assert "ask why something is done this way" in prompt
    assert "[chunk 248] Attention Is All You Need > 3.2.1 Scaled Dot-Product Attention" in prompt
    assert f"<<<\n{CHUNK}\n>>>" in prompt


def test_a_stored_chunk_is_cited_by_its_title_and_section() -> None:
    chunk = Chunk(id=248, section="3.2 Attention", text=CHUNK, content_types=["text"])

    with_section = source_of("Attention Is All You Need", chunk)
    chunk.section = None

    assert with_section.citation == "Attention Is All You Need > 3.2 Attention"
    assert source_of("Attention Is All You Need", chunk).citation == "Attention Is All You Need"


SINGLE = "Why are the dot products scaled before the softmax?"
DOUBLE = "Why are the dot products scaled, and what happens to the softmax without it?"


def answer(quote: str, question: str = SINGLE) -> str:
    return json.dumps(
        {
            "question": question,
            "difficulty": 3,
            "reference_answer": "Their magnitude grows with the key dimension, and a large "
            "softmax input leaves almost no gradient.",
            "key_points": [
                {
                    "text": "large dot products saturate the softmax",
                    "weight": 3,
                    "evidence_quote": quote,
                    "chunk_id": 248,
                },
                {
                    "text": "scaling by the square root of the key dimension undoes it",
                    "weight": 2,
                    "evidence_quote": "To counteract this effect, we scale the dot products",
                    "chunk_id": 248,
                },
            ],
            "misconceptions": ["It keeps the attention weights positive."],
            "source_chunk_ids": [248],
        }
    )


def writer(
    quotes: list[str], seen: list[list[ModelMessage]], questions: list[str] | None = None
) -> FunctionModel:
    """Answers with the next quote in the list, and the next question when a list of them is
    given, recording the conversation it was sent."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        turn = len(seen) - 1
        question = questions[turn] if questions else SINGLE
        return ModelResponse(parts=[TextPart(answer(quotes[turn], question))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def test_a_question_whose_quotes_hold_up_is_written_in_one_call() -> None:
    seen: list[list[ModelMessage]] = []

    generated = asyncio.run(generate_question(writer([GOOD_QUOTE], seen), SOURCES, "why_how"))

    assert (generated.attempts, generated.grounded) == (1, True)
    assert generated.question.question.startswith("Why are the dot products")
    assert generated.usage["requests"] == 1
    assert generated.model == "function:respond:"
    assert generated.prompt_version == "generate-v5"
    assert generated.compound is None


def test_a_question_is_filed_under_the_style_it_was_asked_for() -> None:
    """The model is shown a style's brief, never its name, so a label it picked was a guess.
    It is not asked for one: the question is filed under the style the task asked for."""
    seen: list[list[ModelMessage]] = []

    generated = asyncio.run(generate_question(writer([GOOD_QUOTE], seen), SOURCES, "tradeoffs"))

    assert generated.style == "tradeoffs"
    assert "style" not in GeneratedQuestion.model_json_schema()["properties"]


def test_a_quote_that_is_not_in_the_sources_is_sent_back_once() -> None:
    seen: list[list[ModelMessage]] = []
    invented = "the model learns a separate scaling factor for every attention head"

    generated = asyncio.run(
        generate_question(writer([invented, GOOD_QUOTE], seen), SOURCES, "why_how")
    )

    assert (generated.attempts, generated.grounded) == (2, True)
    # Both calls are charged for, and the second carries the whole conversation back.
    assert generated.usage["requests"] == 2
    assert len(seen[1]) > len(seen[0])
    repair = str(seen[1][-1].parts[-1].content)
    assert invented in repair
    assert "not in chunk 248" in repair


def test_a_quote_that_stays_wrong_is_returned_ungrounded() -> None:
    seen: list[list[ModelMessage]] = []
    invented = "the model learns a separate scaling factor for every attention head"

    generated = asyncio.run(
        generate_question(writer([invented, invented], seen), SOURCES, "why_how")
    )

    assert (generated.attempts, generated.grounded) == (2, False)
    [failed] = [quote for quote in generated.quotes if not quote.grounded]
    assert failed.problem.startswith("the quote is not in chunk 248")
    assert failed.score < 95


@pytest.mark.parametrize(
    ("question", "problem"),
    [
        (SINGLE, None),
        # From the development library: asking about two things is not asking two questions.
        (
            "What problems arise when the mechanisms for controlling what to remember and what to "
            "forget are omitted?",
            None,
        ),
        ("How do layer normalization and batch normalization differ at inference?", None),
        ("When is dot-product attention the better choice?", None),
        ("Compare batch normalization and layer normalization.", None),
        (DOUBLE, 'it joins a second question on with "and what"'),
        (
            "What is the core intuition behind the Transformer and how does relying on attention "
            "change the way it processes sequences?",
            'it joins a second question on with "and how does"',
        ),
        ("Why is the scale needed? What breaks without it?", "it has more than one question mark"),
        (
            "Compare the two pipelines. How does the retrieval score change the ranking?",
            "it gives an instruction and then asks a question",
        ),
    ],
)
def test_a_question_that_asks_two_things_is_found_out(question: str, problem: str | None) -> None:
    assert compound(question) == problem


def test_a_question_that_asks_two_things_is_sent_back_once() -> None:
    seen: list[list[ModelMessage]] = []

    generated = asyncio.run(
        generate_question(
            writer([GOOD_QUOTE, GOOD_QUOTE], seen, [DOUBLE, SINGLE]), SOURCES, "why_how"
        )
    )

    assert (generated.attempts, generated.compound, generated.grounded) == (2, None, True)
    assert generated.question.question == SINGLE
    repair = str(seen[1][-1].parts[-1].content)
    assert 'asks more than one thing: it joins a second question on with "and what"' in repair
    # Nothing was wrong with the quotes, so the repair says nothing about them.
    assert "evidence quotes are not in the sources" not in repair


def test_one_repair_round_mends_two_things_at_once() -> None:
    seen: list[list[ModelMessage]] = []
    invented = "the model learns a separate scaling factor for every attention head"

    generated = asyncio.run(
        generate_question(
            writer([invented, GOOD_QUOTE], seen, [DOUBLE, SINGLE]), SOURCES, "why_how"
        )
    )

    assert (generated.attempts, generated.compound, generated.grounded) == (2, None, True)
    repair = str(seen[1][-1].parts[-1].content)
    assert "asks more than one thing" in repair and invented in repair


def test_a_question_that_still_asks_two_things_is_returned_as_it_is() -> None:
    seen: list[list[ModelMessage]] = []

    generated = asyncio.run(
        generate_question(
            writer([GOOD_QUOTE, GOOD_QUOTE], seen, [DOUBLE, DOUBLE]), SOURCES, "why_how"
        )
    )

    assert generated.attempts == 2
    assert generated.compound == 'it joins a second question on with "and what"'


def test_the_writer_is_shown_what_was_already_asked_from_its_sources() -> None:
    asked = ["Why are the dot products scaled before the softmax?", "What does the\nscale undo?"]

    plain = request(SOURCES, "why_how")
    steered = request(SOURCES, "why_how", asked)

    assert "Already asked" not in plain
    assert steered.startswith(plain)
    assert steered.endswith(
        "Already asked from these sources, so ask about something else:\n"
        "- Why are the dot products scaled before the softmax?\n"
        "- What does the scale undo?\n"
    )


def test_what_was_already_asked_goes_to_the_model() -> None:
    seen: list[list[ModelMessage]] = []
    asked = ["Why is the softmax input scaled?"]

    asyncio.run(generate_question(writer([GOOD_QUOTE], seen), SOURCES, "why_how", asked=asked))

    assert "- Why is the softmax input scaled?" in str(seen[0][-1].parts[-1].content)
