"""The concept inventory: which passages it reads, how it windows them, and what it makes of
the model's answers."""

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from sqlalchemy import func, select

from app.db.models import Chunk
from app.questions.grounding import check_evidence, check_quote
from app.questions.inventory import (
    PROMPT_VERSION,
    Comparison,
    Concept,
    ConceptReading,
    DocumentInventory,
    MergeGroup,
    Passage,
    Review,
    Window,
    apply_review,
    carry_verdicts,
    close_pairs,
    compare,
    concept_of,
    context_only,
    from_json,
    library_groups,
    load_document,
    merge_document,
    merged,
    plan_document,
    read_window,
    resume,
    table_share,
    to_json,
    total_usage,
    window_request,
    windows,
)

SCALE_SENTENCE = "To counteract this effect, we scale the dot products by 1/sqrt(d_k)."
HEADS_SENTENCE = (
    "Multi-head attention allows the model to jointly attend to information from different "
    "representation subspaces at different positions."
)
SCALE = Passage(
    chunk_id=5,
    section="3.2.1 Scaled Dot-Product Attention",
    text="We suspect that for large values of d_k, the dot products grow large in magnitude, "
    "pushing the softmax function into regions where it has extremely small gradients. "
    + SCALE_SENTENCE,
    tokens=60,
)
HEADS = Passage(
    chunk_id=6,
    section="3.2.2 Multi-Head Attention",
    text=f"{HEADS_SENTENCE} With a single attention head, averaging inhibits this.",
    tokens=30,
)


def chunk(
    text: str, section: str | None = "2 Method", types: Sequence[str] = ("text",), chunk_id: int = 1
) -> Chunk:
    return Chunk(
        id=chunk_id,
        position=chunk_id,
        section=section,
        content_types=list(types),
        text=text,
        token_count=len(text.split()),
    )


def idea(name: str, chunk_ids: list[int], evidence: str, **fields: Any) -> dict[str, Any]:
    return {
        "name": name,
        "summary": f"What the chunks say about {name}.",
        "chunk_ids": chunk_ids,
        "kinds": ["reason"],
        "reason": "given",
        "scope": "general",
        "interview": 3,
        "evidence": evidence,
    } | fields


def concept(name: str, chunk_ids: list[int], number: str, **fields: Any) -> Concept:
    values: dict[str, Any] = {
        "document_id": 1,
        "summary": f"About {name}.",
        "kinds": ["reason"],
        "reason": "given",
        "scope": "general",
        "interview": 2,
        "evidence": [check_quote(SCALE_SENTENCE, 5, SCALE.text)],
    } | fields
    return Concept(name=name, chunk_ids=chunk_ids, readings=[number], **values)


def answering(answers: list[dict[str, Any]], seen: list[str]) -> FunctionModel:
    """Answers each request with the next of `answers`, recording the prompt it was sent."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(str(messages[-1].parts[-1].content))
        return ModelResponse(parts=[TextPart(json.dumps(answers[len(seen) - 1]))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def two_windows() -> DocumentInventory:
    return DocumentInventory(
        document_id=1,
        title="Attention Is All You Need",
        held_back={},
        windows=[Window(passages=[SCALE]), Window(passages=[HEADS])],
    )


def test_a_table_is_weighed_by_its_characters_not_its_lines() -> None:
    """Counted by lines, a prose passage holding a short table looked mostly table and took
    usable questions with it: the table was 12% and 13% of its characters."""
    prose = "Retrieval narrows the context to what the question needs, so the reader sees less. "
    table = "\n".join(["| a | b |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |", "| 5 | 6 |"])
    short_table = f"{prose * 4}\n{table}"

    assert table_share(short_table) == pytest.approx(0.12, abs=0.005)
    assert context_only(chunk(short_table)) is None


@pytest.mark.parametrize(("rows", "held_back"), [(12, False), (14, True)])
def test_a_passage_is_a_table_from_half_its_characters(rows: int, held_back: bool) -> None:
    prose = "Each row below is one model, scored on the three aspects. " * 4
    table = "\n".join(f"| model {row:02} | 0.8 |" for row in range(rows))
    text = f"{prose}\n{table}"

    assert (table_share(text) >= 0.5) == held_back
    assert context_only(chunk(text)) == ("mostly table" if held_back else None)


@pytest.mark.parametrize(
    ("text", "section", "types", "reason"),
    [
        ("Proven by " + "\\ " * 8 + "induction.", "2 Method", ["text"], "damaged mathematics"),
        ("Hence (a) b and (c) d and (e) f hold.", "2 Method", ["text"], "damaged mathematics"),
        # Two are an ordinary list.
        ("Either (a) b or (c) d holds.", "2 Method", ["text"], None),
        ("The bound follows.", "Appendix A Proofs > A.1 Lemma 1", ["text"], "back matter"),
        ("We thank them.", "Acknowledgements · Author Contributions", ["text"], "back matter"),
        # It starts in the conclusion, which an idea can rest on.
        ("The method is simple.", "6 Conclusion · Appendix A More Results", ["text"], None),
        # The tagger's rules hold too.
        ("import torch", "2 Method", ["code"], "code only"),
    ],
)
def test_passages_no_idea_can_rest_on_are_held_back(
    text: str, section: str, types: list[str], reason: str | None
) -> None:
    assert context_only(chunk(text, section, types)) == reason


def sized(*tokens: int) -> list[Passage]:
    return [
        Passage(chunk_id=number, section=None, text=f"passage {number}", tokens=count)
        for number, count in enumerate(tokens, start=1)
    ]


def test_windows_hold_whole_passages_in_order_as_evenly_as_they_can() -> None:
    packed = windows(sized(*[800] * 7), budget=5_000)

    # Filled to the budget it would be six passages and one left over.
    assert [[passage.chunk_id for passage in window] for window in packed] == [
        [1, 2, 3, 4],
        [5, 6, 7],
    ]


def test_a_short_document_is_one_window() -> None:
    assert len(windows(sized(300, 400, 500), budget=5_000)) == 1
    assert windows([], budget=5_000) == []


def test_planning_a_document_sets_aside_what_it_holds_back() -> None:
    chunks = [
        chunk("Attention relates every position to every other.", "1 Introduction", chunk_id=1),
        chunk("The bound follows.", "Appendix A Proofs", chunk_id=2),
        chunk("Scaling keeps the softmax away from saturation.", "3 Model", chunk_id=3),
    ]

    document = plan_document(7, "A Paper", chunks)

    assert document.held_back == {2: "back matter"}
    assert [window.chunk_ids for window in document.windows] == [[1, 3]]
    assert not document.needs_merge


def test_the_window_request_shows_every_passage_with_its_chunk_id() -> None:
    untitled = Passage(chunk_id=7, section=None, text="Body.", tokens=1)

    prompt = window_request("Attention Is All You Need", [SCALE, untitled])

    assert prompt.startswith("Document: Attention Is All You Need\n\n")
    assert f"[chunk 5] 3.2.1 Scaled Dot-Product Attention\n<<<\n{SCALE.text}\n>>>" in prompt
    assert "[chunk 7] -\n<<<\nBody.\n>>>" in prompt


def reading(chunk_ids: list[int], evidence: str) -> ConceptReading:
    return ConceptReading.model_validate(idea("scaling the dot products", chunk_ids, evidence))


def test_an_idea_keeps_only_the_chunks_of_its_window() -> None:
    found = concept_of(reading([5, 99], SCALE_SENTENCE), 1, [SCALE, HEADS], "1.1")

    assert found.chunk_ids == [5]
    [evidence] = found.evidence
    assert (evidence.grounded, evidence.chunk_id) == (True, 5)
    assert found.readings == ["1.1"]


def test_a_kind_outside_the_six_is_dropped_rather_than_failing_the_window() -> None:
    answer = idea("scaling the dot products", [5], SCALE_SENTENCE, kinds=["theorem", "reason"])

    found = concept_of(ConceptReading.model_validate(answer), 1, [SCALE], "1.1")

    assert found.kinds == ["reason"]


def test_evidence_found_in_another_chunk_of_the_window_files_the_idea_there_too() -> None:
    found = concept_of(reading([5], HEADS_SENTENCE), 1, [SCALE, HEADS], "1.1")

    assert found.chunk_ids == [5, 6]
    assert found.evidence[0].chunk_id == 6


def test_a_faithfully_shortened_evidence_sentence_files_the_idea() -> None:
    shortened = (
        "We suspect that for large values of d_k ... pushing the softmax function into regions "
        "where it has extremely small gradients."
    )

    found = concept_of(reading([6], shortened), 1, [SCALE, HEADS], "1.1")

    [evidence] = found.evidence
    assert (evidence.grounded, evidence.chunk_id, found.chunk_ids) == (True, 5, [6, 5])


def test_evidence_found_nowhere_is_reported_against_the_first_chunk_cited() -> None:
    invented = "the learning rate is decayed over the first four thousand steps"

    found = concept_of(reading([5], invented), 1, [SCALE, HEADS], "1.1")

    [evidence] = found.evidence
    assert evidence.problem is not None
    assert evidence.problem.startswith("the quote is not in chunk 5")
    assert found.chunk_ids == [5]


EVIDENCE_CHUNK = (
    "The idea of ReAct is simple: we augment the agent's action space to "
    "$\\mathcal{\\hat{A}}=\\mathcal{A}\\cup\\mathcal{L}$, where $\\mathcal{L}$ is the space of "
    "language. An action $\\hat{a}_{t}\\in\\mathcal{L}$ in the language space, which we will "
    "refer to as a thought, does not affect the external environment, thus leading to no "
    "observation feedback. A challenge built into ALFWorld is the need to determine likely "
    "locations for household items (e.g. desklamps will likely be on desks), making this "
    "environment a good fit for LLMs. We inspect two things:\n"
    "- which chunks were retrieved by the search\n"
    "- how each one was scored by the ranker\n\n"
    "The final faithfulness score is computed as $F=\\frac{|V|}{|S|}$, where $|V|$ is the number "
    "of statements that were supported. Let $X$ be the input and $x$ one of its values. Most "
    "simply, we explore zero-shot prompting with GPT-J [45] in the summarization task and 2-shot "
    "prompting with Pythia-2.8B [3] in the dialogue task. With some algebra we obtain:\n\n"
    "$$r(x,y)=\\beta\\log\\frac{\\pi_{r}(y\\mid x)}{\\pi_{\\text{ref}}(y\\mid x)}"
    "+\\beta\\log Z(x).$$ (5)"
)


@pytest.mark.parametrize(
    ("quote", "problem"),
    [
        # Copied whole, prose has a quote's leeway: here a citation mark left out.
        (
            "Most simply, we explore zero-shot prompting with GPT-J in the summarization task and "
            "2-shot prompting with Pythia-2.8B [3] in the dialogue task.",
            None,
        ),
        # Shortened within one sentence, the formula kept, an abbreviation's full stop passed
        (
            "An action $\\hat{a}_{t}\\in\\mathcal{L}$ ... does not affect the external environment",
            None,
        ),
        (
            "the need to determine likely locations for household items ... making this "
            "environment a good fit for LLMs.",
            None,
        ),
        (
            "... does not affect the external environment, thus leading to no observation "
            "feedback.",
            None,
        ),
        (
            "is the space of language ... does not affect the external environment",
            "the quote joins words from different sentences",
        ),
        # A line of a list ends its sentence, full stop or none.
        (
            "which chunks were retrieved ... how each one was scored by the ranker",
            "the quote joins words from different sentences",
        ),
        (
            "An action $\\hat{a}_{t}\\in\\mathcal{L}$ ... affect the external environment, thus "
            "leading to no observation feedback",
            'the words left out include "not"',
        ),
        (
            "we augment the agent's action space to ... is the space of language",
            "the words left out include mathematics",
        ),
        (
            "One challenge built into ALFWorld ... a good fit for LLMs",
            'a piece is not in chunk 7 as written: "One challenge built into ALFWorld"',
        ),
        (
            "does not affect the external environment ... An action $\\hat{a}_{t}\\in\\mathcal{L}$ "
            "in the language space",
            "the pieces are not in chunk 7 in this order",
        ),
        ("we augment ... of language", "the quote is shorter than 6 words"),
    ],
)
def test_a_shortened_evidence_sentence_stands_only_while_it_stays_faithful(
    quote: str, problem: str | None
) -> None:
    check = check_evidence(quote, 7, EVIDENCE_CHUNK)

    assert (check.problem, check.quote) == (problem, quote)


@pytest.mark.parametrize(
    ("quote", "problem"),
    [
        # The full stop that closes a displayed equation belongs to the sentence.
        (
            "With some algebra we obtain: r(x,y)=\\beta\\log\\frac{\\pi_{r}(y\\mid x)}"
            "{\\pi_{\\text{ref}}(y\\mid x)}+\\beta\\log Z(x).",
            None,
        ),
        (
            "The final faithfulness score is computed as F=|V|/|S|, where |V| is the number of "
            "statements",
            "the mathematics is not as chunk 7 writes it: F=|V|/|S|",
        ),
        # The quote check scores this one 96.
        (
            "An action \\hat{a}_t\\in\\mathcal{L} in the language space",
            "the mathematics is not as chunk 7 writes it: \\hat{a}_t\\in\\mathcal{L}",
        ),
        (
            "we augment the agent's action space to Â=A∪L, where L is the space of language",
            "the mathematics is not as chunk 7 writes it: Â=A∪L",
        ),
        # A formula's letters keep their case, though prose is compared without it.
        (
            "Let $x$ be the input and $x$ one of its values.",
            "the mathematics is not as chunk 7 writes it: x",
        ),
        (
            "The final faithfulness score is computed as $F=\\frac{|V|}",
            "the quote starts or ends inside a formula",
        ),
        # The quote check's leeway (96 here) is for prose, not for a sentence with a formula.
        (
            "The final faithfulness score is computed as $F=\\frac{|V|}{|S|}$, where $|V|$ is the "
            "count of statements that were supported.",
            "the quote holds mathematics but is not an exact copy of chunk 7",
        ),
    ],
)
def test_mathematics_in_evidence_is_held_to_more_than_prose(
    quote: str, problem: str | None
) -> None:
    assert check_evidence(quote, 7, EVIDENCE_CHUNK).problem == problem


def test_a_piece_found_twice_is_taken_where_the_shortening_holds() -> None:
    chunk = (
        "We scale the inputs first. Then the model trains for a while. "
        "We scale the dot products by a constant factor."
    )

    check = check_evidence("We scale ... the dot products by a constant factor", 1, chunk)

    assert check.grounded


def test_an_ellipsis_the_chunk_writes_itself_shortens_nothing() -> None:
    chunk = (
        "The encoder maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a "
        "sequence of continuous representations."
    )
    quote = "maps an input sequence of symbol representations $(x_1, ..., x_n)$ to a sequence"

    assert check_evidence(quote, 1, chunk).grounded


def test_a_group_is_folded_into_the_place_of_its_earliest_member() -> None:
    first = concept(
        "a", [1], "1.1", kinds=["mechanism"], reason="stated", scope="document", interview=1
    )
    second = concept("b", [2], "1.2")
    third = concept("c", [3, 1], "2.1", kinds=["reason", "mechanism"], interview=3)
    fourth = concept("d", [4], "2.2")

    result = merged(
        [first, second, third, fourth],
        [MergeGroup(numbers=[3, 1], name=" one idea ", summary="Both of them.")],
    )

    assert [found.name for found in result] == ["one idea", "b", "d"]
    one = result[0]
    assert one.chunk_ids == [1, 3]
    assert one.kinds == ["mechanism", "reason"]
    # The reason is given, and the idea general, when any part says so.
    assert (one.reason, one.scope, one.interview) == ("given", "general", 3)
    assert one.readings == ["1.1", "2.1"]
    assert len(one.evidence) == 2


def test_numbers_out_of_range_or_already_in_a_group_are_passed_over() -> None:
    parts = [concept(str(number), [number], f"1.{number}") for number in (1, 2, 3)]

    result = merged(
        parts,
        [
            MergeGroup(numbers=[1, 2], name="first", summary="."),
            MergeGroup(numbers=[2, 3], name="second", summary="."),
            MergeGroup(numbers=[3, 0, 9], name="third", summary="."),
        ],
    )

    assert [found.name for found in result] == ["first", "3"]


def library() -> dict[str, Concept]:
    return {
        "1.1": concept("dpo loss", [1], "1.1", scope="document"),
        "1.2": concept("bradley-terry", [2], "1.2"),
        "1.3": concept("dpo loss restated", [3, 1], "3.8", kinds=["mechanism"], interview=3),
        "1.4": concept("robustness to temperature", [4], "3.6"),
        "2.1": concept("faithfulness", [9], "1.2", document_id=2),
    }


def named(key: str) -> dict[str, str]:
    return {"key": key, "name": library()[key].name}


def test_a_review_folds_a_merge_into_its_lead_and_keeps_every_other_key() -> None:
    concepts = library()
    review = Review.model_validate(
        {
            "merge": [[named("1.1"), named("1.3")]],
            "scope": [named("1.1") | {"scope": "document"}],
            "exclude": [named("1.4") | {"reason": "the result without the reason"}],
            "keep_apart": [[named("1.2"), named("1.4")]],
            "check": [{"ideas": [named("1.2"), named("2.1")], "note": "a second look"}],
        }
    )

    result = apply_review(concepts, review)

    assert list(result) == ["1.1", "1.2", "1.4", "2.1"]
    lead = result["1.1"]
    assert (lead.name, lead.summary, lead.chunk_ids) == ("dpo loss", "About dpo loss.", [1, 3])
    assert (lead.kinds, lead.interview, lead.readings) == (
        ["mechanism", "reason"],
        3,
        ["1.1", "3.8"],
    )
    # Merged, it reads general as 1.3 does; the review's scope is set after the merge.
    assert lead.scope == "document"
    assert result["1.4"].excluded == "the result without the reason"
    assert result["1.2"] is concepts["1.2"]
    # The ideas as read are left as they were.
    assert list(concepts) == ["1.1", "1.2", "1.3", "1.4", "2.1"]
    assert concepts["1.4"].excluded is None


@pytest.mark.parametrize(
    ("decisions", "error"),
    [
        (
            {"exclude": [{"key": "1.9", "name": "dpo loss", "reason": "."}]},
            "the review names idea 1.9, which this reading does not have",
        ),
        # Read again, the documents may number their ideas otherwise.
        (
            {"exclude": [{"key": "1.2", "name": "dpo loss", "reason": "."}]},
            'the review names idea 1.2 "dpo loss", which this reading calls "bradley-terry"',
        ),
        (
            {"merge": [[named("1.1"), named("1.3")]], "keep_apart": [[named("1.3"), named("1.1")]]},
            "the review keeps 1.3 and 1.1 apart but merges them",
        ),
        (
            {"merge": [[named("1.1"), named("1.3")], [named("1.2"), named("1.3")]]},
            "idea 1.3 is in two merges",
        ),
        ({"merge": [[named("1.2"), named("2.1")]]}, "a merge joins ideas of one document"),
        ({"merge": [[named("1.1"), named("1.1")]]}, "a merge needs two different ideas"),
        (
            {
                "merge": [[named("1.1"), named("1.3")]],
                "scope": [named("1.3") | {"scope": "general"}],
            },
            "idea 1.3 is merged into 1.1: name 1.1",
        ),
        ({"keep_apart": [[named("1.2"), named("1.2")]]}, "ideas kept apart have to be two"),
    ],
)
def test_a_review_that_does_not_fit_the_reading_is_refused(
    decisions: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValueError) as refused:
        apply_review(library(), Review.model_validate(decisions))

    assert str(refused.value).startswith(error)


def test_the_saved_inventory_lists_each_idea_under_its_key_as_the_review_leaves_it() -> None:
    document = two_windows()
    model = answering(
        [
            {"ideas": [idea("scaled attention", [5], SCALE_SENTENCE)]},
            {
                "ideas": [
                    idea("multi-head attention", [6], HEADS_SENTENCE),
                    idea("scaling restated", [5, 6], HEADS_SENTENCE),
                ]
            },
            {"groups": []},
        ],
        [],
    )

    async def scenario() -> None:
        await read_window(model, document, 1)
        await read_window(model, document, 2)
        await merge_document(model, document)

    asyncio.run(scenario())
    review = Review.model_validate(
        {
            "merge": [
                [
                    {"key": "1.1", "name": "scaled attention"},
                    {"key": "1.3", "name": "scaling restated"},
                ]
            ],
            "exclude": [{"key": "1.2", "name": "multi-head attention", "reason": "not asked"}],
        }
    )

    data = json.loads(json.dumps(to_json([document], review=review)))

    saved = data["documents"][0]["concepts"]
    assert [(idea["key"], idea["name"], idea["excluded"]) for idea in saved] == [
        ("1.1", "scaled attention", None),
        ("1.2", "multi-head attention", "not asked"),
    ]
    assert saved[0]["readings"] == ["1.1", "2.2"]
    assert data["review"] == review.model_dump()
    # Only the answers are read back; the review is applied again from where it is kept.
    [again], _, _ = from_json(data)
    assert again.merge == document.merge
    assert [window.reading for window in again.windows] == [
        window.reading for window in document.windows
    ]


def test_a_document_read_in_two_windows_is_merged_from_names_and_summaries() -> None:
    document = two_windows()
    seen: list[str] = []
    model = answering(
        [
            {"ideas": [idea("scaled attention", [5], SCALE_SENTENCE)]},
            {
                "ideas": [
                    idea("multi-head attention", [6], HEADS_SENTENCE, kinds=["mechanism"]),
                    idea("scaling the dot products", [6], HEADS_SENTENCE),
                ]
            },
            {"groups": [{"numbers": [1, 3], "name": "dot-product scaling", "summary": "Why."}]},
        ],
        seen,
    )

    async def scenario() -> None:
        await read_window(model, document, 1)
        await read_window(model, document, 2)
        assert document.needs_merge
        await merge_document(model, document)

    asyncio.run(scenario())

    assert "[chunk 5] 3.2.1 Scaled Dot-Product Attention" in seen[0]
    assert "[chunk 6] 3.2.2 Multi-Head Attention" in seen[1]
    assert "[1] (part 1) scaled attention: What the chunks say about scaled attention." in seen[2]
    assert "[3] (part 2) scaling the dot products:" in seen[2]
    assert [found.name for found in document.concepts()] == [
        "dot-product scaling",
        "multi-head attention",
    ]
    assert document.concepts()[0].readings == ["1.1", "2.2"]
    assert not document.needs_merge
    assert [call.usage["requests"] for call in document.calls] == [1, 1, 1]
    assert {call.version for call in document.calls} == {PROMPT_VERSION}
    assert total_usage(document.calls)["requests"] == 3


def test_an_inventory_read_back_carries_on_where_it_stopped() -> None:
    document = two_windows()
    model = answering([{"ideas": [idea("scaled attention", [5], SCALE_SENTENCE)]}], [])
    asyncio.run(read_window(model, document, 1))

    data = json.loads(json.dumps(to_json([document])))
    [saved], pairs, calls = from_json(data)
    fresh = two_windows()

    assert resume(fresh, saved)
    assert fresh.windows[0].reading == document.windows[0].reading
    assert fresh.windows[0].call == document.windows[0].call
    assert fresh.windows[1].reading is None
    assert (pairs, calls) == ([], [])
    assert data["documents"][0]["concepts"][0]["name"] == "scaled attention"
    assert data["usage"]["requests"] == 1
    # Read again from the start once the windows hold other passages
    other = Passage(chunk_id=7, section=None, text="Another passage.", tokens=2)
    for changed in (
        [Window(passages=[SCALE, HEADS])],
        [Window(passages=[SCALE]), Window(passages=[other])],
    ):
        regrouped = DocumentInventory(
            document_id=1, title="Attention", held_back={}, windows=changed
        )
        assert not resume(regrouped, saved)
        assert regrouped.windows[0].reading is None


class Vectors:
    """Hands out a fixed vector for each text it is asked to embed."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    async def embed_documents(
        self, texts: Sequence[str], progress: Any = None
    ) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def test_each_idea_is_paired_with_its_nearest_in_every_other_document() -> None:
    concepts = {
        "1.1": concept("a", [1], "1.1"),
        "1.2": concept("b", [2], "1.2"),
        "2.1": concept("c", [3], "1.1", document_id=2),
        "2.2": concept("d", [4], "1.2", document_id=2),
        "2.3": concept("e", [5], "1.3", document_id=2),
    }
    vectors = Vectors(
        {
            "a: About a.": [1.0, 0.0],
            # As close to a as can be, but from the same document
            "b: About b.": [1.0, 0.05],
            "c: About c.": [1.0, 0.2],
            "d: About d.": [0.0, 1.0],
            "e: About e.": [1.0, 0.3],
        }
    )

    pairs = asyncio.run(close_pairs(vectors, concepts, threshold=0.9))

    # a and e are close enough, but neither is the other's nearest across the documents.
    assert [(pair.first, pair.second) for pair in pairs] == [
        ("1.2", "2.1"),
        ("1.1", "2.1"),
        ("1.2", "2.3"),
    ]
    assert pairs[0].names == ("b", "c")
    assert pairs[0].similarity == pytest.approx(0.9891, abs=1e-4)
    assert pairs[0].same is None
    # One document has nothing to compare across.
    alone = {key: value for key, value in concepts.items() if key.startswith("1.")}
    assert asyncio.run(close_pairs(vectors, alone, threshold=0.9)) == []


def test_only_pairs_not_yet_judged_go_to_the_model() -> None:
    concepts = {
        "1.1": concept("a", [1], "1.1"),
        "2.1": concept("c", [3], "1.1", document_id=2),
        "2.2": concept("d", [4], "1.2", document_id=2),
    }
    pairs = [
        Comparison("1.1", "2.1", ("a", "c"), 0.95, same=True),
        Comparison("1.1", "2.2", ("a", "d"), 0.85),
        Comparison("2.1", "1.1", ("c", "a"), 0.8),
    ]
    seen: list[str] = []
    model = answering([{"verdicts": [{"pair": 1, "same": False}]}], seen)

    calls = asyncio.run(compare(model, pairs, concepts, {1: "Paper one", 2: "Paper two"}))

    assert len(calls) == 1
    assert seen[0].startswith(
        '[1] a: About a. (from "Paper one")\n    d: About d. (from "Paper two")'
    )
    assert "[2] c: About c." in seen[0]
    # The model left the second pair out, so it is still to be judged.
    assert [pair.same for pair in pairs] == [True, False, None]


def test_a_verdict_carries_over_only_to_the_same_two_ideas() -> None:
    before = [
        Comparison("1.1", "2.1", ("a", "c"), 0.9, same=True),
        Comparison("1.2", "2.2", ("b", "d"), 0.8, same=False),
    ]
    now = [
        Comparison("1.1", "2.1", ("a", "c"), 0.91),
        Comparison("1.2", "2.2", ("b", "renamed"), 0.8),
    ]

    carry_verdicts(now, before)

    assert [pair.same for pair in now] == [True, None]


def test_ideas_found_the_same_are_gathered_across_documents() -> None:
    keys = ["1.1", "1.2", "2.1", "3.1"]
    pairs = [
        Comparison("1.1", "3.1", ("a", "e"), 0.9, same=True),
        # Joined through e, though never compared with a
        Comparison("2.1", "3.1", ("c", "e"), 0.8, same=True),
        Comparison("1.2", "3.1", ("b", "e"), 0.75, same=False),
        Comparison("1.2", "2.1", ("b", "c"), 0.7),
    ]

    assert library_groups(keys, pairs) == [["1.1", "2.1", "3.1"]]


def test_a_document_is_planned_from_its_current_passages(sessions, corpus, embedder) -> None:
    async def scenario() -> tuple[DocumentInventory | None, DocumentInventory | None]:
        async with sessions() as session, session.begin():
            paper = await session.scalar(
                select(Chunk.document_id).where(Chunk.id == corpus.scaling)
            )
            session.add(
                Chunk(
                    document_id=paper,
                    position=0,
                    section="3 Model Architecture",
                    content_types=["text"],
                    text="An older reading of the section.",
                    token_count=6,
                    embedding=embedder.vector("older"),
                    superseded_at=func.now(),
                )
            )
        async with sessions() as session:
            return await load_document(session, paper), await load_document(session, 10_000)

    document, missing = asyncio.run(scenario())

    assert document is not None
    assert document.title == "Attention Is All You Need"
    assert [window.chunk_ids for window in document.windows] == [
        [corpus.scaling, corpus.positions, corpus.softmax]
    ]
    assert document.held_back == {}
    assert missing is None
