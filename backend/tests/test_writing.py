"""Writing a question on one idea of the inventory: the style its passages support, the evidence
copied first, and what the code makes of the answer."""

import asyncio
import json
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RunUsage

from app.llm.pacing import QuotaExhausted
from app.questions.generation import Source
from app.questions.grounding import check_quote
from app.questions.inventory import Call, Concept, DocumentInventory, Passage, Window
from app.questions.validation import AnswerCheck
from app.questions.writing import (
    MAX_TOKENS,
    PASSAGE_TOKENS,
    PROMPT_VERSION,
    STYLE_BRIEFS,
    Entry,
    IdeaQuestion,
    assign_styles,
    check_answer,
    check_question,
    checker_failed,
    entries_json,
    framed,
    library_key_points,
    misfit,
    problems,
    read_entries,
    request,
    sources_of,
    supported_styles,
    write_question,
    writing_settings,
)
from scripts import write_questions

SATURATE = (
    "We suspect that for large values of $d_k$, the dot products grow large in magnitude, "
    "pushing the softmax function into regions where it has extremely small gradients."
)
SCALE = "To counteract this effect, we scale the dot products by $\\frac{1}{\\sqrt{d_k}}$."
HEADS = (
    "Multi-head attention allows the model to jointly attend to information from different "
    "representation subspaces at different positions."
)
SOURCES = [
    Source(
        chunk_id=5,
        citation="Attention Is All You Need > 3.2.1 Scaled Dot-Product Attention",
        text=f"{SATURATE} {SCALE}",
    ),
    Source(
        chunk_id=6,
        citation="Attention Is All You Need > 3.2.2 Multi-Head Attention",
        text=f"{HEADS} With a single attention head, averaging inhibits this.",
    ),
]
SINGLE = "Why are the dot products scaled before the softmax?"
DOUBLE = "Why are the dot products scaled, and what happens to the softmax without it?"


def concept(name: str = "scaling the dot products", **fields: Any) -> Concept:
    values: dict[str, Any] = {
        "document_id": 1,
        "summary": "Large dot products saturate the softmax, so they are scaled down.",
        "chunk_ids": [5, 6],
        "kinds": ["mechanism", "reason"],
        "reason": "given",
        "scope": "general",
        "interview": 3,
        "evidence": [check_quote(SATURATE, 5, SOURCES[0].text)],
        "readings": ["1.1"],
    } | fields
    return Concept(name=name, **values)


def answer(
    evidence: list[tuple[str, int]] | None = None,
    question: str = SINGLE,
    points: list[tuple[str, int, int]] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """An answer as the writer gives it: evidence (sentence, chunk), the question, and key points
    (text, weight, the evidence sentence they rest on)."""
    evidence = evidence if evidence is not None else [(SATURATE, 5), (SCALE, 5)]
    points = (
        points
        if points is not None
        else [("large dot products saturate the softmax", 3, 1), ("scaling undoes it", 2, 2)]
    )
    return {
        "evidence": [{"sentence": sentence, "chunk_id": chunk} for sentence, chunk in evidence],
        "question": question,
        "key_points": [
            {"text": text, "weight": weight, "evidence": number} for text, weight, number in points
        ],
        "reference_answer": "Large dot products saturate the softmax; scaling undoes it.",
        "misconceptions": ["It keeps the attention weights positive."],
        "difficulty": 3,
    } | fields


def writer(
    answers: list[dict[str, Any]], seen: list[list[ModelMessage]], *, reasoning: bool = False
) -> FunctionModel:
    """Answers with the next of `answers`, recording the conversation it was sent; with
    `reasoning`, it shows its reasoning before each answer."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        text = TextPart(json.dumps(answers[len(seen) - 1]))
        return ModelResponse(parts=[ThinkingPart("Let me think."), text] if reasoning else [text])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def last_prompt(conversation: list[ModelMessage]) -> str:
    return str(conversation[-1].parts[-1].content)


def written(**fields: Any) -> IdeaQuestion:
    return IdeaQuestion.model_validate(answer(**fields))


@pytest.mark.parametrize(
    ("kinds", "reason", "styles"),
    [
        (["mechanism"], "given", ["why_how", "intuition"]),
        (["definition"], "given", ["why_how", "intuition"]),
        (["reason"], "given", ["why_how"]),
        # Only told that it is so, a why, a how or the intuition would be invented.
        (["mechanism", "definition"], "stated", []),
        (["comparison"], "stated", ["compare"]),
        (["trade-off", "failure"], "stated", ["tradeoffs", "failure_modes"]),
        (
            ["mechanism", "reason", "trade-off", "failure", "comparison", "definition"],
            "given",
            ["why_how", "intuition", "compare", "tradeoffs", "failure_modes"],
        ),
    ],
)
def test_a_style_is_offered_only_where_the_passages_support_it(
    kinds: list[str], reason: str, styles: list[str]
) -> None:
    assert supported_styles(concept(kinds=kinds, reason=reason)) == styles


def ideas(*kinds: tuple[list[str], str]) -> dict[str, Concept]:
    return {
        f"1.{number}": concept(f"idea {number}", kinds=found, reason=reason)
        for number, (found, reason) in enumerate(kinds, start=1)
    }


def test_the_ideas_with_the_fewest_styles_to_choose_from_choose_first() -> None:
    planned = ideas(
        (["reason"], "given"),
        (["reason", "trade-off"], "given"),
        (["trade-off"], "stated"),
    )

    # Taken in their own order, 1.2 would have the trade-off before 1.3, which has nothing else.
    assert assign_styles(planned) == {"1.1": "why_how", "1.2": "why_how", "1.3": "tradeoffs"}


def test_a_tie_goes_to_the_style_the_fewest_ideas_can_take() -> None:
    planned = ideas(
        (["mechanism"], "given"),
        (["reason"], "given"),
        (["mechanism"], "given"),
    )

    # 1.3 finds why_how and intuition given once each; two ideas can take intuition, three why.
    assert assign_styles(planned) == {"1.1": "intuition", "1.2": "why_how", "1.3": "intuition"}


def test_styles_are_spread_so_that_none_takes_over() -> None:
    # The kinds of the twenty ideas first planned from the three demo papers: half of them can
    # only be asked why, how or for the intuition.
    planned = ideas(
        *[(["mechanism", "definition"], "given")] * 8,
        (["reason", "definition"], "given"),
        (["reason", "comparison"], "given"),
        (["reason", "comparison"], "given"),
        (["mechanism", "trade-off", "definition"], "given"),
        (["mechanism", "trade-off", "definition"], "given"),
        (["mechanism", "failure", "comparison"], "given"),
        (["mechanism", "comparison"], "given"),
        (["reason", "failure", "comparison"], "given"),
        (["reason", "failure", "comparison"], "given"),
        (["reason", "trade-off", "failure"], "given"),
        (["mechanism", "reason", "trade-off", "failure", "comparison", "definition"], "given"),
        (["mechanism", "reason"], "given"),
    )

    styles = assign_styles(planned)

    assert all(style in supported_styles(planned[key]) for key, style in styles.items())
    counts = sorted(list(styles.values()).count(style) for style in STYLE_BRIEFS)
    assert counts == [3, 3, 4, 5, 5]


def test_the_spread_does_not_depend_on_the_order_the_ideas_come_in() -> None:
    planned = ideas(
        (["mechanism"], "given"),
        (["mechanism", "comparison"], "given"),
        (["reason"], "given"),
        (["mechanism"], "given"),
    )
    planned["1.10"] = concept("idea 10", kinds=["mechanism"], reason="given")

    styles = assign_styles(planned)
    backwards = assign_styles(dict(reversed(planned.items())))

    assert backwards == styles
    assert list(backwards) == ["1.10", "1.4", "1.3", "1.2", "1.1"]
    # By key, 1.10 takes its turn after 1.4; by its spelling it would go before, and take 1.4's.
    assert styles == {
        "1.1": "intuition",
        "1.2": "compare",
        "1.3": "why_how",
        "1.4": "intuition",
        "1.10": "why_how",
    }


def test_a_chosen_style_is_taken_in_its_ideas_turn() -> None:
    planned = ideas(*[(["mechanism"], "given")] * 3)
    spread = assign_styles(planned)

    # Chosen as the spread would choose it, it leaves the others as they were.
    assert spread == {"1.1": "why_how", "1.2": "intuition", "1.3": "why_how"}
    assert assign_styles(planned, {"1.2": "intuition"}) == spread
    # Chosen otherwise, the ideas after it make room; those before it keep theirs.
    assert assign_styles(planned, {"1.2": "why_how"}) == {
        "1.1": "why_how",
        "1.2": "why_how",
        "1.3": "intuition",
    }


@pytest.mark.parametrize(
    ("planned", "chosen", "error"),
    [
        (
            ideas((["mechanism"], "given")),
            {"1.1": "compare"},
            "idea 1.1's passages do not support compare, only: why_how, intuition",
        ),
        (ideas((["definition"], "stated")), {}, "idea 1.1's passages support no style"),
        (
            ideas((["mechanism"], "given")),
            {"1.7": "why_how"},
            "a style is chosen for idea 1.7, which is not planned",
        ),
    ],
)
def test_a_style_the_passages_cannot_carry_is_refused(
    planned: dict[str, Concept], chosen: dict[str, str], error: str
) -> None:
    with pytest.raises(ValueError) as refused:
        assign_styles(planned, chosen)

    assert str(refused.value).startswith(error)


def test_the_request_shows_the_idea_its_style_and_every_passage() -> None:
    prompt = request(concept(), SOURCES, "failure_modes")

    assert prompt.startswith(
        "The idea: scaling the dot products. Large dot products saturate the softmax, so they "
        "are scaled down.\nThe passages explain how it works and the reason for it."
        "\n\nWrite the question in this style: ask how or when it goes wrong"
    )
    assert f"[chunk 5] {SOURCES[0].citation}\n<<<\n{SOURCES[0].text}\n>>>" in prompt
    assert f"[chunk 6] {SOURCES[1].citation}\n<<<\n{SOURCES[1].text}\n>>>" in prompt
    # Nothing of what was asked before: the plan asks each idea once.
    assert "already asked" not in prompt.casefold()
    bare = request(concept(kinds=[]), SOURCES, "why_how")
    assert "The passages explain" not in bare
    assert "so they are scaled down.\n\nWrite the question" in bare


def test_an_idea_is_written_from_its_best_passages_as_far_as_they_fit() -> None:
    def passage(chunk_id: int, tokens: int, section: str | None = "2 Method") -> Passage:
        text = f"passage {chunk_id}"
        return Passage(chunk_id=chunk_id, section=section, text=text, tokens=tokens)

    document = DocumentInventory(
        document_id=1,
        title="A Paper",
        held_back={},
        windows=[
            Window(passages=[passage(1, 3_000), passage(2, 1_500, None)]),
            Window(passages=[passage(3, 900), passage(4, 100), passage(5, 6_000)]),
        ],
    )

    sources = sources_of(concept(chunk_ids=[2, 1, 3, 4]), [document])

    # 2 and 1 make 4,500; 3 would pass PASSAGE_TOKENS, and what comes after it is not as good.
    assert [source.chunk_id for source in sources] == [2, 1]
    assert [source.citation for source in sources] == ["A Paper", "A Paper > 2 Method"]
    assert sources[1].text == "passage 1"
    assert PASSAGE_TOKENS == 5_000
    # The best passage is always taken, however long.
    alone = sources_of(concept(chunk_ids=[5, 4]), [document])
    assert [source.chunk_id for source in alone] == [5]


@pytest.mark.parametrize(
    ("question", "problem"),
    [
        (SINGLE, None),
        ("Why does a two-stage pipeline help retrieval?", None),
        ("How does a lookup table speed up decoding?", None),
        ("What does the reward model rank, and on what data?", None),
        ("Why do the authors scale the dot products?", 'it mentions "the authors"'),
        ("According to the ReAct paper, why is acting grounded?", 'it mentions "paper"'),
        ("What does Table 2 say about temperature?", 'it mentions "Table 2"'),
        ("Why does Stage 2 add vector search?", 'it mentions "Stage 2"'),
        ("How does Eq. 5 follow from the objective?", 'it mentions "Eq. 5"'),
        ("What does Figure 3 show about the frontier?", 'it mentions "Figure 3"'),
        ("What does Theorem 1 guarantee?", 'it mentions "Theorem 1"'),
        ("Why is the proposed method faster?", 'it mentions "the proposed"'),
        ("What does Section IV argue?", 'it mentions "Section IV"'),
    ],
)
def test_a_question_framed_on_its_document_is_found_out(question: str, problem: str | None) -> None:
    assert framed(question) == problem


def test_every_evidence_sentence_and_key_point_is_checked() -> None:
    retyped = "To counteract this effect, we scale the dot products by 1/sqrt(d_k)."
    answered = written(
        evidence=[(SATURATE, 5), (retyped, 5), (HEADS, 5)],
        points=[("saturation", 3, 1), ("scaling", 2, 2), ("heads", 1, 3), ("more", 1, 4)],
    )

    checks = check_answer(answered, SOURCES)

    first, second, third = checks.evidence
    assert (first.grounded, first.chunk_id) == (True, 5)
    assert second.problem is not None
    assert second.problem.startswith("the mathematics is not as chunk 5 writes it")
    # Found in the other passage of the idea, it is filed there.
    assert (third.grounded, third.chunk_id) == (True, 6)
    assert checks.points == [
        None,
        "it rests on evidence sentence 2, which does not hold up",
        None,
        "it rests on evidence sentence 4, and there is no such sentence",
    ]
    assert (checks.count, checks.compound, checks.framed) == (None, None, None)
    assert checks.failed == ["evidence", "key_points"]


@pytest.mark.parametrize(
    ("points", "count"),
    [
        ([("saturation", 3, 1)], "there is 1 key point, not two to four"),
        ([("saturation", 3, 1)] * 2, None),
        ([("saturation", 3, 1)] * 4, None),
        ([("saturation", 3, 1)] * 5, "there are 5 key points, not two to four"),
    ],
)
def test_an_answer_has_two_to_four_key_points(
    points: list[tuple[str, int, int]], count: str | None
) -> None:
    checks = check_answer(written(points=points), SOURCES)

    assert checks.count == count
    assert ("key_points" in checks.failed) == (count is not None)


def test_an_answer_with_no_evidence_fails_and_says_so() -> None:
    answered = written(evidence=[], points=[("saturation", 3, 1), ("scaling", 2, 1)])

    checks = check_answer(answered, SOURCES)

    assert checks.failed == ["evidence", "key_points"]
    assert problems(answered, checks)[0] == "no evidence sentence was copied"


def test_key_points_are_kept_as_the_library_keeps_them() -> None:
    answered = written(
        evidence=[(SATURATE, 5), (HEADS, 5)],
        points=[("saturation", 3, 1), ("heads", 2, 2), ("nowhere", 1, 3)],
    )

    points = library_key_points(answered, check_answer(answered, SOURCES))

    assert points == [
        {"text": "saturation", "weight": 3, "evidence_quote": SATURATE, "chunk_id": 5},
        # Filed under the chunk the sentence is in, not the one the writer named
        {"text": "heads", "weight": 2, "evidence_quote": HEADS, "chunk_id": 6},
        {"text": "nowhere", "weight": 1, "evidence_quote": "", "chunk_id": None},
    ]


def test_a_question_whose_evidence_holds_is_written_in_one_call() -> None:
    seen: list[list[ModelMessage]] = []

    result = asyncio.run(write_question(writer([answer()], seen), concept(), SOURCES, "why_how"))

    assert (len(result.answers), result.checks.failed, result.style) == (1, [], "why_how")
    assert result.usage["requests"] == 1
    assert (result.model, result.prompt_version) == ("function:respond:", PROMPT_VERSION)
    assert result.answer.question == SINGLE
    assert "Write the question in this style: ask why it is done this way" in last_prompt(seen[0])


def test_an_answer_has_more_room_than_generate_v5_gave() -> None:
    settings: list[Any] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        settings.append(info.model_settings or {})
        return ModelResponse(parts=[TextPart(json.dumps(answer()))])

    model = FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))

    asyncio.run(write_question(model, concept(), SOURCES, "why_how"))

    # A repair copying the mathematics of an equation ran past generate-v5's 3,000 tokens.
    assert settings[0].get("max_tokens") == MAX_TOKENS == 5_000
    # The reasoning stays as low as generate-v5's; pydantic-ai hands it to the model apart.
    assert writing_settings() == {"thinking": "low", "max_tokens": 5_000}


def test_what_fails_goes_back_once_with_every_problem_named() -> None:
    seen: list[list[ModelMessage]] = []
    retyped = "To counteract this effect, we scale the dot products by 1/sqrt(d_k)."
    first = answer(
        evidence=[(SATURATE, 5), (retyped, 5)],
        question="Why do the authors scale the dot products, and what happens without it?",
    )

    result = asyncio.run(
        write_question(writer([first, answer()], seen), concept(), SOURCES, "why_how")
    )

    assert (len(result.answers), result.checks.failed) == (2, [])
    assert result.usage["requests"] == 2
    repair = last_prompt(seen[1])
    assert f'- evidence sentence 2 ("{retyped}"): the mathematics is not as chunk 5' in repair
    assert '- key point 2 ("scaling undoes it"): it rests on evidence sentence 2,' in repair
    assert "- the question asks more than one thing: it joins a second question on" in repair
    assert '- the question leans on the document: it mentions "the authors"' in repair
    # The first answer goes back with the request to mend it.
    assert len(seen[1]) > len(seen[0])


def test_the_reasoning_is_left_out_of_the_conversation_sent_back() -> None:
    seen: list[list[ModelMessage]] = []

    asyncio.run(
        write_question(
            writer([answer(question=DOUBLE), answer()], seen, reasoning=True),
            concept(),
            SOURCES,
            "why_how",
        )
    )

    [first_answer] = [message for message in seen[1] if isinstance(message, ModelResponse)]
    assert [type(part) for part in first_answer.parts] == [TextPart]
    assert DOUBLE in str(first_answer.parts[0].content)


def test_what_still_fails_after_the_repair_is_returned_as_it_is() -> None:
    seen: list[list[ModelMessage]] = []
    bad = answer(question=DOUBLE)

    result = asyncio.run(write_question(writer([bad, bad], seen), concept(), SOURCES, "why_how"))

    assert (len(result.answers), result.checks.failed) == (2, ["compound"])
    assert len(seen) == 2


def repair_failing(failure: Exception) -> FunctionModel:
    """A writer whose first answer is sent back, and whose repair fails with `failure`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            raise failure
        return ModelResponse(parts=[TextPart(json.dumps(answer(question=DOUBLE)))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


def test_what_a_writing_spent_and_wrote_is_kept_when_the_repair_fails() -> None:
    model = repair_failing(RuntimeError("the provider went away"))
    used = RunUsage()
    kept: list[IdeaQuestion] = []

    with pytest.raises(RuntimeError):
        asyncio.run(write_question(model, concept(), SOURCES, "why_how", usage=used, answers=kept))

    assert used.requests == 1
    assert used.input_tokens > 0
    assert [item.question for item in kept] == [DOUBLE]


def test_a_question_whose_repair_is_turned_down_stands_as_its_first_answer() -> None:
    planned = entry()
    refused = ModelHTTPError(400, "openai/gpt-oss-120b", {"code": "json_validate_failed"})

    with pytest.raises(ModelHTTPError):
        asyncio.run(write_questions.write(repair_failing(refused), planned, concept(), SOURCES))

    assert planned.written
    assert [item.question for item in planned.answers] == [DOUBLE]
    assert (planned.error or "").startswith("ModelHTTPError: status_code: 400")
    assert [call.usage["requests"] for call in planned.calls] == [1]


def test_a_question_the_days_budget_stops_is_written_again() -> None:
    planned = entry()
    gone = QuotaExhausted("groq", "groq is out of tokens for today")

    with pytest.raises(QuotaExhausted):
        asyncio.run(write_questions.write(repair_failing(gone), planned, concept(), SOURCES))

    # The next run, with a day's budget for its repair, writes it from the start.
    assert not planned.written
    assert (planned.error or "").startswith("QuotaExhausted")
    assert [call.usage["requests"] for call in planned.calls] == [1]


def checker(verdict: dict[str, Any], seen: list[str]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(last_prompt(messages))
        return ModelResponse(parts=[TextPart(json.dumps(verdict))])

    return FunctionModel(respond, profile=ModelProfile(supports_json_schema_output=True))


@pytest.mark.parametrize(
    ("kind", "answerable", "failed"),
    [
        ("explain", True, []),
        ("explain", False, ["answerable"]),
        ("recall", True, ["trivia"]),
        ("recall", False, ["answerable", "trivia"]),
    ],
)
def test_the_local_model_turns_down_what_the_passages_do_not_answer_and_recall(
    kind: str, answerable: bool, failed: list[str]
) -> None:
    seen: list[str] = []
    verdict = {"kind": kind, "answer": "Scaling.", "answerable": answerable, "missing": "nothing"}

    check, model = asyncio.run(check_question(checker(verdict, seen), SINGLE, SOURCES))

    assert checker_failed(check) == failed
    assert model == "function:respond:"
    assert seen[0].startswith(f"Question: {SINGLE}\n\nPassages:\n[chunk 5]")
    assert "[chunk 6]" in seen[0]


def entry(**fields: Any) -> Entry:
    values: dict[str, Any] = {
        "key": "1.1",
        "name": "scaling the dot products",
        "document_id": 1,
        "style": "why_how",
        "chunk_ids": [5, 6],
    } | fields
    return Entry(**values)


def test_a_saved_question_is_read_back_and_its_checks_worked_out_again() -> None:
    call = Call(model="openai/gpt-oss-120b", version=PROMPT_VERSION, usage={"requests": 2}, at="t")
    explained = AnswerCheck(kind="explain", answer="Scaling.", answerable=True, missing="nothing")
    accepted = entry(
        answers=[written(question=DOUBLE), written(misconceptions=["a", "b", "c", "d"])],
        calls=[call],
        check=explained,
        checker_model="qwen3.5:4b",
        duplicate={"checked": True, "line": 0.75},
    )
    recalled = entry(
        key="1.2",
        answers=[written()],
        calls=[call],
        check=explained.model_copy(update={"kind": "recall"}),
    )
    unchecked = entry(key="1.3", answers=[written()], calls=[call])
    failed = entry(key="1.4", calls=[call], error="ModelHTTPError: 400")
    unrepaired = entry(
        key="1.5",
        answers=[written(question=DOUBLE)],
        calls=[call],
        check=explained,
        error="ModelHTTPError: 400",
    )

    data = json.loads(
        json.dumps(
            entries_json(
                [accepted, recalled, unchecked, failed, unrepaired],
                {key: SOURCES for key in ("1.1", "1.2", "1.3", "1.4", "1.5")},
            )
        )
    )

    first, second, third, fourth, fifth = data["questions"]
    assert (first["accepted"], first["failed"], first["question"]) == (True, [], SINGLE)
    assert first["misconceptions"] == ["a", "b", "c"]
    assert first["key_points"][0] == {
        "text": "large dot products saturate the softmax",
        "weight": 3,
        "evidence_quote": SATURATE,
        "chunk_id": 5,
        "evidence": 1,
        "problem": None,
    }
    assert [item["problem"] for item in first["evidence"]] == [None, None]
    assert (second["accepted"], second["failed"]) == (False, ["trivia"])
    # Not yet read by the local model, it is not accepted.
    assert (third["accepted"], third["failed"]) == (False, [])
    assert "question" not in fourth
    # Its repair turned down, a question is judged on the answer it kept.
    assert (fifth["question"], fifth["failed"], fifth["accepted"]) == (DOUBLE, ["compound"], False)
    assert fifth["error"] == "ModelHTTPError: 400"
    assert data["usage"]["requests"] == 10
    # Only what was planned and answered is read back.
    assert read_entries(data) == [accepted, recalled, unchecked, failed, unrepaired]


def test_a_question_saved_for_another_plan_does_not_fit_this_one() -> None:
    saved = entry(answers=[written()])

    assert misfit(saved, entry()) is None
    assert misfit(saved, entry(style="intuition")) == (
        "idea 1.1 was written with style why_how, and is planned with intuition"
    )
    assert misfit(saved, entry(chunk_ids=[5])) == (
        "idea 1.1 was written with passages [5, 6], and is planned with [5]"
    )
    assert misfit(saved, entry(version="generate-v7")).endswith(
        f"prompt {PROMPT_VERSION}, and is planned with generate-v7"
    )
    assert misfit(saved, entry(name="dot-product scaling")) is not None
