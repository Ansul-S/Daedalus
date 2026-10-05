"""Writing one question for each idea of the concept inventory, evidence first.

Planned from a passage, a question could ask about anything the passage touched, and the writer
filled in whatever its sources left out: 16 of the 49 questions that passed the checks up to
30 Sep rest part of their weight on a key point their passages do not hold, and 7 of the 9
accepted trade-off questions invent their trade-off. Here the plan is one idea of the inventory,
in a style its passages support, and the writer answers in the order the parts rest on each
other: it copies the sentences that explain the idea, asks the question, writes key points that
each rest on one of those sentences, and makes the reference answer of the key points alone.
It is not shown what was asked before: the plan asks each idea once.

The code checks what it can (`check_answer`): every evidence sentence is in the passages
(`check_evidence`), every key point rests on one that holds up, the question asks one thing
(`compound`) and it stands without the document. The local model reads each answer against the
passages as well (`check_question`): whether they answer it, and whether answering means
explaining rather than recalling. What fails goes back once, with the problems named, a recall
reading among them. Whether a key point says what its sentence says is left to a person: of
three checks measured on 422 key points labelled by hand, none caught enough of the unsupported
ones without flagging too many of the rest.

The style follows the evidence (`supported_styles`): a why or how question only when the passages
give the reason, a comparison only when they draw one, a trade-off only when they name one.
Spread across the ideas planned together (`assign_styles`), no style takes over.
"""

import logging
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from itertools import count
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from app.db.models import QUESTION_STYLES
from app.ingest.chunking import SECTION_SEPARATOR
from app.llm.models import helper_settings
from app.questions import validation
from app.questions.generation import SOURCE, Source, compound, counted
from app.questions.grounding import QuoteCheck, check_evidence
from app.questions.inventory import Call, Concept, DocumentInventory, total_usage
from app.questions.validation import AnswerCheck

log = logging.getLogger(__name__)

PROMPT_VERSION = "generate-v7"
# The passages an idea is written from, the best first, until the next would pass this many
# tokens. A repair sends them again with the first answer, under the 8,000 tokens a minute of
# Groq's free tier; the most any idea of the three demo papers takes is 4,183.
PASSAGE_TOKENS = 5_000
FEWEST_POINTS, MOST_POINTS = 2, 4
MISCONCEPTIONS = 3
# What an answer may take, its reasoning with it. This writer's answers came to 450 to 950
# tokens, but a repair sent back to copy the mathematics of an equation ran past generate-v5's
# 3,000 before its JSON was complete, and Groq turned the whole request down. Groq does not
# reserve the limit against its tokens a minute.
MAX_TOKENS = 5_000

# What a question in each style asks, as the writer is told. The paper style and the connection
# between two passages are gone: the plan is one idea, not one passage or two. Failure modes and
# comparisons ask why: asked how or when it goes wrong, or how it differs, generate-v6 asked for
# what the passages report in 5 of its 7, all of them labelled recall.
STYLE_BRIEFS: dict[str, str] = {
    "why_how": "ask why it is done this way, or how it works",
    "intuition": "ask for the intuition behind it: what it is really doing, in plain terms",
    "compare": "ask why it behaves differently from the alternative the passages compare it with",
    "tradeoffs": "ask about the trade-off the passages name: what it gains, and what it costs",
    "failure_modes": "ask why it goes wrong, as the passages explain it",
}
assert set(STYLE_BRIEFS) <= set(QUESTION_STYLES)

# What the inventory found the passages explain about an idea, in the words the writer is shown
EXPLAINS = {
    "mechanism": "how it works",
    "reason": "the reason for it",
    "trade-off": "what it trades off",
    "failure": "how it fails",
    "comparison": "how it differs from an alternative",
    "definition": "what it is",
}

# What ties a question to the document it was written from: its authors, the paper itself, or
# one of its numbered tables, figures, sections, equations or stages. On the 133 questions
# labelled by hand it finds 15 of the 24 framed on their document, and 6 of the other 109, each
# of them about one document's own system ("Stage 2 (Vector Search Enhancement)", "the proposed
# multi-candidate span selection method") or naming "the Transformer paper". What it cannot see
# is a document's own names for its parts and data ("CCPairs"), which takes reading. "According
# to the analysis" and "according to the passages", which three of generate-v6's twenty leaned
# on, appear in none of the 133.
FRAMED = re.compile(
    r"\b(?:the|these)\s+authors?\b|\bpapers?\b|\barticle\b|\bthis\s+(?:work|study|document)\b"
    r"|\bthe\s+proposed\b|\bas\s+described\b|\bappendix\b"
    r"|\b(?:table|figure|fig\.|section|sec\.|stage|eqn?\.?|equation|algorithm|theorem|lemma)"
    r"\s*\(?(?:\d+|[ivx]+\b)"
    r"|\baccording\s+to\s+the\s+(?:[\w\u2011-]+\s+){0,3}?"
    r"(?:analysis|results?|findings|experiments?|passages?|text)\b",
    re.IGNORECASE,
)

INSTRUCTIONS = """\
You write one question for an AI and machine-learning engineering interview, about one idea,
from the passages you are given and from nothing else. You write its parts in the order they rest
on each other.

evidence: first, the sentences of the passages that explain what a question in your style asks
about the idea -- why it is done, how it works, what it costs, how it fails or how it compares --
one to four of them, each with the chunk it is in. Copy each one whole, character for character,
as it stands: never shorten a sentence or join two with "...", and copy mathematics symbol for
symbol, every command and bracket of it. Where a sentence of words says what a formula says, take
the sentence of words. A label written under or inside a formula is part of the formula, not a
sentence.

question: one question, in the style you are given, that the evidence answers for someone who
has not read the passages. Ask only what the evidence answers: never a reason, a trade-off or a
failure it does not state. It asks about the idea, never about the document: not its sections,
tables, figures, equations or stages, not "the paper", "the authors" or "the analysis". Its answer
explains something; a number, a name or a fact read off the page is not a question, and nor is a
result the passages report -- when something happened, how often, which did better -- however
it is worded: ask what the passages explain about why. It asks one thing: never join a second
question on with "and how", "and what" or "and why", or with a second question mark.

key_points: what an answer to the question has to contain, two to four of them. Each is part of
the answer to what the question asks, says only what one evidence sentence says, gives the
number of that sentence, counted from 1, and has a weight of 1, 2 or 3 for how much of the
answer it carries.

reference_answer: what a strong candidate would say, in a few sentences, made of the key points
and nothing else, however true it would be.

misconceptions: up to three answers that sound right and are not, a sentence each.

difficulty: 1 when the evidence states the answer outright, 3 when the answer has to put a
mechanism together, and 5 when it has to reason across passages.
"""

REQUEST = """\
The idea: {name}. {summary}
{explains}
Write the question in this style: {brief}.

Passages:
{sources}"""

REPAIR = """\
Not all of it holds up:

{problems}

Send the whole question again with these put right. Copy every evidence sentence whole out of
the chunk it is in, character for character, and leave the rest as it is unless it has to change
with them.
"""


class Evidence(BaseModel):
    sentence: str = Field(description="A whole sentence of a chunk, copied exactly as it stands")
    chunk_id: int = Field(description="The chunk it was copied from")


class CitedPoint(BaseModel):
    text: str = Field(description="One thing an answer has to contain")
    weight: Literal[1, 2, 3] = Field(description="How much of the answer it carries")
    evidence: int = Field(description="The number of the evidence sentence it rests on, from 1")


class IdeaQuestion(BaseModel):
    # In the order its parts rest on each other, which is the order the model writes them in.
    # The lists have no bounds here: Groq holds an answer to the schema only once it is written,
    # and turned whole questions down for a fifth key point. The counts are checked in code.
    evidence: list[Evidence] = Field(description="The sentences that explain the idea")
    question: str
    key_points: list[CitedPoint]
    reference_answer: str = Field(description="A strong answer, made of the key points alone")
    misconceptions: list[str]
    difficulty: Literal[1, 2, 3, 4, 5]


@dataclass
class Checks:
    """What the code makes of one answer."""

    # Each evidence sentence as checked, filed under the chunk it was found in
    evidence: list[QuoteCheck]
    # Why each key point does not rest on a sentence that holds up; None for one that does
    points: list[str | None]
    # Why there are too few key points or too many, if there are
    count: str | None
    # Why the question asks more than one thing, if it does
    compound: str | None
    # What ties the question to its document, if anything does
    framed: str | None

    @property
    def failed(self) -> list[str]:
        """The names of the checks the answer fails."""
        failures = {
            "evidence": not self.evidence or not all(check.grounded for check in self.evidence),
            "key_points": self.count is not None or any(self.points),
            "compound": self.compound is not None,
            "framed": self.framed is not None,
        }
        return [name for name, failed in failures.items() if failed]


@dataclass
class Written:
    """One idea's question as the writer left it."""

    # Every answer it gave, the last one standing
    answers: list[IdeaQuestion]
    # What the code makes of the last answer
    checks: Checks
    # The model that answered, which under a fallback chain is not always the first one
    model: str
    style: str
    usage: dict[str, int]
    # The local model's reading of the last answer and the model that read it, if one read it
    check: AnswerCheck | None = None
    checker_model: str | None = None
    prompt_version: str = PROMPT_VERSION

    @property
    def answer(self) -> IdeaQuestion:
        return self.answers[-1]


@dataclass
class Entry:
    """One idea's question as saved: what was planned, every answer the writer gave, what
    writing it cost, and the local model's reading of the last answer."""

    key: str
    name: str
    document_id: int
    style: str
    chunk_ids: list[int]
    version: str = PROMPT_VERSION
    answers: list[IdeaQuestion] = field(default_factory=list)
    # Every request, those of a writing that failed among them: what the day has spent
    calls: list[Call] = field(default_factory=list)
    check: AnswerCheck | None = None
    checker_model: str | None = None
    # Why the last writing failed, if it did. With no answer kept, the next run writes the
    # question again; with one, the question stands as its last answer left it.
    error: str | None = None
    # The questions nearest to this one, reported and not judged
    duplicate: dict[str, Any] | None = None

    @property
    def written(self) -> bool:
        return bool(self.answers)


def supported_styles(concept: Concept) -> list[str]:
    """The styles an idea's passages can carry, from what the inventory found they explain: a why
    or how question, and the intuition behind a mechanism or a definition, only when they give
    the reason; a comparison only when they draw one; a trade-off only when they name one; and
    failure modes only when they describe them."""
    kinds = set(concept.kinds)
    given = concept.reason == "given"
    supported = {
        "why_how": given,
        "intuition": given and bool(kinds & {"mechanism", "definition"}),
        "compare": "comparison" in kinds,
        "tradeoffs": "trade-off" in kinds,
        "failure_modes": "failure" in kinds,
    }
    return [style for style, ok in supported.items() if ok]


def key_order(key: str) -> tuple[int, ...]:
    """An inventory key, "document.idea", in the order of its numbers: 4.2 before 4.10."""
    return tuple(int(part) for part in key.split("."))


def assign_styles(
    concepts: dict[str, Concept], chosen: dict[str, str] | None = None
) -> dict[str, str]:
    """A style for each idea, one its passages support, spread so that no style takes over.

    The ideas take their turn in an order of their own, whatever order they come in: those with
    the fewest styles to choose from first, then by key. Each takes the style given least so
    far, a tie going to the style the fewest ideas can take. A `chosen` style is taken in its
    idea's turn, so that one chosen as the spread would choose it leaves every other idea's
    style as it was.

    Raises ValueError for an idea whose passages support no style, and for a chosen style they
    do not support.
    """
    chosen = chosen or {}
    supported = {key: supported_styles(concept) for key, concept in concepts.items()}
    for key, style in chosen.items():
        if key not in concepts:
            raise ValueError(f"a style is chosen for idea {key}, which is not planned")
        if style not in supported[key]:
            allowed = ", ".join(supported[key]) or "none"
            raise ValueError(f"idea {key}'s passages do not support {style}, only: {allowed}")
    if (bare := next((key for key, styles in supported.items() if not styles), None)) is not None:
        raise ValueError(f"idea {bare}'s passages support no style of question")

    takers = Counter(style for styles in supported.values() for style in styles)
    given: Counter[str] = Counter()
    styles: dict[str, str] = {}
    for key in sorted(concepts, key=lambda key: (len(supported[key]), key_order(key))):
        if key in chosen:
            style = chosen[key]
        else:
            style = min(
                supported[key], key=lambda s: (given[s], takers[s], list(STYLE_BRIEFS).index(s))
            )
        styles[key] = style
        given[style] += 1
    return {key: styles[key] for key in concepts}


def sources_of(concept: Concept, documents: Sequence[DocumentInventory]) -> list[Source]:
    """The passages an idea is written from: those the inventory files it under, the best first,
    until the next would take them past PASSAGE_TOKENS."""
    document = {document.document_id: document for document in documents}[concept.document_id]
    passages = {
        passage.chunk_id: passage for window in document.windows for passage in window.passages
    }
    sources: list[Source] = []
    size = 0
    for chunk_id in concept.chunk_ids:
        passage = passages[chunk_id]
        if sources and size + passage.tokens > PASSAGE_TOKENS:
            break
        sections = [document.title, passage.section] if passage.section else [document.title]
        sources.append(
            Source(chunk_id=chunk_id, citation=SECTION_SEPARATOR.join(sections), text=passage.text)
        )
        size += passage.tokens
    return sources


def listed(items: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def request(concept: Concept, sources: Sequence[Source], style: str) -> str:
    passages = "\n".join(
        SOURCE.format(chunk_id=source.chunk_id, citation=source.citation, text=source.text)
        for source in sources
    )
    explains = [EXPLAINS[kind] for kind in concept.kinds if kind in EXPLAINS]
    return REQUEST.format(
        name=concept.name,
        summary=concept.summary,
        explains=f"The passages explain {listed(explains)}.\n" if explains else "",
        brief=STYLE_BRIEFS[style],
        sources=passages,
    )


def framed(question: str) -> str | None:
    """What ties a question to the document it was written from, or None when it stands alone."""
    found = FRAMED.search(question)
    return f'it mentions "{found.group(0)}"' if found is not None else None


def respelled(sentence: str) -> str:
    """A sentence with its inline mathematics delimited as the passages delimit it: LaTeX's
    \\(...\\) written $...$. generate-v6 copied one formula exactly but in the other spelling."""
    return sentence.replace("\\(", "$").replace("\\)", "$")


def filed(sentence: str, chunk_id: int, texts: dict[int, str]) -> QuoteCheck:
    """An evidence sentence checked against the chunk it names, then against the idea's other
    passages: found in another, it is filed there. Found in none, the check against the chunk it
    names says why."""
    named = check_evidence(sentence, chunk_id, texts.get(chunk_id))
    if named.grounded:
        return named
    others = (
        check_evidence(sentence, other, text) for other, text in texts.items() if other != chunk_id
    )
    return next((check for check in others if check.grounded), named)


def place(sentence: str, chunk_id: int, texts: dict[int, str]) -> QuoteCheck:
    """An evidence sentence filed under the passage it is in, its inline mathematics read as the
    passages write it if it does not hold as written; then it is kept in their spelling."""
    written = filed(sentence, chunk_id, texts)
    if written.grounded or (spelled := respelled(sentence)) == sentence:
        return written
    as_spelled = filed(spelled, chunk_id, texts)
    return as_spelled if as_spelled.grounded else written


def resting(point: CitedPoint, evidence: Sequence[QuoteCheck]) -> str | None:
    """Why a key point does not rest on an evidence sentence that holds up, or None when it
    does."""
    if not 1 <= point.evidence <= len(evidence):
        return f"it rests on evidence sentence {point.evidence}, and there is no such sentence"
    if not evidence[point.evidence - 1].grounded:
        return f"it rests on evidence sentence {point.evidence}, which does not hold up"
    return None


def check_answer(answer: IdeaQuestion, sources: Sequence[Source]) -> Checks:
    texts = {source.chunk_id: source.text for source in sources}
    evidence = [place(item.sentence, item.chunk_id, texts) for item in answer.evidence]
    size = len(answer.key_points)
    counted_points = "there is 1 key point" if size == 1 else f"there are {size} key points"
    return Checks(
        evidence=evidence,
        points=[resting(point, evidence) for point in answer.key_points],
        count=None
        if FEWEST_POINTS <= size <= MOST_POINTS
        else f"{counted_points}, not two to four",
        compound=compound(answer.question),
        framed=framed(answer.question),
    )


def problems(answer: IdeaQuestion, checks: Checks) -> list[str]:
    """What the code finds wrong with an answer, a line each, in words the writer can act on."""
    lines = [] if answer.evidence else ["no evidence sentence was copied"]
    lines += [
        f'evidence sentence {number} ("{check.quote}"): {check.problem}'
        for number, check in enumerate(checks.evidence, start=1)
        if check.problem is not None
    ]
    lines += [
        f'key point {number} ("{point.text}"): {why}'
        for number, (point, why) in enumerate(
            zip(answer.key_points, checks.points, strict=True), start=1
        )
        if why is not None
    ]
    if checks.count is not None:
        lines.append(checks.count)
    if checks.compound is not None:
        lines.append(f"the question asks more than one thing: {checks.compound}")
    if checks.framed is not None:
        lines.append(f"the question leans on the document: {checks.framed}")
    return lines


def repair_request(lines: Sequence[str]) -> str:
    return REPAIR.format(problems="\n".join(f"- {line}" for line in lines))


def library_key_points(answer: IdeaQuestion, checks: Checks) -> list[dict[str, Any]]:
    """The key points as the library keeps them, each with the evidence sentence it rests on as
    its evidence quote, as the passage spells it and filed under the chunk it was found in. One
    resting on no sentence has neither."""
    points = []
    for point in answer.key_points:
        number = point.evidence
        rests = 1 <= number <= len(answer.evidence)
        points.append(
            {
                "text": point.text,
                "weight": point.weight,
                "evidence_quote": checks.evidence[number - 1].quote if rests else "",
                "chunk_id": checks.evidence[number - 1].chunk_id if rests else None,
            }
        )
    return points


def reasoning(part: object) -> bool:
    """Whether a part of an answer is the model's reasoning."""
    return isinstance(part, ThinkingPart)


def without_reasoning(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """The conversation without the model's reasoning. Groq is sent it back inside the answer,
    where it takes room that a repair, sending the passages again, needs under Groq's 8,000
    tokens a minute."""
    return [
        replace(message, parts=[part for part in message.parts if not reasoning(part)])
        if isinstance(message, ModelResponse)
        else message
        for message in messages
    ]


def question_writer(model: Model) -> Agent[None, IdeaQuestion]:
    return Agent(
        model,
        name="writer",
        output_type=NativeOutput(IdeaQuestion, strict=True),
        instructions=INSTRUCTIONS,
    )


def writing_settings() -> ModelSettings:
    return ModelSettings(thinking="low", max_tokens=MAX_TOKENS)


# What goes back when the local model reads a question as recall. The six of generate-v6's twenty
# it read as recall were all labelled recall by hand.
RECALL = (
    "answering the question means recalling what the passages report, not explaining it: ask "
    "why or how, as the evidence explains it"
)

# The local model reading a question against the idea's passages: its reading, and which model
Reader = Callable[[str], Awaitable[tuple[AnswerCheck, str]]]


class ReadingFailed(Exception):
    """The local model could not read an answer. The writing stops, to be done again from the
    start once it can, so that no question goes without the reading its repair rests on."""


async def reading_of(read: Reader, question: str) -> tuple[AnswerCheck, str]:
    try:
        return await read(question)
    except Exception as exc:
        raise ReadingFailed(f"{type(exc).__name__}: {exc}") from exc


async def write_question(
    model: Model,
    concept: Concept,
    sources: list[Source],
    style: str,
    *,
    repairs: int = 1,
    usage: RunUsage | None = None,
    answers: list[IdeaQuestion] | None = None,
    read: Reader | None = None,
) -> Written:
    """Write one question about an idea, sending back what the code finds wrong up to `repairs`
    times, and, given `read`, what the local model reads as recall: it reads every answer before
    anything goes back, so recall goes back in the same round as the code's problems. `usage`
    counts every request as it is made, and `answers` keeps every answer as it comes, so a
    writing that fails part-way still says what it spent and what it wrote.

    The result is returned whether or not the repairs came good: why a question is turned down
    is kept as readily as why it is kept.
    """
    agent = question_writer(model)
    spent = usage if usage is not None else RunUsage()
    given = answers if answers is not None else []
    prompt = request(concept, sources, style)
    history: list[ModelMessage] | None = None
    for attempt in count(1):
        result = await agent.run(
            prompt,
            message_history=history,
            model_settings=writing_settings(),
            usage=spent,
            metadata={"prompt_version": PROMPT_VERSION},
        )
        given.append(result.output)
        checks = check_answer(result.output, sources)
        reading = await reading_of(read, result.output.question) if read is not None else None
        recalled = reading is not None and reading[0].kind == "recall"
        if not (checks.failed or recalled) or attempt > repairs:
            answered = result.all_messages()[-1]
            return Written(
                answers=list(given),
                checks=checks,
                model=getattr(answered, "model_name", None) or model.model_name,
                style=style,
                usage=counted(spent),
                check=reading[0] if reading is not None else None,
                checker_model=reading[1] if reading is not None else None,
            )
        lines = problems(result.output, checks) + [RECALL] * recalled
        log.info("sending back %s: %s", ", ".join(checks.failed + ["trivia"] * recalled), lines)
        history = without_reasoning(result.all_messages())
        prompt = repair_request(lines)
    raise AssertionError("unreachable")


async def check_question(
    model: Model, question: str, sources: list[Source]
) -> tuple[AnswerCheck, str]:
    """The local model's reading of a question against its passages, and the model that read
    it: whether they answer it, and whether answering means explaining or recalling."""
    result = await validation.checker(model).run(
        validation.review(question, sources),
        model_settings=helper_settings(),
        metadata={"prompt_version": validation.PROMPT_VERSION},
    )
    answered = result.all_messages()[-1]
    return result.output, getattr(answered, "model_name", None) or model.model_name


def checker_failed(check: AnswerCheck) -> list[str]:
    """The local model's verdicts that turn a question down: the passages do not answer it, or
    answering it means recalling a fact."""
    failures = {"answerable": not check.answerable, "trivia": check.kind == "recall"}
    return [name for name, failed in failures.items() if failed]


def misfit(saved: Entry, planned: Entry) -> str | None:
    """Why the question saved for an idea is not the one planned for it now, or None when it
    is."""
    for what, before, now in (
        ("name", saved.name, planned.name),
        ("style", saved.style, planned.style),
        ("passages", saved.chunk_ids, planned.chunk_ids),
        ("prompt", saved.version, planned.version),
    ):
        if before != now:
            return f"idea {saved.key} was written with {what} {before}, and is planned with {now}"
    return None


def entry_json(entry: Entry, sources: Sequence[Source]) -> dict[str, Any]:
    """An entry as saved: what was planned and what the models answered, and, worked out from
    them for reading, the last answer with its checks. Only the first part is read back."""
    data: dict[str, Any] = {
        "key": entry.key,
        "name": entry.name,
        "document_id": entry.document_id,
        "style": entry.style,
        "chunk_ids": entry.chunk_ids,
        "version": entry.version,
        "answers": [answer.model_dump() for answer in entry.answers],
        "calls": [vars(call) for call in entry.calls],
        "check": entry.check.model_dump() if entry.check is not None else None,
        "checker_model": entry.checker_model,
        "error": entry.error,
        "duplicate": entry.duplicate,
    }
    if not entry.answers:
        return data
    answer = entry.answers[-1]
    checks = check_answer(answer, sources)
    failed = checks.failed + (checker_failed(entry.check) if entry.check is not None else [])
    points = library_key_points(answer, checks)
    return data | {
        "question": answer.question,
        "difficulty": answer.difficulty,
        "reference_answer": answer.reference_answer,
        "misconceptions": answer.misconceptions[:MISCONCEPTIONS],
        "evidence": [
            {"sentence": check.quote, "chunk_id": check.chunk_id, "problem": check.problem}
            for check in checks.evidence
        ],
        "key_points": [
            point | {"evidence": cited.evidence, "problem": why}
            for point, cited, why in zip(points, answer.key_points, checks.points, strict=True)
        ],
        "problems": problems(answer, checks),
        "failed": failed,
        "accepted": entry.check is not None and not failed,
    }


def entries_json(entries: Sequence[Entry], sources: dict[str, list[Source]]) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "questions": [entry_json(entry, sources[entry.key]) for entry in entries],
        "usage": total_usage([call for entry in entries for call in entry.calls]),
    }


def read_entries(data: dict[str, Any]) -> list[Entry]:
    """The entries of a saved file, without what was worked out from them."""
    return [
        Entry(
            key=saved["key"],
            name=saved["name"],
            document_id=saved["document_id"],
            style=saved["style"],
            chunk_ids=saved["chunk_ids"],
            version=saved["version"],
            answers=[IdeaQuestion.model_validate(answer) for answer in saved["answers"]],
            calls=[Call(**call) for call in saved["calls"]],
            check=AnswerCheck.model_validate(saved["check"]) if saved["check"] else None,
            checker_model=saved["checker_model"],
            error=saved["error"],
            duplicate=saved["duplicate"],
        )
        for saved in data.get("questions", [])
    ]
