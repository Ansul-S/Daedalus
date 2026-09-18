"""Writing one interview question from a handful of source chunks.

The model is given the chunks in delimiters, asked for one question in a named style, and
made to answer in a fixed schema. Every key point it writes has to carry a quote copied from
one of those chunks; when a quote turns out not to be in them, the whole conversation goes
back with the failures named and the model gets one chance to fix it. That repair round cost
about 3.5K tokens in the trial against 2.3K for a question that came out right the first
time, and it took quote grounding from 20 of 24 to 25 of 25.

Structured output goes through a strict JSON schema. The same trial had Groq's own validator
reject a tool call once, while every schema-constrained request came back valid.
"""

import logging
from dataclasses import dataclass, field
from itertools import count
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from app.db.models import QUESTION_STYLES, Chunk
from app.ingest.chunking import SECTION_SEPARATOR
from app.questions.grounding import QuoteCheck, check_quote

log = logging.getLogger(__name__)

PROMPT_VERSION = "generate-v1"
# Enough for a question with its answer and reasoning; a runaway answer would otherwise eat
# a whole minute of the token budget.
MAX_TOKENS = 3000

Style = Literal[
    "intuition", "why_how", "compare", "tradeoffs", "failure_modes", "connection", "paper"
]

STYLE_BRIEFS: dict[str, str] = {
    "intuition": "ask for the intuition behind an idea: what it is really doing, in plain terms",
    "why_how": "ask why something is done this way, or how the mechanism actually works",
    "compare": "ask the candidate to compare two things the sources describe",
    "tradeoffs": "ask about a trade-off, or when one choice is preferable to another",
    "failure_modes": "ask how something breaks, or what goes wrong when it is left out",
    "connection": "ask how two ideas from different parts of the sources fit together",
    "paper": "ask about the problem the paper attacks, its key idea, or where it stops",
}
assert set(STYLE_BRIEFS) == set(QUESTION_STYLES)

INSTRUCTIONS = """\
You write interview questions for an AI and machine-learning engineering interview, one at a
time, from the sources you are given and from nothing else.

The question asks about an idea: why something is done, how it works, what it costs, how it
fails, or how two ideas fit together. Never ask about the document itself -- its sections,
its figures, its numbering or its authors. It has to be answerable from the sources alone by
someone who has not read them, and it is one question, not two joined by "and". A number or
a name is only worth asking about when the sources also say what is behind it.

The reference answer is what a strong candidate would say, in a few sentences, drawn only
from what the sources state. Never add a fact of your own, however true it is.

Key points are what an answer has to contain. Give two to four, each with a weight of 1, 2
or 3 for how much of the answer it carries, and each with an evidence quote: a run of at
least six words copied out of one source character for character, mathematics included, that
shows the point is really there. Never join distant parts of a passage with an ellipsis, and
never stop at a colon -- quote the words that carry the meaning. Say which chunk each quote
came from.

Misconceptions are up to three answers that sound right and are not, a sentence each.

Difficulty is 1 for recalling something stated outright, 3 for explaining a mechanism, and 5
for reasoning across more than one source.
"""

REQUEST = """\
Write one question in this style: {brief}.

Sources:
{sources}
"""

SOURCE = """\
[chunk {chunk_id}] {citation}
<<<
{text}
>>>
"""

REPAIR = """\
These evidence quotes are not in the sources:

{problems}

Send the whole question again with each of them replaced by words copied out of the chunk it
belongs to, character for character. Leave the question and the reference answer as they are
unless a key point has to change with its quote.
"""


class KeyPoint(BaseModel):
    text: str = Field(description="One thing an answer has to contain")
    weight: Literal[1, 2, 3] = Field(description="How much of the answer this point carries")
    evidence_quote: str = Field(description="Six or more words copied from a source exactly")
    chunk_id: int = Field(description="The chunk the quote was copied from")


class GeneratedQuestion(BaseModel):
    question: str
    style: Style
    difficulty: Literal[1, 2, 3, 4, 5]
    reference_answer: str
    key_points: list[KeyPoint] = Field(min_length=2, max_length=4)
    misconceptions: list[str] = Field(max_length=3)
    source_chunk_ids: list[int] = Field(description="The chunks the question was written from")


@dataclass
class Source:
    chunk_id: int
    # Where the passage is from, e.g. "Attention Is All You Need > 3.2 Attention"
    citation: str
    text: str


@dataclass
class Generated:
    question: GeneratedQuestion
    quotes: list[QuoteCheck]
    # The model that answered, which under a fallback chain is not always the first one
    model: str
    prompt_version: str = PROMPT_VERSION
    usage: dict[str, int] = field(default_factory=dict)
    # 1 when the quotes were right the first time, 2 after one repair round
    attempts: int = 1

    @property
    def grounded(self) -> bool:
        return all(quote.grounded for quote in self.quotes)


def source_of(title: str, chunk: Chunk) -> Source:
    """A stored chunk as the model sees it, cited the way the chunk was embedded."""
    citation = SECTION_SEPARATOR.join([title, chunk.section]) if chunk.section else title
    return Source(chunk_id=chunk.id, citation=citation, text=chunk.text)


def generation_settings() -> ModelSettings:
    return ModelSettings(thinking="low", max_tokens=MAX_TOKENS)


def question_agent(model: Model) -> Agent[None, GeneratedQuestion]:
    return Agent(
        model,
        output_type=NativeOutput(GeneratedQuestion, strict=True),
        instructions=INSTRUCTIONS,
    )


def request(sources: list[Source], style: str) -> str:
    passages = "\n".join(
        SOURCE.format(chunk_id=source.chunk_id, citation=source.citation, text=source.text)
        for source in sources
    )
    return REQUEST.format(brief=STYLE_BRIEFS[style], sources=passages)


def repair_request(failed: list[QuoteCheck]) -> str:
    problems = "\n".join(f'- "{quote.quote}": {quote.problem}' for quote in failed)
    return REPAIR.format(problems=problems)


def ground(question: GeneratedQuestion, sources: list[Source]) -> list[QuoteCheck]:
    texts = {source.chunk_id: source.text for source in sources}
    return [
        check_quote(point.evidence_quote, point.chunk_id, texts.get(point.chunk_id))
        for point in question.key_points
    ]


def add_usage(total: dict[str, int], result: AgentRunResult[GeneratedQuestion]) -> dict[str, int]:
    used = result.usage
    counted = {
        "requests": used.requests,
        "input_tokens": used.input_tokens,
        "output_tokens": used.output_tokens,
        "reasoning_tokens": int((used.details or {}).get("reasoning_tokens", 0) or 0),
    }
    return {name: total.get(name, 0) + value for name, value in counted.items()}


async def generate_question(
    model: Model, sources: list[Source], style: str, *, repairs: int = 1
) -> Generated:
    """Write one question, repairing quotes that are not in the sources up to `repairs` times.

    The result is returned whether or not the quotes came good: the caller records why a
    question was turned down as readily as why it was kept.
    """
    agent = question_agent(model)
    prompt = request(sources, style)
    history = None
    usage: dict[str, int] = {}
    for attempt in count(1):
        result = await agent.run(
            prompt, message_history=history, model_settings=generation_settings()
        )
        usage = add_usage(usage, result)
        quotes = ground(result.output, sources)
        failed = [quote for quote in quotes if not quote.grounded]
        if not failed or attempt > repairs:
            answered = result.all_messages()[-1]
            return Generated(
                question=result.output,
                quotes=quotes,
                model=getattr(answered, "model_name", None) or model.model_name,
                usage=usage,
                attempts=attempt,
            )
        log.info("repairing %d quote(s): %s", len(failed), [q.problem for q in failed])
        history = result.all_messages()
        prompt = repair_request(failed)
    raise AssertionError("unreachable")
