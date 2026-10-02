"""The ideas each document explains, listed once, to plan questions by.

Planned passage by passage, the generator asked an idea again from every passage that restates
it: of the 133 questions written up to 30 Sep, 31 repeat an earlier one, every one from the
same document, and 17 of them led by a different passage than the question they repeat. The
inventory lists ideas instead. Each comes with the passages that explain it, what kind of
explanation they give, whether they give the reason or only state the fact, whether the idea
reaches beyond the document, and how likely an interviewer is to ask about it, so that a
question can be planned once for each idea, in a style its evidence supports.

A strong model reads a document in windows of whole passages, about 5,000 tokens each: Groq's
free tier takes 8,000 tokens a minute, prompt and answer together. An idea explained in two
windows is listed twice, so a second call per document merges the two from their names and
summaries alone. Ideas from different documents are compared by embedding, and the pairs that
come close go to the model, which decides whether they are one idea.

Each idea comes with a sentence copied from its first passage, checked in code like an evidence
quote, so that an idea filed under the wrong passage shows up. Passages that can only be
context are not read at all.

An inventory is kept as JSON (`to_json`, `from_json`) with every answer the model gave, so a
run that stops part-way picks up where it stopped instead of paying for a window twice.

A person's review of the ideas -- merges the model missed, scopes it misjudged, ideas to leave
out -- is kept beside it (`Review`) and applied on top of the model's answers (`apply_review`).
Each decision names the idea it is about by key and by name, so that it never lands on another
idea once the documents are read again.
"""

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import combinations
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.ingest.chunking import SECTION_LIST_SEPARATOR
from app.llm.embeddings import Embedder
from app.llm.tracing import traced
from app.questions import tagging
from app.questions.generation import add_usage
from app.questions.grounding import QuoteCheck, check_evidence

log = logging.getLogger(__name__)

# v2: the evidence sentence is copied as it stands, words over formulas. Asked to include the
# mathematics, v1 retyped the LaTeX of three of Ragas's eleven and joined sentences with "..."
# in three, all in the right passage, but none of them a copy the quote check could find.
PROMPT_VERSION = "inventory-v2"
# Passages a window holds at most, counted by the embedding model's tokenizer. With the
# instructions on top the prompt stays near 6,000 tokens, inside Groq's 8,000 a minute.
WINDOW_TOKENS = 5_000
# Room for a window's ideas, at about a hundred tokens each, and the reasoning before them
WINDOW_MAX_TOKENS = 5_000
MERGE_MAX_TOKENS = 3_000
# Ideas from different documents this close, in cosine similarity, go to the model to decide
# whether they are one. A pair below the line is never asked about, so it errs low.
CLOSE = 0.7
# The close pairs sent in one request
PAIRS_PER_REQUEST = 20

# A passage is a table when half its characters sit in table lines.
TABLE_SHARE = 0.5
# What a broken LaTeX parse leaves behind: runs of "\ ", and letters split from their
# brackets, stray primes and spaced-out ellipses. A passage needs three of the latter to count
# as damaged, since one "(i) a" is an ordinary list.
GARBLED = re.compile(r"(?:\\ ){8,}")
BROKEN_MATHS = re.compile(r"\(\s?[a-z]\s?\)\s?[a-z]\b|\s′\s|\.\s\.\s\.")
BROKEN_MATHS_ALLOWED = 2
BACK_MATTER = re.compile(r"^\s*(?:appendix|acknowledg)", re.IGNORECASE)

Kind = Literal["mechanism", "reason", "trade-off", "failure", "comparison", "definition"]
KINDS: tuple[str, ...] = get_args(Kind)

INSTRUCTIONS = """\
You map the ideas a document explains, so that questions for an AI and machine-learning
engineering interview can be planned one idea at a time. You are given part of the document
as numbered chunks, and you work from them alone.

List every idea these chunks explain that an interviewer could ask a candidate about: a
method, a mechanism, a design choice, a principle or a known problem. An idea is what one
interview question asks about -- narrower than a whole method, wider than a single equation --
such as "scaling the dot products in attention", "exposure bias under teacher forcing" or "why
layer normalization steadies training". List each idea once, however many chunks say it
again. Never list the document itself, its sections, figures, tables or authors.

Leave out what only describes how the document tested its work: its datasets, experimental
set-up, prompts, examples and the numbers it reports. A result is an idea only when the chunks
say why it came out that way, and then the idea is the reason, not the number.

For each idea:

name: what a person would call it, in a few words.

summary: one sentence on what the chunks say about it.

chunk_ids: the chunks that explain it, the one that explains it best first. Leave out chunks
that only mention it.

kinds: what the chunks explain about it, and only what they actually give. "mechanism": how
it works. "reason": why it is done, or why it holds. "trade-off": what it gains against what
it costs, only when the chunks name both. "failure": how or when it goes wrong. "comparison":
how it differs from an alternative. "definition": what it is, and no more.

reason: "given" when the chunks say why it is so or how it works, "stated" when they only say
that it is so.

scope: "general" when the idea is known and used beyond this document -- a method in its own
right, a principle, a known problem -- and "document" when it belongs to this document alone:
one of its own components, settings or findings.

interview: how likely an interviewer is to ask about it: 3 likely, 2 possible, 1 unlikely.

evidence: one whole sentence of the first chunk that best shows the idea is explained there,
copied exactly as it stands. Choose a sentence of words over one built around a formula, and
never shorten it or join two sentences with "...".
"""

WINDOW = """\
Document: {title}

{passages}"""

PASSAGE = """\
[chunk {chunk_id}] {section}
<<<
{text}
>>>
"""

MERGE_INSTRUCTIONS = """\
You merge the ideas found in the parts of one document. The document was read a part at a
time, so an idea explained in more than one part was listed more than once. You are given
every idea found, numbered, with the part it came from.

Find the entries that are the same idea: one interview question would ask about all of them,
and the passages behind any of them would answer it. An idea restated, or explained again in
more detail, is the same idea. A part of an idea, a consequence of it, a step built on it or a
comparison it takes part in is a different idea.

For each set of entries that are one idea, give their numbers, a name, and a one-sentence
summary covering them. Leave out the entries that stand alone.
"""

MERGE = """\
Document: {title}

{entries}"""

SAME_INSTRUCTIONS = """\
You compare ideas found in different documents. For each numbered pair, say whether the two
are the same idea: one interview question would ask about both, and what either document says
would answer it. Related ideas are not the same, and neither are an idea and a part of it, or
an idea and an alternative to it.
"""

PAIR = """\
[{number}] {first}
    {second}"""


class ConceptReading(BaseModel):
    """One idea, as the model read it out of a window."""

    name: str = Field(description="What a person would call the idea, in a few words")
    summary: str = Field(description="One sentence on what the chunks say about it")
    chunk_ids: list[int] = Field(description="The chunks that explain it, the best one first")
    # Plain strings, not the six kinds as an enum: Groq checks the answer only once it is
    # written, and turned a whole window down for one "theorem" among them. A kind outside
    # the six is dropped in code instead (`concept_of`).
    kinds: list[str] = Field(
        description='What the chunks explain about it: any of "mechanism", "reason", '
        '"trade-off", "failure", "comparison", "definition"'
    )
    reason: Literal["given", "stated"] = Field(
        description="Whether the chunks say why or how, or only that it is so"
    )
    scope: Literal["general", "document"] = Field(
        description="Whether the idea is known beyond this document"
    )
    interview: Literal[1, 2, 3] = Field(
        description="How likely an interviewer is to ask about it: 3 likely, 1 unlikely"
    )
    evidence: str = Field(description="The sentence of the first chunk that explains it best")


class WindowReading(BaseModel):
    ideas: list[ConceptReading]


class MergeGroup(BaseModel):
    numbers: list[int] = Field(description="The numbers of the entries that are one idea")
    name: str = Field(description="A name for the idea")
    summary: str = Field(description="One sentence covering the entries")


class MergeReading(BaseModel):
    groups: list[MergeGroup]


class PairVerdict(BaseModel):
    pair: int = Field(description="The number of the pair")
    same: bool = Field(description="Whether the two are one idea")


class SameReading(BaseModel):
    verdicts: list[PairVerdict]


class Named(BaseModel):
    """An idea as a review names it: by its key, and by its name so that a decision cannot land
    on another idea once the documents are read again."""

    key: str
    name: str


class ScopeSet(Named):
    scope: Literal["general", "document"]


class Exclusion(Named):
    reason: str


class OpenCheck(BaseModel):
    """Ideas a later step has to look at again before it relies on them."""

    ideas: list[Named]
    note: str


class Review(BaseModel):
    """A person's decisions on one reading of an inventory."""

    # Ideas that are one, each group's lead first: the lead keeps its key, name and summary.
    merge: list[list[Named]] = []
    scope: list[ScopeSet] = []
    # Left out of planning, and still listed
    exclude: list[Exclusion] = []
    # Ideas that stay separate, though close
    keep_apart: list[list[Named]] = []
    check: list[OpenCheck] = []


@dataclass
class Call:
    """One request's answer: who gave it, to which version of the prompt, what it cost and
    when."""

    model: str
    version: str
    usage: dict[str, int]
    at: str


@dataclass
class Passage:
    chunk_id: int
    # The sections it covers, e.g. "4 Direct Preference Optimization"
    section: str | None
    text: str
    tokens: int


@dataclass
class Concept:
    """One idea of one document, after its readings are checked and merged."""

    document_id: int
    name: str
    summary: str
    chunk_ids: list[int]
    kinds: list[str]
    reason: str
    scope: str
    interview: int
    # The sentence each reading behind it copied, checked against the chunks of its window
    evidence: list[QuoteCheck]
    # The readings it stands for, as "window.idea", both counted from 1: ["1.3", "2.1"]
    readings: list[str]
    # Why a review leaves it out of planning, if it does
    excluded: str | None = None


@dataclass
class Window:
    passages: list[Passage]
    reading: WindowReading | None = None
    call: Call | None = None

    @property
    def chunk_ids(self) -> list[int]:
        return [passage.chunk_id for passage in self.passages]

    @property
    def tokens(self) -> int:
        return sum(passage.tokens for passage in self.passages)


@dataclass
class DocumentInventory:
    document_id: int
    title: str
    # Passages no idea can rest on, with the reason
    held_back: dict[int, str]
    windows: list[Window]
    merge: MergeReading | None = None
    merge_call: Call | None = None

    @property
    def read(self) -> bool:
        return all(window.reading is not None for window in self.windows)

    @property
    def needs_merge(self) -> bool:
        return len(self.windows) > 1 and self.merge is None

    @property
    def calls(self) -> list[Call]:
        return [window.call for window in self.windows if window.call] + (
            [self.merge_call] if self.merge_call else []
        )

    def readings(self) -> list[Concept]:
        """Every idea of every window read so far, in reading order."""
        return [
            concept_of(reading, self.document_id, window.passages, f"{number}.{index}")
            for number, window in enumerate(self.windows, start=1)
            if window.reading is not None
            for index, reading in enumerate(window.reading.ideas, start=1)
        ]

    def concepts(self) -> list[Concept]:
        readings = self.readings()
        return merged(readings, self.merge.groups) if self.merge is not None else readings


@dataclass
class Comparison:
    """Two ideas from different documents that came close, and whether they are one idea:
    None until the model has said."""

    # Each idea as "document.concept", the concept counted from 1 in its document's list
    first: str
    second: str
    # Their names when compared: a verdict holds only for the ideas it was given about
    names: tuple[str, str]
    similarity: float
    same: bool | None = None


def table_share(text: str) -> float:
    """The share of a passage's characters that sit in table lines."""
    lines = text.split("\n")
    total = sum(len(line) for line in lines)
    in_tables = sum(len(line) for line in lines if line.lstrip().startswith("|"))
    return in_tables / total if total else 0.0


def context_only(chunk: Chunk) -> str | None:
    """Why no idea can rest on a passage, or None when one can.

    Besides the tagger's rules: a passage that is mostly table, one whose mathematics a PDF
    parse damaged, and back matter -- a passage that starts in an appendix or the
    acknowledgements. A table is weighed by its characters, not its lines: counted by lines,
    two prose passages holding a short table looked mostly table and took two usable
    questions with them, though the table was 12% and 13% of their characters. Weighed by
    characters, the rules hold back 69 of the 322 passages of both libraries (tables 31,
    damaged mathematics 10, back matter 46), and those led 5 of the 133 questions written
    from them, none of them good.
    """
    if table_share(chunk.text) >= TABLE_SHARE:
        return "mostly table"
    if GARBLED.search(chunk.text) or len(BROKEN_MATHS.findall(chunk.text)) > BROKEN_MATHS_ALLOWED:
        return "damaged mathematics"
    first = (chunk.section or "").split(SECTION_LIST_SEPARATOR)[0]
    if BACK_MATTER.match(first):
        return "back matter"
    return tagging.context_only(chunk)


def passage_of(chunk: Chunk) -> Passage:
    return Passage(
        chunk_id=chunk.id, section=chunk.section, text=chunk.text, tokens=chunk.token_count
    )


def windows(passages: Sequence[Passage], budget: int = WINDOW_TOKENS) -> list[list[Passage]]:
    """Consecutive passages in as few windows as `budget` allows, as even as whole passages
    let them be: a document of 5,300 tokens is read as two windows of about 2,650, not as one
    of 5,000 and a last one holding a single passage out of its context."""
    if not passages:
        return []

    def pack(limit: int) -> list[list[Passage]]:
        packed: list[list[Passage]] = []
        size = 0
        for passage in passages:
            if packed and size + passage.tokens <= limit:
                packed[-1].append(passage)
                size += passage.tokens
            else:
                packed.append([passage])
                size = passage.tokens
        return packed

    largest = max(passage.tokens for passage in passages)
    count = len(pack(max(budget, largest)))
    # The smallest window that still needs no more windows than the budget does
    low, high = largest, max(budget, largest)
    while low < high:
        middle = (low + high) // 2
        if len(pack(middle)) <= count:
            high = middle
        else:
            low = middle + 1
    return pack(low)


def plan_document(document_id: int, title: str, chunks: Sequence[Chunk]) -> DocumentInventory:
    """A document's passages split into those an idea can rest on, to be read in windows, and
    those held back as context."""
    held_back: dict[int, str] = {}
    readable: list[Passage] = []
    for chunk in chunks:
        reason = context_only(chunk)
        if reason is None:
            readable.append(passage_of(chunk))
        else:
            held_back[chunk.id] = reason
    return DocumentInventory(
        document_id=document_id,
        title=title,
        held_back=held_back,
        windows=[Window(passages=group) for group in windows(readable)],
    )


async def load_document(session: AsyncSession, document_id: int) -> DocumentInventory | None:
    """A document's current passages, planned into windows; None when there is no such
    document."""
    title = await session.scalar(select(Document.title).where(Document.id == document_id))
    if title is None:
        return None
    chunks = await session.scalars(
        select(Chunk)
        .where(Chunk.document_id == document_id, Chunk.superseded_at.is_(None))
        .order_by(Chunk.position)
    )
    return plan_document(document_id, title, list(chunks))


def resume(planned: DocumentInventory, saved: DocumentInventory | None) -> bool:
    """Carry what a saved run read over to a freshly planned document, when the windows are
    still the same passages. Returns whether they were."""
    if saved is None or [w.chunk_ids for w in saved.windows] != [
        w.chunk_ids for w in planned.windows
    ]:
        return False
    for window, before in zip(planned.windows, saved.windows, strict=True):
        window.reading, window.call = before.reading, before.call
    planned.merge, planned.merge_call = saved.merge, saved.merge_call
    return True


def window_request(title: str, passages: Sequence[Passage]) -> str:
    shown = "\n".join(
        PASSAGE.format(chunk_id=passage.chunk_id, section=passage.section or "-", text=passage.text)
        for passage in passages
    )
    return WINDOW.format(title=title, passages=shown)


def merge_request(title: str, readings: Sequence[Concept]) -> str:
    entries = "\n".join(
        f"[{number}] (part {concept.readings[0].split('.')[0]}) {concept.name}: {concept.summary}"
        for number, concept in enumerate(readings, start=1)
    )
    return MERGE.format(title=title, entries=entries)


def pair_request(pairs: Sequence[tuple[Concept, Concept]], titles: dict[int, str]) -> str:
    def shown(concept: Concept) -> str:
        return f'{concept.name}: {concept.summary} (from "{titles[concept.document_id]}")'

    return "\n".join(
        PAIR.format(number=number, first=shown(first), second=shown(second))
        for number, (first, second) in enumerate(pairs, start=1)
    )


def place_evidence(reading: ConceptReading, passages: Sequence[Passage]) -> QuoteCheck:
    """The evidence sentence, checked against the chunk the idea is filed under first and then
    against the rest of its window: a sentence found in another chunk says where the idea is.
    When it is in none of them, the check against the first chunk cited says why."""
    texts = {passage.chunk_id: passage.text for passage in passages}
    cited = [chunk_id for chunk_id in reading.chunk_ids if chunk_id in texts]
    order = cited + [chunk_id for chunk_id in texts if chunk_id not in cited]
    checks = [check_evidence(reading.evidence, chunk_id, texts[chunk_id]) for chunk_id in order]
    found = next((check for check in checks if check.grounded), None)
    if found is not None:
        return found
    if cited:
        return checks[0]
    return check_evidence(reading.evidence, reading.chunk_ids[0] if reading.chunk_ids else 0, None)


def concept_of(
    reading: ConceptReading, document_id: int, passages: Sequence[Passage], number: str
) -> Concept:
    """A reading checked against its window: chunks outside the window are dropped, and the
    chunk its evidence turned up in is added when the model did not cite it."""
    in_window = {passage.chunk_id for passage in passages}
    chunk_ids = list(dict.fromkeys(chunk for chunk in reading.chunk_ids if chunk in in_window))
    evidence = place_evidence(reading, passages)
    if evidence.grounded and evidence.chunk_id not in chunk_ids:
        chunk_ids.append(evidence.chunk_id)
    return Concept(
        document_id=document_id,
        name=reading.name.strip(),
        summary=reading.summary.strip(),
        chunk_ids=chunk_ids,
        kinds=[kind for kind in KINDS if kind in reading.kinds],
        reason=reading.reason,
        scope=reading.scope,
        interview=reading.interview,
        evidence=[evidence],
        readings=[number],
    )


def combine(parts: Sequence[Concept], name: str, summary: str) -> Concept:
    """Several readings of one idea as one: every chunk and kind any of them found, the reason
    given when any part gives it, general when any part reads it so, and the likeliest of
    their interview odds."""
    return Concept(
        document_id=parts[0].document_id,
        name=name.strip(),
        summary=summary.strip(),
        chunk_ids=list(dict.fromkeys(chunk for part in parts for chunk in part.chunk_ids)),
        kinds=[kind for kind in KINDS if any(kind in part.kinds for part in parts)],
        reason="given" if any(part.reason == "given" for part in parts) else "stated",
        scope="general" if any(part.scope == "general" for part in parts) else "document",
        interview=max(part.interview for part in parts),
        evidence=[check for part in parts for check in part.evidence],
        readings=[number for part in parts for number in part.readings],
    )


def merged(readings: Sequence[Concept], groups: Sequence[MergeGroup]) -> list[Concept]:
    """The readings with each group the model found folded into one, in the place of its
    earliest member. Numbers count from 1; one out of range or already in a group is passed
    over, and so is a group left with fewer than two members."""
    taken: set[int] = set()
    folded: dict[int, Concept] = {}
    for group in groups:
        indexes = {number - 1 for number in group.numbers if 0 < number <= len(readings)}
        members = sorted(indexes - taken)
        if len(members) < 2:
            continue
        taken.update(members)
        parts = [readings[index] for index in members]
        folded[members[0]] = combine(parts, group.name, group.summary)
    return [
        folded.get(index, reading)
        for index, reading in enumerate(readings)
        if index in folded or index not in taken
    ]


def keyed(documents: Sequence[DocumentInventory]) -> dict[str, Concept]:
    """Every idea of the library under its key, "document.concept"."""
    return {
        f"{document.document_id}.{number}": concept
        for document in documents
        for number, concept in enumerate(document.concepts(), start=1)
    }


def apply_review(concepts: dict[str, Concept], review: Review) -> dict[str, Concept]:
    """The ideas as a review leaves them, in their order and under their keys.

    Each merge folds its ideas into the lead, which keeps its key, name and summary and takes
    the rest as the merge per document does (`combine`); the other keys go. Scopes are set
    after merging, and an excluded idea stays listed, marked with the reason. Raises ValueError
    when the review names an idea this reading does not have or calls something else, merges
    ideas of two documents or an idea twice, merges ideas it keeps apart, or sets anything on
    an idea merged into another.
    """
    named = [idea for group in review.merge for idea in group]
    named += [*review.scope, *review.exclude]
    named += [idea for group in review.keep_apart for idea in group]
    named += [idea for check in review.check for idea in check.ideas]
    for idea in named:
        concept = concepts.get(idea.key)
        if concept is None:
            raise ValueError(f"the review names idea {idea.key}, which this reading does not have")
        if concept.name != idea.name:
            raise ValueError(
                f'the review names idea {idea.key} "{idea.name}", '
                f'which this reading calls "{concept.name}"'
            )
    for group in review.keep_apart:
        if len({idea.key for idea in group}) < 2:
            raise ValueError("ideas kept apart have to be two different ideas")

    result = dict(concepts)
    merged_into: dict[str, str] = {}
    in_a_merge: set[str] = set()
    for group in review.merge:
        keys = [idea.key for idea in group]
        if len(set(keys)) < max(len(keys), 2):
            raise ValueError(f"a merge needs two different ideas: {', '.join(keys)}")
        if (twice := next((key for key in keys if key in in_a_merge), None)) is not None:
            raise ValueError(f"idea {twice} is in two merges")
        in_a_merge.update(keys)
        if len({concepts[key].document_id for key in keys}) > 1:
            raise ValueError(f"a merge joins ideas of one document, not {', '.join(keys)}")
        for apart in review.keep_apart:
            if len({idea.key for idea in apart} & set(keys)) > 1:
                kept = " and ".join(idea.key for idea in apart)
                raise ValueError(f"the review keeps {kept} apart but merges them")
        lead = concepts[keys[0]]
        result[keys[0]] = combine([concepts[key] for key in keys], lead.name, lead.summary)
        for key in keys[1:]:
            merged_into[key] = keys[0]
            del result[key]

    for setting in [*review.scope, *review.exclude]:
        if setting.key in merged_into:
            lead_key = merged_into[setting.key]
            raise ValueError(f"idea {setting.key} is merged into {lead_key}: name {lead_key}")
    for scope in review.scope:
        result[scope.key] = replace(result[scope.key], scope=scope.scope)
    for exclusion in review.exclude:
        result[exclusion.key] = replace(result[exclusion.key], excluded=exclusion.reason)
    return result


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norms if norms else 0.0


async def close_pairs(
    embedder: Embedder, concepts: dict[str, Concept], threshold: float = CLOSE
) -> list[Comparison]:
    """Each idea with its nearest idea in every other document, when their names and summaries
    come within `threshold` in cosine similarity; closest first.

    Only the nearest: two documents rarely explain one idea twice, and on one subject the
    embedding model keeps most short texts close, so every pair over the line would put
    dozens of related ideas in front of the model for each one that is really the same.
    """
    keys = list(concepts)
    if len({concept.document_id for concept in concepts.values()}) < 2:
        return []
    vectors = await embedder.embed_documents(
        [f"{concepts[key].name}: {concepts[key].summary}" for key in keys]
    )
    similarity: dict[tuple[int, int], float] = {}
    # (idea, other document) -> the idea of that document nearest to it
    nearest: dict[tuple[int, int], int] = {}
    for first, second in combinations(range(len(keys)), 2):
        one, other = concepts[keys[first]], concepts[keys[second]]
        if one.document_id == other.document_id:
            continue
        similarity[first, second] = cosine(vectors[first], vectors[second])
        for index, partner, document in (
            (first, second, other.document_id),
            (second, first, one.document_id),
        ):
            known = nearest.get((index, document))
            pair = (min(index, known), max(index, known)) if known is not None else None
            if pair is None or similarity[first, second] > similarity[pair]:
                nearest[index, document] = partner
    chosen = {(min(index, partner), max(index, partner)) for (index, _), partner in nearest.items()}
    close = [
        Comparison(
            first=keys[first],
            second=keys[second],
            names=(concepts[keys[first]].name, concepts[keys[second]].name),
            similarity=round(similarity[first, second], 4),
        )
        for first, second in sorted(chosen)
        if similarity[first, second] >= threshold
    ]
    return sorted(close, key=lambda pair: -pair.similarity)


def carry_verdicts(pairs: Sequence[Comparison], before: Sequence[Comparison]) -> None:
    """Give each pair the verdict an earlier run reached on the same two ideas."""
    known = {(pair.first, pair.second, pair.names): pair.same for pair in before}
    for pair in pairs:
        pair.same = known.get((pair.first, pair.second, pair.names))


def groups_of(count: int, joined: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Indexes 0..count-1 gathered into the groups that `joined` links up, directly or through
    each other: each group in index order, and the groups by their first index."""
    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in joined:
        low, high = sorted((root(first), root(second)))
        parent[high] = low
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def library_groups(keys: Sequence[str], pairs: Sequence[Comparison]) -> list[list[str]]:
    """The ideas that the model found to be one across documents, in groups of two or more."""
    index = {key: number for number, key in enumerate(keys)}
    joined = [
        (index[pair.first], index[pair.second])
        for pair in pairs
        if pair.same and pair.first in index and pair.second in index
    ]
    return [[keys[i] for i in group] for group in groups_of(len(keys), joined) if len(group) > 1]


def inventory_settings(max_tokens: int) -> ModelSettings:
    return ModelSettings(thinking="low", max_tokens=max_tokens)


def call_of(result: AgentRunResult[Any], model: Model) -> Call:
    """What a run cost, and the model that answered: under a fallback chain, not always the
    first one."""
    answered = getattr(result.all_messages()[-1], "model_name", None)
    return Call(
        model=answered or model.model_name,
        version=PROMPT_VERSION,
        usage=add_usage({}, result),
        at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def reader(model: Model) -> Agent[None, WindowReading]:
    # No second try at an answer that fails validation: it would send the whole window again.
    # The window is left unread instead, for the next run to pick up.
    return Agent(
        model,
        name="inventory reader",
        output_type=NativeOutput(WindowReading, strict=True),
        instructions=INSTRUCTIONS,
        retries=0,
    )


def merger(model: Model) -> Agent[None, MergeReading]:
    return Agent(
        model,
        name="inventory merger",
        output_type=NativeOutput(MergeReading, strict=True),
        instructions=MERGE_INSTRUCTIONS,
        retries=0,
    )


def comparer(model: Model) -> Agent[None, SameReading]:
    return Agent(
        model,
        name="inventory comparer",
        output_type=NativeOutput(SameReading, strict=True),
        instructions=SAME_INSTRUCTIONS,
        retries=0,
    )


async def read_window(model: Model, document: DocumentInventory, number: int) -> Window:
    """Read one window of a document, counted from 1, and keep the answer on it."""
    window = document.windows[number - 1]
    with traced("inventory", version=PROMPT_VERSION, document=document.document_id, window=number):
        result = await reader(model).run(
            window_request(document.title, window.passages),
            model_settings=inventory_settings(WINDOW_MAX_TOKENS),
            metadata={"prompt_version": PROMPT_VERSION},
        )
    window.reading, window.call = result.output, call_of(result, model)
    return window


async def merge_document(model: Model, document: DocumentInventory) -> MergeReading:
    """Ask which of a document's readings are one idea, and keep the answer on it."""
    with traced("inventory merge", version=PROMPT_VERSION, document=document.document_id):
        result = await merger(model).run(
            merge_request(document.title, document.readings()),
            model_settings=inventory_settings(MERGE_MAX_TOKENS),
            metadata={"prompt_version": PROMPT_VERSION},
        )
    document.merge, document.merge_call = result.output, call_of(result, model)
    return result.output


async def compare(
    model: Model, pairs: Sequence[Comparison], concepts: dict[str, Concept], titles: dict[int, str]
) -> list[Call]:
    """Ask whether each pair the model has not judged yet is one idea, PAIRS_PER_REQUEST to a
    request. A pair the model leaves out stays unjudged."""
    open_pairs = [pair for pair in pairs if pair.same is None]
    calls: list[Call] = []
    for start in range(0, len(open_pairs), PAIRS_PER_REQUEST):
        batch = open_pairs[start : start + PAIRS_PER_REQUEST]
        shown = [(concepts[pair.first], concepts[pair.second]) for pair in batch]
        with traced("inventory compare", version=PROMPT_VERSION, pairs=len(batch)):
            result = await comparer(model).run(
                pair_request(shown, titles),
                model_settings=inventory_settings(MERGE_MAX_TOKENS),
                metadata={"prompt_version": PROMPT_VERSION},
            )
        said = {verdict.pair: verdict.same for verdict in result.output.verdicts}
        for number, pair in enumerate(batch, start=1):
            pair.same = said.get(number)
        calls.append(call_of(result, model))
    return calls


def total_usage(calls: Sequence[Call]) -> dict[str, int]:
    total: dict[str, int] = {}
    for call in calls:
        for name, value in call.usage.items():
            total[name] = total.get(name, 0) + value
    return total


def _call(data: dict[str, Any] | None) -> Call | None:
    return Call(**data) if data else None


def concept_json(key: str, concept: Concept) -> dict[str, Any]:
    return {
        "key": key,
        "name": concept.name,
        "summary": concept.summary,
        "chunk_ids": concept.chunk_ids,
        "kinds": concept.kinds,
        "reason": concept.reason,
        "scope": concept.scope,
        "interview": concept.interview,
        "evidence": [
            {"quote": check.quote, "chunk_id": check.chunk_id, "problem": check.problem}
            for check in concept.evidence
        ],
        "readings": concept.readings,
        "excluded": concept.excluded,
    }


def to_json(
    documents: Sequence[DocumentInventory],
    pairs: Sequence[Comparison] = (),
    compare_calls: Sequence[Call] = (),
    review: Review | None = None,
) -> dict[str, Any]:
    """The inventory as saved: every answer the model gave, with the ideas worked out from
    them, and from the review when there is one, alongside for reading. Only the answers are
    read back."""
    concepts = keyed(documents)
    if review is not None:
        concepts = apply_review(concepts, review)
    calls = [call for document in documents for call in document.calls] + list(compare_calls)
    return {
        "prompt_version": PROMPT_VERSION,
        "documents": [
            {
                "document_id": document.document_id,
                "title": document.title,
                "held_back": {str(chunk): reason for chunk, reason in document.held_back.items()},
                "windows": [
                    {
                        "chunk_ids": window.chunk_ids,
                        "tokens": window.tokens,
                        "reading": window.reading.model_dump() if window.reading else None,
                        "call": vars(window.call) if window.call else None,
                    }
                    for window in document.windows
                ],
                "merge": document.merge.model_dump() if document.merge else None,
                "merge_call": vars(document.merge_call) if document.merge_call else None,
                "concepts": [
                    concept_json(key, concept)
                    for key, concept in concepts.items()
                    if concept.document_id == document.document_id
                ],
            }
            for document in documents
        ],
        "comparisons": [
            {
                "first": pair.first,
                "second": pair.second,
                "names": list(pair.names),
                "similarity": pair.similarity,
                "same": pair.same,
            }
            for pair in pairs
        ],
        "compare_calls": [vars(call) for call in compare_calls],
        "review": review.model_dump() if review is not None else None,
        "library": library_groups(list(concepts), pairs),
        "usage": total_usage(calls),
    }


def from_json(
    data: dict[str, Any],
) -> tuple[list[DocumentInventory], list[Comparison], list[Call]]:
    """The documents, comparisons and comparison calls of a saved inventory. A window's
    passages are not saved, only their chunk ids: a document read back is good for `resume`,
    which takes the passages from a fresh plan."""
    documents = [
        DocumentInventory(
            document_id=saved["document_id"],
            title=saved["title"],
            held_back={int(chunk): reason for chunk, reason in saved["held_back"].items()},
            windows=[
                Window(
                    passages=[
                        Passage(chunk_id=chunk, section=None, text="", tokens=0)
                        for chunk in window["chunk_ids"]
                    ],
                    reading=WindowReading.model_validate(window["reading"])
                    if window["reading"]
                    else None,
                    call=_call(window["call"]),
                )
                for window in saved["windows"]
            ],
            merge=MergeReading.model_validate(saved["merge"]) if saved["merge"] else None,
            merge_call=_call(saved["merge_call"]),
        )
        for saved in data.get("documents", [])
    ]
    pairs = [
        Comparison(
            first=pair["first"],
            second=pair["second"],
            names=(pair["names"][0], pair["names"][1]),
            similarity=pair["similarity"],
            same=pair["same"],
        )
        for pair in data.get("comparisons", [])
    ]
    calls = [Call(**call) for call in data.get("compare_calls", [])]
    return documents, pairs, calls
