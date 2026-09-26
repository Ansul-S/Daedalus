"""Stand-ins for every model, for the end-to-end test (`FAKE_MODELS=true`).

Each one answers in the schema the real model answers in, from nothing but the prompt it is
sent, so the pipeline around it runs as it does for real -- quote grounding, the checks, the
duplicate test, scoring -- without a request leaving the machine or a token of anyone's quota:

* the writer asks why the first sentence of a passage holds, and quotes whole sentences of it
  as the key points' evidence, so every quote is found;
* the helper tags a passage with its section, so that each section becomes a topic, finds a
  passage worth asking about when two of its sentences are long enough to quote, and passes a
  question whose words the passages hold;
* the grader counts a key point covered by how many of its words the answer uses, and supports
  a claim with the question's passage that holds most of its words;
* the embedder counts a text's words, each in a dimension picked by hashing it, the same in
  every process, so the passages the worker stores and the queries the API sends agree;
* passages are measured in words, so no tokenizer is downloaded.

What they write is made up, so they belong on a throwaway database; the settings refuse them
in production.
"""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from app.db.models import EMBEDDING_DIMENSIONS

WRITER = "fake-writer"
HELPER = "fake-helper"
GRADER = "fake-grader"

# A key point's quote has to be six words or more; a sentence this long leaves room
QUOTABLE_WORDS = 8
# Shares of a key point's words an answer has to use for it to count as covered, or partly
COVERED = 0.7
PARTIAL = 0.35
# Share of a claim's words a passage has to hold for it to support the claim
SUPPORTED = 0.6

# Words that say nothing about what a text is about
COMMON_WORDS = """
a about after all also an and any are as at be because been before being both but by can could
did do does each even for from had has have how if in into is it its itself just may more most
much must no not of on once one only or other our out over same should so some such than that
the their them then there these they this those through to too under until up very was we were
what when where whether which while who why will with would you your
"""
STOPWORDS = frozenset(COMMON_WORDS.split())

WORD = re.compile(r"[a-z0-9]+")
TOKEN = re.compile(r"\w+|[^\w\s]")
FENCE = re.compile(r"^(`{3,}|~{3,}).*?^\1", re.MULTILINE | re.DOTALL)
HEADING = re.compile(r"^#{1,6} .*$", re.MULTILINE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# A passage as the writer, the checker and the grader are shown it
SOURCE = re.compile(r"^\[chunk (\d+)\] ([^\n]*)\n<<<\n(.*?)\n>>>$", re.MULTILINE | re.DOTALL)
EXPLAIN = re.compile(r"\s*(why|how|explain|compare|when|what happens|what goes wrong)\b", re.I)


# ---------- reading text ----------


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def content_words(text: str) -> set[str]:
    return {word for word in words(text) if word not in STOPWORDS and len(word) > 2}


def sentences(text: str) -> list[str]:
    """The prose sentences of a passage or an answer: code and headings left out, lines
    joined up, and a list item or a line of its own taken as a sentence."""
    prose = HEADING.sub("", FENCE.sub("", text))
    found: list[str] = []
    for paragraph in re.split(r"\n\s*\n", prose):
        flat = " ".join(paragraph.split())
        found += [sentence for sentence in SENTENCE_END.split(flat) if sentence]
    return found


def quotable(text: str) -> list[str]:
    """The sentences long enough to be a key point's evidence."""
    return [sentence for sentence in sentences(text) if len(sentence.split()) >= QUOTABLE_WORDS]


def share(wanted: set[str], found: set[str]) -> float:
    return len(wanted & found) / len(wanted) if wanted else 0.0


def inside(sentence: str) -> str:
    """A sentence as it reads inside another: its full stop dropped, and its first letter small
    unless the first word is an acronym."""
    body = sentence.strip().rstrip(".!?")
    first = body.split(maxsplit=1)[0] if body else ""
    if first == "A" or first[1:2].islower():
        body = body[0].lower() + body[1:]
    return body


def shorten(text: str, count: int = 8) -> str:
    kept = text.split()
    return " ".join(kept[:count]) + ("…" if len(kept) > count else "")


def count_tokens(text: str) -> int:
    """Words and punctuation marks, near enough to what a tokenizer counts."""
    return len(TOKEN.findall(text))


# ---------- reading the prompt ----------


def prompt(messages: list[ModelMessage]) -> str:
    """The first thing the model was asked: the request that carries the passages. A repair
    round is answered the same way, since the answer to the first request was already right."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    raise ValueError("there is no request to answer")


def passages(text: str) -> list[tuple[int, str]]:
    """(chunk id, text) of every passage in a prompt, in order."""
    return [(int(chunk_id), body) for chunk_id, _, body in SOURCE.findall(text)]


def between(text: str, start: str, end: str) -> str:
    """What lies between the first `start` and the last `end`."""
    return text.split(start, 1)[1].rsplit(end, 1)[0]


# ---------- the writer ----------


def write_question(request: str) -> dict[str, Any]:
    """A GeneratedQuestion: why the first sentence of the first passage holds, with up to four
    of the passages' sentences as key points, each quoting itself."""
    sources = passages(request)
    if not sources:
        raise ValueError("no passages to write a question from")
    per_source = 3 if len(sources) == 1 else 2
    points = [
        (chunk_id, sentence)
        for chunk_id, text in sources
        for sentence in quotable(text)[:per_source]
    ][:4]
    if len(points) < 2:
        raise ValueError(f"chunk {sources[0][0]} has fewer than two sentences to quote")
    lead = quotable(sources[0][1])
    return {
        "question": f"Why is it that {inside(lead[0])}?",
        "difficulty": 1 if len(sources) == 1 else 5,
        "reference_answer": " ".join(lead[:2]),
        "key_points": [
            {
                "text": sentence,
                "weight": 2 if number == 0 else 1,
                "evidence_quote": sentence,
                "chunk_id": chunk_id,
            }
            for number, (chunk_id, sentence) in enumerate(points)
        ],
        "misconceptions": [],
        "source_chunk_ids": [chunk_id for chunk_id, _ in sources],
    }


# ---------- the helper: tags and checks ----------


def read_passage(request: str) -> dict[str, Any]:
    """A TagReading: the sections the passage covers as its tags, or its most frequent words
    when it has none. A concept name of each section's own keeps the topics apart: tags
    shared between passages would merge them, and a question joins the topic its passage
    shares most."""
    section = re.search(r"^Section: (.*)$", request, re.MULTILINE)
    text = between(request, "Passage:\n<<<\n", "\n>>>")
    # "3 Model > 3.1 Encoder · 3.2 Decoder" covers the encoder and the decoder
    covered = section.group(1).split(" > ")[-1] if section else "-"
    tags = [] if covered == "-" else [heading.lower() for heading in covered.split(" · ")]
    if not tags:
        frequent = Counter(word for word in words(text) if word not in STOPWORDS and len(word) > 3)
        tags = [word for word, _ in frequent.most_common(3)] or ["passage"]
    worth_asking = len(quotable(text)) >= 2
    return {
        "explains": tags[0] if worth_asking else "nothing",
        "tags": tags,
        "worth_asking": worth_asking,
    }


def check_question(request: str) -> dict[str, Any]:
    """An AnswerCheck: answerable when the passages hold nearly every word the question asks
    about, and an explanation rather than a fact when it asks why or how."""
    question = between(request, "Question: ", "\n\nPassages:")
    sources = passages(request)
    held = content_words(" ".join(text for _, text in sources))
    asked = content_words(question)
    missing = sorted(asked - held)
    answerable = bool(sources) and len(missing) <= len(asked) // 4
    lead = quotable(sources[0][1]) if sources else []
    return {
        "kind": "explain" if EXPLAIN.match(question) else "recall",
        "answer": " ".join(lead[:2]) or "The passages do not say.",
        "answerable": answerable,
        "missing": "nothing" if answerable else ", ".join(missing),
    }


# ---------- the grader ----------


def grade(request: str) -> dict[str, Any]:
    """A grade: key points labelled by the share of their words the answer uses, and each
    sentence of the answer supported by the passage that holds most of its words."""
    shown, answer = request.split("<<<answer\n", 1)
    answer = answer.rsplit("\nanswer>>>", 1)[0]
    points = re.findall(r"^- (k\d+): (.*)$", between(shown, "Key points:\n", "\n\nPassages:"), re.M)
    sources = [(chunk_id, content_words(text)) for chunk_id, text in passages(shown)]
    said = [sentence for sentence in sentences(answer) if len(sentence.split()) >= 3]
    used = content_words(answer)

    labels = []
    for point_id, text in points:
        wanted = content_words(text)
        covered = share(wanted, used)
        status = "covered" if covered >= COVERED else "partial" if covered >= PARTIAL else "missing"
        closest = max(said, key=lambda sentence: len(wanted & content_words(sentence)), default="")
        labels.append(
            {
                "id": point_id,
                "status": status,
                "answer_quote": "" if status == "missing" else closest,
            }
        )

    claims = []
    for sentence in said:
        claimed = content_words(sentence)
        best = max(sources, key=lambda source: share(claimed, source[1]), default=None)
        if best is not None and share(claimed, best[1]) >= SUPPORTED:
            claims.append(
                {
                    "claim": sentence,
                    "verdict": "supported",
                    "chunk_id": best[0],
                    "why": f"Chunk {best[0]} says as much.",
                }
            )
        else:
            claims.append(
                {
                    "claim": sentence,
                    "verdict": "unverified",
                    "chunk_id": 0,
                    "why": "The passages do not address it.",
                }
            )

    texts = dict(points)
    gaps = [texts[label["id"]] for label in labels if label["status"] != "covered"]
    return {
        "key_points": labels,
        "claims": claims,
        "clarity": 4 if len(said) >= 2 else 2,
        "strengths": [
            shorten(texts[label["id"]]) for label in labels if label["status"] == "covered"
        ],
        "gaps": [shorten(gap) for gap in gaps],
        "errors": [],
        "improved_answer": " ".join(texts.values()),
        "follow_up": f"The passage also says: “{gaps[0]}” Why does that hold?"
        if gaps
        else "Which of these points would you lead with in an interview, and why?",
    }


# ---------- as models ----------


def _model(
    name: str, answer: Callable[[list[ModelMessage], AgentInfo], dict[str, Any]]
) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(json.dumps(answer(messages, info)))])

    return FunctionModel(
        respond, model_name=name, profile=ModelProfile(supports_json_schema_output=True)
    )


def writer() -> FunctionModel:
    """Writes questions, in place of Groq, Gemini and the local grader model."""
    return _model(WRITER, lambda messages, info: write_question(prompt(messages)))


def helper() -> FunctionModel:
    """Tags passages and checks questions, in place of the small local model."""

    def answer(messages: list[ModelMessage], info: AgentInfo) -> dict[str, Any]:
        asked = info.model_request_parameters.output_object
        fields = set(asked.json_schema.get("properties", {})) if asked else set()
        if "tags" in fields:
            return read_passage(prompt(messages))
        if "answerable" in fields:
            return check_question(prompt(messages))
        raise ValueError(f"the helper has no answer for {asked.name if asked else 'plain text'}")

    return _model(HELPER, answer)


def grader() -> FunctionModel:
    """Grades answers, in place of Qwen on Groq and its fallbacks."""
    return _model(GRADER, lambda messages, info: grade(prompt(messages)))


# ---------- the embedder ----------


def dimension(word: str) -> int:
    """The same dimension for the same word in every process: Python's own hash of a string
    changes from one process to the next."""
    digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
    return int.from_bytes(digest) % EMBEDDING_DIMENSIONS


def vector(text: str) -> list[float]:
    """A unit vector counting the text's words. A text with no words counts as one empty word,
    since a vector of zeros has no direction to compare."""
    counts = [0.0] * EMBEDDING_DIMENSIONS
    for word in words(text) or [""]:
        counts[dimension(word)] += 1.0
    norm = math.sqrt(sum(count * count for count in counts))
    return [count / norm for count in counts]


def ollama_embeddings() -> httpx.AsyncClient:
    """A client that answers Ollama's embedding endpoint itself, so the real `Embedder` runs
    unchanged on top of it."""

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/api/embed":
            return httpx.Response(404, json={"error": f"no fake for {request.url.path}"})
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"embeddings": [vector(text) for text in inputs]})

    return httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="http://fake-ollama")
