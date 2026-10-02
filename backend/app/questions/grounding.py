"""Checking that an evidence quote really is in the chunk it claims to come from.

A question is only worth keeping when every key point can be traced to words that are in the
sources. The comparison is deliberately lenient about how a passage is written down: models
reproduce it faithfully but re-wrap the lines, drop the dollar signs around inline maths or
turn a non-breaking hyphen into a plain one. It is strict about what hides a paraphrase --
an ellipsis inside a quote can stand for anything, and six words can be found in any text.
The punctuation a quote ends on hides nothing, so a quote that stops at the colon opening a
list, or trails off into an ellipsis, is judged on its words alone.

An evidence sentence (`check_evidence`) may also be shortened, when what is left out cannot
change what it says, and is held to more than prose wherever it touches mathematics.
"""

import re
import string
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

# partial_ratio scores the best-matching window of the chunk against the quote. In the trial
# a quote that only lost its dollar signs still scored 98-99, while an ellipsis quote reached
# 90.5 and a bullet list rejoined with commas 91.1, so the bar sits between them.
THRESHOLD = 95.0
MIN_WORDS = 6

HYPHENS = dict.fromkeys([0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212], "-")
MARKDOWN = re.compile(r"[$`*_#]+")
ELLIPSIS = re.compile(r"\.\.\.|…")
# What a normalized quote can end on without it being part of what it says. Refusing a colon
# on sight turned away quotes copied character for character up to the list they introduce.
TRAILING = " .,:;!?-"


@dataclass
class QuoteCheck:
    quote: str
    chunk_id: int
    score: float
    # Why the quote was not accepted, in words the model can act on; None when it was
    problem: str | None

    @property
    def grounded(self) -> bool:
        return self.problem is None


def normalize(text: str) -> str:
    """Strip away everything that is spelling rather than wording."""
    folded = unicodedata.normalize("NFKC", text).translate(HYPHENS)
    return re.sub(r"\s+", " ", MARKDOWN.sub(" ", folded)).strip().casefold()


def check_quote(quote: str, chunk_id: int, chunk_text: str | None) -> QuoteCheck:
    def failed(problem: str, score: float = 0.0) -> QuoteCheck:
        return QuoteCheck(quote=quote, chunk_id=chunk_id, score=score, problem=problem)

    if chunk_text is None:
        return failed(f"chunk {chunk_id} is not one of the sources")
    # Trimmed before anything is judged: a detached mark at the end would count as a word, and
    # one the source does not have would cost a faithful copy its match.
    words = normalize(quote).rstrip(TRAILING)
    if len(words.split()) < MIN_WORDS:
        return failed(f"the quote is shorter than {MIN_WORDS} words")
    # The score decides whether an ellipsis is the model eliding a span or the source's own
    # notation: a paper that writes a sequence as (x_1, ..., x_n) is quoted faithfully with
    # the dots in it, while a quote that really leaves words out stops matching the chunk.
    score = float(fuzz.partial_ratio(words, normalize(chunk_text)))
    if score < THRESHOLD:
        if ELLIPSIS.search(words):
            return failed("the quote leaves words out instead of running on", score)
        return failed(f"the quote is not in chunk {chunk_id} (closest match {score:.0f}%)", score)
    return QuoteCheck(quote=quote, chunk_id=chunk_id, score=score, problem=None)


# What a shortened sentence may not leave out: losing one of these can turn its meaning round.
NEGATION = re.compile(
    r"\b(?:not|no|never|none|nor|neither|cannot|without|unless|except)\b|n['’]t\b"
)
# Abbreviations whose full stop does not end a sentence
ABBREVIATIONS = frozenset(
    {"e.g", "i.e", "al", "vs", "cf", "etc", "fig", "figs", "eq", "eqs", "sec", "tab", "viz"}
)
# A full stop, question or exclamation mark, with any closing quote or bracket, before a space
SENTENCE_END = re.compile(r"(\S*?)[.!?]+[\"')\]]*(?=\s|$)")
# A word is mathematics when it holds LaTeX, an operator, a subscript, a dollar sign or a symbol
# from the Greek, arrow or mathematical-operator blocks; so is everything between dollar signs.
MATHS = re.compile(r"[\\^_{}=<>|$]|[\u0370-\u03ff\u2190-\u21ff\u2200-\u22ff]")
FORMULA = re.compile(r"\$\$[^$]+?\$\$|\$[^$\n]+?\$")
SPELLING = (
    HYPHENS
    | dict.fromkeys([0x2018, 0x2019, 0x201A, 0x201B], "'")
    | dict.fromkeys([0x201C, 0x201D, 0x201E, 0x201F], '"')
)
# What a copy may drop without dropping a word: spaces, and Markdown's emphasis, code and
# headings. Underscores stay, since in mathematics they are subscripts.
LAYOUT = frozenset("$`*#")
# What a copy may start or end on without it being part of what it copies
ENDS = ".,:;!?\"'"
# What stands around a formula in a sentence without being part of it
AROUND = ",.;:!?\"'$"


def folded(text: str) -> str:
    """Text in its compatibility forms and with one kind of hyphen and quote mark, its case
    kept."""
    return unicodedata.normalize("NFKC", text).translate(SPELLING)


def squashed(text: str) -> str:
    """Text with every space and dollar sign taken out, for comparing mathematics as written."""
    return re.sub(r"\s+", "", folded(text)).replace("$", "")


@dataclass
class Copy:
    """Text with its spelling taken out -- spaces, Markdown, case, compatibility forms, kinds of
    hyphen and quote mark -- and, for each character left, where it stood in `raw`."""

    raw: str
    text: str
    origins: list[int]


def copy_of(raw: str, trim: bool = False) -> Copy:
    """`raw` as a copy is compared; with `trim`, without the punctuation it starts or ends on."""
    kept: list[str] = []
    origins: list[int] = []
    for index, char in enumerate(raw):
        for part in folded(char):
            if part.isspace() or part in LAYOUT or unicodedata.category(part) == "Cf":
                continue
            for lower in part.casefold():
                kept.append(lower)
                origins.append(index)
    start, end = 0, len(kept)
    while trim and start < end and kept[start] in ENDS:
        start += 1
    while trim and end > start and kept[end - 1] in ENDS:
        end -= 1
    return Copy(raw=raw, text="".join(kept[start:end]), origins=origins[start:end])


def maths_marks(text: str) -> list[bool]:
    """Which characters of `text` are mathematics: those between dollar signs and those of a
    word that holds LaTeX or a mathematical symbol, but never a space, a dollar sign, or the
    punctuation a formula or a word ends on -- the full stop that closes a displayed equation
    belongs to the sentence."""
    marked = [False] * len(text)
    stretches = list(FORMULA.finditer(text)) + [
        word for word in re.finditer(r"\S+", text) if MATHS.search(word.group())
    ]
    for stretch in stretches:
        written = stretch.group()
        start = stretch.start() + len(written) - len(written.lstrip(AROUND + string.whitespace))
        end = stretch.end() - len(written) + len(written.rstrip(AROUND + string.whitespace))
        for index in range(start, end):
            marked[index] = marked[index] or not (text[index].isspace() or text[index] == "$")
    return marked


def sentence_ends(text: str) -> list[int]:
    """Where the sentences of `text` end: the index of each full stop, question or exclamation
    mark that ends one, leaving out those that close an abbreviation."""
    return [
        found.end(1)
        for found in SENTENCE_END.finditer(text)
        if found.group(1).lstrip("([\"'").casefold() not in ABBREVIATIONS
    ]


def retyped_maths(quote: str, chunk_text: str) -> str | None:
    """The first formula in a quote that the chunk does not write, if any: compared as written,
    case and all, with only spaces and dollar signs taken out."""
    source = squashed(chunk_text)
    for piece in ELLIPSIS.split(quote):
        for word in piece.split():
            if MATHS.search(word) and (bare := word.strip(AROUND)) and squashed(bare) not in source:
                return bare
    return None


def placements(pieces: Sequence[Copy], text: str) -> Iterator[list[tuple[int, int]]]:
    """Every way the pieces stand in `text` in order: the first piece wherever it is, each of
    the others at its nearest place after the one before."""
    start = text.find(pieces[0].text)
    while start >= 0:
        spans = [(start, start + len(pieces[0].text))]
        for piece in pieces[1:]:
            at = text.find(piece.text, spans[-1][1])
            if at < 0:
                return
            spans.append((at, at + len(piece.text)))
        yield spans
        start = text.find(pieces[0].text, start + 1)


def beside_maths(text: str, maths: Sequence[bool], index: int, step: int) -> bool:
    """Whether the nearest character beside `index`, going by `step` past spaces and dollar
    signs, is mathematics."""
    index += step
    while 0 <= index < len(text) and (text[index].isspace() or text[index] == "$"):
        index += step
    return 0 <= index < len(text) and maths[index]


def word_at(text: str, index: int) -> str:
    """The word of `text` that holds `index`, without the punctuation around it."""
    for found in re.finditer(r"\S+", text):
        if found.start() <= index < found.end():
            return found.group().strip(AROUND)
    return ""


def unfaithful(
    pieces: Sequence[Copy], spans: Sequence[tuple[int, int]], chunk: Copy, chunk_id: int
) -> str | None:
    """What keeps the pieces of a quote, standing at `spans` of the chunk, from being a faithful
    copy of it, or None when nothing does."""
    maths = maths_marks(chunk.raw)
    ends = sentence_ends(chunk.raw)
    raw = [(chunk.origins[start], chunk.origins[end - 1] + 1) for start, end in spans]
    for (_, left), (right, _) in zip(raw, raw[1:], strict=False):
        left_out = chunk.raw[left:right]
        if "\n" in left_out or any(left <= end < right for end in ends):
            return "the quote joins words from different sentences"
        if (negation := NEGATION.search(normalize(left_out))) is not None:
            return f'the words left out include "{negation.group(0)}"'
        if any(maths[left:right]):
            return "the words left out include mathematics"
    for piece, (start, end) in zip(pieces, spans, strict=True):
        first, last = chunk.origins[start], chunk.origins[end - 1]
        if (maths[first] and beside_maths(chunk.raw, maths, first, -1)) or (
            maths[last] and beside_maths(chunk.raw, maths, last, 1)
        ):
            return "the quote starts or ends inside a formula"
        for at, origin in zip(piece.origins, chunk.origins[start:end], strict=True):
            if maths[origin] and folded(chunk.raw[origin]) != folded(piece.raw[at]):
                formula = word_at(piece.raw, at)
                return f"the mathematics is not as chunk {chunk_id} writes it: {formula}"
    return None


def check_evidence(quote: str, chunk_id: int, chunk_text: str | None) -> QuoteCheck:
    """An evidence sentence, copied whole or shortened, held to what a copy has to keep.

    Copied whole, its prose is checked like any quote. Shortened with "...", it stands only
    while it stays faithful and traceable: every piece is in the chunk as written, spelling
    aside, in order, and the words left out between two pieces lie within one sentence and hold
    no negation and no mathematics. Asked never to shorten, gpt-oss still shortened 16 of the
    74 sentences it copied in reading three papers.

    Mathematics is held to more than prose. A sentence that holds or touches a formula has to
    be an exact copy, the formula's case and all, and may not start or end inside one: a
    formula retyped -- `F=|V|/|S|` for `F=\\frac{|V|}{|S|}`, `y_w` for `y_{w}` -- says
    something the source does not write, however close it scores.
    """
    if chunk_text is None:
        return check_quote(quote, chunk_id, None)

    def failed(problem: str, score: float = 0.0) -> QuoteCheck:
        return QuoteCheck(quote=quote, chunk_id=chunk_id, score=score, problem=problem)

    if (formula := retyped_maths(quote, chunk_text)) is not None:
        return failed(f"the mathematics is not as chunk {chunk_id} writes it: {formula}")
    chunk = copy_of(chunk_text)
    whole = copy_of(quote, trim=True)
    # An ellipsis the chunk writes itself, as in (x_1, ..., x_n), shortens nothing.
    if not ELLIPSIS.search(quote) or whole.text in chunk.text:
        checked = check_quote(quote, chunk_id, chunk_text)
        if not checked.grounded:
            return checked
        if whole.text not in chunk.text:
            # A near copy: the quote check allows one for prose, never for mathematics.
            found = fuzz.partial_ratio_alignment(whole.text, chunk.text)
            near = (
                chunk.origins[found.dest_start : found.dest_end]
                if found is not None and len(whole.text) <= len(chunk.text)
                else chunk.origins
            )
            maths = maths_marks(chunk_text)
            if MATHS.search(quote) or any(maths[origin] for origin in near):
                return failed(
                    f"the quote holds mathematics but is not an exact copy of chunk {chunk_id}",
                    checked.score,
                )
            return checked
        pieces = [whole]
    else:
        parts = [part for part in ELLIPSIS.split(quote) if normalize(part).strip(TRAILING)]
        if sum(len(normalize(part).strip(TRAILING).split()) for part in parts) < MIN_WORDS:
            return failed(f"the quote is shorter than {MIN_WORDS} words")
        pieces = [copy_of(part, trim=True) for part in parts]

    problems = [
        unfaithful(pieces, spans, chunk, chunk_id) for spans in placements(pieces, chunk.text)
    ]
    if None in problems:
        return QuoteCheck(quote=quote, chunk_id=chunk_id, score=100.0, problem=None)
    if problems:
        return failed(problems[0])
    missing = next((piece for piece in pieces if piece.text not in chunk.text), None)
    if missing is not None:
        return failed(f'a piece is not in chunk {chunk_id} as written: "{missing.raw.strip()}"')
    return failed(f"the pieces are not in chunk {chunk_id} in this order")
