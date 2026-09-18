"""Checking that an evidence quote really is in the chunk it claims to come from.

A question is only worth keeping when every key point can be traced to words that are in the
sources. The comparison is deliberately lenient about how a passage is written down: models
reproduce it faithfully but re-wrap the lines, drop the dollar signs around inline maths or
turn a non-breaking hyphen into a plain one. It is strict about what hides a paraphrase --
an ellipsis can stand for anything, six words can be found in any text, and a quote that
stops at a colon states nothing.
"""

import re
import unicodedata
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
    if len(normalize(quote).split()) < MIN_WORDS:
        return failed(f"the quote is shorter than {MIN_WORDS} words")
    if quote.rstrip().endswith(":"):
        return failed("the quote stops at a colon, so it states nothing on its own")
    # The score decides whether an ellipsis is the model eliding a span or the source's own
    # notation: a paper that writes a sequence as (x_1, ..., x_n) is quoted faithfully with
    # the dots in it, while a quote that really leaves words out stops matching the chunk.
    score = float(fuzz.partial_ratio(normalize(quote), normalize(chunk_text)))
    if score < THRESHOLD:
        if ELLIPSIS.search(quote):
            return failed("the quote leaves words out instead of running on", score)
        return failed(f"the quote is not in chunk {chunk_id} (closest match {score:.0f}%)", score)
    return QuoteCheck(quote=quote, chunk_id=chunk_id, score=score, problem=None)
