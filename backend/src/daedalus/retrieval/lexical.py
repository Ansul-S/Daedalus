"""
Lexical retrieval — BM25 over the FTS5 index.

Answers the half embeddings are bad at. Study material for AI/ML is dense
with exact tokens — ``softmax``, ``LoRA``, ``RMSNorm`` — and an embedding
of a rare term is a weak signal, while an exact match is a decisive one.

Most of this module is about the query string. FTS5's ``MATCH`` takes a
query *language*, not a bag of words, so raw user text is both a crash and
an injection surface: ``C++`` is a syntax error, and ``NEAR(a b)`` parses
as an operator and quietly changes what the search means. Every query is
therefore rebuilt from its tokens rather than passed through.
"""

from __future__ import annotations

import logging
import re
import sqlite3

from daedalus.config import constants
from daedalus.interfaces.retrieval import Retriever, SearchHit

__all__ = ["LexicalRetriever"]


logger = logging.getLogger(__name__)


# Runs of word characters, which is close to what the unicode61 tokenizer
# keeps when it builds the index. Everything else — quotes, parentheses,
# the operators that make up FTS5's query language — is dropped.
_TOKEN = re.compile(r"\w+", re.UNICODE)

_BM25 = """
SELECT rowid AS chunk_id, bm25(chunks_fts) AS score
  FROM chunks_fts
 WHERE chunks_fts MATCH ?
 ORDER BY score
 LIMIT ?
"""


def to_match_expression(query: str) -> str | None:
    """
    Rebuild a user query as a safe FTS5 expression, or ``None`` if empty.

    Each token is emitted as a quoted phrase. Quoting is what disarms the
    query language: inside quotes FTS5 tokenizes the contents instead of
    parsing operators, and the tokens cannot contain a quote themselves
    because the pattern above never matches one.

    Tokens are joined with ``OR``. FTS5 defaults to ``AND``, which is the
    wrong default for natural-language questions — requiring every word of
    "why is the dot product scaled before the softmax" to appear in one
    chunk matches almost nothing. BM25 already handles the ranking; the
    match only needs to decide what is a candidate.
    """

    tokens = _TOKEN.findall(query)

    if not tokens:
        return None

    return " OR ".join(f'"{token}"' for token in tokens)


class LexicalRetriever(Retriever):
    """Searches ``chunks_fts`` for keyword matches, ranked by BM25."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def search(self, query: str, top_k: int = constants.DEFAULT_TOP_K) -> list[SearchHit]:
        if top_k <= 0:
            return []

        expression = to_match_expression(query)

        # A query of pure punctuation has no terms to look up. That is an
        # empty result, not an error.
        if expression is None:
            logger.debug("Query %r contains no searchable terms", query)
            return []

        rows = self._connection.execute(_BM25, (expression, top_k)).fetchall()

        # bm25() returns increasingly negative numbers for better matches.
        # Negating leaves the ordering untouched and satisfies the port's
        # higher-is-better rule.
        return [SearchHit(chunk_id=row["chunk_id"], score=-row["score"]) for row in rows]
