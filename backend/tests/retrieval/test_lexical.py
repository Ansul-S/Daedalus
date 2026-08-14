"""
Lexical retrieval, and the query sanitization that keeps FTS5 usable.

The sanitizer tests are the important ones here: every string in
``test_hostile_queries_do_not_reach_fts5`` raises ``OperationalError`` if
passed to MATCH unchanged.
"""

from __future__ import annotations

import sqlite3

import pytest

from daedalus.retrieval.lexical import LexicalRetriever, to_match_expression

# Query Rebuilding


def test_words_become_quoted_terms() -> None:
    assert to_match_expression("softmax scaling") == '"softmax" OR "scaling"'


def test_punctuation_is_dropped() -> None:
    assert to_match_expression("what is *attention*?") == '"what" OR "is" OR "attention"'


def test_fts5_operators_are_reduced_to_words() -> None:
    """NEAR parses as an operator unquoted; quoted it is just a term."""

    assert to_match_expression("NEAR(a b)") == '"NEAR" OR "a" OR "b"'


def test_quotes_cannot_escape_the_quoting() -> None:
    expression = to_match_expression('say "hello"')

    assert expression == '"say" OR "hello"'


def test_a_query_with_no_words_has_no_expression() -> None:
    assert to_match_expression("?!*()") is None


def test_an_empty_query_has_no_expression() -> None:
    assert to_match_expression("") is None


# Searching


def test_an_exact_term_finds_its_chunk(
    indexed: sqlite3.Connection, chunk_ids: list[int]
) -> None:
    hits = LexicalRetriever(indexed).search("convolution")

    assert [hit.chunk_id for hit in hits] == [chunk_ids[3]]


def test_a_shared_term_finds_every_chunk_containing_it(
    indexed: sqlite3.Connection, chunk_ids: list[int]
) -> None:
    hits = LexicalRetriever(indexed).search("softmax")

    assert {hit.chunk_id for hit in hits} == {chunk_ids[0], chunk_ids[1]}


def test_scores_are_higher_is_better(indexed: sqlite3.Connection) -> None:
    """bm25() is negative and ascending; the port promises the opposite."""

    hits = LexicalRetriever(indexed).search("softmax")

    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
    assert all(hit.score > 0 for hit in hits)


@pytest.mark.parametrize(
    "query",
    ["C++", 'what is "attention"?', "", "   ", "?!*", "NEAR(a b)", "softmax OR", "a AND"],
)
def test_hostile_queries_do_not_reach_fts5(
    indexed: sqlite3.Connection, query: str
) -> None:
    """Every one of these is a syntax error if passed to MATCH raw."""

    LexicalRetriever(indexed).search(query)


def test_a_query_of_pure_punctuation_finds_nothing(indexed: sqlite3.Connection) -> None:
    assert LexicalRetriever(indexed).search("?!*") == []


def test_a_term_absent_from_the_corpus_finds_nothing(
    indexed: sqlite3.Connection,
) -> None:
    assert LexicalRetriever(indexed).search("photosynthesis") == []


def test_results_are_limited_to_top_k(indexed: sqlite3.Connection) -> None:
    hits = LexicalRetriever(indexed).search("softmax", top_k=1)

    assert len(hits) == 1


def test_asking_for_no_results_returns_none(indexed: sqlite3.Connection) -> None:
    assert LexicalRetriever(indexed).search("softmax", top_k=0) == []


def test_deleted_chunks_stop_matching(
    indexed: sqlite3.Connection, chunk_ids: list[int]
) -> None:
    """Guards the FTS delete trigger from the storage layer."""

    indexed.execute("DELETE FROM chunks WHERE id = ?", (chunk_ids[3],))
    indexed.commit()

    assert LexicalRetriever(indexed).search("convolution") == []
