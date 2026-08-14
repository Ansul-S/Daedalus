"""
Application exceptions.

A small hierarchy so callers can distinguish "this document is bad" from
"this component is broken" without matching on error strings.
"""

from __future__ import annotations

__all__ = [
    "DaedalusError",
    "DocumentNotFoundError",
    "DuplicateDocumentError",
    "EmbeddingError",
    "ExtractionError",
    "RetrievalError",
    "StorageError",
    "UnsupportedFileTypeError",
]


class DaedalusError(Exception):
    """Base class for every error raised deliberately by Daedalus."""


class UnsupportedFileTypeError(DaedalusError):
    """The uploaded file has an extension the ingestion pipeline cannot handle."""


class ExtractionError(DaedalusError):
    """A document could not be converted to text."""


class EmbeddingError(DaedalusError):
    """Text could not be turned into vectors.

    Raised when the embedding backend is unavailable or produces vectors of
    a dimension the vector index cannot store.
    """


class StorageError(DaedalusError):
    """A write to the database was rejected before it was attempted."""


class RetrievalError(DaedalusError):
    """A search could not be run against the index.

    Raised when the query cannot be posed at all — an embedder whose
    vectors do not fit the index, for instance — never when a well-formed
    query simply matches nothing.
    """


class DuplicateDocumentError(StorageError):
    """A document with the same content has already been ingested."""


class DocumentNotFoundError(StorageError):
    """No document record exists for the given identifier."""
