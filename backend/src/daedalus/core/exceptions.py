"""
Application exceptions.

A small hierarchy so callers can distinguish "this document is bad" from
"this component is broken" without matching on error strings.
"""

from __future__ import annotations

__all__ = [
    "DaedalusError",
    "ExtractionError",
    "UnsupportedFileTypeError",
]


class DaedalusError(Exception):
    """Base class for every error raised deliberately by Daedalus."""


class UnsupportedFileTypeError(DaedalusError):
    """The uploaded file has an extension the ingestion pipeline cannot handle."""


class ExtractionError(DaedalusError):
    """A document could not be converted to text."""
