"""
Grounded question answering.

The endpoint the whole pipeline exists to serve. Defined with ``def`` and
not ``async def`` for the same reason as ingestion (ADR-008): generation
blocks for tens of seconds on a local model, and on the event loop that
would stall every other request in the process.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status

from daedalus.api.dependencies import get_db, get_embedder, get_llm
from daedalus.api.schemas import AnswerRequest, AnswerResponse
from daedalus.core.exceptions import LLMError
from daedalus.generation import answer_question
from daedalus.interfaces.embedding import Embedder
from daedalus.interfaces.llm import LLM

__all__ = ["router"]


logger = logging.getLogger(__name__)


router = APIRouter(tags=["answer"])


@router.post("/answer")
def answer(
    request: AnswerRequest,
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    llm: Annotated[LLM, Depends(get_llm)],
) -> AnswerResponse:
    """
    Answer a question from the indexed material, with citations.

    A question the corpus does not cover comes back as a refusal with no
    citations — a normal ``200``, because refusing correctly is the right
    answer rather than an error.
    """

    try:
        result = answer_question(db, embedder, llm, request.question, top_k=request.top_k)
    except LLMError as error:
        # The model backend being down is not the client's fault, and it is
        # transient in a way a 500 would not communicate.
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return AnswerResponse.from_answer(result)
