"""
A language model that needs no model.

Generation is the hardest part of the pipeline to test against the real
thing: the output is nondeterministic, slow, and different on every
machine. Asserting that a real model cites its sources correctly means
asserting on text nobody can predict.

This substitutes a scripted one. Because it records every prompt it
receives, tests can check the half that *is* deterministic and does matter
— that retrieved context actually reached the model, and that whatever it
answered was turned into the right citations.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from daedalus.interfaces.llm import LLM

__all__ = ["FakeLLM"]


logger = logging.getLogger(__name__)


DEFAULT_RESPONSE = "The material covers this. [1]"


class FakeLLM(LLM):
    """
    Returns scripted answers and remembers what it was asked.

    ``responses`` maps a substring to the answer it triggers. Substring
    rather than exact match because the prompt a caller builds contains
    the retrieved context, so keying on the whole thing would mean
    rewriting the test whenever the prompt template changes.
    """

    def __init__(
        self,
        *,
        responses: Mapping[str, str] | None = None,
        default: str = DEFAULT_RESPONSE,
    ) -> None:
        self._responses = dict(responses or {})
        self._default = default

        # Every prompt and system message this was called with, in order.
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    @property
    def model(self) -> str:
        return "fake"

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)

        # First match in insertion order, so a test with overlapping keys
        # gets the one it declared first rather than an arbitrary winner.
        for trigger, response in self._responses.items():
            if trigger in prompt:
                return response

        return self._default
