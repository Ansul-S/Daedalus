"""
The language model port.

Generation is the one place where swapping the backend is most likely to
happen — Ollama today, a hosted API if this ever needs to run somewhere
without a GPU — so pipeline code depends on this and never on a client
library.

The contract is deliberately one method. Streaming, tool calling, and
structured output are all real features of real backends, and none of them
has a caller yet; adding them now would mean guessing at signatures no
code has to satisfy. Sampling settings are fixed when an implementation is
constructed rather than passed per call, which keeps a run reproducible:
the evaluation harness cannot accidentally compare two answers generated
at different temperatures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["LLM"]


class LLM(ABC):
    """Generates text from a prompt."""

    @property
    @abstractmethod
    def model(self) -> str:
        """
        Identifier of the model being used.

        Recorded alongside evaluation results — a metric is meaningless
        without knowing which model produced it. Reading this must not
        load or contact anything.
        """

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """
        Answer a single prompt.

        Implementations must satisfy the following, which the shared
        contract test enforces:

        - The return value is the model's text alone, with any provider
          framing removed.
        - ``system`` sets the instruction context and is honoured
          separately from ``prompt``, because backends that distinguish
          the two weight them differently.
        - Raises ``LLMError`` when the backend cannot be reached or
          returns something unusable. An empty answer is a failure, not a
          valid completion.
        """
