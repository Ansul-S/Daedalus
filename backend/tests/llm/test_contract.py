"""
The contract every LLM must satisfy.

The real backend is gated behind an environment variable: it needs
`ollama serve` running with the model pulled, and a completion takes
seconds. Set DAEDALUS_TEST_REAL_LLM=1 to include it.
"""

from __future__ import annotations

import os

import httpx
import pytest

from daedalus.interfaces.llm import LLM
from daedalus.llm import FakeLLM, OllamaLLM

QUESTION = "Answer with the single word: yes"


def _real_llm() -> LLM:
    """Build the Ollama-backed model, or skip."""

    if os.environ.get("DAEDALUS_TEST_REAL_LLM") != "1":
        pytest.skip("set DAEDALUS_TEST_REAL_LLM=1 to test against a running Ollama")

    llm = OllamaLLM()

    try:
        httpx.get("http://localhost:11434/api/tags", timeout=2).raise_for_status()
    except httpx.HTTPError:  # pragma: no cover - depends on the developer's machine
        pytest.skip("Ollama is not reachable")

    return llm


@pytest.fixture(params=["fake", "real"])
def llm(request: pytest.FixtureRequest) -> LLM:
    if request.param == "fake":
        return FakeLLM()

    return _real_llm()


def test_reports_which_model_it_is(llm: LLM) -> None:
    assert llm.model


def test_reading_the_model_name_contacts_nothing() -> None:
    """Callers record it alongside results, so it must stay free."""

    assert OllamaLLM(model="never-pulled").model == "never-pulled"


def test_a_prompt_produces_text(llm: LLM) -> None:
    answer = llm.complete(QUESTION)

    assert isinstance(answer, str)
    assert answer.strip()


def test_a_system_prompt_is_accepted(llm: LLM) -> None:
    answer = llm.complete(QUESTION, system="You are terse.")

    assert answer.strip()


def test_llm_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        LLM()  # type: ignore[abstract]
