"""Tracing model calls: on only with both keys, one trace per piece of work, and nothing of
what was asked or answered in it unless content is asked for. Spans go to an in-memory
exporter in place of Langfuse, through the same path they take to it."""

import asyncio
import base64
from collections.abc import Callable, Iterator, Sequence

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from app.core.checks import check_traces_sent, check_tracing
from app.core.config import Settings
from app.db.models import EMBEDDING_DIMENSIONS
from app.grading.grader import grader
from app.llm import tracing
from app.llm.embeddings import Embedder
from app.questions.generation import question_agent
from app.questions.tagging import tagger
from app.questions.validation import checker

# Written into every prompt, reply, instruction and error below: wherever it turns up in an
# exported span, text has left the machine
SECRET = "zebra-quartz-7731"
KEYS = {"langfuse_public_key": "pk-lf-test", "langfuse_secret_key": "sk-lf-test"}


def tracing_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **{"langfuse_tracing_enabled": True, **KEYS, **overrides})


@pytest.fixture
def traced_into() -> Iterator[Callable[..., InMemorySpanExporter]]:
    """Starts tracing into memory with the settings given; stops it after the test."""
    started: list[tracing.Tracing | None] = []

    def start(**overrides) -> InMemorySpanExporter:
        exporter = InMemorySpanExporter()
        started.append(tracing.start(tracing_settings(**overrides), "test", exporter=exporter))
        assert started[-1] is not None
        return exporter

    yield start
    for each in started:
        tracing.stop(each)


def replying(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(f"the reply names {SECRET}")])


def failing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # As Groq answers a reply that broke the schema: the reply comes back inside the error
    body = {"error": {"code": "json_validate_failed", "failed_generation": f"{{{SECRET}"}}
    raise ModelHTTPError(status_code=400, model_name="stand-in", body=body)


def writer_run(respond=replying) -> None:
    agent = Agent(FunctionModel(respond), name="writer", instructions=f"Instructions {SECRET}")
    asyncio.run(agent.run(f"A prompt about {SECRET}", metadata={"prompt_version": "test-v1"}))


def everything(spans: Sequence[ReadableSpan]) -> str:
    return "\n".join(span.to_json() for span in spans)


def by_name(spans: Sequence[ReadableSpan]) -> dict[str, ReadableSpan]:
    return {span.name: span for span in spans}


def test_tracing_is_off_without_both_keys_with_the_stand_ins_or_when_switched_off() -> None:
    cases = {
        "no Langfuse keys": Settings(_env_file=None, langfuse_tracing_enabled=True),
        "LANGFUSE_SECRET_KEY is not set": Settings(
            _env_file=None, langfuse_tracing_enabled=True, langfuse_public_key="pk-lf-test"
        ),
        "the stand-in models (FAKE_MODELS) are not traced": tracing_settings(fake_models=True),
        "LANGFUSE_TRACING_ENABLED is false": tracing_settings(langfuse_tracing_enabled=False),
    }
    for reason, settings in cases.items():
        assert tracing.switched_off(settings) == reason
        assert tracing.start(settings, "test", exporter=InMemorySpanExporter()) is None
        with tracing.traced("generate") as span:
            assert not span.is_recording()


def test_the_check_says_what_a_trace_holds_and_warns_about_a_key_alone() -> None:
    on = check_tracing(tracing_settings())
    assert (on.status, on.detail) == (
        "ok",
        "to Langfuse at https://cloud.langfuse.com: timings, tokens and models, "
        "no prompts or answers",
    )
    assert (
        "prompts and answers included"
        in check_tracing(tracing_settings(langfuse_content=True)).detail
    )
    alone = check_tracing(
        Settings(_env_file=None, langfuse_tracing_enabled=True, langfuse_public_key="pk-lf-x")
    )
    assert (alone.status, alone.detail) == ("warn", "off: LANGFUSE_SECRET_KEY is not set")
    off = check_tracing(Settings(_env_file=None, langfuse_tracing_enabled=True))
    assert (off.status, off.detail) == ("ok", "off: no Langfuse keys")


def test_spans_go_to_the_project_s_endpoint_with_its_keys(monkeypatch) -> None:
    # The name Langfuse's older snippets use for the address
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com/")
    url, headers = tracing.endpoint(tracing_settings())

    assert url == "https://us.cloud.langfuse.com/api/public/otel/v1/traces"
    scheme, token = headers["Authorization"].split()
    assert (scheme, base64.b64decode(token).decode()) == ("Basic", "pk-lf-test:sk-lf-test")
    # Without it, Langfuse shows what it was sent only after a while
    assert headers["x-langfuse-ingestion-version"] == "4"


def test_a_piece_of_work_is_one_trace_with_its_model_calls_inside(traced_into) -> None:
    exported = traced_into()

    with tracing.traced("generate", version="generate-v3", job=4, task=17, style="why_how"):
        writer_run()

    spans = by_name(exported.get_finished_spans())
    root, run = spans["generate"], spans["invoke_agent writer"]
    [chat] = [span for name, span in spans.items() if name.startswith("chat ")]
    assert {span.context.trace_id for span in spans.values()} == {root.context.trace_id}
    assert run.parent.span_id == root.context.span_id
    assert chat.parent.span_id == run.context.span_id
    assert root.attributes["langfuse.trace.name"] == "generate"
    assert root.attributes["langfuse.version"] == "generate-v3"
    assert tuple(root.attributes["langfuse.trace.tags"]) == ("generate",)
    assert root.attributes["langfuse.trace.metadata.task"] == 17
    assert root.attributes["langfuse.trace.metadata.style"] == "why_how"
    # What each model call cost, and the prompt version its run was made with
    assert chat.attributes["gen_ai.usage.input_tokens"] > 0
    assert chat.attributes["gen_ai.usage.output_tokens"] > 0
    assert chat.attributes["gen_ai.request.model"] == "function:replying:"
    assert '"prompt_version":"test-v1"' in run.attributes["metadata"]
    assert root.resource.attributes["service.name"] == "daedalus-test"
    assert root.resource.attributes["deployment.environment.name"] == "local"


def test_prompts_replies_and_instructions_stay_here_by_default(traced_into) -> None:
    exported = traced_into()

    with tracing.traced("generate"):
        writer_run()

    spans = exported.get_finished_spans()
    assert len(spans) == 3
    assert SECRET not in everything(spans)


def test_an_error_keeps_its_type_and_loses_its_text(traced_into) -> None:
    exported = traced_into()

    with pytest.raises(ModelHTTPError), tracing.traced("generate"):
        writer_run(failing)

    spans = exported.get_finished_spans()
    assert SECRET not in everything(spans)
    for span in spans:
        assert span.status.status_code is StatusCode.ERROR
        assert span.status.description is None
        [event] = span.events
        assert event.name == "exception"
        assert set(event.attributes) == {"exception.type", "exception.escaped"}
        assert event.attributes["exception.type"] == "pydantic_ai.exceptions.ModelHTTPError"


def test_with_content_asked_for_everything_goes(traced_into) -> None:
    exported = traced_into(langfuse_content=True)

    with pytest.raises(ModelHTTPError), tracing.traced("generate"):
        writer_run(failing)
    with tracing.traced("generate"):
        writer_run()

    text = everything(exported.get_finished_spans())
    assert f"A prompt about {SECRET}" in text
    assert f"the reply names {SECRET}" in text
    assert "failed_generation" in text


def test_an_embedding_is_traced_with_its_model_and_tokens_but_not_its_texts(traced_into) -> None:
    exported = traced_into()

    def ollama(request: httpx.Request) -> httpx.Response:
        vectors = [[0.0] * EMBEDDING_DIMENSIONS] * 2
        return httpx.Response(200, json={"embeddings": vectors, "prompt_eval_count": 11})

    http = httpx.AsyncClient(transport=httpx.MockTransport(ollama), base_url="http://ollama")
    embedder = Embedder(tracing_settings(), http)
    asyncio.run(embedder.embed_documents([f"a passage on {SECRET}", "another passage"]))

    [span] = exported.get_finished_spans()
    assert span.name == "embeddings qwen3-embedding:0.6b"
    assert span.attributes["langfuse.observation.type"] == "embedding"
    assert span.attributes["gen_ai.request.model"] == "qwen3-embedding:0.6b"
    assert span.attributes["gen_ai.usage.input_tokens"] == 11
    assert span.attributes["daedalus.embedding.inputs"] == 2
    assert SECRET not in everything([span])


def test_stopping_leaves_agents_and_work_untraced(traced_into) -> None:
    exported = traced_into()
    tracing.stop(None)

    with tracing.traced("generate") as span:
        writer_run()

    assert not span.is_recording()
    assert exported.get_finished_spans() == ()


class Refusing(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return SpanExportResult.FAILURE


def test_the_sender_counts_the_batches_taken_and_refused() -> None:
    taken = tracing.Sender(InMemorySpanExporter(), content=False)
    refused = tracing.Sender(Refusing(), content=False)

    for sender in (taken, refused, refused):
        sender.export([])

    assert taken.results == {"sent": 1}
    assert refused.results == {"refused": 2}


def batched(started: tracing.Tracing | None) -> InMemorySpanExporter:
    """What a batch that waits a minute has been sent, as the one Langfuse gets would."""
    assert started is not None
    exporter = InMemorySpanExporter()
    started.provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=60_000))
    return exporter


@pytest.mark.parametrize("environment", ["local", "production"])
def test_deployed_a_piece_of_work_is_sent_as_it_ends(traced_into, environment) -> None:
    """A deployed API may be frozen once it has answered, before the batch goes out, so there
    a piece of work's spans are sent as it ends. Locally they wait for the batch."""
    every_span = traced_into(environment=environment)
    batch = batched(tracing.current())

    with tracing.traced("grade", attempt=3):
        writer_run()

    ended = sorted(span.name for span in every_span.get_finished_spans())
    assert "grade" in ended and len(ended) >= 3
    sent = sorted(span.name for span in batch.get_finished_spans())
    assert sent == (ended if environment == "production" else [])


def test_deployed_a_failed_piece_of_work_is_sent_too(traced_into) -> None:
    traced_into(environment="production")
    batch = batched(tracing.current())

    with pytest.raises(RuntimeError), tracing.traced("grade"):
        raise RuntimeError("no model answered")

    assert [span.name for span in batch.get_finished_spans()] == ["grade"]


def test_the_api_says_what_its_tracing_has_sent(client, settings, traced_into) -> None:
    def traces_sent() -> dict:
        checks = client.get("/health/deps").json()
        return next(check for check in checks if check["name"] == "traces sent")

    settings.langfuse_public_key = KEYS["langfuse_public_key"]
    settings.langfuse_secret_key = SecretStr(KEYS["langfuse_secret_key"])
    settings.langfuse_tracing_enabled = True
    assert traces_sent() == {
        "name": "traces sent",
        "status": "warn",
        "detail": "tracing was not started here",
    }

    exported = traced_into()
    with tracing.traced("grade"):
        writer_run()
    spans = len(exported.get_finished_spans())

    # Each span goes on its own to the stand-in exporter
    assert traces_sent()["detail"] == f"since start: {spans} batches taken, 0 refused"
    refused = tracing.Tracing(TracerProvider(), tracing.Sender(Refusing(), content=False))
    refused.sender.export([])
    assert (check_traces_sent(refused).status, check_traces_sent(refused).detail) == (
        "warn",
        "since start: 0 batches taken, 1 refused",
    )


def test_each_agent_runs_under_its_own_name() -> None:
    model = TestModel()
    names = [
        question_agent(model).name,
        checker(model).name,
        tagger(model).name,
        grader(model, [1], 2).name,
    ]
    assert names == ["writer", "checker", "tagger", "grader"]
