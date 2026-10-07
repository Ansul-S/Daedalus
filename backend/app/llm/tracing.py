"""Tracing every model call to Langfuse, when its keys are set.

Pydantic AI already describes each agent run and each request to a model in OpenTelemetry
spans: the provider, the model, the tokens in and out, and how long it took. Those spans, the
embedder's own, and one span for each piece of work -- a question written, an answer graded, a
passage tagged -- go to Langfuse's OpenTelemetry endpoint, one trace per piece of work. Without
both keys nothing is set up and every span is a no-op. With FAKE_MODELS nothing is traced
either: what the stand-ins write is made up.

What leaves the machine is timings, token counts, models, providers, prompt versions and ids.
Prompts, passages, answers and the models' replies stay here: Pydantic AI is told to leave them
out, and an error keeps its type but loses its message, which from a provider can quote what
the model wrote. LANGFUSE_CONTENT=1 sends all of it, for looking into a prompt.

Plain OpenTelemetry rather than Langfuse's own client, so that everything on the way out
passes through `Sender` here: the client's masking hook sees span attributes but not the
exception events that carry an error's message.

Spans wait in a batch that a background thread sends every few seconds. A deployed API runs
as a serverless function, frozen as soon as it has answered, so the thread may never get to
them; there, each piece of work's spans are sent as it ends, before the answer goes out.
"""

import base64
import logging
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import NoOpTracer, Span, Status, Tracer
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

from app.core.config import Settings

# Langfuse's OpenTelemetry endpoint, under the project's address
OTLP_PATH = "/api/public/otel/v1/traces"
# Without it, Langfuse can take up to 15 minutes to show what it was sent
INGESTION_HEADER = {"x-langfuse-ingestion-version": "4"}
# A batch that cannot be sent is given up on after this long, so that a script ending without
# a network does not hang on its last one
EXPORT_TIMEOUT = 5.0
# All that an error keeps of itself when content is left out
ERROR_ATTRIBUTES = ("exception.type", "exception.escaped")

log = logging.getLogger(__name__)

_tracer: Tracer = NoOpTracer()
_started: "Tracing | None" = None


def switched_off(settings: Settings) -> str | None:
    """Why nothing is traced, or None when tracing is on."""
    if settings.fake_models:
        return "the stand-in models (FAKE_MODELS) are not traced"
    if not settings.langfuse_tracing_enabled:
        return "LANGFUSE_TRACING_ENABLED is false"
    public, secret = settings.langfuse_public_key, settings.langfuse_secret_key
    if public is None and secret is None:
        return "no Langfuse keys"
    if public is None or secret is None:
        missing = "LANGFUSE_PUBLIC_KEY" if public is None else "LANGFUSE_SECRET_KEY"
        return f"{missing} is not set"
    return None


def endpoint(settings: Settings) -> tuple[str, dict[str, str]]:
    """Where spans go and the headers that let them in: the project's keys, as Basic auth."""
    assert settings.langfuse_public_key and settings.langfuse_secret_key
    pair = f"{settings.langfuse_public_key}:{settings.langfuse_secret_key.get_secret_value()}"
    token = base64.b64encode(pair.encode()).decode()
    url = settings.langfuse_base_url.rstrip("/") + OTLP_PATH
    return url, {"Authorization": f"Basic {token}", **INGESTION_HEADER}


def kept_error_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (attributes or {}).items() if key in ERROR_ATTRIBUTES}


def without_text(span: ReadableSpan) -> ReadableSpan:
    """The span with every error reduced to its type -- no message, no stack trace, no status
    description -- and any other event to its name. The rest is left as it is."""
    events = [
        Event(event.name, kept_error_attributes(event.attributes), event.timestamp)
        for event in span.events
    ]
    attributes = {
        key: value
        for key, value in (span.attributes or {}).items()
        if not key.startswith("exception.") or key in ERROR_ATTRIBUTES
    }
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=events,
        links=span.links,
        kind=span.kind,
        status=Status(span.status.status_code),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class Sender(SpanExporter):
    """Everything on its way out: errors stripped of their text unless content is wanted, and
    a count of the batches the other end took and refused."""

    def __init__(self, exporter: SpanExporter, *, content: bool) -> None:
        self.exporter = exporter
        self.content = content
        self.results: Counter[str] = Counter()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        outgoing = spans if self.content else [without_text(span) for span in spans]
        result = self.exporter.export(outgoing)
        self.results["sent" if result is SpanExportResult.SUCCESS else "refused"] += 1
        return result

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self.exporter.force_flush(timeout_millis)


@dataclass
class Tracing:
    provider: TracerProvider
    sender: Sender
    # Each piece of work's spans sent as it ends, not left to the batch: in production
    each_trace: bool = False

    def flush(self) -> bool:
        """Send what is waiting; False when it could not be sent in time."""
        return self.provider.force_flush()


def current() -> Tracing | None:
    """The tracing this process started, or None while it traces nothing."""
    return _started


def start(
    settings: Settings, service: str, *, exporter: SpanExporter | None = None
) -> Tracing | None:
    """Trace every agent run and model request from here on, or do nothing when tracing is
    off. `exporter` stands in for Langfuse, and gets each span as it ends."""
    global _tracer, _started
    if switched_off(settings):
        return None
    resource = Resource.create(
        {
            "service.name": f"daedalus-{service}",
            "deployment.environment.name": settings.environment,
        }
    )
    provider = TracerProvider(resource=resource)
    if exporter is None:
        url, headers = endpoint(settings)
        sender = Sender(
            OTLPSpanExporter(endpoint=url, headers=headers, timeout=EXPORT_TIMEOUT),
            content=settings.langfuse_content,
        )
        provider.add_span_processor(BatchSpanProcessor(sender))
    else:
        sender = Sender(exporter, content=settings.langfuse_content)
        provider.add_span_processor(SimpleSpanProcessor(sender))
    Agent.instrument_all(
        InstrumentationSettings(
            tracer_provider=provider,
            include_content=settings.langfuse_content,
            # The output schema and the instructions, sent with every request
            include_model_request_parameters=settings.langfuse_content,
        )
    )
    _tracer = provider.get_tracer("daedalus")
    _started = Tracing(provider, sender, each_trace=settings.environment == "production")
    return _started


def stop(tracing: Tracing | None) -> None:
    """Send what is left and stop tracing."""
    global _tracer, _started
    Agent.instrument_all(False)
    _tracer = NoOpTracer()
    _started = None
    if tracing is not None:
        tracing.provider.shutdown()


@contextmanager
def tracing(settings: Settings, service: str) -> Iterator[Tracing | None]:
    """Tracing for as long as a process runs: the API, the worker, a script."""
    started = start(settings, service)
    try:
        yield started
    finally:
        stop(started)


@contextmanager
def traced(
    name: str, *, version: str | None = None, **metadata: str | int | None
) -> Iterator[Span]:
    """One piece of work, as a trace of its own: the model calls made inside it are its
    children. `version` is the prompt version, and `metadata` ids that say what it was about;
    neither should hold any text."""
    attributes: dict[str, str | int | list[str]] = {
        "langfuse.trace.name": name,
        "langfuse.trace.tags": [name],
    }
    if version is not None:
        attributes["langfuse.version"] = version
    for key, value in metadata.items():
        if value is not None:
            attributes[f"langfuse.trace.metadata.{key}"] = value
    try:
        with _tracer.start_as_current_span(name, attributes=attributes) as span:
            yield span
    finally:
        if _started is not None and _started.each_trace and not _started.flush():
            log.warning("The spans of %r were not sent to Langfuse in time", name)


@contextmanager
def embedding(model: str, inputs: int) -> Iterator[Span]:
    """A request to the embedding model: which model, how many texts and, once known, how
    many tokens. Never the texts."""
    attributes = {
        "langfuse.observation.type": "embedding",
        "gen_ai.operation.name": "embeddings",
        "gen_ai.provider.name": "ollama",
        "gen_ai.request.model": model,
        "daedalus.embedding.inputs": inputs,
    }
    with _tracer.start_as_current_span(f"embeddings {model}", attributes=attributes) as span:
        yield span
