import json
import os
import subprocess
import sys
import textwrap

from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import BACKEND


def test_health_reports_ok() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_behind_a_proxy_the_api_answers_under_its_path() -> None:
    """Vercel serves the API under /api and passes each request on with that path: with
    ROOT_PATH=/api the routes still match, and the docs ask for the schema under /api. The app
    is made when its module is imported, so it is imported afresh with the setting."""
    script = (
        "import json; from fastapi.testclient import TestClient; from app.main import app; "
        "client = TestClient(app); "
        "print(json.dumps([client.get('/api/health').json(), "
        "client.get('/api/openapi.json').json()['servers']]))"
    )
    elsewhere = subprocess.run(
        [sys.executable, "-c", script],
        cwd=BACKEND,
        env={**os.environ, "ROOT_PATH": "/api"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(elsewhere.stdout) == [{"status": "ok"}, [{"url": "/api"}]]


def test_the_api_starts_without_the_model_providers() -> None:
    """A deployed API imports itself on every cold start, and the providers' libraries are
    more than a third of that: they are loaded only once a model is built. The import is made
    afresh, as a new process would."""
    providers = ["openai", "google.genai", "groq", "aiohttp"]
    script = (
        "import json, sys; import app.main; "
        f"print(json.dumps([name for name in {providers!r} if name in sys.modules]))"
    )
    elsewhere = subprocess.run(
        [sys.executable, "-c", script], cwd=BACKEND, capture_output=True, text=True, check=True
    )

    assert json.loads(elsewhere.stdout) == []


def test_the_api_starts_without_langgraph() -> None:
    """LangGraph, with langchain-core and LangSmith, adds about a third of a second to an
    import: only an interview loads it (`app.interview.graph`)."""
    libraries = ["langgraph", "langchain_core", "langsmith"]
    script = (
        "import json, sys; import app.main; "
        f"print(json.dumps([name for name in {libraries!r} if name in sys.modules]))"
    )
    elsewhere = subprocess.run(
        [sys.executable, "-c", script], cwd=BACKEND, capture_output=True, text=True, check=True
    )

    assert json.loads(elsewhere.stdout) == []


def test_fastapi_records_nothing_of_a_request() -> None:
    """FastAPI's own OpenTelemetry is off: with global providers in place, as an exporter set up
    from OTEL_* variables would have them, a request, a validation failure and an error leave no
    span, no log and no measurement, so neither a body sent nor a stack trace can leave. The
    providers can be set once in a process, so the app is imported afresh in a new one."""
    script = textwrap.dedent(
        """
        import json

        from fastapi.testclient import TestClient
        from opentelemetry import _logs, metrics, trace
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import (
            InMemoryLogRecordExporter,
            SimpleLogRecordProcessor,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        spans, logs, readings = (
            InMemorySpanExporter(),
            InMemoryLogRecordExporter(),
            InMemoryMetricReader(),
        )
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
        trace.set_tracer_provider(tracer_provider)
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(SimpleLogRecordProcessor(logs))
        _logs.set_logger_provider(logger_provider)
        metrics.set_meter_provider(MeterProvider(metric_readers=[readings]))

        from app.main import app

        @app.get("/probe")
        def probe(n: int) -> None:
            raise RuntimeError("an error with a stack trace")

        client = TestClient(app, raise_server_exceptions=False)
        answered = [
            client.get("/health").status_code,
            client.get("/probe", params={"n": "a visitor's answer"}).status_code,
            client.get("/probe", params={"n": 1}).status_code,
        ]
        measured = readings.get_metrics_data()
        print(json.dumps({
            "answered": answered,
            "spans": len(spans.get_finished_spans()),
            "logs": len(logs.get_finished_logs()),
            "measurements": 0 if measured is None else sum(
                len(scope.metrics)
                for resource in measured.resource_metrics
                for scope in resource.scope_metrics
            ),
        }))
        """
    )
    elsewhere = subprocess.run(
        [sys.executable, "-c", script], cwd=BACKEND, capture_output=True, text=True, check=True
    )

    assert json.loads(elsewhere.stdout) == {
        "answered": [200, 422, 500],
        "spans": 0,
        "logs": 0,
        "measurements": 0,
    }
