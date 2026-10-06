import json
import os
import subprocess
import sys

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
