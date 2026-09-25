"""Write the API's OpenAPI schema to a file: the frontend's typed client is generated from it.

FastAPI builds the schema from the routes alone, so neither the server nor the database has
to be running.

Run from backend/:  uv run python -m scripts.openapi ../frontend/openapi.json
"""

import argparse
import json
from pathlib import Path

from app.main import app


def write_schema(path: Path) -> None:
    path.write_text(json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n", "utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the API's OpenAPI schema to a file.")
    parser.add_argument("path", type=Path, help="where to write it, e.g. ../frontend/openapi.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    write_schema(arguments.path)
    print(f"Wrote the API schema to {arguments.path}")
