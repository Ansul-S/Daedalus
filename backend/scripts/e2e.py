"""Run the end-to-end test: the whole app on fake models, in a database of its own.

Run from the repo root:  make e2e
                         make e2e ARGS=--headed    to watch it in a browser window

It makes `daedalus_e2e` afresh on the Postgres server of `make db-up`, and a temporary folder
for uploads, then runs Playwright (frontend/playwright.config.ts). Playwright starts the API,
the worker and the frontend on them with FAKE_MODELS on, and walks from the landing page to a
graded answer and its room on the dashboard. The database and the folder go when it ends,
whether it passed or not.

The servers take the usual ports, 8000 and 3000, because the frontend is built with the API's
address in it, so `make api` and `make web` or `pnpm start` have to be stopped first. The
frontend is built as `pnpm build` builds it, so `pnpm start` serves the same app afterwards.
"""

import logging
import os
import socket
import subprocess
import sys
import tempfile

import psycopg
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from app.core.config import REPO_ROOT, get_settings

DATABASE = "daedalus_e2e"
PORTS = {8000: "the API: stop `make api`", 3000: "the frontend: stop `make web` or `pnpm start`"}


def in_use(port: int) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=0.5):
            return True
    except OSError:
        return False


def recreate(admin: str, url: str) -> None:
    """A fresh database, migrated to the latest schema."""
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(f"DROP DATABASE IF EXISTS {DATABASE} WITH (FORCE)")
        connection.execute(f"CREATE DATABASE {DATABASE}")
    config = Config(toml_file=REPO_ROOT / "backend" / "pyproject.toml")
    config.attributes["database_url"] = url
    command.upgrade(config, "head")


def drop(admin: str) -> None:
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(f"DROP DATABASE IF EXISTS {DATABASE} WITH (FORCE)")


def main(playwright_args: list[str]) -> int:
    busy = [f"port {port} is in use ({what})" for port, what in PORTS.items() if in_use(port)]
    if busy:
        print("\n".join(busy) + ". The end-to-end test runs its own servers there.")
        return 1

    server = make_url(get_settings().database_url)
    admin = server.set(drivername="postgresql", database="postgres")
    admin_url = admin.render_as_string(hide_password=False)
    url = server.set(database=DATABASE).render_as_string(hide_password=False)
    try:
        recreate(admin_url, url)
    except psycopg.OperationalError:
        print("Postgres is not running; start it with `make db-up`.")
        return 1

    try:
        with tempfile.TemporaryDirectory(prefix="daedalus-e2e-") as data_dir:
            environment = os.environ | {"E2E_DATABASE_URL": url, "E2E_DATA_DIR": data_dir}
            run = subprocess.run(
                ["pnpm", "exec", "playwright", "test", *playwright_args],
                cwd=REPO_ROOT / "frontend",
                env=environment,
                check=False,
            )
            return run.returncode
    finally:
        drop(admin_url)


if __name__ == "__main__":
    # Quiet: the migrations would otherwise report every step
    logging.basicConfig(level=logging.WARNING)
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
