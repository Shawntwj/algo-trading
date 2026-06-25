"""Apply ``execution/schema.sql`` against the project's ClickHouse instance.

The data layer already owns connection bootstrap
(``data.clickhouse_client.get_client``). This module just splits the SQL file
on ``;`` boundaries and runs each statement — keeping the DDL in a real
``.sql`` file (rather than a Python string) so it can be diffed and read by a
DBA.

CLI::

    python -m execution.migrate              # applies to the configured DB
    python -m execution.migrate --database other_db  # override DB name

The migration is idempotent (each DDL uses ``IF NOT EXISTS``).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from clickhouse_connect.driver.client import Client

from data.clickhouse_client import get_client

log = logging.getLogger(__name__)


SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _split_statements(sql_text: str) -> list[str]:
    """Split a SQL file on ``;`` boundaries, skipping pure-whitespace
    fragments. Line comments (``-- …``) are stripped BEFORE splitting so a
    semicolon inside a comment doesn't fragment a statement.

    Good enough for our hand-written DDL — no string literals contain
    semicolons. If the schema grows to need a real parser, swap this for
    ``sqlparse.split``.
    """
    # 1. Drop ``--`` line comments first so commented semicolons don't split.
    decommented_lines = []
    for line in sql_text.splitlines():
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        decommented_lines.append(line)
    decommented = "\n".join(decommented_lines)

    # 2. Split on ``;`` and trim. Skip empty fragments.
    statements: list[str] = []
    for raw in decommented.split(";"):
        stmt = raw.strip()
        if stmt:
            statements.append(stmt)
    return statements


def apply_schema(client: Client | None = None, schema_path: Path = SCHEMA_PATH) -> list[str]:
    """Apply ``schema_path`` against ``client`` (defaults to the configured
    ClickHouse). Returns the list of statements executed.
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"schema file not found: {schema_path}")
    client = client or get_client()
    sql_text = schema_path.read_text()
    statements = _split_statements(sql_text)
    for stmt in statements:
        log.info("applying DDL: %s", stmt.splitlines()[0][:80])
        client.command(stmt)
    return statements


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply execution audit-table schema.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=SCHEMA_PATH,
        help=f"Path to SQL file (default: {SCHEMA_PATH}).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="DEBUG-level logging."
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    applied = apply_schema(schema_path=args.schema)
    print(f"applied {len(applied)} DDL statement(s) from {args.schema}")


if __name__ == "__main__":  # pragma: no cover — CLI
    main()
