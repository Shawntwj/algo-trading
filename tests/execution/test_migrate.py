"""Tests for ``execution.migrate`` — runs the real schema against a tmp DB.

Per the BRIEF's working principles, we do NOT mock ClickHouse. If the local
ClickHouse isn't reachable, the test module is skipped with a clear reason.

The tmp database is created with a random suffix per test session and is
DROPped on teardown so we never pollute the configured ``algo`` DB.
"""
from __future__ import annotations

import uuid

import pytest

from execution.migrate import _split_statements, apply_schema, SCHEMA_PATH


def _ch_client():
    try:
        import clickhouse_connect
        from config import load_settings
        cfg = load_settings().clickhouse
        client = clickhouse_connect.get_client(
            host=cfg.host,
            port=cfg.port,
            username=cfg.user,
            password=cfg.password,
        )
        # Probe.
        client.command("SELECT 1")
        return client, cfg
    except Exception:  # noqa: BLE001
        return None, None


CH_CLIENT, CH_CFG = _ch_client()
ch_required = pytest.mark.skipif(
    CH_CLIENT is None,
    reason="ClickHouse unreachable — skipping migrate tests.",
)


# ─── Pure-Python tests (no DB) ───────────────────────────────────────────────
def test_split_statements_drops_comments_and_blanks():
    sql = """
    -- top level comment
    CREATE TABLE foo (a Int64) ENGINE = MergeTree ORDER BY a;

    -- another
    CREATE TABLE bar (b Int64) ENGINE = MergeTree ORDER BY b;
    ;
    """
    out = _split_statements(sql)
    assert len(out) == 2
    assert out[0].startswith("CREATE TABLE foo")
    assert out[1].startswith("CREATE TABLE bar")


def test_schema_file_exists_and_has_both_tables():
    text = SCHEMA_PATH.read_text()
    assert "CREATE TABLE IF NOT EXISTS orders" in text
    assert "CREATE TABLE IF NOT EXISTS decisions" in text


# ─── DB tests (skipped without ClickHouse) ───────────────────────────────────
@ch_required
def test_apply_schema_creates_orders_and_decisions(tmp_path):
    """Create a tmp DB, apply the schema, verify the tables exist, DROP."""
    tmp_db = f"algo_test_{uuid.uuid4().hex[:8]}"
    CH_CLIENT.command(f"CREATE DATABASE {tmp_db}")
    try:
        import clickhouse_connect
        scoped = clickhouse_connect.get_client(
            host=CH_CFG.host,
            port=CH_CFG.port,
            username=CH_CFG.user,
            password=CH_CFG.password,
            database=tmp_db,
        )
        applied = apply_schema(client=scoped)
        assert len(applied) == 2

        rows = scoped.query(
            f"SELECT name FROM system.tables WHERE database = '{tmp_db}' ORDER BY name"
        ).result_rows
        names = [r[0] for r in rows]
        assert names == ["decisions", "orders"]

        # Idempotent — second apply must not raise.
        apply_schema(client=scoped)
    finally:
        CH_CLIENT.command(f"DROP DATABASE IF EXISTS {tmp_db}")


@ch_required
def test_orders_table_columns(tmp_path):
    tmp_db = f"algo_test_{uuid.uuid4().hex[:8]}"
    CH_CLIENT.command(f"CREATE DATABASE {tmp_db}")
    try:
        import clickhouse_connect
        scoped = clickhouse_connect.get_client(
            host=CH_CFG.host,
            port=CH_CFG.port,
            username=CH_CFG.user,
            password=CH_CFG.password,
            database=tmp_db,
        )
        apply_schema(client=scoped)
        rows = scoped.query(
            f"DESCRIBE TABLE {tmp_db}.orders"
        ).result_rows
        cols = {r[0] for r in rows}
        for must_have in (
            "order_id", "submitted_at", "strategy", "ticker", "side", "qty",
            "order_type", "limit_price", "stop_price", "broker",
            "broker_order_id", "status", "filled_qty", "avg_fill_price",
            "last_updated",
        ):
            assert must_have in cols, f"missing column {must_have} in orders"

        rows = scoped.query(
            f"DESCRIBE TABLE {tmp_db}.decisions"
        ).result_rows
        cols = {r[0] for r in rows}
        for must_have in (
            "decision_id", "decided_at", "strategy", "ticker", "bar_timestamp",
            "target_position", "current_position", "diff_qty", "explanation",
            "risk_blocked", "risk_reason",
        ):
            assert must_have in cols, f"missing column {must_have} in decisions"
    finally:
        CH_CLIENT.command(f"DROP DATABASE IF EXISTS {tmp_db}")
