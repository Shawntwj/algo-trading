from __future__ import annotations

import logging
from functools import lru_cache

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from config import load_settings

log = logging.getLogger(__name__)


BARS_DDL = """
CREATE TABLE IF NOT EXISTS bars (
    ticker     LowCardinality(String),
    timestamp  DateTime64(3, 'UTC'),
    open       Float64,
    high       Float64,
    low        Float64,
    close      Float64,
    volume     Float64,
    interval   LowCardinality(String),
    ingested_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(ingested_at)
PARTITION BY (ticker, toYYYYMM(timestamp))
ORDER BY (ticker, interval, timestamp)
SETTINGS index_granularity = 8192
"""


@lru_cache(maxsize=1)
def get_client() -> Client:
    cfg = load_settings().clickhouse
    client = clickhouse_connect.get_client(
        host=cfg.host,
        port=cfg.port,
        username=cfg.user,
        password=cfg.password,
        database=cfg.database,
    )
    return client


def ensure_schema() -> None:
    """Create the database (if needed) and the bars table."""
    cfg = load_settings().clickhouse
    # Connect to default DB first so we can CREATE DATABASE.
    bootstrap = clickhouse_connect.get_client(
        host=cfg.host,
        port=cfg.port,
        username=cfg.user,
        password=cfg.password,
    )
    bootstrap.command(f"CREATE DATABASE IF NOT EXISTS {cfg.database}")
    bootstrap.close()

    client = get_client()
    client.command(BARS_DDL)
    log.info("ClickHouse schema ready (db=%s)", cfg.database)
