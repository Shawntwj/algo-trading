-- ClickHouse schema for the live runner's audit tables (Task 5b).
--
-- Two tables: ``orders`` (one row per broker submission, mutated on status
-- updates via ReplacingMergeTree) and ``decisions`` (immutable per-bar
-- decision log; one row per (strategy, ticker) decision regardless of
-- whether an order followed).
--
-- Both tables live in the same DB as ``bars`` — picked up from
-- ``config.load_settings().clickhouse.database`` by ``execution/migrate.py``.

CREATE TABLE IF NOT EXISTS orders (
    order_id          String,
    submitted_at      DateTime64(3, 'UTC'),
    strategy          LowCardinality(String),
    ticker            LowCardinality(String),
    side              Enum8('buy' = 1, 'sell' = 2),
    qty               Float64,
    order_type        Enum8('market' = 1, 'limit' = 2, 'stop' = 3, 'stop_limit' = 4),
    limit_price       Nullable(Float64),
    stop_price        Nullable(Float64),
    broker            LowCardinality(String),
    broker_order_id   Nullable(String),
    status            Enum8('submitted' = 1, 'filled' = 2, 'partial' = 3, 'cancelled' = 4, 'rejected' = 5),
    filled_qty        Float64 DEFAULT 0,
    avg_fill_price    Nullable(Float64),
    last_updated      DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(last_updated)
PARTITION BY toYYYYMM(submitted_at)
ORDER BY (strategy, ticker, submitted_at, order_id)
SETTINGS index_granularity = 8192;


CREATE TABLE IF NOT EXISTS decisions (
    decision_id       UUID,
    decided_at        DateTime64(3, 'UTC'),
    strategy          LowCardinality(String),
    ticker            LowCardinality(String),
    bar_timestamp     DateTime64(3, 'UTC'),
    target_position   Float64,
    current_position  Float64,
    diff_qty          Float64,
    explanation       String,
    risk_blocked      UInt8,
    risk_reason       Nullable(String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(decided_at)
ORDER BY (strategy, ticker, decided_at)
SETTINGS index_granularity = 8192;
