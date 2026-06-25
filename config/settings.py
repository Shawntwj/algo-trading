from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parent / "tickers.yaml"


@dataclass
class ClickHouseConfig:
    host: str = "localhost"
    port: int = 8123
    database: str = "algo"
    user: str = "default"
    password: str = ""


@dataclass
class CostConfig:
    commission: float = 0.0005
    slippage: float = 0.0005


@dataclass
class Settings:
    universe: list[str] = field(default_factory=list)
    intervals: list[str] = field(default_factory=lambda: ["1d"])
    backfill_start: str = "2018-01-01"
    backfill_end: str | None = None
    costs: CostConfig = field(default_factory=CostConfig)
    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)

    @property
    def end_date(self) -> str:
        return self.backfill_end or date.today().isoformat()


def load_settings(path: Path | str = CONFIG_PATH) -> Settings:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Settings(
        universe=list(raw.get("universe", [])),
        intervals=list(raw.get("intervals", ["1d"])),
        backfill_start=raw.get("backfill", {}).get("start", "2018-01-01"),
        backfill_end=raw.get("backfill", {}).get("end"),
        costs=CostConfig(**raw.get("costs", {})),
        clickhouse=ClickHouseConfig(**raw.get("clickhouse", {})),
    )
