from __future__ import annotations

from dagster import Definitions, OpExecutionContext, job, op

from .assets import bars
from .schedules import daily_update_schedule, run_daily_update


@op
def daily_update_op(context: OpExecutionContext) -> dict:
    counts = run_daily_update()
    context.log.info("daily update wrote: %s", counts)
    return counts


@job(name="daily_update_job")
def daily_update_job():
    daily_update_op()


defs = Definitions(
    assets=[bars],
    jobs=[daily_update_job],
    schedules=[daily_update_schedule],
)
