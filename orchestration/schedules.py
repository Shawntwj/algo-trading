from __future__ import annotations

from dagster import (
    DefaultScheduleStatus,
    RunRequest,
    ScheduleEvaluationContext,
    schedule,
)

from config import load_settings
from data import update_latest


@schedule(
    cron_schedule="30 21 * * 1-5",  # 21:30 UTC weekdays — after US market close
    job_name="daily_update_job",
    default_status=DefaultScheduleStatus.STOPPED,
    execution_timezone="UTC",
)
def daily_update_schedule(context: ScheduleEvaluationContext):
    return RunRequest(run_key=context.scheduled_execution_time.isoformat())


def run_daily_update() -> dict[str, int]:
    """Plain entry point used by the scheduled job; also callable directly."""
    settings = load_settings()
    return update_latest(settings.universe, interval=settings.intervals[0])
