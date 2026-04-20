from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.config import settings
from app.services.pipeline import NewsPipeline

logger = logging.getLogger(__name__)


class SchedulerService:
    def __init__(self, pipeline: NewsPipeline) -> None:
        self.pipeline = pipeline
        self.scheduler = BackgroundScheduler()
        self.started = False

    def start(self) -> None:
        if self.started or not settings.enable_scheduler:
            return
        self._register_jobs()
        self.scheduler.start()
        self.started = True
        self._schedule_bootstrap_cycle()

    def shutdown(self) -> None:
        if self.started:
            self.scheduler.shutdown(wait=False)
            self.started = False

    def update_collect_interval(self, minutes: int) -> None:
        settings.collect_interval_minutes = minutes
        if self.started:
            self.scheduler.reschedule_job("collect_news_job", trigger="interval", minutes=minutes)

    def status(self) -> dict:
        collect_job = self.scheduler.get_job("collect_news_job") if self.started else None
        analyze_job = self.scheduler.get_job("analyze_issues_job") if self.started else None
        now = datetime.now(timezone.utc)
        return {
            "started": self.started,
            "collect_interval_minutes": settings.collect_interval_minutes,
            "analyze_interval_minutes": settings.analyze_interval_minutes,
            "collect_next_run_at": _iso_or_none(getattr(collect_job, "next_run_time", None)),
            "analyze_next_run_at": _iso_or_none(getattr(analyze_job, "next_run_time", None)),
            "collect_seconds_remaining": _seconds_remaining(getattr(collect_job, "next_run_time", None), now),
            "analyze_seconds_remaining": _seconds_remaining(getattr(analyze_job, "next_run_time", None), now),
        }

    def _register_jobs(self) -> None:
        self.scheduler.add_job(
            self.pipeline.collect_and_refresh,
            "interval",
            minutes=settings.collect_interval_minutes,
            id="collect_news_job",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self.pipeline.analyze_only,
            "interval",
            minutes=settings.analyze_interval_minutes,
            id="analyze_issues_job",
            replace_existing=True,
        )

    def _schedule_bootstrap_cycle(self) -> None:
        self.scheduler.add_job(
            self._run_bootstrap_cycle,
            "date",
            run_date=datetime.now(timezone.utc) + timedelta(seconds=1),
            id="bootstrap_collect_job",
            replace_existing=True,
        )

    def _run_bootstrap_cycle(self) -> None:
        try:
            self.pipeline.collect_and_refresh()
        except Exception as exc:  # pragma: no cover - startup safety
            logger.warning("Bootstrap collection cycle failed: %s", exc)


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _seconds_remaining(next_run_time: datetime | None, now: datetime) -> int | None:
    if next_run_time is None:
        return None
    return max(int((next_run_time - now).total_seconds()), 0)
