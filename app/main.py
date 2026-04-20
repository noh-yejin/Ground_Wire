from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.config import settings
from app.repository import IssueRepository
from app.services.clustering import canonicalize_topic
from app.services.pipeline import NewsPipeline
from app.services.scheduler import SchedulerService
from app.services.slack_reporter import send_issue

repository = IssueRepository()
pipeline = NewsPipeline(repository=repository)
scheduler_service = SchedulerService(pipeline=pipeline)
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))
KST = timezone(timedelta(hours=9))


class SchedulerUpdateRequest(BaseModel):
    collect_interval_minutes: int = Field(ge=5, le=60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    scheduler_service.start()
    yield
    scheduler_service.shutdown()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    issues = repository.list_issues()
    ready_issues = [issue for issue in issues if issue.status.value == "READY"]
    hold_issues = [issue for issue in issues if issue.status.value == "HOLD"]
    priority_issues = [issue for issue in ready_issues if issue.analysis.priority.value == "priority"]
    general_issues = [issue for issue in ready_issues if issue.analysis.priority.value == "general"]
    keyword_hub = _build_keyword_hub(issues)
    market_pulse = _build_market_pulse(issues, minutes=15)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "issues": issues,
            "priority_issues": priority_issues,
            "general_issues": general_issues,
            "hold_issues": hold_issues,
            "keyword_hub": keyword_hub,
            "market_pulse": market_pulse,
            "issues_json": json.dumps(_serialize_dashboard_issues(issues), ensure_ascii=False),
            "app_name": settings.app_name,
            "hold_threshold": settings.hold_threshold,
            "scheduler_enabled": settings.enable_scheduler,
            "runtime_status": _build_runtime_status(),
        },
    )


@app.get("/api/issues")
def list_issues() -> list[dict]:
    issues = repository.list_issues()
    return _serialize_issue_cards(issues)


@app.get("/api/dashboard-data")
def dashboard_data() -> dict:
    issues = repository.list_issues()
    return {
        "issues": _serialize_dashboard_issues(issues),
        "issue_cards": _serialize_issue_cards(issues),
        "keyword_hub": _build_keyword_hub(issues),
        "market_pulse": _build_market_pulse(issues, minutes=15),
        "runtime_status": _build_runtime_status(),
    }


@app.post("/api/jobs/collect")
def collect_news() -> dict:
    article_ids = pipeline.collect_only()
    return {"count": len(article_ids), "articles": article_ids, "runtime_status": _build_runtime_status()}


@app.post("/api/jobs/analyze")
def analyze_issues() -> dict:
    issues = pipeline.analyze_only()
    return {"count": len(issues), "issues": [issue.id for issue in issues], "runtime_status": _build_runtime_status()}


@app.post("/api/pipeline/run")
def run_pipeline() -> dict:
    issues = pipeline.run()
    return {"count": len(issues), "issues": [issue.id for issue in issues], "runtime_status": _build_runtime_status()}


@app.get("/api/system/status")
def system_status() -> dict:
    return _build_runtime_status()


@app.post("/api/system/scheduler")
def update_scheduler(request: SchedulerUpdateRequest) -> dict:
    scheduler_service.update_collect_interval(request.collect_interval_minutes)
    repository.save_job_run(
        "scheduler_update",
        "SUCCESS",
        {"collect_interval_minutes": request.collect_interval_minutes},
    )
    return _build_runtime_status()


@app.post("/api/issues/{issue_id}/report")
def report_issue(issue_id: str) -> dict:
    issue = repository.get_issue(issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="Issue not found")
    return send_issue(issue)


def _build_runtime_status() -> dict:
    latest_collect = repository.get_latest_job_run("collect_news_job")
    latest_analyze = repository.get_latest_job_run("analyze_issues_job")
    scheduler = scheduler_service.status()
    recent_jobs = repository.list_recent_job_runs(limit=6)
    return {
        "scheduler": {
            **scheduler,
            "collect_next_run_display": _format_compact_datetime(scheduler.get("collect_next_run_at")),
            "analyze_next_run_display": _format_compact_datetime(scheduler.get("analyze_next_run_at")),
        },
        "latest_collect": _with_display_time(latest_collect),
        "latest_analyze": _with_display_time(latest_analyze),
        "recent_jobs": recent_jobs,
        "funnel_metrics": _build_funnel_metrics(),
        "article_window_hours": settings.article_window_hours,
        "llm": pipeline.analyzer.debug_status(),
    }


def _build_keyword_hub(issues: list) -> dict:
    repeated_priority = Counter()
    repeated_general = Counter()
    signal_priority = Counter()
    signal_general = Counter()

    for issue in issues:
        keyword_counter = repeated_priority if issue.analysis.priority.value == "priority" else repeated_general
        signal_counter = signal_priority if issue.analysis.priority.value == "priority" else signal_general
        keyword_counter.update(issue.analysis.keywords[:5])
        signal_counter.update(issue.analysis.key_signals[:5])

    return {
        "priority_keywords": repeated_priority.most_common(8),
        "general_keywords": repeated_general.most_common(8),
        "priority_signals": signal_priority.most_common(8),
        "general_signals": signal_general.most_common(8),
    }


def _serialize_dashboard_issues(issues: list) -> list[dict]:
    payload = []
    for issue in issues:
        hourly_counts: dict[str, int] = defaultdict(int)
        for article in issue.articles:
            bucket = article.published_at.strftime("%Y-%m-%d %H:00")
            hourly_counts[bucket] += 1
        payload.append(
            {
                "id": issue.id,
                "topic": _display_topic(issue.topic),
                "status": issue.status.value,
                "priority": issue.analysis.priority.value,
                "keywords": issue.analysis.keywords,
                "key_signals": issue.analysis.key_signals,
                "summary": issue.analysis.summary,
                "sentiment": issue.analysis.sentiment.value,
                "reliability": issue.reliability.value,
                "key_points": issue.analysis.key_points,
                "trend_summary": issue.analysis.trend_summary,
                "risk_points": issue.analysis.risk_points,
                "market_impact": issue.analysis.market_impact.value,
                "policy_risk": issue.analysis.policy_risk.value,
                "volatility_risk": issue.analysis.volatility_risk.value,
                "hold_reason": issue.analysis.hold_reason,
                "reliability_breakdown": {
                    "source_diversity": issue.reliability.source_diversity,
                    "recency": issue.reliability.recency,
                    "evidence_coverage": issue.reliability.evidence_coverage,
                    "cross_source_confirmation": issue.reliability.cross_source_confirmation,
                },
                "articles": [
                    {
                        "source": article.source,
                        "title": article.title,
                        "url": article.url,
                        "published_at": article.published_at.isoformat(),
                        "collected_at": article.collected_at.isoformat() if article.collected_at else None,
                    }
                    for article in issue.articles[:6]
                ],
                "hourly_counts": dict(sorted(hourly_counts.items())),
            }
        )
    return payload


def _serialize_issue_cards(issues: list) -> list[dict]:
    return [
        {
            "id": issue.id,
            "topic": _display_topic(issue.topic),
            "status": issue.status.value,
            "reliability": issue.reliability.value,
            "keywords": issue.keywords,
            "summary": issue.analysis.summary,
            "key_signals": issue.analysis.key_signals,
            "priority": issue.analysis.priority.value,
            "sentiment": issue.analysis.sentiment.value,
            "market_impact": issue.analysis.market_impact.value,
            "policy_risk": issue.analysis.policy_risk.value,
            "volatility_risk": issue.analysis.volatility_risk.value,
            "risk_points": issue.analysis.risk_points,
            "sources": sorted({article.source for article in issue.articles}),
            "article_count": len(issue.articles),
            "updated_at": issue.updated_at.isoformat(),
            "articles": [
                {
                    "source": article.source,
                    "title": article.title,
                    "url": article.url,
                    "published_at": article.published_at.isoformat(),
                    "collected_at": article.collected_at.isoformat() if article.collected_at else None,
                }
                for article in issue.articles
            ],
        }
        for issue in issues
    ]


def _build_market_pulse(issues: list, minutes: int = 15) -> dict:
    threshold = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    rise_now_threshold = datetime.now(timezone.utc) - timedelta(hours=1)
    rise_prev_threshold = datetime.now(timezone.utc) - timedelta(hours=2)
    recent_keyword_counter: Counter[str] = Counter()
    recent_signal_counter: Counter[str] = Counter()
    recent_topic_counter: Counter[str] = Counter()
    rising_now_counter: Counter[str] = Counter()
    rising_prev_counter: Counter[str] = Counter()
    recent_issue_scores: list[tuple[float, object]] = []
    recent_article_count = 0

    for issue in issues:
        recent_articles = []
        for article in issue.articles:
            timestamp = article.collected_at or article.published_at
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
            if timestamp >= threshold:
                recent_articles.append(article)
            for keyword in issue.analysis.keywords[:4]:
                if timestamp >= rise_now_threshold:
                    rising_now_counter.update([keyword])
                elif timestamp >= rise_prev_threshold:
                    rising_prev_counter.update([keyword])

        if not recent_articles:
            continue

        recent_article_count += len(recent_articles)
        recent_keyword_counter.update(issue.analysis.keywords[:4])
        recent_signal_counter.update(issue.analysis.key_signals[:4])
        recent_topic_counter.update(_display_topic(issue.topic) for _ in recent_articles)

        issue_score = issue.reliability.value + (0.08 if issue.status.value == "READY" else 0)
        issue_score += min(len(recent_articles) * 0.03, 0.18)
        recent_issue_scores.append((issue_score, issue))

    if recent_article_count == 0 and issues:
        return _build_market_pulse_fallback(issues, minutes)

    focus_issue = max(recent_issue_scores, default=(0, None), key=lambda item: item[0])[1]
    promotion_candidate = _pick_promotion_candidate(issues)
    rising_keyword = _pick_rising_keyword(rising_now_counter, rising_prev_counter)
    summary = _summarize_market_pulse(
        recent_signal_counter=recent_signal_counter,
        recent_keyword_counter=recent_keyword_counter,
        recent_article_count=recent_article_count,
        minutes=minutes,
    )

    return {
        "window_minutes": minutes,
        "recent_article_count": recent_article_count,
        "summary": summary,
        "summary_parts": _build_market_pulse_summary_parts(
            recent_article_count=recent_article_count,
            top_signals=recent_signal_counter.most_common(2),
            top_keywords=recent_keyword_counter.most_common(3),
        ),
        "top_signals": recent_signal_counter.most_common(6),
        "top_keywords": recent_keyword_counter.most_common(8),
        "top_topics": recent_topic_counter.most_common(4),
        "focus_issue_id": focus_issue.id if focus_issue else None,
        "focus_issue_topic": _display_topic(focus_issue.topic) if focus_issue else None,
        "rising_keyword": rising_keyword,
        "promotion_candidate": promotion_candidate,
    }


def _build_funnel_metrics() -> dict:
    now = datetime.now(timezone.utc)
    runs = repository.list_recent_job_runs(limit=200)
    collect_runs = []
    analyze_runs = []
    for run in runs:
        created_at = run.get("created_at")
        if not created_at:
            continue
        try:
            dt = datetime.fromisoformat(created_at)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        if dt < now - timedelta(hours=24):
            continue
        if run["job_name"] == "collect_news_job":
            collect_runs.append(run)
        elif run["job_name"] == "analyze_issues_job":
            analyze_runs.append(run)

    latest_collect = collect_runs[0]["details"] if collect_runs else {}
    latest_analyze = analyze_runs[0]["details"] if analyze_runs else {}
    avg_collected = round(sum(run["details"].get("stored_count", 0) for run in collect_runs) / max(len(collect_runs), 1), 1)
    avg_clustered = round(sum(run["details"].get("issue_count", 0) for run in analyze_runs) / max(len(analyze_runs), 1), 1)
    ready_rates = [
        run["details"].get("ready_count", 0) / max(run["details"].get("issue_count", 1), 1)
        for run in analyze_runs
        if run["details"].get("issue_count", 0) > 0
    ]
    avg_ready_rate = round((sum(ready_rates) / max(len(ready_rates), 1)) * 100, 1) if ready_rates else 0.0
    return {
        "current_collected": latest_collect.get("stored_count", 0),
        "current_clustered": latest_analyze.get("issue_count", 0),
        "current_ready": latest_analyze.get("ready_count", 0),
        "current_hold": max(latest_analyze.get("issue_count", 0) - latest_analyze.get("ready_count", 0), 0),
        "avg_collected_24h": avg_collected,
        "avg_clustered_24h": avg_clustered,
        "avg_ready_rate_24h": avg_ready_rate,
    }


def _summarize_market_pulse(
    recent_signal_counter: Counter[str],
    recent_keyword_counter: Counter[str],
    recent_article_count: int,
    minutes: int,
) -> str:
    if recent_article_count == 0:
        return f"최근 {minutes}분 기준 새로 반영된 기사 흐름이 아직 충분하지 않습니다."

    top_signals = [label for label, _ in recent_signal_counter.most_common(2)]
    top_keywords = [label for label, _ in recent_keyword_counter.most_common(3)]

    if top_signals:
        return (
            f"최근 {minutes}분 동안 {recent_article_count}건이 반영됐고, "
            f"지금은 {' · '.join(top_signals)} 시그널이 중심입니다. "
            f"반복 키워드는 {' · '.join(top_keywords or top_signals)}입니다."
        )

    return (
        f"최근 {minutes}분 동안 {recent_article_count}건이 반영됐고, "
        f"반복 키워드는 {' · '.join(top_keywords)}입니다."
    )


def _build_market_pulse_fallback(issues: list, minutes: int) -> dict:
    keyword_counter: Counter[str] = Counter()
    signal_counter: Counter[str] = Counter()
    topic_counter: Counter[str] = Counter()

    for issue in issues[:12]:
        keyword_counter.update(issue.analysis.keywords[:4])
        signal_counter.update(issue.analysis.key_signals[:4])
        topic_counter.update([_display_topic(issue.topic)])

    focus_issue = issues[0] if issues else None
    promotion_candidate = _pick_promotion_candidate(issues)
    return {
        "window_minutes": minutes,
        "recent_article_count": 0,
        "summary": f"최근 {minutes}분 내 신규 반영은 적지만, 직전 분석 기준으로는 {' · '.join(label for label, _ in signal_counter.most_common(2)) or '핵심 이슈'} 흐름이 이어지고 있습니다.",
        "summary_parts": _build_market_pulse_summary_parts(
            recent_article_count=0,
            top_signals=signal_counter.most_common(2),
            top_keywords=keyword_counter.most_common(3),
        ),
        "top_signals": signal_counter.most_common(6),
        "top_keywords": keyword_counter.most_common(8),
        "top_topics": topic_counter.most_common(4),
        "focus_issue_id": focus_issue.id if focus_issue else None,
        "focus_issue_topic": _display_topic(focus_issue.topic) if focus_issue else None,
        "rising_keyword": None,
        "promotion_candidate": promotion_candidate,
    }


def _pick_rising_keyword(now_counter: Counter[str], prev_counter: Counter[str]) -> dict | None:
    best: tuple[str, int, int, int] | None = None
    for keyword, now_count in now_counter.items():
        previous = prev_counter.get(keyword, 0)
        delta = now_count - previous
        if delta <= 0 and now_count < 2:
            continue
        score = delta * 10 + now_count
        if best is None or score > best[3]:
            best = (keyword, now_count, previous, score)
    if best is None:
        return None
    return {"keyword": best[0], "current_count": best[1], "previous_count": best[2], "delta": best[1] - best[2]}


def _pick_promotion_candidate(issues: list) -> dict | None:
    candidates: list[tuple[float, object]] = []
    now = datetime.now(timezone.utc)
    for issue in issues:
        if issue.status.value != "HOLD":
            continue
        article_count = len(issue.articles)
        signal_count = len(issue.analysis.key_signals)
        unique_sources = len({article.source for article in issue.articles})
        recent_30m_count = 0
        for article in issue.articles:
            timestamp = article.collected_at or article.published_at
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
            if timestamp >= now - timedelta(minutes=30):
                recent_30m_count += 1
        score = issue.reliability.value * 58
        score += issue.reliability.cross_source_confirmation * 14
        score += issue.reliability.evidence_coverage * 12
        score += min(article_count * 3.5, 14)
        score += min(signal_count * 4.5, 13.5)
        score += min(unique_sources * 2.5, 10)
        score += min(recent_30m_count * 4.0, 12)
        hold_reason = issue.analysis.hold_reason or ""
        if "독립 출처 부족" in hold_reason:
            score -= 11
        elif "기사 수 부족" in hold_reason:
            score -= 7
        elif "신뢰도 점수 부족" in hold_reason:
            score -= 5
        candidates.append((round(max(min(score, 99), 0), 1), issue))
    if not candidates:
        return None
    top_score, top_issue = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return {
        "issue_id": top_issue.id,
        "topic": _display_topic(top_issue.topic),
        "score": top_score,
        "recent_30m_count": sum(
            1
            for article in top_issue.articles
            if (
                ((article.collected_at or article.published_at).replace(tzinfo=timezone.utc)
                 if (article.collected_at or article.published_at).tzinfo is None
                 else (article.collected_at or article.published_at).astimezone(timezone.utc))
                >= now - timedelta(minutes=30)
            )
        ),
        "reason_text": _build_promotion_reason(top_issue),
    }


def _build_market_pulse_summary_parts(
    recent_article_count: int,
    top_signals: list[tuple[str, int]],
    top_keywords: list[tuple[str, int]],
) -> dict:
    return {
        "article_label": f"{recent_article_count}건 반영" if recent_article_count else "신규 반영 적음",
        "signal_label": " · ".join(label for label, _ in top_signals) or "시그널 대기",
        "keyword_label": " · ".join(label for label, _ in top_keywords) or "키워드 대기",
    }


def _build_promotion_reason(issue) -> str:
    recent_count = 0
    now = datetime.now(timezone.utc)
    for article in issue.articles:
        timestamp = article.collected_at or article.published_at
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        if timestamp >= now - timedelta(minutes=30):
            recent_count += 1
    return (
        f"최근 30분 신규 {recent_count}건, "
        f"출처 {len({article.source for article in issue.articles})}개, "
        f"시그널 {len(issue.analysis.key_signals)}개"
    )


def _with_display_time(job: dict | None) -> dict | None:
    if job is None:
        return None
    return {**job, "created_at_display": _format_compact_datetime(job.get("created_at"))}


def _format_compact_datetime(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(KST)
    return dt.strftime("%m-%d %H:%M")


def _display_topic(value: str) -> str:
    if not value:
        return value
    return canonicalize_topic(value)
