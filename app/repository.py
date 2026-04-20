from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.models import (
    AnalysisResult,
    Article,
    EvidenceSnippet,
    ImpactLabel,
    Issue,
    IssuePriority,
    IssueStatus,
    ReliabilityScore,
    RiskLevel,
    SentimentLabel,
)


class IssueRepository:
    def __init__(self, database_path: str | None = None) -> None:
        self.database_path = database_path or settings.database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _initialize(self) -> None:
        Path(self.database_path).touch(exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    language TEXT NOT NULL,
                    collected_at TEXT NOT NULL,
                    content_quality REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS issues (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reliability REAL NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feed_states (
                    feed_url TEXT PRIMARY KEY,
                    etag TEXT,
                    modified TEXT,
                    last_checked_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_articles(self, articles: list[Article]) -> None:
        with self._connect() as conn:
            for article in articles:
                conn.execute(
                    """
                    INSERT INTO articles (
                        id, source, published_at, url, title, content, language, collected_at, content_quality
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        source = excluded.source,
                        published_at = excluded.published_at,
                        url = excluded.url,
                        title = excluded.title,
                        content = excluded.content,
                        language = excluded.language,
                        collected_at = excluded.collected_at,
                        content_quality = excluded.content_quality
                    """,
                    (
                        article.id,
                        article.source,
                        article.published_at.isoformat(),
                        article.url,
                        article.title,
                        article.content,
                        article.language,
                        (article.collected_at or article.published_at).isoformat(),
                        article.content_quality,
                    ),
                )
            conn.commit()

    def list_articles(self) -> list[Article]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, source, published_at, url, content, language, collected_at, content_quality
                FROM articles
                ORDER BY published_at DESC
                """
            ).fetchall()
        return [
            Article(
                id=row[0],
                title=row[1],
                source=row[2],
                published_at=datetime.fromisoformat(row[3]),
                url=row[4],
                content=row[5],
                language=row[6],
                collected_at=datetime.fromisoformat(row[7]),
                content_quality=row[8],
            )
            for row in rows
        ]

    def save_job_run(self, job_name: str, status: str, details: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_runs (job_name, status, details, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (job_name, status, json.dumps(details, ensure_ascii=False), datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_latest_job_run(self, job_name: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_name, status, details, created_at
                FROM job_runs
                WHERE job_name = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (job_name,),
            ).fetchone()
        if not row:
            return None
        return {
            "job_name": row[0],
            "status": row[1],
            "details": json.loads(row[2]),
            "created_at": row[3],
        }

    def list_recent_job_runs(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_name, status, details, created_at
                FROM job_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "job_name": row[0],
                "status": row[1],
                "details": json.loads(row[2]),
                "created_at": row[3],
            }
            for row in rows
        ]

    def get_feed_state(self, feed_url: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT etag, modified, last_checked_at FROM feed_states WHERE feed_url = ?",
                (feed_url,),
            ).fetchone()
        if not row:
            return None
        return {"etag": row[0], "modified": row[1], "last_checked_at": row[2]}

    def save_feed_state(self, feed_url: str, etag: str | None, modified: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feed_states (feed_url, etag, modified, last_checked_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(feed_url) DO UPDATE SET
                    etag = excluded.etag,
                    modified = excluded.modified,
                    last_checked_at = excluded.last_checked_at
                """,
                (feed_url, etag, modified, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def save_issues(self, issues: list[Issue]) -> None:
        with self._connect() as conn:
            for issue in issues:
                payload = json.dumps(self._serialize_issue(issue), ensure_ascii=False)
                conn.execute(
                    """
                    INSERT INTO issues (id, topic, status, reliability, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        topic = excluded.topic,
                        status = excluded.status,
                        reliability = excluded.reliability,
                        payload = excluded.payload,
                        updated_at = excluded.updated_at
                    """,
                    (
                        issue.id,
                        issue.topic,
                        issue.status.value,
                        issue.reliability.value,
                        payload,
                        issue.updated_at.isoformat(),
                    ),
                )
            conn.commit()

    def list_issues(self) -> list[Issue]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM issues ORDER BY reliability DESC, updated_at DESC"
            ).fetchall()
        return [self._deserialize_issue(json.loads(row[0])) for row in rows]

    def get_issue(self, issue_id: str) -> Issue | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM issues WHERE id = ?", (issue_id,)).fetchone()
        if not row:
            return None
        return self._deserialize_issue(json.loads(row[0]))

    def _serialize_issue(self, issue: Issue) -> dict:
        return {
            "id": issue.id,
            "topic": issue.topic,
            "keywords": issue.keywords,
            "status": issue.status.value,
            "updated_at": issue.updated_at.isoformat(),
            "articles": [
                {
                    **asdict(article),
                    "published_at": article.published_at.isoformat(),
                    "collected_at": (article.collected_at or article.published_at).isoformat(),
                }
                for article in issue.articles
            ],
            "evidence": [asdict(item) for item in issue.evidence],
            "reliability": {
                **asdict(issue.reliability),
            },
            "analysis": {
                "summary": issue.analysis.summary,
                "keywords": issue.analysis.keywords,
                "key_signals": issue.analysis.key_signals,
                "key_points": issue.analysis.key_points,
                "trend_summary": issue.analysis.trend_summary,
                "sentiment": issue.analysis.sentiment.value,
                "market_impact": issue.analysis.market_impact.value,
                "policy_risk": issue.analysis.policy_risk.value,
                "volatility_risk": issue.analysis.volatility_risk.value,
                "risk_points": issue.analysis.risk_points,
                "grounded": issue.analysis.grounded,
                "priority": issue.analysis.priority.value,
                "hold_reason": issue.analysis.hold_reason,
            },
        }

    def _deserialize_issue(self, data: dict) -> Issue:
        articles = [
            Article(
                id=item["id"],
                title=item["title"],
                source=item["source"],
                published_at=datetime.fromisoformat(item["published_at"]),
                url=item["url"],
                content=item["content"],
                language=item.get("language", "ko"),
                collected_at=datetime.fromisoformat(item["collected_at"])
                if item.get("collected_at")
                else None,
                content_quality=item.get("content_quality", 0.0),
            )
            for item in data["articles"]
        ]
        evidence = [EvidenceSnippet(**item) for item in data["evidence"]]
        reliability = ReliabilityScore(**data["reliability"])
        legacy_summary = data.get("summary")
        legacy_keywords = data.get("keywords", [])
        analysis_data = data.get("analysis")

        if analysis_data is None:
            analysis = AnalysisResult(
                summary=legacy_summary or "기존 데이터에는 상세 분석 필드가 없습니다.",
                keywords=legacy_keywords,
                key_signals=[],
                key_points=[],
                trend_summary="기존 데이터에는 추이 설명이 없습니다.",
                sentiment=SentimentLabel.NEUTRAL,
                market_impact=ImpactLabel.NEUTRAL,
                policy_risk=RiskLevel.MEDIUM,
                volatility_risk=RiskLevel.MEDIUM,
                risk_points=["구버전 이슈 데이터라 상세 분석 정보가 없습니다."],
                grounded=False,
                priority=IssuePriority.GENERAL,
                hold_reason="legacy_issue_payload",
            )
        else:
            analysis = AnalysisResult(
                summary=analysis_data["summary"],
                keywords=analysis_data["keywords"],
                key_signals=analysis_data.get("key_signals", []),
                key_points=analysis_data.get("key_points", []),
                trend_summary=analysis_data.get("trend_summary", "추이 설명이 아직 생성되지 않았습니다."),
                sentiment=SentimentLabel(analysis_data["sentiment"]),
                market_impact=ImpactLabel(analysis_data.get("market_impact", analysis_data["sentiment"])),
                policy_risk=RiskLevel(analysis_data.get("policy_risk", "medium")),
                volatility_risk=RiskLevel(analysis_data.get("volatility_risk", "medium")),
                risk_points=analysis_data["risk_points"],
                grounded=analysis_data["grounded"],
                priority=IssuePriority(analysis_data.get("priority", "general")),
                hold_reason=analysis_data.get("hold_reason"),
            )
        return Issue(
            id=data["id"],
            topic=data["topic"],
            keywords=data["keywords"],
            articles=articles,
            evidence=evidence,
            reliability=reliability,
            analysis=analysis,
            status=IssueStatus(data["status"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
