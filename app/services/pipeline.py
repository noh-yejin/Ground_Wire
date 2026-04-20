from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from hashlib import sha1

from app.config import settings
from app.models import Issue
from app.repository import IssueRepository
from app.services.collection import NewsCollector
from app.services.clustering import canonical_topic_key, cluster_articles, _extract_keywords, label_topic
from app.services.llm_analyzer import LLMAnalyzer
from app.services.preprocessing import preprocess_articles
from app.services.rag import EvidenceRetriever
from app.services.reliability import score_issue
from app.services.source_normalizer import is_trusted_ready_source


class NewsPipeline:
    def __init__(self, repository: IssueRepository | None = None) -> None:
        self.repository = repository or IssueRepository()
        self.collector = NewsCollector(repository=self.repository)
        self.retriever = EvidenceRetriever()
        self.analyzer = LLMAnalyzer()

    def collect_only(self) -> list[str]:
        raw_articles = self.collector.collect()
        articles = preprocess_articles(raw_articles)
        self.repository.save_articles(articles)
        self.repository.save_job_run(
            "collect_news_job",
            "SUCCESS",
            {"raw_count": len(raw_articles), "stored_count": len(articles)},
        )
        return [article.id for article in articles]

    def analyze_only(self) -> list[Issue]:
        articles = self.repository.list_articles()
        if not articles:
            articles = preprocess_articles(self.collector.collect())
            self.repository.save_articles(articles)

        grouped_articles = _merge_equivalent_groups(cluster_articles(articles))
        issues: list[Issue] = []

        for group in grouped_articles:
            topic = _derive_topic(group)
            keywords = _extract_keywords(" ".join(article.title for article in group))[:5]
            evidence = self.retriever.retrieve(group)
            reliability = score_issue(group, evidence)
            hold_reason = _build_hold_reason(group, evidence, reliability)
            analysis = self.analyzer.analyze(
                topic=topic,
                articles=group,
                evidence=evidence,
                reliability=reliability,
                hold_reason=hold_reason,
            )
            issue_id = sha1(topic.encode("utf-8")).hexdigest()[:10]
            issues.append(
                Issue(
                    id=issue_id,
                    topic=topic,
                    keywords=keywords,
                    articles=group,
                    evidence=evidence,
                    reliability=reliability,
                    analysis=analysis,
                    status=_status_from_hold_reason(hold_reason),
                    updated_at=datetime.utcnow(),
                )
            )

        self.repository.save_issues(issues)
        self.repository.save_job_run(
            "analyze_issues_job",
            "SUCCESS",
            {"issue_count": len(issues), "ready_count": sum(issue.status.value == "READY" for issue in issues)},
        )
        return issues

    def collect_and_refresh(self) -> list[Issue]:
        article_ids = self.collect_only()
        if not article_ids:
            return []
        return self.analyze_only()

    def run(self) -> list[Issue]:
        self.collect_only()
        return self.analyze_only()


def _derive_topic(group: list) -> str:
    return label_topic(group)


def _merge_equivalent_groups(groups: list[list]) -> list[list]:
    merged: dict[str, list] = defaultdict(list)
    for group in groups:
        merged[canonical_topic_key(group)].extend(group)

    result: list[list] = []
    for articles in merged.values():
        deduped = {article.id: article for article in articles}
        result.append(list(deduped.values()))
    return result


def _build_hold_reason(group: list, evidence: list, reliability) -> str | None:
    unique_sources = len({article.source for article in group})
    if len(group) < settings.min_articles_per_issue:
        return f"기사 수 부족: {len(group)}건"
    if unique_sources < settings.min_unique_sources:
        return f"독립 출처 부족: {unique_sources}개"
    if len({item.source for item in evidence}) < settings.min_unique_sources:
        return "근거 문서의 출처 다양성이 부족합니다."
    if not any(is_trusted_ready_source(item.source) for item in evidence):
        return "신뢰 가능한 핵심 출처 근거가 부족합니다."
    if reliability.value < settings.hold_threshold:
        return f"신뢰도 점수 부족: {reliability.value}"
    return None


def _status_from_hold_reason(hold_reason: str | None):
    from app.models import IssueStatus

    return IssueStatus.HOLD if hold_reason else IssueStatus.READY
