from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import sha1
import logging
import json

from app.config import settings
from app.models import Issue
from app.repository import IssueRepository
from app.services.collection import NewsCollector
from app.services.clustering import canonical_topic_key, cluster_articles, _extract_keywords, label_topic
from app.services.crawling import is_google_news_url
from app.services.llm_analyzer import LLMAnalyzer
from app.services.preprocessing import preprocess_articles
from app.services.rag import EvidenceRetriever
from app.services.reliability import score_issue
from app.services.source_normalizer import is_trusted_ready_source

logger = logging.getLogger(__name__)


class NewsPipeline:
    def __init__(self, repository: IssueRepository | None = None) -> None:
        self.repository = repository or IssueRepository()
        self.collector = NewsCollector(repository=self.repository)
        self.retriever = EvidenceRetriever()
        self.analyzer = LLMAnalyzer()

    def collect_only(self) -> list[str]:
        try:
            raw_articles = self.collector.collect()
            articles = preprocess_articles(raw_articles)
            articles = self.collector.resolve_article_links(articles)
            articles = preprocess_articles(articles)
            self.repository.save_articles(articles)
            self.repository.save_job_run(
                "collect_news_job",
                "SUCCESS",
                {"raw_count": len(raw_articles), "stored_count": len(articles)},
            )
            return [article.id for article in articles]
        except Exception as exc:
            self._record_failure("collect_news_job", exc)
            raise

    def analyze_only(self) -> list[Issue]:
        try:
            articles = preprocess_articles(
                _without_placeholder_links(_within_article_window(self.repository.list_articles()))
            )
            if articles:
                self.repository.save_articles(articles)
            else:
                articles = preprocess_articles(self.collector.collect())
                articles = self.collector.resolve_article_links(articles)
                articles = preprocess_articles(articles)
                self.repository.save_articles(articles)

            grouped_articles = _merge_equivalent_groups(cluster_articles(articles))
            issues = self._analyze_groups(grouped_articles, articles)
            issues = self._run_second_pass_reviews(issues)

            self.repository.save_issues(issues)
            self.repository.save_job_run(
                "analyze_issues_job",
                "SUCCESS",
                {"issue_count": len(issues), "ready_count": sum(issue.status.value == "READY" for issue in issues)},
            )
            return issues
        except Exception as exc:
            self._record_failure("analyze_issues_job", exc)
            raise

    def collect_and_refresh(self) -> list[Issue]:
        article_ids = self.collect_only()
        if not article_ids:
            return []
        return self.analyze_only()

    def run(self) -> list[Issue]:
        self.collect_only()
        return self.analyze_only()

    def _record_failure(self, job_name: str, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        logger.exception("%s failed: %s", job_name, message)
        self.repository.save_job_run(job_name, "FAILED", {"error": message})

    def _analyze_groups(self, grouped_articles: list[list], corpus_articles: list) -> list[Issue]:
        worker_count = min(settings.max_parallel_issue_analysis, max(len(grouped_articles), 1))
        if worker_count <= 1 or len(grouped_articles) <= 1:
            return [self._analyze_group(group, corpus_articles) for group in grouped_articles]
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(lambda group: self._analyze_group(group, corpus_articles), grouped_articles))

    def _analyze_group(self, group: list, corpus_articles: list) -> Issue:
        topic = _derive_topic(group)
        keywords = _extract_keywords(" ".join(article.title for article in group))[:5]
        evidence = self.retriever.retrieve(group)
        reliability = score_issue(group, evidence)
        hold_reason = _build_hold_reason(group, evidence, reliability)
        cache_key = _issue_analysis_cache_key(topic, group, evidence, reliability, hold_reason)
        analysis = self.repository.get_issue_analysis_cache(cache_key)
        if analysis is not None:
            analysis = _with_cache_metadata(analysis)
        elif _should_use_lightweight_hold_path(group, evidence, reliability, hold_reason):
            analysis = self.analyzer.analyze_lightweight(
                topic=topic,
                articles=group,
                evidence=evidence,
                reliability=reliability,
                hold_reason=hold_reason or "보류 기준 충족",
            )
            self.repository.save_issue_analysis_cache(cache_key, analysis)
        else:
            analysis = self.analyzer.analyze(
                topic=topic,
                articles=group,
                evidence=evidence,
                reliability=reliability,
                hold_reason=hold_reason,
                corpus_articles=corpus_articles,
            )
            self.repository.save_issue_analysis_cache(cache_key, analysis)
        effective_hold_reason = analysis.hold_reason or hold_reason
        issue_id = sha1(topic.encode("utf-8")).hexdigest()[:10]
        return Issue(
            id=issue_id,
            topic=topic,
            keywords=keywords,
            articles=group,
            evidence=evidence,
            reliability=reliability,
            analysis=analysis,
            status=_status_from_hold_reason(effective_hold_reason),
            updated_at=datetime.utcnow(),
        )

    def _run_second_pass_reviews(self, issues: list[Issue]) -> list[Issue]:
        reviewed = 0
        updated: list[Issue] = []
        for issue in issues:
            if reviewed >= settings.max_second_pass_reviews:
                updated.append(issue)
                continue
            if not _is_second_pass_candidate(issue):
                updated.append(issue)
                continue
            promoted = self.analyzer.review_hold_for_promotion(
                topic=issue.topic,
                articles=issue.articles,
                evidence=issue.evidence,
                reliability=issue.reliability,
                analysis=issue.analysis,
            )
            cache_key = _issue_analysis_cache_key(
                issue.topic,
                issue.articles,
                issue.evidence,
                issue.reliability,
                issue.analysis.hold_reason,
            )
            if promoted is None:
                reviewed_issue = Issue(
                    id=issue.id,
                    topic=issue.topic,
                    keywords=issue.keywords,
                    articles=issue.articles,
                    evidence=issue.evidence,
                    reliability=issue.reliability,
                    analysis=_mark_second_pass_reviewed(issue.analysis, promoted=False),
                    status=issue.status,
                    updated_at=datetime.utcnow(),
                )
                self.repository.save_issue_analysis_cache(cache_key, reviewed_issue.analysis)
                updated.append(reviewed_issue)
                reviewed += 1
                continue
            reviewed += 1
            self.repository.save_issue_analysis_cache(cache_key, promoted)
            updated.append(
                Issue(
                    id=issue.id,
                    topic=issue.topic,
                    keywords=issue.keywords,
                    articles=issue.articles,
                    evidence=issue.evidence,
                    reliability=issue.reliability,
                    analysis=promoted,
                    status=_status_from_hold_reason(promoted.hold_reason),
                    updated_at=datetime.utcnow(),
                )
            )
        return updated


def _derive_topic(group: list) -> str:
    return label_topic(group)


def _within_article_window(articles: list) -> list:
    threshold = datetime.now(timezone.utc) - timedelta(hours=settings.article_window_hours)
    selected = []
    for article in articles:
        published_at = article.published_at
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        else:
            published_at = published_at.astimezone(timezone.utc)
        if published_at >= threshold:
            selected.append(article)
    return selected


def _without_placeholder_links(articles: list) -> list:
    return [
        article
        for article in articles
        if not is_google_news_url(article.url)
        and "example.com" not in article.url
    ]


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
    if settings.require_trusted_ready_source and not any(is_trusted_ready_source(item.source) for item in evidence):
        return "신뢰 가능한 핵심 출처 근거가 부족합니다."
    if reliability.value < settings.hold_threshold:
        return f"신뢰도 점수 부족: {reliability.value}"
    return None


def _status_from_hold_reason(hold_reason: str | None):
    from app.models import IssueStatus

    return IssueStatus.HOLD if hold_reason else IssueStatus.READY


ANALYSIS_CACHE_VERSION = "issue-analysis-v3"


def _issue_analysis_cache_key(
    topic: str,
    group: list,
    evidence: list,
    reliability,
    hold_reason: str | None,
) -> str:
    payload = {
        "version": ANALYSIS_CACHE_VERSION,
        "model": settings.openai_model,
        "topic": topic,
        "hold_reason": hold_reason,
        "reliability": round(reliability.value, 3),
        "articles": [
            {
                "id": article.id,
                "title": article.title,
                "source": article.source,
                "published_at": article.published_at.isoformat(),
            }
            for article in group[:10]
        ],
        "evidence": [
            {
                "article_id": item.article_id,
                "source": item.source,
                "quote": item.quote[:180],
            }
            for item in evidence[:8]
        ],
    }
    return sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _should_use_lightweight_hold_path(group: list, evidence: list, reliability, hold_reason: str | None) -> bool:
    if not hold_reason:
        return False
    unique_sources = len({article.source for article in group})
    evidence_sources = len({item.source for item in evidence})
    if len(group) < settings.min_articles_per_issue:
        return True
    if unique_sources < settings.min_unique_sources:
        return True
    if evidence_sources < settings.min_unique_sources:
        return True
    if reliability.value < max(settings.hold_threshold - 0.08, 0.5):
        return True
    if "신뢰 가능한 핵심 출처 근거가 부족합니다." in hold_reason and reliability.value < settings.hold_threshold + 0.03:
        return True
    return False


def _with_cache_metadata(analysis):
    details = dict(analysis.grounding_details or {})
    llm = dict(details.get("llm", {}))
    llm["cache_hit"] = True
    details["llm"] = llm
    analysis.grounding_details = details
    return analysis


def _is_second_pass_candidate(issue: Issue) -> bool:
    if issue.status.value != "HOLD":
        return False
    details = issue.analysis.grounding_details or {}
    llm = details.get("llm", {})
    if llm.get("analysis_mode") != "combined_remote":
        return False
    if llm.get("second_pass_reviewed"):
        return False
    hold_reason = issue.analysis.hold_reason or ""
    if "grounding 검증 부족" not in hold_reason:
        return False
    grounding = details.get("grounding", {})
    decision = details.get("decision", {})
    grounded_ratio = float(grounding.get("grounded_ratio", 0.0) or 0.0)
    issue_score = float(grounding.get("issue_score", 0.0) or 0.0)
    contradiction_ratio = float(grounding.get("contradiction_ratio", 0.0) or 0.0)
    ready_claim_count = int(decision.get("ready_claim_count", 0) or 0)
    return grounded_ratio >= 0.65 and issue_score >= 0.73 and contradiction_ratio == 0.0 and ready_claim_count >= 3


def _mark_second_pass_reviewed(analysis, promoted: bool):
    details = dict(analysis.grounding_details or {})
    llm = dict(details.get("llm", {}))
    llm["second_pass_reviewed"] = True
    llm["second_pass_promoted"] = promoted
    details["llm"] = llm
    analysis.grounding_details = details
    return analysis
