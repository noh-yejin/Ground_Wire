import pytest
from datetime import datetime, timezone

from app.models import AnalysisResult, Article, EvidenceSnippet, ImpactLabel, IssuePriority, ReliabilityScore, RiskLevel, SentimentLabel
from app.repository import IssueRepository
from app.sample_data import load_sample_articles
from app.services import pipeline as pipeline_module
from app.services.pipeline import NewsPipeline


def test_collect_only_records_failed_job_run(tmp_path, monkeypatch):
    repository = IssueRepository(database_path=str(tmp_path / "pipeline.db"))
    pipeline = NewsPipeline(repository=repository)

    def boom():
        raise RuntimeError("collect exploded")

    monkeypatch.setattr(pipeline.collector, "collect", boom)

    with pytest.raises(RuntimeError, match="collect exploded"):
        pipeline.collect_only()

    latest = repository.get_latest_job_run("collect_news_job")
    assert latest is not None
    assert latest["status"] == "FAILED"
    assert "RuntimeError: collect exploded" == latest["details"]["error"]


def test_analyze_only_records_failed_job_run(tmp_path, monkeypatch):
    repository = IssueRepository(database_path=str(tmp_path / "pipeline.db"))
    pipeline = NewsPipeline(repository=repository)

    sample_articles = load_sample_articles()
    monkeypatch.setattr(repository, "list_articles", lambda: sample_articles)
    monkeypatch.setattr(pipeline_module, "_within_article_window", lambda items: items)
    monkeypatch.setattr(pipeline_module, "_without_placeholder_links", lambda items: items)
    monkeypatch.setattr(pipeline_module, "preprocess_articles", lambda items: items)
    monkeypatch.setattr(
        pipeline.retriever,
        "retrieve",
        lambda group: [
            EvidenceSnippet(
                article_id=group[0].id,
                source=group[0].source,
                quote=group[0].content,
                url=group[0].url,
            )
        ],
    )

    def boom(**_kwargs):
        raise RuntimeError("analysis exploded")

    monkeypatch.setattr(pipeline.analyzer, "analyze", boom)

    with pytest.raises(RuntimeError, match="analysis exploded"):
        pipeline.analyze_only()

    latest = repository.get_latest_job_run("analyze_issues_job")
    assert latest is not None
    assert latest["status"] == "FAILED"
    assert "RuntimeError: analysis exploded" == latest["details"]["error"]


def test_analyze_only_parallelizes_multiple_groups(tmp_path, monkeypatch):
    repository = IssueRepository(database_path=str(tmp_path / "pipeline.db"))
    pipeline = NewsPipeline(repository=repository)
    article_one = Article(
        id="1",
        title="반도체 투자 확대",
        source="연합뉴스",
        published_at=datetime.now(timezone.utc),
        url="https://example.com/1",
        content="정부가 반도체 투자 확대 계획을 공개했다.",
    )
    article_two = Article(
        id="2",
        title="금리 인하 기대 약화",
        source="Reuters",
        published_at=datetime.now(timezone.utc),
        url="https://example.com/2",
        content="미국 CPI 발표 이후 금리 인하 기대가 약화됐다.",
    )

    monkeypatch.setattr(repository, "list_articles", lambda: [article_one, article_two])
    monkeypatch.setattr(pipeline_module, "_within_article_window", lambda items: items)
    monkeypatch.setattr(pipeline_module, "_without_placeholder_links", lambda items: items)
    monkeypatch.setattr(pipeline_module, "preprocess_articles", lambda items: items)
    monkeypatch.setattr(pipeline_module, "cluster_articles", lambda _articles: [[article_one], [article_two]])
    monkeypatch.setattr(
        pipeline.retriever,
        "retrieve",
        lambda group: [EvidenceSnippet(article_id=group[0].id, source=group[0].source, quote=group[0].content, url=group[0].url)],
    )
    monkeypatch.setattr(
        pipeline_module,
        "score_issue",
        lambda *_args, **_kwargs: ReliabilityScore(0.8, 0.8, 0.8, 0.8, 0.8),
    )
    monkeypatch.setattr(
        pipeline.analyzer,
        "analyze",
        lambda **_kwargs: AnalysisResult(
            summary="요약",
            keywords=["반도체"],
            key_signals=["투자 확대"],
            key_points=["포인트"],
            trend_summary="안정",
            sentiment=SentimentLabel.NEUTRAL,
            market_impact=ImpactLabel.NEUTRAL,
            policy_risk=RiskLevel.LOW,
            volatility_risk=RiskLevel.LOW,
            risk_points=[],
            grounded=True,
            priority=IssuePriority.GENERAL,
            hold_reason=None,
        ),
    )

    called: dict[str, int] = {}

    class FakeExecutor:
        def __init__(self, max_workers: int):
            called["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, func, iterable):
            return [func(item) for item in iterable]

    monkeypatch.setattr(pipeline_module, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(pipeline_module.settings, "max_parallel_issue_analysis", 4)

    issues = pipeline.analyze_only()

    assert len(issues) == 2
    assert called["max_workers"] == 2


def test_analyze_group_uses_cached_analysis(tmp_path, monkeypatch):
    repository = IssueRepository(database_path=str(tmp_path / "pipeline.db"))
    pipeline = NewsPipeline(repository=repository)
    article = load_sample_articles()[0]
    monkeypatch.setattr(pipeline.retriever, "retrieve", lambda _group: [EvidenceSnippet(article_id=article.id, source=article.source, quote=article.content, url=article.url)])
    monkeypatch.setattr(
        pipeline_module,
        "score_issue",
        lambda *_args, **_kwargs: ReliabilityScore(0.82, 0.8, 0.8, 0.8, 0.8),
    )
    monkeypatch.setattr(pipeline_module, "_derive_topic", lambda _group: "반도체 투자 확대")
    monkeypatch.setattr(pipeline_module, "_build_hold_reason", lambda *_args, **_kwargs: None)

    cached = AnalysisResult(
        summary="캐시된 요약",
        keywords=["반도체"],
        key_signals=["투자 확대"],
        key_points=["포인트"],
        trend_summary="안정",
        sentiment=SentimentLabel.NEUTRAL,
        market_impact=ImpactLabel.NEUTRAL,
        policy_risk=RiskLevel.LOW,
        volatility_risk=RiskLevel.LOW,
        risk_points=[],
        grounded=True,
        priority=IssuePriority.GENERAL,
        hold_reason=None,
        grounding_details={"llm": {"analysis_used": True}},
    )
    cache_key = pipeline_module._issue_analysis_cache_key(
        "반도체 투자 확대",
        [article],
        [EvidenceSnippet(article_id=article.id, source=article.source, quote=article.content, url=article.url)],
        ReliabilityScore(0.82, 0.8, 0.8, 0.8, 0.8),
        None,
    )
    repository.save_issue_analysis_cache(cache_key, cached)

    def fail(**_kwargs):
        raise AssertionError("analysis should not run on cache hit")

    monkeypatch.setattr(pipeline.analyzer, "analyze", fail)

    issue = pipeline._analyze_group([article], [article])

    assert issue.analysis.summary == "캐시된 요약"
    assert issue.analysis.grounding_details["llm"]["cache_hit"] is True


def test_analyze_group_uses_lightweight_hold_path(tmp_path, monkeypatch):
    repository = IssueRepository(database_path=str(tmp_path / "pipeline.db"))
    pipeline = NewsPipeline(repository=repository)
    article = load_sample_articles()[0]
    evidence = [EvidenceSnippet(article_id=article.id, source=article.source, quote=article.content, url=article.url)]
    monkeypatch.setattr(pipeline.retriever, "retrieve", lambda _group: evidence)
    monkeypatch.setattr(
        pipeline_module,
        "score_issue",
        lambda *_args, **_kwargs: ReliabilityScore(0.52, 0.5, 0.8, 0.5, 0.5),
    )
    monkeypatch.setattr(pipeline_module.settings, "min_articles_per_issue", 2)
    monkeypatch.setattr(pipeline_module, "_derive_topic", lambda _group: "물가 · 둔화")
    monkeypatch.setattr(pipeline_module, "_build_hold_reason", lambda *_args, **_kwargs: "기사 수 부족: 1건")

    called = {"lightweight": 0}

    def lightweight(**_kwargs):
        called["lightweight"] += 1
        return AnalysisResult(
            summary="보류된 이슈입니다. 사유: 기사 수 부족: 1건. 요약",
            keywords=["물가"],
            key_signals=["물가"],
            key_points=["포인트"],
            trend_summary="안정",
            sentiment=SentimentLabel.NEUTRAL,
            market_impact=ImpactLabel.NEUTRAL,
            policy_risk=RiskLevel.LOW,
            volatility_risk=RiskLevel.LOW,
            risk_points=["기사 수 부족: 1건"],
            grounded=False,
            priority=IssuePriority.GENERAL,
            hold_reason="기사 수 부족: 1건",
            grounding_details={"llm": {"analysis_mode": "lightweight_hold"}},
        )

    monkeypatch.setattr(pipeline.analyzer, "analyze_lightweight", lightweight)
    monkeypatch.setattr(
        pipeline.analyzer,
        "analyze",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("full analysis should not run")),
    )

    issue = pipeline._analyze_group([article], [article])

    assert called["lightweight"] == 1
    assert issue.analysis.grounding_details["llm"]["analysis_mode"] == "lightweight_hold"


def test_issue_analysis_cache_key_changes_with_hold_reason() -> None:
    article = load_sample_articles()[0]
    evidence = [EvidenceSnippet(article_id=article.id, source=article.source, quote=article.content, url=article.url)]
    reliability = ReliabilityScore(0.8, 0.8, 0.8, 0.8, 0.8)

    first = pipeline_module._issue_analysis_cache_key("반도체", [article], evidence, reliability, None)
    second = pipeline_module._issue_analysis_cache_key("반도체", [article], evidence, reliability, "기사 수 부족: 1건")

    assert first != second


def test_second_pass_candidate_filters_borderline_grounding() -> None:
    article = load_sample_articles()[0]
    issue = pipeline_module.Issue(
        id="1",
        topic="cpi · 금리 · 인하",
        keywords=["금리"],
        articles=[article],
        evidence=[],
        reliability=ReliabilityScore(0.8, 0.8, 0.8, 0.8, 0.8),
        analysis=AnalysisResult(
            summary="보류",
            keywords=["금리"],
            key_signals=["금리 인하"],
            key_points=["포인트"],
            trend_summary="안정",
            sentiment=SentimentLabel.NEUTRAL,
            market_impact=ImpactLabel.NEUTRAL,
            policy_risk=RiskLevel.LOW,
            volatility_risk=RiskLevel.LOW,
            risk_points=[],
            grounded=False,
            priority=IssuePriority.GENERAL,
            hold_reason="grounding 검증 부족",
            grounding_details={
                "grounding": {"grounded_ratio": 0.667, "issue_score": 0.739, "contradiction_ratio": 0.0},
                "decision": {"ready_claim_count": 3},
                "llm": {"analysis_mode": "combined_remote"},
            },
        ),
        status=pipeline_module._status_from_hold_reason("grounding 검증 부족"),
        updated_at=article.published_at,
    )

    assert pipeline_module._is_second_pass_candidate(issue) is True


def test_run_second_pass_reviews_respects_limit(tmp_path, monkeypatch):
    repository = IssueRepository(database_path=str(tmp_path / "pipeline.db"))
    pipeline = NewsPipeline(repository=repository)
    article = load_sample_articles()[0]

    def make_issue(issue_id: str) -> pipeline_module.Issue:
        return pipeline_module.Issue(
            id=issue_id,
            topic=f"cpi · 금리 · 인하 {issue_id}",
            keywords=["금리"],
            articles=[article],
            evidence=[],
            reliability=ReliabilityScore(0.8, 0.8, 0.8, 0.8, 0.8),
            analysis=AnalysisResult(
                summary="보류",
                keywords=["금리"],
                key_signals=["금리 인하"],
                key_points=["포인트"],
                trend_summary="안정",
                sentiment=SentimentLabel.NEUTRAL,
                market_impact=ImpactLabel.NEUTRAL,
                policy_risk=RiskLevel.LOW,
                volatility_risk=RiskLevel.LOW,
                risk_points=[],
                grounded=False,
                priority=IssuePriority.GENERAL,
                hold_reason="grounding 검증 부족",
                grounding_details={
                    "grounding": {"grounded_ratio": 0.667, "issue_score": 0.739, "contradiction_ratio": 0.0},
                    "decision": {"ready_claim_count": 3},
                    "llm": {"analysis_mode": "combined_remote"},
                },
            ),
            status=pipeline_module._status_from_hold_reason("grounding 검증 부족"),
            updated_at=article.published_at,
        )

    issues = [make_issue("1"), make_issue("2")]
    calls = {"count": 0}

    def review(**_kwargs):
        calls["count"] += 1
        return None

    monkeypatch.setattr(pipeline.analyzer, "review_hold_for_promotion", review)
    monkeypatch.setattr(pipeline_module.settings, "max_second_pass_reviews", 1)

    reviewed = pipeline._run_second_pass_reviews(issues)

    assert len(reviewed) == 2
    assert calls["count"] == 1
