from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app import main as main_module
from app.models import AnalysisResult, Article, ImpactLabel, Issue, IssuePriority, IssueStatus, ReliabilityScore, RiskLevel, SentimentLabel
from app.repository import IssueRepository


def _build_issue() -> Issue:
    now = datetime.now(timezone.utc)
    article = Article(
        id="a1",
        title="반도체 투자 확대",
        source="Reuters",
        published_at=now,
        url="https://example.com/a1",
        content="정부가 반도체 투자 확대 계획을 발표했다.",
        collected_at=now,
    )
    return Issue(
        id="issue-1",
        topic="반도체 · 투자 확대",
        keywords=["legacy-keyword"],
        articles=[article],
        evidence=[],
        reliability=ReliabilityScore(0.82, 0.8, 0.9, 0.7, 0.75, 0.64, 0.0),
        analysis=AnalysisResult(
            summary="정부가 반도체 투자 확대 계획을 발표했다.",
            keywords=["반도체", "투자"],
            key_signals=["투자 확대"],
            key_points=["투자 확대 계획 발표"],
            trend_summary="관련 보도가 이어지는 흐름이다.",
            sentiment=SentimentLabel.POSITIVE,
            market_impact=ImpactLabel.POSITIVE,
            policy_risk=RiskLevel.MEDIUM,
            volatility_risk=RiskLevel.LOW,
            risk_points=["세부 일정은 추가 확인이 필요하다."],
            grounded=True,
            priority=IssuePriority.PRIORITY,
            hold_reason=None,
            grounding_details={
                "claims": [
                    {
                        "claim": "정부가 반도체 투자 확대 계획을 발표했다.",
                        "support_count": 2,
                        "trusted_support_count": 1,
                        "external_support_count": 1,
                        "reference_support_count": 1,
                        "authoritative_reference_count": 1,
                        "contradiction_count": 0,
                        "score": 0.81,
                        "ready": True,
                    }
                ],
                "grounding": {"issue_score": 0.81, "grounded_ratio": 1.0, "reference_grounded_ratio": 1.0},
            },
        ),
        status=IssueStatus.READY,
        updated_at=now,
    )


def test_api_issue_payload_is_canonical_and_enveloped(tmp_path, monkeypatch) -> None:
    repository = IssueRepository(database_path=str(tmp_path / "api.db"))
    repository.save_issues([_build_issue()])
    monkeypatch.setattr(main_module, "repository", repository)
    monkeypatch.setattr(main_module, "pipeline", main_module.NewsPipeline(repository=repository))
    monkeypatch.setattr(main_module.scheduler_service, "start", lambda: None)
    monkeypatch.setattr(main_module.scheduler_service, "shutdown", lambda: None)

    client = TestClient(main_module.app)
    response = client.get("/api/issues")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    item = payload["items"][0]
    assert item["keywords"] == ["반도체", "투자"]
    assert "legacy-keyword" not in item["keywords"]
    assert item["article_count"] == 1
    assert "hourly_counts" in item


def test_dashboard_data_uses_single_issue_shape_without_issue_cards(tmp_path, monkeypatch) -> None:
    repository = IssueRepository(database_path=str(tmp_path / "dashboard.db"))
    repository.save_issues([_build_issue()])
    monkeypatch.setattr(main_module, "repository", repository)
    monkeypatch.setattr(main_module, "pipeline", main_module.NewsPipeline(repository=repository))
    monkeypatch.setattr(main_module.scheduler_service, "start", lambda: None)
    monkeypatch.setattr(main_module.scheduler_service, "shutdown", lambda: None)

    client = TestClient(main_module.app)
    response = client.get("/api/dashboard-data")

    assert response.status_code == 200
    payload = response.json()
    assert payload["issues"]["count"] == 1
    assert payload["issues"]["items"][0]["grounding_details"]["claims"][0]["reference_support_count"] == 1
    assert "issue_cards" not in payload
