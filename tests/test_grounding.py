from datetime import datetime, timedelta

from app.services.llm_analyzer import LLMAnalyzer
from app.services.reliability import build_evidence, score_grounding, score_issue
from app.models import Article


def test_grounding_details_present_for_supported_issue() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="반도체 지원 확대 발표",
            source="Reuters",
            published_at=now - timedelta(hours=1),
            url="https://example.com/1",
            content="정부가 반도체 지원 확대를 발표했고 생산 투자 확대를 추진한다고 밝혔다.",
        ),
        Article(
            id="2",
            title="정부 반도체 투자 확대",
            source="연합뉴스",
            published_at=now - timedelta(hours=2),
            url="https://example.com/2",
            content="정부가 반도체 지원 확대를 발표했고 생산 투자 확대 계획을 공개했다.",
        ),
        Article(
            id="3",
            title="반도체 생산 투자 확대 계획",
            source="매일경제",
            published_at=now - timedelta(hours=3),
            url="https://example.com/3",
            content="반도체 지원 확대 발표 이후 생산 투자 확대가 이어질 것이라는 분석이 나왔다.",
        ),
    ]
    evidence = build_evidence(articles)
    reliability = score_issue(articles, evidence)
    analysis = LLMAnalyzer().analyze("반도체 지원 확대", articles, evidence, reliability, hold_reason=None)

    assert analysis.grounding_details["claims"]
    assert analysis.grounding_details["grounding"]["grounded_ratio"] > 0
    assert analysis.grounded is True
    assert analysis.hold_reason is None


def test_grounding_score_penalizes_contradictions() -> None:
    reliability = score_issue(
        [
            Article(
                id="1",
                title="기사1",
                source="Reuters",
                published_at=datetime.utcnow(),
                url="https://example.com/1",
                content="정부가 금리 인하를 단행했다.",
            ),
            Article(
                id="2",
                title="기사2",
                source="연합뉴스",
                published_at=datetime.utcnow(),
                url="https://example.com/2",
                content="정부는 금리 인하 보도를 부인했다.",
            ),
        ],
        [],
    )
    grounded = score_grounding(
        [
            {"claim": "정부가 금리 인하를 단행했다.", "score": 0.82, "ready": True, "contradiction_count": 0},
            {"claim": "정부는 금리 인하 보도를 부인했다.", "score": 0.41, "ready": False, "contradiction_count": 1},
        ],
        reliability,
    )

    assert grounded["contradiction_ratio"] > 0
    assert grounded["issue_score"] < 0.82
    assert grounded["reasons"]


def test_grounding_uses_external_corpus_support() -> None:
    now = datetime.utcnow()
    group = [
        Article(
            id="1",
            title="삼성전자 10% 투자 확대",
            source="Reuters",
            published_at=now - timedelta(hours=1),
            url="https://example.com/1",
            content="삼성전자가 반도체 투자 10% 확대를 발표했다.",
        ),
        Article(
            id="2",
            title="반도체 투자 확대 발표",
            source="연합뉴스",
            published_at=now - timedelta(hours=2),
            url="https://example.com/2",
            content="삼성전자가 반도체 투자 10% 확대 계획을 공개했다.",
        ),
    ]
    corpus = [
        *group,
        Article(
            id="3",
            title="삼성전자 투자 확대 공식 확인",
            source="매일경제",
            published_at=now - timedelta(hours=3),
            url="https://example.com/3",
            content="삼성전자가 반도체 투자 10% 확대 방침을 공식 확인했다.",
        ),
    ]
    evidence = build_evidence(group)
    reliability = score_issue(group, evidence)
    analysis = LLMAnalyzer().analyze("삼성전자 반도체 투자 확대", group, evidence, reliability, None, corpus_articles=corpus)

    claims = analysis.grounding_details["claims"]
    assert claims
    assert any(item.get("external_support_count", 0) >= 1 for item in claims)
