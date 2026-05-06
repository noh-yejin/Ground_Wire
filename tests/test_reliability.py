from datetime import datetime, timedelta

from app.models import Article, EvidenceSnippet
from app.services.reliability import build_evidence, score_issue
from app.services.summarizer import summarize_issue


def test_issue_with_multiple_sources_is_ready() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="AI 반도체 수요 증가",
            source="연합뉴스",
            published_at=now - timedelta(hours=1),
            url="https://example.com/1",
            content="AI 반도체 수요 증가가 수출 회복으로 이어지고 있다.",
        ),
        Article(
            id="2",
            title="HBM 수요와 메모리 업황 개선",
            source="Reuters",
            published_at=now - timedelta(hours=2),
            url="https://example.com/2",
            content="메모리 업황 개선의 배경으로 AI 서버 투자가 지목된다.",
            language="en",
        ),
        Article(
            id="3",
            title="국내 반도체 실적 기대 확대",
            source="매일경제",
            published_at=now - timedelta(hours=3),
            url="https://example.com/3",
            content="국내 반도체 기업 실적 개선 기대가 높아지고 있다.",
        ),
    ]
    evidence = build_evidence(articles)
    reliability = score_issue(articles, evidence)
    summary, status = summarize_issue(
        "반도체 / ai / 수요",
        articles,
        evidence,
        reliability,
        hold_threshold=0.7,
        min_articles=3,
        min_sources=2,
    )

    assert status.value == "READY"
    assert reliability.value >= 0.7
    assert "교차 확인" in summary


def test_single_source_issue_is_held() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="단일 출처 기사",
            source="연합뉴스",
            published_at=now - timedelta(hours=1),
            url="https://example.com/1",
            content="이 기사는 단일 출처만으로 구성되어 있다.",
        ),
        Article(
            id="2",
            title="비슷한 기사",
            source="연합뉴스",
            published_at=now - timedelta(hours=2),
            url="https://example.com/2",
            content="같은 출처의 후속 보도다.",
        ),
    ]
    evidence = build_evidence(articles)
    reliability = score_issue(articles, evidence)
    summary, status = summarize_issue(
        "단일 / 출처",
        articles,
        evidence,
        reliability,
        hold_threshold=0.7,
        min_articles=3,
        min_sources=2,
    )

    assert status.value == "HOLD"
    assert "보류" in summary


def test_reference_evidence_increases_reliability_components() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="기준금리 동결 가능성",
            source="Reuters",
            published_at=now - timedelta(hours=1),
            url="https://example.com/1",
            content="시장에서는 기준금리 동결 가능성을 높게 보고 있다.",
        ),
        Article(
            id="2",
            title="환율과 수입물가 점검",
            source="연합뉴스",
            published_at=now - timedelta(hours=2),
            url="https://example.com/2",
            content="통화당국은 환율과 수입물가 흐름을 함께 점검하고 있다.",
        ),
    ]
    evidence = [
        EvidenceSnippet(article_id="1", source="Reuters", quote=articles[0].content, url=articles[0].url),
        EvidenceSnippet(article_id="2", source="연합뉴스", quote=articles[1].content, url=articles[1].url),
        EvidenceSnippet(
            article_id="ref-1",
            source="BOK Notes",
            quote="기준금리 결정 시 환율과 수입물가 반응을 함께 점검해야 한다.",
            url="https://example.com/ref",
            evidence_type="reference",
            source_id="bok-notes",
            source_type="government",
            authority_score=0.95,
            freshness_score=0.92,
        ),
    ]

    reliability = score_issue(articles, evidence)

    assert reliability.reference_strength > 0.5
    assert reliability.cross_source_confirmation > 0.7
