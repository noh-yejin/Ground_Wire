from datetime import datetime

from app.models import Article
from app.services.preprocessing import preprocess_articles


def test_preprocess_articles_removes_duplicates_and_low_quality() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="AI 반도체 수요 증가",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/1",
            content="<p>AI 반도체 수요 증가가 수출 회복으로 이어지고 있다.</p>",
            collected_at=now,
        ),
        Article(
            id="2",
            title="AI 반도체 수요 증가",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/1",
            content="중복 기사",
            collected_at=now,
        ),
        Article(
            id="3",
            title="짧음",
            source="블로그",
            published_at=now,
            url="https://example.com/3",
            content="광고",
            collected_at=now,
        ),
    ]

    processed = preprocess_articles(articles)

    assert len(processed) == 1
    assert processed[0].content.startswith("AI 반도체 수요 증가")
    assert processed[0].content_quality >= 0.45
