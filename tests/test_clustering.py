from datetime import datetime

from app.models import Article
from app.services.clustering import _extract_keywords, label_topic
from app.services.source_normalizer import normalize_source_name, source_weight


def test_topic_labeling_filters_generic_tokens() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="AI 반도체 수출 회복, HBM 수요 확대",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/1",
            content="AI 서버 수요와 반도체 수출 회복이 이어진다.",
        ),
        Article(
            id="2",
            title="반도체 업황 개선, AI 메모리 투자 확대",
            source="Reuters",
            published_at=now,
            url="https://example.com/2",
            content="HBM과 메모리 투자 확대가 업황 개선을 이끈다.",
            language="en",
        ),
    ]

    topic = label_topic(articles)

    assert "the" not in _extract_keywords("the ai market")
    assert "반도체" in topic
    assert "HBM" in topic or "수출" in topic


def test_source_normalization_maps_aliases() -> None:
    assert normalize_source_name("Reuters - World News") == "Reuters"
    assert normalize_source_name("hankyung") == "한국경제"
    assert source_weight("Reuters - World News") > source_weight("unknown source")
