from datetime import datetime, timedelta

from app.models import Article
from app.services.clustering import _extract_keywords, cluster_articles, label_topic
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


def test_cluster_articles_requires_stronger_title_alignment() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="중국 LPR 11개월째 동결…1년물 3.0%·5년물 3.5%",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/1",
            content="중국이 LPR을 11개월째 동결했다.",
        ),
        Article(
            id="2",
            title="독일, 이스라엘에 114억 무기 수출",
            source="Reuters",
            published_at=now,
            url="https://example.com/2",
            content="독일의 대이스라엘 무기 수출 규모가 114억으로 집계됐다.",
        ),
        Article(
            id="3",
            title="中, 사실상 기준금리 LPR 동결",
            source="한국경제",
            published_at=now,
            url="https://example.com/3",
            content="중국이 사실상 기준금리 역할을 하는 LPR을 동결했다.",
        ),
    ]

    clusters = cluster_articles(articles)
    cluster_titles = [sorted(article.title for article in cluster) for cluster in clusters]

    assert len(clusters) == 2
    assert any(
        "중국 LPR 11개월째 동결…1년물 3.0%·5년물 3.5%" in titles
        and "中, 사실상 기준금리 LPR 동결" in titles
        for titles in cluster_titles
    )


def test_cluster_articles_avoids_generic_ai_overlap_merge() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="AI 서버용 메모리 주문 증가, 반도체 업황 개선",
            source="한국경제",
            published_at=now,
            url="https://example.com/1",
            content="AI 서버용 메모리 주문이 늘며 반도체 업황이 개선되고 있다.",
        ),
        Article(
            id="2",
            title="AI 의료 영상 판독 소프트웨어 투자 확대",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/2",
            content="의료 AI 소프트웨어 투자 확대가 이어지고 있다.",
        ),
    ]

    clusters = cluster_articles(articles)

    assert len(clusters) == 2


def test_cluster_articles_respects_large_time_gap() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="미 CPI 발표 후 증시 변동성 확대",
            source="한국경제",
            published_at=now,
            url="https://example.com/1",
            content="CPI 발표 이후 증시 변동성이 확대됐다.",
        ),
        Article(
            id="2",
            title="미 CPI 발표 후 증시 변동성 확대",
            source="Reuters",
            published_at=now - timedelta(days=2),
            url="https://example.com/2",
            content="이틀 전 CPI 발표 당시 증시 변동성이 확대됐다.",
        ),
    ]

    clusters = cluster_articles(articles)

    assert len(clusters) == 2


def test_cluster_articles_merges_semantic_macro_paraphrases() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="미 CPI 발표 이후 금리 인하 기대 약화",
            source="Reuters",
            published_at=now,
            url="https://example.com/1",
            content="미국 CPI 발표 이후 시장의 금리 인하 기대가 약화됐고 연준의 정책 전환 시점이 늦춰질 수 있다는 분석이 나왔다.",
        ),
        Article(
            id="2",
            title="월가, CPI 이후 연준 금리 인하 시점 재조정",
            source="연합뉴스",
            published_at=now + timedelta(hours=1),
            url="https://example.com/2",
            content="월가는 CPI 발표 뒤 연준의 금리 인하 시점이 늦춰질 수 있다고 봤고 물가 압력이 여전히 남아 있다고 평가했다.",
        ),
    ]

    clusters = cluster_articles(articles)

    assert len(clusters) == 1


def test_cluster_articles_does_not_merge_only_broad_macro_overlap() -> None:
    now = datetime.utcnow()
    articles = [
        Article(
            id="1",
            title="미 CPI 발표 이후 금리 인하 기대 약화",
            source="Reuters",
            published_at=now,
            url="https://example.com/1",
            content="미국 CPI 발표 이후 시장의 금리 인하 기대가 약화됐다.",
        ),
        Article(
            id="2",
            title="중국 LPR 동결에도 대출 수요 회복 지연",
            source="연합뉴스",
            published_at=now,
            url="https://example.com/2",
            content="중국은 사실상 기준금리 역할을 하는 LPR을 동결했지만 대출 수요 회복은 여전히 더딘 상황이다.",
        ),
    ]

    clusters = cluster_articles(articles)

    assert len(clusters) == 2
