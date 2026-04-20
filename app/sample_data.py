from datetime import datetime, timedelta

from app.models import Article


def load_sample_articles() -> list[Article]:
    now = datetime.utcnow()
    return [
        Article(
            id="a1",
            title="한국 반도체 수출 증가세, AI 서버 수요 확대 영향",
            source="연합뉴스",
            published_at=now - timedelta(hours=2),
            url="https://example.com/a1",
            content="한국 반도체 수출이 AI 서버 수요 증가에 힘입어 회복세를 보인다는 분석이 나왔다. 메모리 가격 반등과 수출 증가가 이어지고 있다.",
        ),
        Article(
            id="a2",
            title="AI 인프라 투자 확대로 메모리 반도체 업황 개선",
            source="Reuters",
            published_at=now - timedelta(hours=3),
            url="https://example.com/a2",
            content="글로벌 클라우드 기업들의 AI 인프라 투자 확대가 메모리 반도체 수요를 자극하고 있으며 아시아 공급망 전반에 긍정적 신호가 나타났다.",
            language="en",
        ),
        Article(
            id="a3",
            title="삼성전자·SK하이닉스, HBM 수요 지속 기대",
            source="매일경제",
            published_at=now - timedelta(hours=1),
            url="https://example.com/a3",
            content="HBM 중심의 고부가 메모리 수요가 지속되며 국내 반도체 기업 실적 기대가 커지고 있다. 증권가는 업황 회복 근거로 AI 서버 확산을 들었다.",
        ),
        Article(
            id="a7",
            title="반도체 업계, AI 서버용 메모리 주문 증가 확인",
            source="한국경제",
            published_at=now - timedelta(hours=2, minutes=30),
            url="https://example.com/a7",
            content="국내 반도체 업계는 AI 서버용 메모리 주문 증가가 이어지고 있다고 밝혔다. 시장에서는 수출 회복과 고부가 메모리 수요 확대를 함께 거론했다.",
        ),
        Article(
            id="a4",
            title="미국 금리 인하 기대 후퇴, 뉴욕 증시 혼조",
            source="Bloomberg",
            published_at=now - timedelta(hours=4),
            url="https://example.com/a4",
            content="예상보다 높은 물가 지표로 금리 인하 기대가 약해졌고 기술주 중심으로 증시 변동성이 확대됐다.",
            language="en",
        ),
        Article(
            id="a5",
            title="미 CPI 발표 후 증시 변동성 확대",
            source="한국경제",
            published_at=now - timedelta(hours=5),
            url="https://example.com/a5",
            content="미국 CPI 발표 이후 기준금리 인하 기대가 줄면서 국내외 증시의 변동성이 확대됐다는 분석이 나온다.",
        ),
        Article(
            id="a6",
            title="연준 인사들, 물가 둔화 확인 필요 강조",
            source="CNBC",
            published_at=now - timedelta(hours=6),
            url="https://example.com/a6",
            content="연준 관계자들은 금리 인하에 앞서 물가 둔화가 더 명확해져야 한다고 언급했다. 시장은 긴축 장기화 가능성을 반영하고 있다.",
            language="en",
        ),
        Article(
            id="a8",
            title="월가, CPI 이후 금리 인하 시점 재조정",
            source="WSJ",
            published_at=now - timedelta(hours=4, minutes=30),
            url="https://example.com/a8",
            content="월가는 CPI 발표 이후 연준의 금리 인하 시점이 늦춰질 수 있다고 봤다. 이에 따라 미국 증시와 채권 시장 변동성이 커졌다는 평가가 나온다.",
            language="en",
        ),
    ]
