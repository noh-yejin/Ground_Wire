import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    for name in (".env", ".env.local"):
        path = Path(name)
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)


_load_dotenv()


def _split_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(slots=True)
class Settings:
    app_name: str = "GroundWire"
    database_path: str = "news_agent.db"
    hold_threshold: float = 0.65
    min_articles_per_issue: int = 2
    min_unique_sources: int = 2
    slack_webhook_url: str | None = None
    enable_scheduler: bool = True
    collect_interval_minutes: int = 15
    analyze_interval_minutes: int = 60
    article_window_hours: int = 24
    max_articles_per_feed: int = 30
    crawl_timeout_seconds: int = 8
    crawl_max_articles_per_run: int = 10
    llm_timeout_seconds: int = 25
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini"
    embedding_model: str = "text-embedding-3-small"
    rss_feed_urls: tuple[str, ...] = (
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=AI+when:1d&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=%EB%B0%98%EB%8F%84%EC%B2%B4+when:1d&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=%EA%B8%88%EB%A6%AC+when:1d&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=%EC%88%98%EC%B6%9C+when:1d&hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=inflation+OR+fed+OR+rate+cut+when:1d&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=chip+OR+semiconductor+OR+HBM+when:1d&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=tariff+OR+sanction+OR+conflict+when:1d&hl=en-US&gl=US&ceid=US:en",
    )
    monitored_topic_keywords: tuple[str, ...] = (
        "ai",
        "semiconductor",
        "chip",
        "hbm",
        "반도체",
        "수출",
        "금리",
        "인플레이션",
        "inflation",
        "fed",
        "rate",
        "tariff",
        "sanction",
        "conflict",
        "war",
        "제재",
        "전쟁",
        "에너지",
        "환율",
        "경제",
    )
    high_signal_keywords: tuple[str, ...] = (
        "semiconductor",
        "chip",
        "hbm",
        "반도체",
        "수출",
        "금리",
        "인플레이션",
        "inflation",
        "fed",
        "rate cut",
        "tariff",
        "sanction",
        "conflict",
        "war",
        "제재",
        "전쟁",
        "에너지",
        "환율",
    )
    priority_keywords: tuple[str, ...] = (
        "속보",
        "긴급",
        "breaking",
        "반도체",
        "semiconductor",
        "hbm",
        "금리",
        "fed",
        "rate cut",
        "inflation",
        "tariff",
        "sanction",
        "conflict",
        "war",
        "전쟁",
        "제재",
        "수출",
        "환율",
        "실적",
    )
    crawl_priority_keywords: tuple[str, ...] = (
        "속보",
        "긴급",
        "금리",
        "반도체",
        "전쟁",
        "제재",
        "실적",
        "ai",
    )


settings = Settings(
    slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
    enable_scheduler=os.getenv("ENABLE_SCHEDULER", "true").lower() == "true",
    collect_interval_minutes=int(os.getenv("COLLECT_INTERVAL_MINUTES", "15")),
    analyze_interval_minutes=int(os.getenv("ANALYZE_INTERVAL_MINUTES", "60")),
    article_window_hours=int(os.getenv("ARTICLE_WINDOW_HOURS", "24")),
    max_articles_per_feed=int(os.getenv("MAX_ARTICLES_PER_FEED", "30")),
    crawl_timeout_seconds=int(os.getenv("CRAWL_TIMEOUT_SECONDS", "8")),
    crawl_max_articles_per_run=int(os.getenv("CRAWL_MAX_ARTICLES_PER_RUN", "10")),
    llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "25")),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
    embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    rss_feed_urls=_split_csv_env(
        "RSS_FEED_URLS",
        (
            "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=AI+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=%EB%B0%98%EB%8F%84%EC%B2%B4+when:1d&hl=ko&gl=KR&ceid=KR:ko",
            "https://news.google.com/rss/search?q=%EA%B8%88%EB%A6%AC+when:1d&hl=ko&gl=KR&ceid=KR:ko",
            "https://news.google.com/rss/search?q=%EC%88%98%EC%B6%9C+when:1d&hl=ko&gl=KR&ceid=KR:ko",
            "https://news.google.com/rss/search?q=inflation+OR+fed+OR+rate+cut+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=chip+OR+semiconductor+OR+HBM+when:1d&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=tariff+OR+sanction+OR+conflict+when:1d&hl=en-US&gl=US&ceid=US:en",
        ),
    ),
    monitored_topic_keywords=_split_csv_env(
        "MONITORED_TOPIC_KEYWORDS",
        (
            "ai",
            "semiconductor",
            "chip",
            "hbm",
            "반도체",
            "수출",
            "금리",
            "인플레이션",
            "inflation",
            "fed",
            "rate",
            "tariff",
            "sanction",
            "conflict",
            "war",
            "제재",
            "전쟁",
            "에너지",
            "환율",
            "경제",
        ),
    ),
    high_signal_keywords=_split_csv_env(
        "HIGH_SIGNAL_KEYWORDS",
        (
            "semiconductor",
            "chip",
            "hbm",
            "반도체",
            "수출",
            "금리",
            "인플레이션",
            "inflation",
            "fed",
            "rate cut",
            "tariff",
            "sanction",
            "conflict",
            "war",
            "제재",
            "전쟁",
            "에너지",
            "환율",
        ),
    ),
    priority_keywords=_split_csv_env(
        "PRIORITY_KEYWORDS",
        (
            "속보",
            "긴급",
            "breaking",
            "반도체",
            "semiconductor",
            "hbm",
            "금리",
            "fed",
            "rate cut",
            "inflation",
            "tariff",
            "sanction",
            "conflict",
            "war",
            "전쟁",
            "제재",
            "수출",
            "환율",
            "실적",
        ),
    ),
)
