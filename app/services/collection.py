from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha1
from time import struct_time

import feedparser

from app.config import settings
from app.models import Article
from app.repository import IssueRepository
from app.sample_data import load_sample_articles
from app.services.crawling import fetch_article_body, is_supported_for_crawl
from app.services.source_normalizer import normalize_source_name


class NewsCollector:
    """Live RSS collector with a sample-data fallback."""

    def __init__(self, repository: IssueRepository | None = None) -> None:
        self.repository = repository or IssueRepository()

    def collect(self) -> list[Article]:
        collected_at = datetime.utcnow()
        live_articles = self._collect_live_articles(collected_at)
        if live_articles:
            return live_articles

        # Fallback keeps the project runnable offline, but live RSS is tried first.
        return self._collect_sample_articles(collected_at)

    def _collect_live_articles(self, collected_at: datetime) -> list[Article]:
        articles: list[Article] = []
        crawled_count = 0
        seen_ids: set[str] = set()
        threshold = datetime.now(timezone.utc) - timedelta(hours=settings.article_window_hours)

        for feed_url in settings.rss_feed_urls:
            state = self.repository.get_feed_state(feed_url) or {}
            parsed = feedparser.parse(
                feed_url,
                etag=state.get("etag"),
                modified=state.get("modified"),
            )
            self.repository.save_feed_state(
                feed_url,
                getattr(parsed, "etag", None),
                getattr(parsed, "modified", None),
            )

            if getattr(parsed, "status", None) == 304:
                continue

            feed_source = parsed.feed.get("title", feed_url)
            for entry in parsed.entries[: settings.max_articles_per_feed]:
                article = self._entry_to_article(entry, feed_source, collected_at)
                if article is None:
                    continue
                if self._to_utc(article.published_at) < threshold:
                    continue
                if not self._is_relevant(article):
                    continue
                if article.id in seen_ids:
                    continue
                seen_ids.add(article.id)

                if (
                    crawled_count < settings.crawl_max_articles_per_run
                    and self.should_fetch_full_content(article.title, article.content)
                    and is_supported_for_crawl(article.url)
                ):
                    crawled_body = self.fetch_full_content(article.url)
                    if crawled_body:
                        article.content = crawled_body
                        crawled_count += 1

                articles.append(article)

        return articles

    def _collect_sample_articles(self, collected_at: datetime) -> list[Article]:
        articles: list[Article] = []
        for article in load_sample_articles():
            content = article.content
            if self.should_fetch_full_content(article.title, content):
                content = self.fetch_full_content(article.url) or content

            articles.append(
                Article(
                    id=article.id or sha1(article.url.encode("utf-8")).hexdigest()[:12],
                    title=article.title,
                    source=article.source,
                    published_at=article.published_at,
                    url=article.url,
                    content=content,
                    language=article.language,
                    collected_at=collected_at,
                )
            )
        return articles

    def should_fetch_full_content(self, title: str, content: str = "") -> bool:
        lowered = f"{title} {content}".lower()
        return any(keyword.lower() in lowered for keyword in settings.crawl_priority_keywords)

    def fetch_full_content(self, url: str) -> str | None:
        return fetch_article_body(url)

    def _entry_to_article(self, entry, feed_source: str, collected_at: datetime) -> Article | None:
        url = getattr(entry, "link", None)
        title = self._clean_entry_title(getattr(entry, "title", None), feed_source)
        if not url or not title:
            return None

        published_at = self._parse_entry_datetime(entry)
        content = self._extract_entry_content(entry)
        source = self._extract_source(entry, feed_source)
        digest = sha1(f"{url}|{title}".encode("utf-8")).hexdigest()[:12]
        language = "ko" if any("\uac00" <= char <= "\ud7a3" for char in title) else "en"
        return Article(
            id=digest,
            title=title,
            source=source,
            published_at=published_at,
            url=url,
            content=content,
            language=language,
            collected_at=collected_at,
        )

    def _parse_entry_datetime(self, entry) -> datetime:
        parsed_value = (
            getattr(entry, "published_parsed", None)
            or getattr(entry, "updated_parsed", None)
            or getattr(entry, "created_parsed", None)
        )
        if parsed_value is None:
            return datetime.now(timezone.utc)
        if isinstance(parsed_value, struct_time):
            return datetime(*parsed_value[:6], tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    def _extract_entry_content(self, entry) -> str:
        if getattr(entry, "summary", None):
            return str(entry.summary)
        if getattr(entry, "description", None):
            return str(entry.description)
        if getattr(entry, "content", None):
            parts = entry.content
            if parts and isinstance(parts, list):
                return str(parts[0].get("value", ""))
        return ""

    def _extract_source(self, entry, fallback: str) -> str:
        source = getattr(entry, "source", None)
        if source and isinstance(source, dict):
            return normalize_source_name(source.get("title", fallback))
        return normalize_source_name(fallback)

    def _to_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _is_relevant(self, article: Article) -> bool:
        haystack = f"{article.title} {article.content}".lower()
        high_signal = any(keyword.lower() in haystack for keyword in settings.high_signal_keywords)
        topic_hits = sum(1 for keyword in settings.monitored_topic_keywords if keyword.lower() in haystack)
        return high_signal or topic_hits >= 2

    def _clean_entry_title(self, title: str | None, feed_source: str) -> str | None:
        if not title:
            return None
        cleaned = title.strip()
        if " - " in cleaned:
            head, tail = cleaned.rsplit(" - ", 1)
            if normalize_source_name(tail) != tail or normalize_source_name(tail) == normalize_source_name(feed_source):
                return head.strip()
        return cleaned
