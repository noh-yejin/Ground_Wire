from __future__ import annotations

import re
from hashlib import sha1

from app.models import Article

WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def preprocess_articles(articles: list[Article]) -> list[Article]:
    deduped = _dedupe_articles(articles)
    cleaned: list[Article] = []
    for article in deduped:
        content = clean_text(article.content)
        quality = score_article_quality(article.title, content)
        if quality < 0.45:
            continue
        cleaned.append(
            Article(
                id=article.id,
                title=clean_text(article.title),
                source=article.source,
                published_at=article.published_at,
                url=article.url,
                content=content,
                language=article.language,
                collected_at=article.collected_at,
                content_quality=quality,
            )
        )
    return cleaned


def clean_text(text: str) -> str:
    no_html = HTML_TAG_PATTERN.sub(" ", text)
    normalized = WHITESPACE_PATTERN.sub(" ", no_html).strip()
    return normalized


def score_article_quality(title: str, content: str) -> float:
    content_length_score = min(len(content) / 280, 1.0)
    title_length_score = min(len(title) / 25, 1.0)
    has_sentence_like_shape = 1.0 if any(end in content for end in (".", "다", "했다")) else 0.4
    return round(content_length_score * 0.5 + title_length_score * 0.2 + has_sentence_like_shape * 0.3, 3)


def _dedupe_articles(articles: list[Article]) -> list[Article]:
    seen: set[str] = set()
    result: list[Article] = []
    for article in articles:
        digest = sha1(f"{article.url}|{article.title}".encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        result.append(article)
    return result
