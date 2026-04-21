from __future__ import annotations

import html
import re
from hashlib import sha1

from app.models import Article

WHITESPACE_PATTERN = re.compile(r"\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
FULL_COVERAGE_PATTERN = re.compile(r"view full coverage on google news", re.IGNORECASE)
LOW_SIGNAL_TITLE_PATTERNS = (
    re.compile(r"\blive updates?\b", re.IGNORECASE),
    re.compile(r"\bmarkets wrap\b", re.IGNORECASE),
    re.compile(r"\bgoogle news\b", re.IGNORECASE),
    re.compile(r"\bv\.daum\.net\b", re.IGNORECASE),
)


def preprocess_articles(articles: list[Article]) -> list[Article]:
    deduped = _dedupe_articles(articles)
    cleaned: list[Article] = []
    for article in deduped:
        title = clean_text(article.title)
        content = clean_article_content(article.title, article.content, article.source)
        if is_low_signal_aggregator_article(title, content, article.source, article.url):
            continue
        quality = score_article_quality(article.title, content)
        if quality < 0.45:
            continue
        cleaned.append(
            Article(
                id=article.id,
                title=title,
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
    unescaped = html.unescape(str(text or ""))
    no_html = HTML_TAG_PATTERN.sub(" ", unescaped)
    normalized = WHITESPACE_PATTERN.sub(" ", no_html).strip()
    return normalized


def clean_article_content(title: str, content: str, source: str) -> str:
    cleaned_title = clean_text(title)
    cleaned_content = clean_text(content)
    cleaned_source = clean_text(source)

    if not cleaned_content:
        return cleaned_title

    lowered = FULL_COVERAGE_PATTERN.sub("", cleaned_content).strip()
    if cleaned_source:
        lowered = re.sub(rf"\b{re.escape(cleaned_source)}\b", " ", lowered, flags=re.IGNORECASE)
        lowered = WHITESPACE_PATTERN.sub(" ", lowered).strip()
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\s{2,}", lowered) if part.strip()]
    normalized_parts: list[str] = []

    for part in parts:
        candidate = _strip_source_suffix(part, cleaned_source)
        candidate = FULL_COVERAGE_PATTERN.sub("", candidate).strip()
        if not candidate:
            continue
        if _looks_like_source_only(candidate, cleaned_source):
            continue
        if _is_duplicateish(candidate, cleaned_title):
            continue
        if any(_is_duplicateish(candidate, existing) for existing in normalized_parts):
            continue
        normalized_parts.append(candidate)

    if normalized_parts:
        return " ".join(normalized_parts[:3])

    fallback = _strip_source_suffix(lowered, cleaned_source).strip()
    if fallback and not _looks_like_source_only(fallback, cleaned_source) and not _is_duplicateish(fallback, cleaned_title):
        return fallback
    return cleaned_title


def score_article_quality(title: str, content: str) -> float:
    content_length_score = min(len(content) / 280, 1.0)
    title_length_score = min(len(title) / 25, 1.0)
    has_sentence_like_shape = 1.0 if any(end in content for end in (".", "다", "했다")) else 0.4
    aggregator_penalty = 0.28 if is_low_signal_aggregator_article(title, content, "", "") else 0.0
    return round(max(0.0, content_length_score * 0.5 + title_length_score * 0.2 + has_sentence_like_shape * 0.3 - aggregator_penalty), 3)


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


def _strip_source_suffix(text: str, source: str) -> str:
    candidate = text.strip()
    if not source:
        return candidate
    escaped = re.escape(source)
    candidate = re.sub(rf"(?:\s*[-|·]\s*|\s+)({escaped})$", "", candidate, flags=re.IGNORECASE).strip()
    return candidate


def _normalize_compare_text(text: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]", "", clean_text(text).lower())


def _is_duplicateish(left: str, right: str) -> bool:
    normalized_left = _normalize_compare_text(left)
    normalized_right = _normalize_compare_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    if shorter in longer:
        return (len(shorter) / max(len(longer), 1)) >= 0.88
    return False


def _looks_like_source_only(text: str, source: str) -> bool:
    normalized = _normalize_compare_text(text)
    normalized_source = _normalize_compare_text(source)
    return bool(normalized and normalized_source and normalized == normalized_source)


def is_low_signal_aggregator_article(title: str, content: str, source: str, url: str) -> bool:
    title_text = clean_text(title)
    content_text = clean_text(content)
    url_text = clean_text(url)
    source_text = clean_text(source)
    joined = " ".join(filter(None, [title_text, content_text, source_text, url_text]))

    if any(pattern.search(joined) for pattern in LOW_SIGNAL_TITLE_PATTERNS):
        return True
    if "google news" in joined.lower():
        return True
    if "news.google.com" in url_text.lower() and _is_google_news_wrapper_low_value(title_text, content_text):
        return True
    if "v.daum.net" in joined.lower() and _is_duplicateish(content_text, title_text):
        return True
    if "live" in title_text.lower() and "update" in title_text.lower():
        return True
    return False


def _is_google_news_wrapper_low_value(title: str, content: str) -> bool:
    if not content:
        return True
    if _is_duplicateish(content, title):
        return True
    if len(content) < 80:
        return True
    sentence_count = len([part for part in re.split(r"(?<=[.!?。！？])\s+", content) if part.strip()])
    return sentence_count < 2 and len(content) < 140
