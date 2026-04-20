from __future__ import annotations

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.config import settings

USER_AGENT = (
    "ReliabilityNewsAgent/0.1 (+https://localhost; contact=local-dev)"
)


def fetch_article_body(url: str) -> str | None:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=settings.crawl_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    _remove_noise(soup)

    selectors = (
        "article p",
        "main p",
        "[itemprop='articleBody'] p",
        ".article-body p",
        ".story-body p",
        "p",
    )
    paragraphs: list[str] = []
    for selector in selectors:
        paragraphs = [node.get_text(" ", strip=True) for node in soup.select(selector)]
        paragraphs = [text for text in paragraphs if len(text) >= 40]
        if paragraphs:
            break

    if not paragraphs:
        return None

    body = " ".join(paragraphs)
    return body[:6000]


def is_supported_for_crawl(url: str) -> bool:
    hostname = urlparse(url).hostname or ""
    return hostname.endswith(
        (
            "google.com",
            "reuters.com",
            "yonhapnews.co.kr",
            "yna.co.kr",
            "hankyung.com",
            "mk.co.kr",
            "wsj.com",
            "bloomberg.com",
            "cnbc.com",
        )
    )


def _remove_noise(soup: BeautifulSoup) -> None:
    for tag in soup.select("script, style, nav, footer, aside, form"):
        tag.decompose()
