from __future__ import annotations

import json
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.config import settings

USER_AGENT = (
    "ReliabilityNewsAgent/0.1 (+https://localhost; contact=local-dev)"
)
GOOGLE_NEWS_BATCH_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je"


def resolve_article_url(url: str) -> str:
    if not _is_google_news_url(url):
        return url

    batched_url = _resolve_google_news_batched_url(url)
    if batched_url:
        return batched_url

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=settings.crawl_timeout_seconds,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException:
        return url

    final_url = response.url or url
    if not _is_google_news_url(final_url):
        return final_url

    soup = BeautifulSoup(response.text, "html.parser")
    for selector in (
        "link[rel='canonical']",
        "link[rel='amphtml']",
        "meta[property='og:url']",
    ):
        node = soup.select_one(selector)
        href = (node.get("href") or node.get("content") or "").strip() if node else ""
        if href and not _is_google_news_url(href):
            return href

    refresh = soup.select_one("meta[http-equiv='refresh']")
    refresh_content = (refresh.get("content") or "").strip() if refresh else ""
    refresh_match = re.search(r"url=['\"]?([^'\";]+)", refresh_content, flags=re.IGNORECASE)
    if refresh_match:
        candidate = refresh_match.group(1).strip()
        if candidate and not _is_google_news_url(candidate):
            return candidate

    text_match = re.search(r"https?://[^\s\"'<>]+", response.text)
    if text_match:
        candidate = text_match.group(0).strip()
        if candidate and not _is_google_news_url(candidate):
            return candidate

    return final_url


def is_google_news_url(url: str) -> bool:
    return _is_google_news_url(url)


def fetch_article_body(url: str) -> str | None:
    target_url = resolve_article_url(url)
    try:
        response = requests.get(
            target_url,
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
    if hostname.endswith("news.google.com"):
        return True
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


def _is_google_news_url(url: str) -> bool:
    hostname = urlparse(url).hostname or ""
    return hostname.endswith("news.google.com")


def _resolve_google_news_batched_url(url: str) -> str | None:
    try:
        session = requests.Session()
        response = session.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=settings.crawl_timeout_seconds,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    if not _is_google_news_url(response.url or url):
        return response.url

    soup = BeautifulSoup(response.text, "html.parser")
    params = soup.select_one("[data-n-a-sg][data-n-a-ts][data-n-a-id]")
    if params is None:
        return None

    article_id = params.get("data-n-a-id", "").strip()
    timestamp = params.get("data-n-a-ts", "").strip()
    signature = params.get("data-n-a-sg", "").strip()
    if not article_id or not timestamp or not signature:
        return None

    locale = _google_news_locale_from_url(response.url or url)
    request_payload = [
        "garturlreq",
        [
            [
                locale["hl"],
                locale["gl"],
                ["FINANCE_TOP_INDICES", "WEB_TEST_1_0_0"],
                None,
                None,
                1,
                1,
                locale["ceid"],
                None,
                180,
                None,
                None,
                None,
                None,
                None,
                0,
                None,
                None,
                [1608992183, 723341000],
            ],
            locale["hl"],
            locale["gl"],
            1,
            [2, 3, 4, 8],
            1,
            0,
            "655000234",
            0,
            0,
            None,
            0,
        ],
        article_id,
        int(timestamp),
        signature,
    ]
    batched_payload = [
        [
            [
                "Fbv4je",
                json.dumps(request_payload, separators=(",", ":")),
                None,
                "generic",
            ]
        ]
    ]

    try:
        batched_response = session.post(
            GOOGLE_NEWS_BATCH_URL,
            data={"f.req": json.dumps(batched_payload, separators=(",", ":"))},
            headers={
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                "User-Agent": USER_AGENT,
                "Referer": response.url or url,
            },
            timeout=settings.crawl_timeout_seconds,
        )
        batched_response.raise_for_status()
    except requests.RequestException:
        return None

    for candidate in re.findall(r'https?://[^"\\\]]+', batched_response.text):
        resolved = candidate.replace("\\u003d", "=").replace("\\u0026", "&")
        if not _is_google_news_url(resolved):
            return resolved
    return None


def _google_news_locale_from_url(url: str) -> dict[str, str]:
    query = urlparse(url).query
    hl_match = re.search(r"(?:^|&)hl=([^&]+)", query)
    gl_match = re.search(r"(?:^|&)gl=([^&]+)", query)
    ceid_match = re.search(r"(?:^|&)ceid=([^&]+)", query)
    hl = hl_match.group(1) if hl_match else "en-US"
    gl = gl_match.group(1) if gl_match else "US"
    ceid = ceid_match.group(1) if ceid_match else f"{gl}:en"
    return {"hl": hl, "gl": gl, "ceid": ceid}
