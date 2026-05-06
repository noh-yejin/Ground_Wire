from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from app.models import Article

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9가-힣]{2,}")
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?(?:%|년|개월|월|일|원|달러|bp)?")
ENTITY_PATTERN = re.compile(r"[A-Z][A-Za-z&.-]{1,}|[가-힣]{2,}|\d+(?:\.\d+)?%?")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "are",
    "new",
    "its",
    "over",
    "more",
    "than",
    "that",
    "this",
    "will",
    "into",
    "after",
    "amid",
    "says",
    "say",
    "news",
    "google",
    "nbsp",
    "update",
    "latest",
    "video",
    "live",
    "breaking",
    "net",
    "daum",
    "com",
    "그리고",
    "있다",
    "대한",
    "확대",
    "발표",
    "기대",
    "분석",
    "영향",
    "중심",
    "관련",
    "기자",
    "속보",
    "오늘",
    "정부",
    "대통령",
    "기준",
    "가능성",
    "전망",
    "브리핑",
    "global",
}

DISPLAY_ALIASES = {
    "ai": "AI",
    "semiconductor": "반도체",
    "chip": "반도체",
    "hbm": "HBM",
    "fed": "연준",
    "rate": "금리",
    "inflation": "인플레이션",
    "tariff": "관세",
    "sanction": "제재",
    "conflict": "분쟁",
    "war": "전쟁",
    "iran": "이란",
    "export": "수출",
    "수출": "수출",
    "금리": "금리",
    "반도체": "반도체",
    "환율": "환율",
    "energy": "에너지",
    "oil": "유가",
    "earnings": "실적",
    "tsmc": "TSMC",
    "nvidia": "엔비디아",
    "lpr": "금리",
    "cpi": "물가",
    "fomc": "연준",
    "rate": "금리",
    "rates": "금리",
    "cut": "인하",
    "cuts": "인하",
    "hike": "인상",
    "hikes": "인상",
    "inflation": "물가",
    "price": "물가",
    "prices": "물가",
    "earnings": "실적",
    "guidance": "가이던스",
    "export": "수출",
    "exports": "수출",
}

SEMANTIC_ALIASES = {
    "lpr": "금리",
    "금리인하": "금리 인하",
    "ratecut": "금리 인하",
    "ratecuts": "금리 인하",
    "rate": "금리",
    "rates": "금리",
    "cut": "인하",
    "cuts": "인하",
    "hike": "인상",
    "hikes": "인상",
    "cpi": "물가",
    "inflation": "물가",
    "price": "물가",
    "prices": "물가",
    "fed": "연준",
    "fomc": "연준",
    "semiconductor": "반도체",
    "chip": "반도체",
    "chips": "반도체",
    "memory": "메모리",
    "exports": "수출",
    "earnings": "실적",
}

GENERIC_TOPIC_TOKENS = {
    "ai",
    "iran",
    "middle",
    "east",
    "stock",
    "market",
    "world",
    "news",
    "update",
    "latest",
    "foodnavigator",
    "cnbc",
    "reuters",
    "wsj",
    "chain",
    "supply",
    "risk",
    "amid",
    "hits",
    "hit",
    "server",
    "servers",
    "투자",
    "시장",
    "업계",
}

MAX_CLUSTER_TIME_GAP = timedelta(hours=18)


def cluster_articles(articles: list[Article]) -> list[list[Article]]:
    clusters: list[dict[str, object]] = []
    for article in articles:
        tokens = set(_extract_keywords(article.title + " " + article.content)[:8])
        title_tokens = set(_extract_keywords(article.title)[:6])
        salient_title_tokens = set(_extract_salient_tokens(article.title))
        number_tokens = set(_extract_numbers(article.title + " " + article.content))
        concept_tokens = set(_extract_concepts(article.title + " " + article.content))
        entity_tokens = set(_extract_entities(article.title + " " + article.content))
        title_ngrams = _char_ngrams(_normalize_text_for_similarity(article.title))
        matched_cluster: dict[str, object] | None = None

        for cluster in clusters:
            overlap = len(tokens & cluster["tokens"])
            title_overlap = len(title_tokens & cluster["title_tokens"])
            salient_title_overlap = len(salient_title_tokens & cluster["salient_title_tokens"])
            similarity = overlap / max(len(tokens | cluster["tokens"]), 1)
            concept_overlap = len(concept_tokens & cluster["concept_tokens"])
            concept_similarity = concept_overlap / max(len(concept_tokens | cluster["concept_tokens"]), 1)
            entity_overlap = len(entity_tokens & cluster["entity_tokens"])
            title_ngram_similarity = _set_similarity(title_ngrams, cluster["title_ngrams"])
            number_compatible = _numbers_compatible(number_tokens, cluster["number_tokens"])
            time_compatible = _cluster_time_compatible(article.published_at, cluster["published_range"])
            if _should_merge_cluster(
                overlap=overlap,
                title_overlap=title_overlap,
                salient_title_overlap=salient_title_overlap,
                similarity=similarity,
                concept_overlap=concept_overlap,
                concept_similarity=concept_similarity,
                entity_overlap=entity_overlap,
                title_ngram_similarity=title_ngram_similarity,
                number_compatible=number_compatible,
                time_compatible=time_compatible,
            ):
                matched_cluster = cluster
                break

        if matched_cluster is None:
            clusters.append(
                {
                    "tokens": set(tokens),
                    "title_tokens": set(title_tokens),
                    "salient_title_tokens": set(salient_title_tokens),
                    "number_tokens": set(number_tokens),
                    "concept_tokens": set(concept_tokens),
                    "entity_tokens": set(entity_tokens),
                    "title_ngrams": set(title_ngrams),
                    "published_range": _published_range(article.published_at, article.published_at),
                    "articles": [article],
                }
            )
            continue

        matched_cluster["articles"].append(article)
        matched_cluster["tokens"].update(tokens)
        matched_cluster["title_tokens"].update(title_tokens)
        matched_cluster["salient_title_tokens"].update(salient_title_tokens)
        matched_cluster["number_tokens"].update(number_tokens)
        matched_cluster["concept_tokens"].update(concept_tokens)
        matched_cluster["entity_tokens"].update(entity_tokens)
        matched_cluster["title_ngrams"].update(title_ngrams)
        matched_cluster["published_range"] = _published_range_extend(matched_cluster["published_range"], article.published_at)

    return [cluster["articles"] for cluster in clusters]


def label_topic(articles: list[Article]) -> str:
    title_text = " ".join(article.title for article in articles)
    title_keywords = _extract_keywords(title_text)
    content_keywords = _extract_keywords(" ".join(article.content for article in articles))
    keywords = _rank_topic_keywords(title_keywords, content_keywords)
    if not keywords:
        return "Untitled issue"
    return canonicalize_topic(" · ".join(keywords[:3]))


def _rank_topic_keywords(title_keywords: list[str], content_keywords: list[str]) -> list[str]:
    ranked: list[str] = []
    merged = title_keywords[:10] + content_keywords[:14]
    for token in merged:
        if token in GENERIC_TOPIC_TOKENS:
            continue
        label = DISPLAY_ALIASES.get(token, token.upper() if token.isupper() else token)
        if label not in ranked:
            ranked.append(label)

    if not ranked:
        fallback = [DISPLAY_ALIASES.get(token, token) for token in title_keywords[:3] if token not in GENERIC_TOPIC_TOKENS]
        ranked.extend(fallback)
    return ranked


def canonicalize_topic(value: str) -> str:
    if not value:
        return value
    raw_parts = re.split(r"\s*[·/]\s*", value)
    normalized: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        token = part.strip()
        if not token:
            continue
        alias = DISPLAY_ALIASES.get(token.lower(), token)
        key = alias.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(alias)
    normalized.sort(key=lambda item: item.lower())
    return " · ".join(normalized[:3])


def canonical_topic_key(articles: list[Article]) -> str:
    return canonicalize_topic(label_topic(articles))


def _extract_keywords(text: str) -> list[str]:
    frequencies: dict[str, int] = defaultdict(int)
    for token in TOKEN_PATTERN.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) <= 2 and not any("\uac00" <= char <= "\ud7a3" for char in token):
            continue
        frequencies[token] += 1
    return [token for token, _ in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))]


def _extract_numbers(text: str) -> list[str]:
    return [token.lower() for token in NUMBER_PATTERN.findall(text.lower())]


def _extract_salient_tokens(text: str) -> list[str]:
    return [token for token in _extract_keywords(text) if token not in GENERIC_TOPIC_TOKENS][:5]


def _extract_concepts(text: str) -> list[str]:
    concepts: list[str] = []
    for token in _extract_keywords(text):
        normalized = _normalize_concept_token(token)
        if not normalized or normalized in GENERIC_TOPIC_TOKENS:
            continue
        if normalized not in concepts:
            concepts.append(normalized)
    return concepts[:10]


def _normalize_concept_token(token: str) -> str:
    lowered = token.lower().strip()
    if not lowered:
        return ""
    lowered = re.sub(r"(ing|ed|es|s)$", "", lowered) if lowered.isascii() else lowered
    return SEMANTIC_ALIASES.get(lowered, DISPLAY_ALIASES.get(lowered, lowered))


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    for token in ENTITY_PATTERN.findall(text):
        normalized = _normalize_concept_token(token)
        if len(normalized) < 2 or normalized in STOPWORDS or normalized in GENERIC_TOPIC_TOKENS:
            continue
        if normalized not in entities:
            entities.append(normalized)
    return entities[:12]


def _normalize_text_for_similarity(text: str) -> str:
    concepts = _extract_concepts(text)
    if concepts:
        return " ".join(concepts)
    return " ".join(_extract_keywords(text))


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[index:index + n] for index in range(len(compact) - n + 1)}


def _set_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _numbers_compatible(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return True
    return bool(left & right)


def _cluster_time_compatible(published_at: datetime, published_range: tuple[datetime, datetime]) -> bool:
    timestamp = _to_utc(published_at)
    start, end = published_range
    return not (timestamp < start - MAX_CLUSTER_TIME_GAP or timestamp > end + MAX_CLUSTER_TIME_GAP)


def _published_range(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    return (_to_utc(start), _to_utc(end))


def _published_range_extend(published_range: tuple[datetime, datetime], timestamp: datetime) -> tuple[datetime, datetime]:
    current = _to_utc(timestamp)
    return (min(published_range[0], current), max(published_range[1], current))


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _should_merge_cluster(
    overlap: int,
    title_overlap: int,
    salient_title_overlap: int,
    similarity: float,
    concept_overlap: int,
    concept_similarity: float,
    entity_overlap: int,
    title_ngram_similarity: float,
    number_compatible: bool,
    time_compatible: bool,
) -> bool:
    if not number_compatible or not time_compatible:
        return False
    if salient_title_overlap >= 2:
        return True
    if salient_title_overlap >= 1 and title_overlap >= 2:
        return True
    if salient_title_overlap >= 1 and overlap >= 3 and similarity >= 0.22:
        return True
    if title_overlap >= 2 and overlap >= 4 and similarity >= 0.35:
        return True
    if entity_overlap >= 2 and concept_overlap >= 2 and concept_similarity >= 0.3:
        return True
    if entity_overlap >= 1 and concept_overlap >= 3 and concept_similarity >= 0.42:
        return True
    if title_ngram_similarity >= 0.42 and concept_overlap >= 2:
        return True
    return False
