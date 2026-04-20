from __future__ import annotations

import re

SOURCE_ALIAS_MAP = {
    "reuters": "Reuters",
    "bloomberg": "Bloomberg",
    "wsj": "WSJ",
    "wall street journal": "WSJ",
    "cnbc": "CNBC",
    "연합뉴스": "연합뉴스",
    "연합뉴스tv": "연합뉴스TV",
    "yonhap": "연합뉴스",
    "yna": "연합뉴스",
    "한국경제": "한국경제",
    "hankyung": "한국경제",
    "매일경제": "매일경제",
    "mk.co.kr": "매일경제",
    "한겨레": "한겨레",
    "경향신문": "경향신문",
    "조선일보": "조선일보",
}

SOURCE_WEIGHTS = {
    "Reuters": 0.96,
    "Bloomberg": 0.95,
    "WSJ": 0.93,
    "CNBC": 0.9,
    "연합뉴스": 0.9,
    "연합뉴스TV": 0.84,
    "한국경제": 0.84,
    "매일경제": 0.82,
    "한겨레": 0.78,
    "경향신문": 0.78,
    "조선일보": 0.8,
}

TRUSTED_READY_SOURCES = {
    "Reuters",
    "Bloomberg",
    "WSJ",
    "CNBC",
    "연합뉴스",
    "한국경제",
    "매일경제",
    "조선일보",
    "한겨레",
    "경향신문",
}


def normalize_source_name(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value).strip().lower()
    for alias, canonical in SOURCE_ALIAS_MAP.items():
        if alias in lowered:
            return canonical
    cleaned = re.sub(r"\s*-\s*google news$", "", value, flags=re.IGNORECASE).strip()
    return cleaned


def source_weight(value: str) -> float:
    return SOURCE_WEIGHTS.get(normalize_source_name(value), 0.42)


def is_trusted_ready_source(value: str) -> bool:
    return normalize_source_name(value) in TRUSTED_READY_SOURCES
