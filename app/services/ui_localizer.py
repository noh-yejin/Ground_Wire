from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from app.config import settings

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional in tests
    OpenAI = None

logger = logging.getLogger(__name__)


class UILocalizationSchema(BaseModel):
    text: str = Field(description="Natural Korean UI text")


class UIDisplayLocalizer:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds) if (
            settings.openai_api_key and OpenAI is not None
        ) else None
        self._cache: dict[tuple[str, str, int], str] = {}

    def localize_label(self, text: str) -> str:
        return self._localize(text, mode="label", max_chars=28)

    def localize_summary(self, text: str) -> str:
        return self._localize(text, mode="summary", max_chars=140)

    def localize_point(self, text: str, max_chars: int = 30) -> str:
        return self._localize(text, mode="point", max_chars=max_chars)

    def _localize(self, text: str, *, mode: str, max_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
        if not cleaned:
            return ""
        cache_key = (mode, cleaned, max_chars)
        if cache_key in self._cache:
            return self._cache[cache_key]

        needs_llm = bool(re.search(r"[A-Za-z]", cleaned)) or (mode == "point" and len(cleaned) > max_chars)
        if not needs_llm or self.client is None:
            result = self._fallback(cleaned, mode=mode, max_chars=max_chars)
            self._cache[cache_key] = result
            return result

        try:
            response = self.client.responses.parse(
                model=settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You convert financial/news UI text into natural Korean for display only. "
                            "Do not add facts. Keep meaning faithful. "
                            "For labels, keep them very short. "
                            "For points, compress the whole sentence into one concise Korean phrase instead of just truncating the first sentence. "
                            "Return only one Korean string."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"mode={mode}\n"
                            f"max_chars={max_chars}\n"
                            f"text={cleaned}\n"
                            "규칙:\n"
                            "- 영어는 자연스러운 한국어로 번역\n"
                            "- 고유명사는 필요할 때 유지\n"
                            "- point 모드에서는 문장 전체 의미를 유지한 짧은 한 문장으로 압축\n"
                            "- summary 모드는 핵심만 자연스럽게 한국어로 정리\n"
                        ),
                    },
                ],
                text_format=UILocalizationSchema,
            )
            parsed = getattr(response, "output_parsed", None)
            if parsed and parsed.text:
                result = self._fallback(parsed.text, mode=mode, max_chars=max_chars)
                self._cache[cache_key] = result
                return result
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning("UI localization failed: %s: %s", type(exc).__name__, exc)

        result = self._fallback(cleaned, mode=mode, max_chars=max_chars)
        self._cache[cache_key] = result
        return result

    def _fallback(self, text: str, *, mode: str, max_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if mode == "point":
            compact = re.sub(r"\([^)]*\)", "", cleaned)
            compact = re.sub(r"\[[^\]]*\]", "", compact)
            compact = re.sub(r"^[\-\u2022]+\s*", "", compact)
            compact = compact.strip()
            if len(compact) <= max_chars:
                return compact
            return f"{compact[:max_chars].rstrip()}..."
        if len(cleaned) <= max_chars:
            return cleaned
        return f"{cleaned[:max_chars].rstrip()}..."


ui_localizer = UIDisplayLocalizer()
