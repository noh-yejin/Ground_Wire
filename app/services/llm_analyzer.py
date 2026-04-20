from __future__ import annotations

import logging
from pydantic import BaseModel, Field

from app.config import settings
from app.models import (
    AnalysisResult,
    Article,
    EvidenceSnippet,
    ImpactLabel,
    IssuePriority,
    ReliabilityScore,
    RiskLevel,
    SentimentLabel,
)
from app.services.clustering import _extract_keywords
from app.services.source_normalizer import normalize_source_name

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional in tests
    OpenAI = None

logger = logging.getLogger(__name__)

NEGATIVE_HINTS = ("위험", "후퇴", "둔화", "긴축", "변동성", "제재")
POSITIVE_HINTS = ("증가", "개선", "회복", "확대", "기대", "성장")


class IssueAnalysisSchema(BaseModel):
    summary: str = Field(description="Evidence-grounded issue summary")
    keywords: list[str] = Field(description="Issue keywords")
    key_signals: list[str] = Field(description="Important priority signals")
    key_points: list[str] = Field(description="Top short bullet points in Korean")
    trend_summary: str = Field(description="Short Korean explanation of whether issue attention is rising, flat, or falling")
    sentiment: SentimentLabel = Field(description="Overall sentiment")
    market_impact: ImpactLabel = Field(description="Estimated market impact direction")
    policy_risk: RiskLevel = Field(description="Estimated policy risk level")
    volatility_risk: RiskLevel = Field(description="Estimated volatility risk level")
    risk_points: list[str] = Field(description="Key risks or caveats")
    grounded: bool = Field(description="Whether the answer is grounded in evidence")
    priority: IssuePriority = Field(description="priority or general")
    hold_reason: str | None = Field(default=None, description="Why the issue should be held if not ready")


class LLMAnalyzer:
    """Backend-only analysis interface.

    Uses OpenAI Responses API on the backend when an API key is configured.
    Falls back to deterministic local analysis when no API key is available.
    """

    def __init__(self) -> None:
        self.last_remote_error: str | None = None
        self.last_remote_success: bool = False
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds) if (
            settings.openai_api_key and OpenAI is not None
        ) else None

    def analyze(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        reliability: ReliabilityScore,
        hold_reason: str | None,
    ) -> AnalysisResult:
        joined_titles = " ".join(article.title for article in articles)
        joined_content = " ".join(article.content for article in articles)
        keywords = _extract_keywords(f"{joined_titles} {joined_content}")[:6]
        key_signals = derive_key_signals(topic, articles, keywords)
        key_points = derive_key_points(evidence, articles)
        trend_summary = derive_trend_summary(articles)
        sentiment = detect_sentiment(joined_titles + " " + joined_content)
        risk_points = derive_risk_points(articles, reliability, hold_reason)
        priority = derive_priority(topic, articles, reliability, key_signals)
        market_impact = derive_market_impact(topic, articles, sentiment)
        policy_risk = derive_policy_risk(topic, articles, key_signals)
        volatility_risk = derive_volatility_risk(topic, articles, sentiment)

        if hold_reason:
            return AnalysisResult(
                summary=f"보류된 이슈입니다. 사유: {hold_reason}",
                keywords=keywords,
                key_signals=key_signals,
                key_points=key_points,
                trend_summary=trend_summary,
                sentiment=sentiment,
                market_impact=market_impact,
                policy_risk=policy_risk,
                volatility_risk=volatility_risk,
                risk_points=risk_points,
                grounded=False,
                priority=priority,
                hold_reason=hold_reason,
            )

        if self.client is not None:
            remote = self._analyze_with_openai(topic, articles, evidence, reliability)
            if remote is not None:
                return remote

        self.last_remote_success = False
        return AnalysisResult(
            summary=build_local_summary(topic, articles, evidence, reliability),
            keywords=keywords,
            key_signals=key_signals,
            key_points=key_points,
            trend_summary=trend_summary,
            sentiment=sentiment,
            market_impact=market_impact,
            policy_risk=policy_risk,
            volatility_risk=volatility_risk,
            risk_points=risk_points,
            grounded=True,
            priority=priority,
            hold_reason=None,
        )

    def debug_status(self) -> dict:
        return {
            "openai_package_available": OpenAI is not None,
            "api_key_configured": bool(settings.openai_api_key),
            "client_initialized": self.client is not None,
            "model": settings.openai_model,
            "last_remote_success": self.last_remote_success,
            "last_remote_error": self.last_remote_error,
        }

    def _analyze_with_openai(
        self,
        topic: str,
        articles: list[Article],
        evidence: list[EvidenceSnippet],
        reliability: ReliabilityScore,
    ) -> AnalysisResult | None:
        self.last_remote_error = None
        try:
            response = self.client.responses.parse(
                model=settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are a reliability-first issue analysis model. "
                            "Only use the provided evidence. If evidence is insufficient, say so in hold_reason. "
                            "All outputs must be written in Korean, even when the source articles are in English. "
                            "Translate the source meaning into Korean before summarizing. "
                            "Return concise Korean JSON-compatible structured output."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_issue_prompt(topic, articles, evidence, reliability),
                    },
                ],
                text_format=IssueAnalysisSchema,
            )
        except Exception as exc:
            self.last_remote_success = False
            self.last_remote_error = f"{type(exc).__name__}: {exc}"
            logger.warning("OpenAI issue analysis failed: %s", self.last_remote_error)
            return None

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            self.last_remote_success = False
            self.last_remote_error = "No output_parsed returned from Responses API."
            logger.warning("OpenAI issue analysis failed: %s", self.last_remote_error)
            return None
        self.last_remote_success = True
        return AnalysisResult(
            summary=parsed.summary,
            keywords=parsed.keywords[:6],
            key_signals=parsed.key_signals[:5],
            key_points=parsed.key_points[:4],
            trend_summary=parsed.trend_summary,
            sentiment=parsed.sentiment,
            market_impact=parsed.market_impact,
            policy_risk=parsed.policy_risk,
            volatility_risk=parsed.volatility_risk,
            risk_points=parsed.risk_points[:5],
            grounded=parsed.grounded,
            priority=parsed.priority,
            hold_reason=parsed.hold_reason,
        )


def detect_sentiment(text: str) -> SentimentLabel:
    lowered = text.lower()
    positive = sum(1 for token in POSITIVE_HINTS if token in lowered)
    negative = sum(1 for token in NEGATIVE_HINTS if token in lowered)
    if positive and negative:
        return SentimentLabel.MIXED
    if positive > negative:
        return SentimentLabel.POSITIVE
    if negative > positive:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


def derive_risk_points(
    articles: list[Article],
    reliability: ReliabilityScore,
    hold_reason: str | None,
) -> list[str]:
    risks: list[str] = []
    if hold_reason:
        risks.append(hold_reason)
    if reliability.source_diversity < 0.6:
        risks.append("출처 다양성이 충분하지 않을 수 있습니다.")
    if reliability.cross_source_confirmation < 0.75:
        risks.append("복수 출처 교차 검증이 약합니다.")
    if any("변동성" in article.title for article in articles):
        risks.append("시장 변동성 확대 가능성에 유의해야 합니다.")
    return risks or ["현재 기준에서 두드러진 리스크는 제한적입니다."]


def derive_key_signals(topic: str, articles: list[Article], keywords: list[str]) -> list[str]:
    haystack = f"{topic} " + " ".join(article.title for article in articles)
    signals = [keyword for keyword in settings.priority_keywords if keyword.lower() in haystack.lower()]
    ordered: list[str] = []
    for signal in signals + keywords:
        if signal == "nbsp":
            continue
        if signal not in ordered:
            ordered.append(signal)
    return ordered[:5]


def derive_key_points(evidence: list[EvidenceSnippet], articles: list[Article]) -> list[str]:
    if evidence:
        return [item.quote[:120] for item in evidence[:3]]
    fallback: list[str] = []
    for article in articles[:3]:
        sentence = article.content.split(".")[0].strip()
        if sentence:
            fallback.append(sentence[:120])
    return fallback


def derive_market_impact(topic: str, articles: list[Article], sentiment: SentimentLabel) -> ImpactLabel:
    haystack = f"{topic} " + " ".join(article.title for article in articles)
    lowered = haystack.lower()
    if any(token in lowered for token in ("금리", "inflation", "tariff", "oil", "유가", "관세", "연준", "환율")):
        if sentiment in (SentimentLabel.NEGATIVE, SentimentLabel.MIXED):
            return ImpactLabel.NEGATIVE
        if sentiment == SentimentLabel.POSITIVE:
            return ImpactLabel.POSITIVE
    return ImpactLabel(sentiment.value) if sentiment != SentimentLabel.MIXED else ImpactLabel.MIXED


def derive_policy_risk(topic: str, articles: list[Article], key_signals: list[str]) -> RiskLevel:
    haystack = f"{topic} " + " ".join(article.title for article in articles) + " " + " ".join(key_signals)
    lowered = haystack.lower()
    high_tokens = ("정책", "regulation", "sanction", "제재", "관세", "금리", "연준", "정부", "국회")
    medium_tokens = ("guidance", "실적", "전망", "협상", "정상회담")
    if any(token in lowered for token in high_tokens):
        return RiskLevel.HIGH
    if any(token in lowered for token in medium_tokens):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def derive_volatility_risk(topic: str, articles: list[Article], sentiment: SentimentLabel) -> RiskLevel:
    haystack = f"{topic} " + " ".join(article.title for article in articles)
    lowered = haystack.lower()
    if any(token in lowered for token in ("분쟁", "conflict", "war", "attack", "tariff", "제재", "유가", "환율")):
        return RiskLevel.HIGH
    if sentiment == SentimentLabel.MIXED or any(token in lowered for token in ("실적", "guidance", "변동성", "긴축")):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def derive_trend_summary(articles: list[Article]) -> str:
    if len(articles) < 2:
        return "표본이 적어 추세를 판단하기 어렵습니다."
    ordered = sorted(articles, key=lambda article: article.published_at)
    midpoint = max(len(ordered) // 2, 1)
    early_count = len(ordered[:midpoint])
    late_count = len(ordered[midpoint:])
    if late_count >= early_count * 1.5:
        return "최근 시간대로 갈수록 기사 출현이 늘어나는 상승 추세입니다."
    if early_count >= late_count * 1.5:
        return "초반 대비 최근 기사 출현이 줄어드는 둔화 추세입니다."
    return "최근 시간대 기준 기사 출현 빈도가 비교적 안정적으로 유지되고 있습니다."


def derive_priority(
    topic: str,
    articles: list[Article],
    reliability: ReliabilityScore,
    key_signals: list[str],
) -> IssuePriority:
    haystack = f"{topic} " + " ".join(article.title for article in articles)
    signal_count = sum(1 for keyword in settings.priority_keywords if keyword.lower() in haystack.lower())
    if signal_count >= 2 or (signal_count >= 1 and reliability.value >= 0.75):
        return IssuePriority.PRIORITY
    return IssuePriority.GENERAL


def build_local_summary(
    topic: str,
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    reliability: ReliabilityScore,
) -> str:
    if evidence:
        first = evidence[0].quote.strip()
        second = evidence[1].quote.strip() if len(evidence) > 1 else ""
        summary = first[:170]
        if second:
            summary += f" 또한 {second[:110]}"
        return summary

    lead = articles[0].content.split(".")[0].strip() if articles and articles[0].content else ""
    if lead:
        return lead[:220]

    return (
        f"{topic} 관련 보도가 {len(articles)}건 수집됐고 "
        f"{len({article.source for article in articles})}개 출처에서 확인됐습니다. "
        f"현재 신뢰도 점수는 {reliability.value}입니다."
    )


def _build_issue_prompt(
    topic: str,
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    reliability: ReliabilityScore,
) -> str:
    article_lines = [
        f"- [{normalize_source_name(article.source)}] {article.title} | {article.published_at.isoformat()} | {article.url}"
        for article in articles[:8]
    ]
    evidence_lines = [
        f"- [{normalize_source_name(item.source)}] score={item.score}: {item.quote} ({item.url})"
        for item in evidence[:6]
    ]
    return (
        f"이슈 주제: {topic}\n"
        f"신뢰도 점수: {reliability.value}\n"
        f"기사 목록:\n" + "\n".join(article_lines) + "\n"
        f"근거 스니펫:\n" + "\n".join(evidence_lines) + "\n"
        "요구사항:\n"
        "1. 근거 기반으로만 종합 요약\n"
        "2. 키워드 3~6개\n"
        "3. key_signals에는 속보성/시장성/정책성 핵심 시그널만 1~5개\n"
        "4. key_points에는 핵심 bullet 2~4개를 한국어로 제공\n"
        "5. trend_summary에는 추이를 한국어 한 문장으로 설명\n"
        "6. sentiment는 positive, neutral, negative, mixed 중 하나\n"
        "7. market_impact는 positive, neutral, negative, mixed 중 하나\n"
        "8. policy_risk와 volatility_risk는 low, medium, high 중 하나\n"
        "9. priority는 priority 또는 general\n"
        "10. 리스크 포인트 1~5개\n"
        "11. 근거 부족이면 grounded=false 또는 hold_reason 명시\n"
    )
