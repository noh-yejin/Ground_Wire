from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class IssueStatus(str, Enum):
    READY = "READY"
    HOLD = "HOLD"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class IssuePriority(str, Enum):
    PRIORITY = "priority"
    GENERAL = "general"


class ImpactLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class Article:
    id: str
    title: str
    source: str
    published_at: datetime
    url: str
    content: str
    language: str = "ko"
    collected_at: datetime | None = None
    content_quality: float = 0.0


@dataclass(slots=True)
class EvidenceSnippet:
    article_id: str
    source: str
    quote: str
    url: str
    score: float = 0.0


@dataclass(slots=True)
class ReliabilityScore:
    value: float
    source_diversity: float
    recency: float
    evidence_coverage: float
    cross_source_confirmation: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisResult:
    summary: str
    keywords: list[str]
    key_signals: list[str]
    key_points: list[str]
    trend_summary: str
    sentiment: SentimentLabel
    market_impact: ImpactLabel
    policy_risk: RiskLevel
    volatility_risk: RiskLevel
    risk_points: list[str]
    grounded: bool
    priority: IssuePriority
    hold_reason: str | None = None


@dataclass(slots=True)
class Issue:
    id: str
    topic: str
    keywords: list[str]
    articles: list[Article]
    evidence: list[EvidenceSnippet]
    reliability: ReliabilityScore
    analysis: AnalysisResult
    status: IssueStatus
    updated_at: datetime
