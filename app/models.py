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
    evidence_type: str = "news"
    document_id: str | None = None
    title: str | None = None
    source_id: str | None = None
    source_type: str | None = None
    authority_score: float | None = None
    freshness_score: float | None = None
    contradiction_hint: bool = False


@dataclass(slots=True)
class ReferenceSource:
    id: str
    name: str
    kind: str
    location: str
    authority_score: float = 0.8
    is_active: bool = True
    notes: str | None = None
    last_synced_at: datetime | None = None
    seed_urls: list[str] = field(default_factory=list)
    refresh_minutes: int = 60
    fetch_config: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ReferenceDocument:
    id: str
    source_id: str
    title: str
    source: str
    content: str
    doc_type: str
    updated_at: datetime
    url: str = ""
    file_path: str | None = None
    source_type: str = "manual"
    authority_score: float = 0.8
    content_hash: str = ""


@dataclass(slots=True)
class ReferenceChunk:
    id: str
    document_id: str
    source_id: str
    title: str
    source: str
    text: str
    chunk_index: int
    updated_at: datetime
    url: str = ""
    source_type: str = "manual"
    authority_score: float = 0.8
    content_hash: str = ""


@dataclass(slots=True)
class ReliabilityScore:
    value: float
    source_diversity: float
    recency: float
    evidence_coverage: float
    cross_source_confirmation: float
    reference_strength: float = 0.0
    contradiction_penalty: float = 0.0
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
    grounding_details: dict[str, object] = field(default_factory=dict)


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
