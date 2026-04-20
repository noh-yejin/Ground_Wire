from __future__ import annotations

from datetime import datetime, timezone

from app.models import Article, EvidenceSnippet, ReliabilityScore


def build_evidence(articles: list[Article]) -> list[EvidenceSnippet]:
    evidence: list[EvidenceSnippet] = []
    for article in articles[:5]:
        sentence = article.content.split(".")[0].strip()
        if sentence:
            evidence.append(
                EvidenceSnippet(
                    article_id=article.id,
                    source=article.source,
                    quote=sentence,
                    url=article.url,
                )
            )
    return evidence


def score_issue(articles: list[Article], evidence: list[EvidenceSnippet]) -> ReliabilityScore:
    if not articles:
        return ReliabilityScore(0.0, 0.0, 0.0, 0.0, 0.0, ["No articles"])

    unique_sources = len({article.source for article in articles})
    source_diversity = min(unique_sources / 4, 1.0)

    now = datetime.now(timezone.utc)
    latest_age_hours = min(
        max((now - _to_utc(article.published_at)).total_seconds() / 3600, 0) for article in articles
    )
    recency = max(0.0, 1 - (latest_age_hours / 24))

    evidence_coverage = min(len(evidence) / max(len(articles), 1), 1.0)
    cross_source_confirmation = min(unique_sources / max(len(articles), 1) + 0.35, 1.0)

    value = round(
        (
            source_diversity * 0.30
            + recency * 0.20
            + evidence_coverage * 0.25
            + cross_source_confirmation * 0.25
        ),
        3,
    )

    reasons: list[str] = []
    if unique_sources < 2:
        reasons.append("Need at least two independent sources")
    if len(evidence) < 2:
        reasons.append("Evidence coverage is too thin")
    if recency < 0.5:
        reasons.append("Articles are not recent enough")

    return ReliabilityScore(
        value=value,
        source_diversity=round(source_diversity, 3),
        recency=round(recency, 3),
        evidence_coverage=round(evidence_coverage, 3),
        cross_source_confirmation=round(cross_source_confirmation, 3),
        reasons=reasons,
    )


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
