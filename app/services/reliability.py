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


def score_grounding(claim_results: list[dict], base_reliability: ReliabilityScore) -> dict:
    if not claim_results:
        return {
            "grounded_ratio": 0.0,
            "avg_claim_score": 0.0,
            "contradiction_ratio": 0.0,
            "issue_score": round(base_reliability.value * 0.4, 3),
            "reasons": ["검증 가능한 claim이 없습니다."],
        }

    grounded_results = [item for item in claim_results if item.get("ready")]
    contradiction_results = [item for item in claim_results if item.get("contradiction_count", 0) > 0]
    grounded_ratio = len(grounded_results) / len(claim_results)
    contradiction_ratio = len(contradiction_results) / len(claim_results)
    avg_claim_score = sum(float(item.get("score", 0.0)) for item in claim_results) / len(claim_results)

    issue_score = (
        avg_claim_score * 0.52
        + grounded_ratio * 0.28
        + base_reliability.value * 0.20
        - min(contradiction_ratio * 0.35, 0.35)
    )
    issue_score = round(max(0.0, min(issue_score, 1.0)), 3)

    reasons: list[str] = []
    if grounded_ratio < 0.5:
        reasons.append("검증 통과 claim 비율이 낮습니다.")
    if contradiction_ratio > 0:
        reasons.append("상충 근거가 감지된 claim이 있습니다.")
    if base_reliability.value < 0.7:
        reasons.append("출처/최신성 기반 기본 신뢰도가 높지 않습니다.")

    return {
        "grounded_ratio": round(grounded_ratio, 3),
        "avg_claim_score": round(avg_claim_score, 3),
        "contradiction_ratio": round(contradiction_ratio, 3),
        "issue_score": issue_score,
        "reasons": reasons,
    }


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
