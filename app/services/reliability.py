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
        return ReliabilityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ["No articles"])

    unique_sources = len({article.source for article in articles})
    reference_evidence = [item for item in evidence if item.evidence_type == "reference"]
    news_evidence = [item for item in evidence if item.evidence_type != "reference"]
    unique_reference_sources = len({item.source_id or item.source for item in reference_evidence})
    source_diversity = min((unique_sources / 3) + (min(unique_reference_sources, 2) * 0.12), 1.0)

    now = datetime.now(timezone.utc)
    latest_age_hours = min(
        max((now - _to_utc(article.published_at)).total_seconds() / 3600, 0) for article in articles
    )
    article_recency = max(0.0, 1 - (latest_age_hours / 24))
    reference_recency = (
        sum(float(item.freshness_score or 0.45) for item in reference_evidence) / len(reference_evidence)
        if reference_evidence
        else 0.0
    )
    recency = round((article_recency * 0.78) + (reference_recency * 0.22), 3) if reference_evidence else round(article_recency, 3)

    weighted_evidence = len(news_evidence) + sum((item.authority_score or 0.75) * 1.2 for item in reference_evidence)
    evidence_coverage = min(weighted_evidence / max(len(articles) + 1, 1), 1.0)
    reference_strength = (
        min(
            (
                sum((item.authority_score or 0.75) * (item.freshness_score or 0.55) for item in reference_evidence)
                / max(len(reference_evidence), 1)
            )
            + min(unique_reference_sources * 0.12, 0.24),
            1.0,
        )
        if reference_evidence
        else 0.0
    )
    contradiction_penalty = min(sum(0.12 for item in evidence if item.contradiction_hint), 0.3)
    cross_source_confirmation = min(
        (unique_sources / max(len(articles), 1)) * 0.60
        + (reference_strength * 0.20)
        + (min(unique_reference_sources, 2) * 0.08)
        + 0.26,
        1.0,
    )

    value = round(
        (
            source_diversity * 0.28
            + recency * 0.20
            + evidence_coverage * 0.24
            + cross_source_confirmation * 0.18
            + reference_strength * 0.10
            - contradiction_penalty
        ),
        3,
    )
    value = max(0.0, min(value, 1.0))

    reasons: list[str] = []
    if unique_sources < 2:
        reasons.append("Need at least two independent sources")
    if len(evidence) < 2:
        reasons.append("Evidence coverage is too thin")
    if recency < 0.5:
        reasons.append("Articles are not recent enough")
    if reference_evidence and reference_strength < 0.55:
        reasons.append("Reference evidence is present but not strong enough")
    if contradiction_penalty > 0:
        reasons.append("Counter or correction-style evidence needs review")

    return ReliabilityScore(
        value=round(value, 3),
        source_diversity=round(source_diversity, 3),
        recency=round(recency, 3),
        evidence_coverage=round(evidence_coverage, 3),
        cross_source_confirmation=round(cross_source_confirmation, 3),
        reference_strength=round(reference_strength, 3),
        contradiction_penalty=round(contradiction_penalty, 3),
        reasons=reasons,
    )


def score_grounding(claim_results: list[dict], base_reliability: ReliabilityScore) -> dict:
    if not claim_results:
        return {
            "grounded_ratio": 0.0,
            "avg_claim_score": 0.0,
            "contradiction_ratio": 0.0,
            "reference_grounded_ratio": 0.0,
            "counter_weight": 0.0,
            "issue_score": round(base_reliability.value * 0.4, 3),
            "reasons": ["검증 가능한 claim이 없습니다."],
        }

    grounded_results = [item for item in claim_results if item.get("ready")]
    contradiction_results = [item for item in claim_results if item.get("contradiction_count", 0) > 0]
    reference_backed_results = [item for item in claim_results if item.get("reference_support_count", 0) > 0]
    grounded_ratio = len(grounded_results) / len(claim_results)
    contradiction_ratio = len(contradiction_results) / len(claim_results)
    reference_grounded_ratio = len(reference_backed_results) / len(claim_results)
    counter_weight = min(
        sum(float(item.get("contradiction_weight", 0.0)) for item in claim_results) / max(len(claim_results), 1),
        1.0,
    )
    avg_claim_score = sum(float(item.get("score", 0.0)) for item in claim_results) / len(claim_results)

    issue_score = (
        avg_claim_score * 0.42
        + grounded_ratio * 0.20
        + reference_grounded_ratio * 0.18
        + base_reliability.value * 0.15
        + min(base_reliability.reference_strength, 1.0) * 0.10
        - min(contradiction_ratio * 0.22, 0.22)
        - min(counter_weight * 0.32, 0.32)
        - min(base_reliability.contradiction_penalty * 0.25, 0.12)
    )
    issue_score = round(max(0.0, min(issue_score, 1.0)), 3)

    reasons: list[str] = []
    if grounded_ratio < 0.5:
        reasons.append("검증 통과 claim 비율이 낮습니다.")
    if reference_grounded_ratio < 0.35:
        reasons.append("참조 문서 기반 claim 뒷받침이 충분하지 않습니다.")
    if contradiction_ratio > 0:
        reasons.append("상충 근거가 감지된 claim이 있습니다.")
    if counter_weight > 0.22:
        reasons.append("정정/반박 성격의 근거 비중이 높습니다.")
    if base_reliability.value < 0.7:
        reasons.append("출처/최신성 기반 기본 신뢰도가 높지 않습니다.")

    return {
        "grounded_ratio": round(grounded_ratio, 3),
        "avg_claim_score": round(avg_claim_score, 3),
        "contradiction_ratio": round(contradiction_ratio, 3),
        "reference_grounded_ratio": round(reference_grounded_ratio, 3),
        "counter_weight": round(counter_weight, 3),
        "issue_score": issue_score,
        "reasons": reasons,
    }


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
