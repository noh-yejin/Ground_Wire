from __future__ import annotations

from app.models import Article, EvidenceSnippet, IssueStatus, ReliabilityScore


def summarize_issue(
    topic: str,
    articles: list[Article],
    evidence: list[EvidenceSnippet],
    reliability: ReliabilityScore,
    hold_threshold: float,
    min_articles: int,
    min_sources: int,
) -> tuple[str, IssueStatus]:
    unique_sources = len({article.source for article in articles})

    if (
        reliability.value < hold_threshold
        or len(articles) < min_articles
        or unique_sources < min_sources
        or len(evidence) < 2
    ):
        reason_text = "; ".join(reliability.reasons) or "Insufficient grounded evidence"
        return f"보류: {reason_text}", IssueStatus.HOLD

    headline_titles = ", ".join(article.title for article in articles[:3])
    evidence_sources = ", ".join(sorted({item.source for item in evidence}))
    summary = (
        f"{topic} 이슈는 현재 {len(articles)}건의 기사와 {unique_sources}개 출처에서 교차 확인되었습니다. "
        f"핵심 근거 출처는 {evidence_sources}이며, 기사들 전반에서 공통적으로 확인되는 흐름은 다음과 같습니다: "
        f"{headline_titles}."
    )
    return summary, IssueStatus.READY
