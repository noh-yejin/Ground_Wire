from __future__ import annotations

from datetime import datetime, timezone

from app.models import Article, EvidenceSnippet
from app.repository import IssueRepository
from app.services.retrieval.vector_store import build_store
from app.services.source_normalizer import is_trusted_ready_source, source_weight


class NewsEvidenceRetriever:
    def __init__(self, repository: IssueRepository | None = None) -> None:
        self.repository = repository

    def retrieve(
        self,
        articles: list[Article],
        query_text: str,
        top_k: int = 12,
        corpus_articles: list[Article] | None = None,
    ) -> list[EvidenceSnippet]:
        article_pool = self._merge_articles(articles, corpus_articles or [])
        store = build_store(repository=self.repository)
        try:
            store.add_articles(article_pool)
            retrieved = store.query(query_text, top_k=top_k)
        except Exception:
            return []

        evidence = [
            EvidenceSnippet(
                article_id=doc.article_id,
                source=doc.source,
                quote=doc.text,
                url=doc.url,
                score=round(score, 3),
                evidence_type="news",
                document_id=doc.document_id,
                title=doc.title,
                source_type="news",
                freshness_score=_freshness_score(doc.updated_at, horizon_hours=24),
            )
            for doc, score in retrieved
        ]
        return self.rerank(self.filter(evidence))[:5]

    def filter(self, evidence: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        filtered = [
            item for item in evidence if source_weight(item.source) >= 0.75 and item.score >= 0.08 and len(item.quote) >= 40
        ]
        trusted = [item for item in filtered if is_trusted_ready_source(item.source)]
        return trusted or filtered

    def rerank(self, evidence: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        return sorted(
            evidence,
            key=lambda item: (
                source_weight(item.source) * 0.50
                + item.score * 0.35
                + (item.freshness_score or 0.0) * 0.15,
                len(item.quote),
            ),
            reverse=True,
        )

    def _merge_articles(self, articles: list[Article], extra_articles: list[Article]) -> list[Article]:
        merged: dict[str, Article] = {}
        for article in [*articles, *extra_articles]:
            merged[article.id] = article
        return list(merged.values())


def _freshness_score(value: datetime | None, horizon_hours: int) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    age_hours = max((datetime.now(timezone.utc) - value).total_seconds() / 3600, 0.0)
    return round(max(0.1, 1 - (age_hours / max(horizon_hours, 1))), 3)
