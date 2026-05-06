from __future__ import annotations

from datetime import datetime, timezone

from app.models import EvidenceSnippet
from app.repository import IssueRepository
from app.services.retrieval.vector_store import build_store


class ReferenceEvidenceRetriever:
    def __init__(self, repository: IssueRepository | None = None) -> None:
        self.repository = repository or IssueRepository()

    def retrieve(self, query_text: str, top_k: int = 10) -> list[EvidenceSnippet]:
        reference_chunks = self.repository.list_reference_chunks(active_only=True)
        if not reference_chunks:
            return []
        store = build_store(repository=self.repository)
        try:
            store.add_reference_chunks(reference_chunks)
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
                evidence_type="reference",
                document_id=doc.document_id,
                title=doc.title,
                source_id=getattr(doc, "source_id", None),
                source_type=doc.source_type,
                authority_score=doc.authority_score,
                freshness_score=_freshness_score(doc.updated_at, horizon_days=45),
                contradiction_hint=_looks_like_counter_update(doc.text, doc.title),
            )
            for doc, score in retrieved
        ]
        return self.rerank(self.filter(evidence))[:5]

    def filter(self, evidence: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        filtered = [
            item
            for item in evidence
            if (item.authority_score or 0.0) >= 0.55 and item.score >= 0.08 and len(item.quote) >= 40
        ]
        return filtered or evidence

    def rerank(self, evidence: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        return sorted(
            evidence,
            key=lambda item: (
                ((item.authority_score or 0.8) * 0.48)
                + item.score * 0.30
                + ((item.freshness_score or 0.0) * 0.14)
                + (0.08 if item.contradiction_hint else 0.0),
                len(item.quote),
            ),
            reverse=True,
        )


def _freshness_score(value: datetime | None, horizon_days: int) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    age_days = max((datetime.now(timezone.utc) - value).total_seconds() / 86400, 0.0)
    return round(max(0.2, 1 - (age_days / max(horizon_days, 1))), 3)


def _looks_like_counter_update(text: str, title: str | None) -> bool:
    lowered = f"{title or ''} {text}".lower()
    return any(
        token in lowered
        for token in (
            "부인",
            "반박",
            "정정",
            "철회",
            "해명",
            "번복",
            "사실무근",
            "clarified",
            "clarification",
            "denied",
            "deny",
            "revised",
            "withdrawn",
            "withdraw",
            "correction",
        )
    )
