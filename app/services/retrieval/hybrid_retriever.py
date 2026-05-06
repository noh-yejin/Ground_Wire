from __future__ import annotations

from app.models import Article, EvidenceSnippet
from app.repository import IssueRepository
from app.services.retrieval.news_retriever import NewsEvidenceRetriever
from app.services.retrieval.reference_retriever import ReferenceEvidenceRetriever
from app.services.source_normalizer import source_weight


class HybridEvidenceRetriever:
    def __init__(self, repository: IssueRepository | None = None) -> None:
        self.repository = repository or IssueRepository()
        self.news_retriever = NewsEvidenceRetriever(repository=self.repository)
        self.reference_retriever = ReferenceEvidenceRetriever(repository=self.repository)

    def retrieve(self, articles: list[Article]) -> list[EvidenceSnippet]:
        query_text = " ".join(article.title for article in articles[:4])
        return self.retrieve_with_query(articles, query_text=query_text)

    def retrieve_with_query(
        self,
        articles: list[Article],
        query_text: str,
        top_k: int = 12,
        corpus_articles: list[Article] | None = None,
    ) -> list[EvidenceSnippet]:
        news = self.news_retriever.retrieve(articles, query_text=query_text, top_k=top_k, corpus_articles=corpus_articles)
        reference = self.reference_retriever.retrieve(query_text, top_k=max(4, top_k // 2))
        combined = self._merge_evidence_lists(news, reference)
        return self.rerank(combined)[:5]

    def retrieve_for_claim(
        self,
        claim_text: str,
        articles: list[Article],
        corpus_articles: list[Article] | None = None,
    ) -> list[EvidenceSnippet]:
        return self.retrieve_with_query(articles, query_text=claim_text, top_k=14, corpus_articles=corpus_articles)

    def retrieve_counter_evidence(
        self,
        claim_text: str,
        articles: list[Article],
        corpus_articles: list[Article] | None = None,
    ) -> list[EvidenceSnippet]:
        counter_query = (
            f"{claim_text} 반박 부인 철회 정정 해명 번복 부정 상충 아니다 사실무근 "
            "clarified denied revised correction withdrawn no plan not considering"
        )
        news = self.news_retriever.retrieve(articles, query_text=counter_query, top_k=10, corpus_articles=corpus_articles)
        reference = self.reference_retriever.retrieve(counter_query, top_k=6)
        return self.rerank(self._merge_evidence_lists(news, reference))[:5]

    def retrieve_external_for_claim(self, claim_text: str, articles: list[Article], corpus_articles: list[Article]) -> list[EvidenceSnippet]:
        return self.reference_retriever.retrieve(claim_text, top_k=10)

    def source_weight(self, source: str) -> float:
        return source_weight(source)

    def multi_source_verified(self, evidence: list[EvidenceSnippet]) -> bool:
        return len({item.source for item in evidence}) >= 2

    def rerank(self, evidence: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        return sorted(
            evidence,
            key=lambda item: (
                self._evidence_weight(item) * 0.50
                + item.score * 0.30
                + ((item.freshness_score or 0.0) * 0.12)
                + (0.08 if item.contradiction_hint else 0.0),
                len(item.quote),
            ),
            reverse=True,
        )

    def filter(self, evidence: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        filtered = [
            item
            for item in evidence
            if self._evidence_weight(item) >= 0.55 and item.score >= 0.08 and len(item.quote) >= 40
        ]
        return filtered or evidence

    def _merge_evidence_lists(self, *lists: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        merged: dict[tuple[str, str], EvidenceSnippet] = {}
        for items in lists:
            for item in items:
                key = (item.article_id, item.quote)
                if key not in merged or item.score > merged[key].score:
                    merged[key] = item
        return list(merged.values())

    def _evidence_weight(self, evidence: EvidenceSnippet) -> float:
        if evidence.evidence_type == "reference":
            return evidence.authority_score or 0.92
        return source_weight(evidence.source)
