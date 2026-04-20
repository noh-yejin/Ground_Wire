from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt

from app.config import settings
from app.models import Article, EvidenceSnippet
from app.services.clustering import _extract_keywords
from app.services.source_normalizer import is_trusted_ready_source, normalize_source_name, source_weight

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


@dataclass(slots=True)
class VectorDocument:
    article_id: str
    source: str
    url: str
    text: str
    vector: Counter[str] | list[float]


class SimpleVectorStore:
    def __init__(self) -> None:
        self.documents: list[VectorDocument] = []

    def add_articles(self, articles: list[Article]) -> None:
        for article in articles:
            for chunk in _article_chunks(article):
                self.documents.append(
                    VectorDocument(
                        article_id=article.id,
                        source=normalize_source_name(article.source),
                        url=article.url,
                        text=chunk,
                        vector=Counter(_extract_keywords(chunk)),
                    )
                )

    def query(self, text: str, top_k: int = 10) -> list[tuple[VectorDocument, float]]:
        query_vector = Counter(_extract_keywords(text))
        scored: list[tuple[VectorDocument, float]] = []
        for doc in self.documents:
            score = _cosine_similarity(query_vector, doc.vector)
            if score > 0:
                scored.append((doc, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


class OpenAIEmbeddingStore:
    def __init__(self) -> None:
        if OpenAI is None or not settings.openai_api_key:
            raise RuntimeError("OpenAI embedding client is unavailable")
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds)
        self.documents: list[VectorDocument] = []

    def add_articles(self, articles: list[Article]) -> None:
        chunks: list[tuple[Article, str]] = []
        for article in articles:
            for chunk in _article_chunks(article):
                chunks.append((article, chunk))
        if not chunks:
            return

        response = self.client.embeddings.create(
            model=settings.embedding_model,
            input=[chunk for _, chunk in chunks],
        )
        for (article, chunk), item in zip(chunks, response.data):
            self.documents.append(
                VectorDocument(
                    article_id=article.id,
                    source=normalize_source_name(article.source),
                    url=article.url,
                    text=chunk,
                    vector=item.embedding,
                )
            )

    def query(self, text: str, top_k: int = 10) -> list[tuple[VectorDocument, float]]:
        if not self.documents:
            return []
        response = self.client.embeddings.create(
            model=settings.embedding_model,
            input=text,
        )
        query_vector = response.data[0].embedding
        scored: list[tuple[VectorDocument, float]] = []
        for doc in self.documents:
            score = _cosine_similarity(query_vector, doc.vector)
            if score > 0:
                scored.append((doc, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


class EvidenceRetriever:
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
        store = self._build_store()
        article_pool = self._merge_articles(articles, corpus_articles or [])
        store.add_articles(article_pool)
        retrieved = store.query(query_text, top_k=top_k)

        evidence = [
            EvidenceSnippet(
                article_id=doc.article_id,
                source=doc.source,
                quote=doc.text,
                url=doc.url,
                score=round(score, 3),
            )
            for doc, score in retrieved
        ]
        filtered = self.filter(evidence)
        reranked = self.rerank(filtered)
        return reranked[:5]

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
        counter_query = f"{claim_text} 반박 부인 철회 논란 상충 아니다 사실무근"
        return self.retrieve_with_query(articles, query_text=counter_query, top_k=10, corpus_articles=corpus_articles)

    def retrieve_external_for_claim(self, claim_text: str, articles: list[Article], corpus_articles: list[Article]) -> list[EvidenceSnippet]:
        cluster_ids = {article.id for article in articles}
        external_articles = [article for article in corpus_articles if article.id not in cluster_ids]
        if not external_articles:
            return []
        return self.retrieve_with_query(external_articles, query_text=claim_text, top_k=10)

    def source_weight(self, source: str) -> float:
        return source_weight(source)

    def multi_source_verified(self, evidence: list[EvidenceSnippet]) -> bool:
        return len({item.source for item in evidence}) >= 2

    def rerank(self, evidence: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        return sorted(
            evidence,
            key=lambda item: (self.source_weight(item.source) * 0.65 + item.score * 0.35, len(item.quote)),
            reverse=True,
        )

    def filter(self, evidence: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
        filtered = [
            item
            for item in evidence
            if self.source_weight(item.source) >= 0.75 and item.score >= 0.08 and len(item.quote) >= 40
        ]
        trusted = [item for item in filtered if is_trusted_ready_source(item.source)]
        return trusted or filtered

    def _build_store(self):
        if settings.openai_api_key and OpenAI is not None:
            try:
                return OpenAIEmbeddingStore()
            except Exception:
                return SimpleVectorStore()
        return SimpleVectorStore()

    def _merge_articles(self, articles: list[Article], extra_articles: list[Article]) -> list[Article]:
        merged: dict[str, Article] = {}
        for article in [*articles, *extra_articles]:
            merged[article.id] = article
        return list(merged.values())


def _article_chunks(article: Article) -> list[str]:
    parts = [part.strip() for part in article.content.replace("\n", " ").split(".") if part.strip()]
    chunks = [part for part in parts if len(part) >= 40]
    if not chunks:
        fallback = f"{article.title}. {article.content}".strip()
        return [fallback[:500]]
    return chunks[:6]


def _cosine_similarity(left: Counter[str] | list[float], right: Counter[str] | list[float]) -> float:
    if isinstance(left, list) and isinstance(right, list):
        return _cosine_similarity_dense(left, right)
    if not left or not right:
        return 0.0
    intersection = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in intersection)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _cosine_similarity_dense(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
