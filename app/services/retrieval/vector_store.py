from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
from math import sqrt

from app.config import settings
from app.models import Article, ReferenceChunk
from app.repository import IssueRepository
from app.services.clustering import _extract_keywords
from app.services.source_normalizer import normalize_source_name

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
    evidence_type: str = "news"
    document_id: str | None = None
    title: str | None = None
    source_id: str | None = None
    source_type: str | None = None
    authority_score: float | None = None
    updated_at: datetime | None = None
    keywords: tuple[str, ...] = ()


class SimpleVectorStore:
    def __init__(self) -> None:
        self.documents: list[VectorDocument] = []

    def add_articles(self, articles: list[Article]) -> None:
        self.documents.extend(article_documents(articles))

    def add_reference_chunks(self, chunks: list[ReferenceChunk]) -> None:
        self.documents.extend(reference_documents(chunks))

    def query(self, text: str, top_k: int = 10) -> list[tuple[VectorDocument, float]]:
        query_vector = Counter(_extract_keywords(text))
        if not query_vector:
            return []
        candidates = self._candidate_documents(query_vector, limit=max(top_k * 8, 24))
        scored: list[tuple[VectorDocument, float]] = []
        for doc in candidates:
            score = cosine_similarity(query_vector, doc.vector)
            if score > 0:
                scored.append((doc, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def _candidate_documents(self, query_vector: Counter[str], limit: int) -> list[VectorDocument]:
        if len(self.documents) <= limit:
            return self.documents
        query_terms = set(query_vector)
        ranked = sorted(
            self.documents,
            key=lambda doc: (
                len(query_terms & set(doc.keywords)),
                sum(query_vector.get(term, 0) for term in doc.keywords),
                len(doc.text),
            ),
            reverse=True,
        )
        return ranked[:limit]


class OpenAIEmbeddingStore:
    def __init__(self, repository: IssueRepository | None = None) -> None:
        if OpenAI is None or not settings.openai_api_key:
            raise RuntimeError("OpenAI embedding client is unavailable")
        self.client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds, max_retries=0)
        self.repository = repository
        self.documents: list[VectorDocument] = []

    def add_articles(self, articles: list[Article]) -> None:
        for doc in article_documents(articles):
            doc.vector = []
            self.documents.append(doc)
        self._embed_pending_documents()

    def add_reference_chunks(self, chunks: list[ReferenceChunk]) -> None:
        for doc in reference_documents(chunks):
            doc.vector = []
            self.documents.append(doc)
        self._embed_pending_documents()

    def _embed_pending_documents(self) -> None:
        pending = [doc for doc in self.documents if not doc.vector]
        if not pending:
            return
        uncached: list[VectorDocument] = []
        for doc in pending:
            cache_key = _embedding_cache_key(doc.text)
            cached = self.repository.get_embedding_cache(cache_key) if self.repository else None
            if cached is not None:
                doc.vector = cached
            else:
                uncached.append(doc)
        if not uncached:
            return
        response = self.client.embeddings.create(
            model=settings.embedding_model,
            input=[doc.text for doc in uncached],
        )
        for doc, item in zip(uncached, response.data):
            doc.vector = item.embedding
            if self.repository is not None:
                self.repository.save_embedding_cache(_embedding_cache_key(doc.text), item.embedding)

    def query(self, text: str, top_k: int = 10) -> list[tuple[VectorDocument, float]]:
        if not self.documents:
            return []
        query_vector = self._query_embedding(text)
        query_terms = set(_extract_keywords(text))
        candidates = self.documents
        if query_terms and len(self.documents) > max(top_k * 10, 40):
            candidates = sorted(
                self.documents,
                key=lambda doc: (
                    len(query_terms & set(doc.keywords)),
                    doc.authority_score or 0.0,
                    len(doc.text),
                ),
                reverse=True,
            )[: max(top_k * 10, 40)]
        scored: list[tuple[VectorDocument, float]] = []
        for doc in candidates:
            score = cosine_similarity(query_vector, doc.vector)
            if score > 0:
                scored.append((doc, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    def _query_embedding(self, text: str) -> list[float]:
        cache_key = _embedding_cache_key(f"query::{text}")
        cached = self.repository.get_embedding_cache(cache_key) if self.repository else None
        if cached is not None:
            return cached
        response = self.client.embeddings.create(model=settings.embedding_model, input=text)
        query_vector = response.data[0].embedding
        if self.repository is not None:
            self.repository.save_embedding_cache(cache_key, query_vector)
        return query_vector


def build_store(repository: IssueRepository | None = None):
    if settings.openai_api_key and OpenAI is not None:
        try:
            return OpenAIEmbeddingStore(repository=repository)
        except Exception:
            return SimpleVectorStore()
    return SimpleVectorStore()


def article_documents(articles: list[Article]) -> list[VectorDocument]:
    documents: list[VectorDocument] = []
    for article in articles:
        for chunk in article_chunks(article):
            documents.append(
                VectorDocument(
                    article_id=article.id,
                    source=normalize_source_name(article.source),
                    url=article.url,
                    text=chunk,
                    vector=Counter(_extract_keywords(chunk)),
                    evidence_type="news",
                    document_id=article.id,
                    title=article.title,
                    source_id=None,
                    source_type="news",
                    authority_score=None,
                    updated_at=article.published_at,
                    keywords=tuple(_extract_keywords(chunk)),
                )
            )
    return documents


def reference_documents(chunks: list[ReferenceChunk]) -> list[VectorDocument]:
    return [
        VectorDocument(
            article_id=chunk.id,
            source=chunk.source,
            url=chunk.url,
            text=chunk.text,
            vector=Counter(_extract_keywords(chunk.text)),
            evidence_type="reference",
            document_id=chunk.document_id,
            title=chunk.title,
            source_id=chunk.source_id,
            source_type=chunk.source_type,
            authority_score=chunk.authority_score,
            updated_at=chunk.updated_at,
            keywords=tuple(_extract_keywords(chunk.text)),
        )
        for chunk in chunks
    ]


def article_chunks(article: Article) -> list[str]:
    parts = [part.strip() for part in article.content.replace("\n", " ").split(".") if part.strip()]
    chunks = [part for part in parts if len(part) >= 40]
    if not chunks:
        fallback = f"{article.title}. {article.content}".strip()
        return [fallback[:500]]
    return chunks[:6]


def cosine_similarity(left: Counter[str] | list[float], right: Counter[str] | list[float]) -> float:
    if isinstance(left, list) and isinstance(right, list):
        return cosine_similarity_dense(left, right)
    if not left or not right:
        return 0.0
    intersection = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in intersection)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def cosine_similarity_dense(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _embedding_cache_key(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()
