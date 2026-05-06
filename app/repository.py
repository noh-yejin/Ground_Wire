from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.models import (
    AnalysisResult,
    Article,
    EvidenceSnippet,
    ImpactLabel,
    Issue,
    IssuePriority,
    IssueStatus,
    ReferenceChunk,
    ReferenceDocument,
    ReferenceSource,
    ReliabilityScore,
    RiskLevel,
    SentimentLabel,
)


class IssueRepository:
    def __init__(self, database_path: str | None = None) -> None:
        self.database_path = database_path or settings.database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _initialize(self) -> None:
        Path(self.database_path).touch(exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    language TEXT NOT NULL,
                    collected_at TEXT NOT NULL,
                    content_quality REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS issues (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reliability REAL NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feed_states (
                    feed_url TEXT PRIMARY KEY,
                    etag TEXT,
                    modified TEXT,
                    last_checked_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS issue_analysis_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_documents (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    url TEXT NOT NULL,
                    file_path TEXT,
                    doc_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'manual',
                    authority_score REAL NOT NULL DEFAULT 0.8,
                    content_hash TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    url TEXT NOT NULL,
                    text TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    source_type TEXT NOT NULL DEFAULT 'manual',
                    authority_score REAL NOT NULL DEFAULT 0.8,
                    content_hash TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(document_id) REFERENCES reference_documents(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_sources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    location TEXT NOT NULL,
                    authority_score REAL NOT NULL,
                    is_active INTEGER NOT NULL,
                    notes TEXT,
                    last_synced_at TEXT,
                    config_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self._ensure_column(conn, "reference_sources", "config_json", "TEXT NOT NULL DEFAULT '{}'")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_sync_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reference_document_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(document_id, content_hash)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "reference_documents", "source_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "reference_documents", "source_type", "TEXT NOT NULL DEFAULT 'manual'")
            self._ensure_column(conn, "reference_documents", "authority_score", "REAL NOT NULL DEFAULT 0.8")
            self._ensure_column(conn, "reference_documents", "content_hash", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "reference_chunks", "source_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "reference_chunks", "source_type", "TEXT NOT NULL DEFAULT 'manual'")
            self._ensure_column(conn, "reference_chunks", "authority_score", "REAL NOT NULL DEFAULT 0.8")
            self._ensure_column(conn, "reference_chunks", "content_hash", "TEXT NOT NULL DEFAULT ''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reference_documents_source_id ON reference_documents(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reference_documents_content_hash ON reference_documents(source_id, content_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reference_chunks_source_id ON reference_chunks(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reference_chunks_document_id ON reference_chunks(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reference_sync_runs_source_created ON reference_sync_runs(source_id, created_at DESC)")
            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row[1] for row in rows}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def save_articles(self, articles: list[Article]) -> None:
        with self._connect() as conn:
            for article in articles:
                conn.execute(
                    """
                    INSERT INTO articles (
                        id, source, published_at, url, title, content, language, collected_at, content_quality
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        source = excluded.source,
                        published_at = excluded.published_at,
                        url = excluded.url,
                        title = excluded.title,
                        content = excluded.content,
                        language = excluded.language,
                        collected_at = excluded.collected_at,
                        content_quality = excluded.content_quality
                    """,
                    (
                        article.id,
                        article.source,
                        article.published_at.isoformat(),
                        article.url,
                        article.title,
                        article.content,
                        article.language,
                        (article.collected_at or article.published_at).isoformat(),
                        article.content_quality,
                    ),
                )
            conn.commit()

    def list_articles(self) -> list[Article]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, source, published_at, url, content, language, collected_at, content_quality
                FROM articles
                ORDER BY published_at DESC
                """
            ).fetchall()
        return [
            Article(
                id=row[0],
                title=row[1],
                source=row[2],
                published_at=datetime.fromisoformat(row[3]),
                url=row[4],
                content=row[5],
                language=row[6],
                collected_at=datetime.fromisoformat(row[7]),
                content_quality=row[8],
            )
            for row in rows
        ]

    def save_job_run(self, job_name: str, status: str, details: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO job_runs (job_name, status, details, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (job_name, status, json.dumps(details, ensure_ascii=False), datetime.utcnow().isoformat()),
            )
            conn.commit()

    def get_latest_job_run(self, job_name: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_name, status, details, created_at
                FROM job_runs
                WHERE job_name = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (job_name,),
            ).fetchone()
        if not row:
            return None
        return {
            "job_name": row[0],
            "status": row[1],
            "details": json.loads(row[2]),
            "created_at": row[3],
        }

    def list_recent_job_runs(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_name, status, details, created_at
                FROM job_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "job_name": row[0],
                "status": row[1],
                "details": json.loads(row[2]),
                "created_at": row[3],
            }
            for row in rows
        ]

    def get_feed_state(self, feed_url: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT etag, modified, last_checked_at FROM feed_states WHERE feed_url = ?",
                (feed_url,),
            ).fetchone()
        if not row:
            return None
        return {"etag": row[0], "modified": row[1], "last_checked_at": row[2]}

    def save_feed_state(self, feed_url: str, etag: str | None, modified: str | None) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feed_states (feed_url, etag, modified, last_checked_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(feed_url) DO UPDATE SET
                    etag = excluded.etag,
                    modified = excluded.modified,
                    last_checked_at = excluded.last_checked_at
                """,
                (feed_url, etag, modified, datetime.utcnow().isoformat()),
            )
            conn.commit()

    def save_issues(self, issues: list[Issue]) -> None:
        with self._connect() as conn:
            current_ids = {issue.id for issue in issues}
            if current_ids:
                placeholders = ",".join("?" for _ in current_ids)
                conn.execute(f"DELETE FROM issues WHERE id NOT IN ({placeholders})", tuple(current_ids))
            else:
                conn.execute("DELETE FROM issues")
            for issue in issues:
                payload = json.dumps(self._serialize_issue(issue), ensure_ascii=False)
                conn.execute(
                    """
                    INSERT INTO issues (id, topic, status, reliability, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        topic = excluded.topic,
                        status = excluded.status,
                        reliability = excluded.reliability,
                        payload = excluded.payload,
                        updated_at = excluded.updated_at
                    """,
                    (
                        issue.id,
                        issue.topic,
                        issue.status.value,
                        issue.reliability.value,
                        payload,
                        issue.updated_at.isoformat(),
                    ),
                )
            conn.commit()

    def get_issue_analysis_cache(self, cache_key: str) -> AnalysisResult | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM issue_analysis_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        return self._deserialize_analysis(json.loads(row[0]))

    def save_issue_analysis_cache(self, cache_key: str, analysis: AnalysisResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO issue_analysis_cache (cache_key, payload, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (
                    cache_key,
                    json.dumps(self._serialize_analysis(analysis), ensure_ascii=False),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

    def save_reference_corpus(
        self,
        documents: list[ReferenceDocument],
        chunks: list[ReferenceChunk],
        source_ids: list[str] | None = None,
    ) -> None:
        with self._connect() as conn:
            effective_source_ids = source_ids or sorted(
                {document.source_id for document in documents} | {chunk.source_id for chunk in chunks}
            )
            if effective_source_ids:
                placeholders = ",".join("?" for _ in effective_source_ids)
                conn.execute(f"DELETE FROM reference_chunks WHERE source_id IN ({placeholders})", tuple(effective_source_ids))
                conn.execute(f"DELETE FROM reference_documents WHERE source_id IN ({placeholders})", tuple(effective_source_ids))
            for document in documents:
                conn.execute(
                    """
                    INSERT INTO reference_documents (
                        id, source_id, title, source, url, file_path, doc_type, content, updated_at, source_type, authority_score, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.source_id,
                        document.title,
                        document.source,
                        document.url,
                        document.file_path,
                        document.doc_type,
                        document.content,
                        document.updated_at.isoformat(),
                        document.source_type,
                        document.authority_score,
                        document.content_hash,
                    ),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO reference_document_versions (
                        document_id, source_id, title, url, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.source_id,
                        document.title,
                        document.url,
                        document.content_hash,
                        datetime.utcnow().isoformat(),
                    ),
                )
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO reference_chunks (
                        id, document_id, source_id, title, source, url, text, chunk_index, updated_at, source_type, authority_score, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.source_id,
                        chunk.title,
                        chunk.source,
                        chunk.url,
                        chunk.text,
                        chunk.chunk_index,
                        chunk.updated_at.isoformat(),
                        chunk.source_type,
                        chunk.authority_score,
                        chunk.content_hash,
                    ),
                )
            conn.commit()

    def replace_reference_source_data(
        self,
        source_id: str,
        documents: list[ReferenceDocument],
        chunks: list[ReferenceChunk],
    ) -> None:
        with self._connect() as conn:
            document_ids = [document.id for document in documents]
            chunk_ids = [chunk.id for chunk in chunks]
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                conn.execute(
                    f"DELETE FROM reference_chunks WHERE source_id = ? AND id NOT IN ({placeholders})",
                    (source_id, *chunk_ids),
                )
            else:
                conn.execute("DELETE FROM reference_chunks WHERE source_id = ?", (source_id,))
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                conn.execute(
                    f"DELETE FROM reference_documents WHERE source_id = ? AND id NOT IN ({placeholders})",
                    (source_id, *document_ids),
                )
            else:
                conn.execute("DELETE FROM reference_documents WHERE source_id = ?", (source_id,))

            for document in documents:
                conn.execute(
                    """
                    INSERT INTO reference_documents (
                        id, source_id, title, source, url, file_path, doc_type, content, updated_at, source_type, authority_score, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        source_id = excluded.source_id,
                        title = excluded.title,
                        source = excluded.source,
                        url = excluded.url,
                        file_path = excluded.file_path,
                        doc_type = excluded.doc_type,
                        content = excluded.content,
                        updated_at = excluded.updated_at,
                        source_type = excluded.source_type,
                        authority_score = excluded.authority_score,
                        content_hash = excluded.content_hash
                    """,
                    (
                        document.id,
                        document.source_id,
                        document.title,
                        document.source,
                        document.url,
                        document.file_path,
                        document.doc_type,
                        document.content,
                        document.updated_at.isoformat(),
                        document.source_type,
                        document.authority_score,
                        document.content_hash,
                    ),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO reference_document_versions (
                        document_id, source_id, title, url, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.source_id,
                        document.title,
                        document.url,
                        document.content_hash,
                        datetime.utcnow().isoformat(),
                    ),
                )

            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO reference_chunks (
                        id, document_id, source_id, title, source, url, text, chunk_index, updated_at, source_type, authority_score, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        document_id = excluded.document_id,
                        source_id = excluded.source_id,
                        title = excluded.title,
                        source = excluded.source,
                        url = excluded.url,
                        text = excluded.text,
                        chunk_index = excluded.chunk_index,
                        updated_at = excluded.updated_at,
                        source_type = excluded.source_type,
                        authority_score = excluded.authority_score,
                        content_hash = excluded.content_hash
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.source_id,
                        chunk.title,
                        chunk.source,
                        chunk.url,
                        chunk.text,
                        chunk.chunk_index,
                        chunk.updated_at.isoformat(),
                        chunk.source_type,
                        chunk.authority_score,
                        chunk.content_hash,
                    ),
                )
            conn.commit()

    def save_reference_sources(self, sources: list[ReferenceSource]) -> None:
        with self._connect() as conn:
            for source in sources:
                conn.execute(
                    """
                    INSERT INTO reference_sources (
                        id, name, kind, location, authority_score, is_active, notes, last_synced_at, config_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        kind = excluded.kind,
                        location = excluded.location,
                        authority_score = excluded.authority_score,
                        is_active = excluded.is_active,
                        notes = excluded.notes,
                        last_synced_at = excluded.last_synced_at,
                        config_json = excluded.config_json
                    """,
                    (
                        source.id,
                        source.name,
                        source.kind,
                        source.location,
                        source.authority_score,
                        int(source.is_active),
                        source.notes,
                        source.last_synced_at.isoformat() if source.last_synced_at else None,
                        json.dumps(
                            {
                                "seed_urls": source.seed_urls,
                                "refresh_minutes": source.refresh_minutes,
                                "fetch_config": source.fetch_config,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            conn.commit()

    def list_reference_sources(self, active_only: bool = False) -> list[ReferenceSource]:
        query = """
            SELECT id, name, kind, location, authority_score, is_active, notes, last_synced_at, config_json
            FROM reference_sources
        """
        params: tuple = ()
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY authority_score DESC, name ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        sources: list[ReferenceSource] = []
        for row in rows:
            config = json.loads(row[8] or "{}")
            sources.append(
                ReferenceSource(
                    id=row[0],
                    name=row[1],
                    kind=row[2],
                    location=row[3],
                    authority_score=row[4],
                    is_active=bool(row[5]),
                    notes=row[6],
                    last_synced_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    seed_urls=config.get("seed_urls", []),
                    refresh_minutes=int(config.get("refresh_minutes", 60)),
                    fetch_config=config.get("fetch_config", {}),
                )
            )
        return sources

    def save_reference_sync_run(self, source_id: str, status: str, details: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reference_sync_runs (source_id, status, details, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, status, json.dumps(details, ensure_ascii=False), datetime.utcnow().isoformat()),
            )
            conn.execute(
                "UPDATE reference_sources SET last_synced_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), source_id),
            )
            conn.commit()

    def list_reference_sync_runs(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_id, status, details, created_at
                FROM reference_sync_runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {"source_id": row[0], "status": row[1], "details": json.loads(row[2]), "created_at": row[3]}
            for row in rows
        ]

    def list_reference_document_versions(self, document_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT document_id, source_id, title, url, content_hash, created_at
                FROM reference_document_versions
                WHERE document_id = ?
                ORDER BY id ASC
                """,
                (document_id,),
            ).fetchall()
        return [
            {
                "document_id": row[0],
                "source_id": row[1],
                "title": row[2],
                "url": row[3],
                "content_hash": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def list_reference_documents(self) -> list[ReferenceDocument]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, source_id, title, source, content, doc_type, updated_at, url, file_path, source_type, authority_score
                    , content_hash
                FROM reference_documents
                ORDER BY updated_at DESC, title ASC
                """
            ).fetchall()
        return [
            ReferenceDocument(
                id=row[0],
                source_id=row[1],
                title=row[2],
                source=row[3],
                content=row[4],
                doc_type=row[5],
                updated_at=datetime.fromisoformat(row[6]),
                url=row[7],
                file_path=row[8],
                source_type=row[9],
                authority_score=row[10],
                content_hash=row[11],
            )
            for row in rows
        ]

    def list_reference_chunks(self, active_only: bool = False) -> list[ReferenceChunk]:
        query = """
                SELECT c.id, c.document_id, c.source_id, c.title, c.source, c.text, c.chunk_index, c.updated_at, c.url, c.source_type, c.authority_score, c.content_hash
                FROM reference_chunks c
        """
        if active_only:
            query += " JOIN reference_sources s ON s.id = c.source_id WHERE s.is_active = 1"
        query += " ORDER BY c.updated_at DESC, c.document_id ASC, c.chunk_index ASC"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [
            ReferenceChunk(
                id=row[0],
                document_id=row[1],
                source_id=row[2],
                title=row[3],
                source=row[4],
                text=row[5],
                chunk_index=row[6],
                updated_at=datetime.fromisoformat(row[7]),
                url=row[8],
                source_type=row[9],
                authority_score=row[10],
                content_hash=row[11],
            )
            for row in rows
        ]

    def get_embedding_cache(self, cache_key: str) -> list[float] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT vector_json FROM embedding_cache WHERE cache_key = ?", (cache_key,)).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def save_embedding_cache(self, cache_key: str, vector: list[float]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO embedding_cache (cache_key, vector_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    vector_json = excluded.vector_json,
                    created_at = excluded.created_at
                """,
                (cache_key, json.dumps(vector), datetime.utcnow().isoformat()),
            )
            conn.commit()

    def list_issues(self) -> list[Issue]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM issues ORDER BY reliability DESC, updated_at DESC"
            ).fetchall()
        return [self._deserialize_issue(json.loads(row[0])) for row in rows]

    def get_issue(self, issue_id: str) -> Issue | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM issues WHERE id = ?", (issue_id,)).fetchone()
        if not row:
            return None
        return self._deserialize_issue(json.loads(row[0]))

    def _serialize_issue(self, issue: Issue) -> dict:
        return {
            "id": issue.id,
            "topic": issue.topic,
            "keywords": issue.keywords,
            "status": issue.status.value,
            "updated_at": issue.updated_at.isoformat(),
            "articles": [
                {
                    **asdict(article),
                    "published_at": article.published_at.isoformat(),
                    "collected_at": (article.collected_at or article.published_at).isoformat(),
                }
                for article in issue.articles
            ],
            "evidence": [asdict(item) for item in issue.evidence],
            "reliability": {
                **asdict(issue.reliability),
            },
            "analysis": self._serialize_analysis(issue.analysis),
        }

    def _deserialize_issue(self, data: dict) -> Issue:
        articles = [
            Article(
                id=item["id"],
                title=item["title"],
                source=item["source"],
                published_at=datetime.fromisoformat(item["published_at"]),
                url=item["url"],
                content=item["content"],
                language=item.get("language", "ko"),
                collected_at=datetime.fromisoformat(item["collected_at"])
                if item.get("collected_at")
                else None,
                content_quality=item.get("content_quality", 0.0),
            )
            for item in data["articles"]
        ]
        evidence = [EvidenceSnippet(**item) for item in data["evidence"]]
        reliability = ReliabilityScore(**data["reliability"])
        legacy_summary = data.get("summary")
        legacy_keywords = data.get("keywords", [])
        analysis_data = data.get("analysis")

        if analysis_data is None:
            analysis = AnalysisResult(
                summary=legacy_summary or "기존 데이터에는 상세 분석 필드가 없습니다.",
                keywords=legacy_keywords,
                key_signals=[],
                key_points=[],
                trend_summary="기존 데이터에는 추이 설명이 없습니다.",
                sentiment=SentimentLabel.NEUTRAL,
                market_impact=ImpactLabel.NEUTRAL,
                policy_risk=RiskLevel.MEDIUM,
                volatility_risk=RiskLevel.MEDIUM,
                risk_points=["구버전 이슈 데이터라 상세 분석 정보가 없습니다."],
                grounded=False,
                priority=IssuePriority.GENERAL,
                hold_reason="legacy_issue_payload",
                grounding_details={},
            )
        else:
            analysis = self._deserialize_analysis(analysis_data)
        return Issue(
            id=data["id"],
            topic=data["topic"],
            keywords=data["keywords"],
            articles=articles,
            evidence=evidence,
            reliability=reliability,
            analysis=analysis,
            status=IssueStatus(data["status"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def _serialize_analysis(self, analysis: AnalysisResult) -> dict:
        return {
            "summary": analysis.summary,
            "keywords": analysis.keywords,
            "key_signals": analysis.key_signals,
            "key_points": analysis.key_points,
            "trend_summary": analysis.trend_summary,
            "sentiment": analysis.sentiment.value,
            "market_impact": analysis.market_impact.value,
            "policy_risk": analysis.policy_risk.value,
            "volatility_risk": analysis.volatility_risk.value,
            "risk_points": analysis.risk_points,
            "grounded": analysis.grounded,
            "priority": analysis.priority.value,
            "hold_reason": analysis.hold_reason,
            "grounding_details": analysis.grounding_details,
        }

    def _deserialize_analysis(self, analysis_data: dict) -> AnalysisResult:
        return AnalysisResult(
            summary=analysis_data["summary"],
            keywords=analysis_data["keywords"],
            key_signals=analysis_data.get("key_signals", []),
            key_points=analysis_data.get("key_points", []),
            trend_summary=analysis_data.get("trend_summary", "추이 설명이 아직 생성되지 않았습니다."),
            sentiment=SentimentLabel(analysis_data["sentiment"]),
            market_impact=ImpactLabel(analysis_data.get("market_impact", analysis_data["sentiment"])),
            policy_risk=RiskLevel(analysis_data.get("policy_risk", "medium")),
            volatility_risk=RiskLevel(analysis_data.get("volatility_risk", "medium")),
            risk_points=analysis_data["risk_points"],
            grounded=analysis_data["grounded"],
            priority=IssuePriority(analysis_data.get("priority", "general")),
            hold_reason=analysis_data.get("hold_reason"),
            grounding_details=analysis_data.get("grounding_details", {}),
        )
