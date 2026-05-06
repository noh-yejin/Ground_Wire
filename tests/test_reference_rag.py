from datetime import datetime, timezone

from app.models import Article
from app.repository import IssueRepository
from app.services.rag import EvidenceRetriever
from app.services.reference_ingestion import ReferenceCorpusIngestor
from app.services.reference_registry import DEFAULT_SOURCE_ID, ReferenceSourceRegistry


def test_reference_ingestion_persists_documents_and_chunks(tmp_path) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "policy.md").write_text(
        "# 반도체 정책 메모\n\n정부는 반도체 인프라와 전력 공급 지원을 확대한다.\n\n세제 지원도 함께 검토한다.\n",
        encoding="utf-8",
    )
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))

    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert len(result.documents) == 1
    assert len(result.chunks) >= 1
    stored_documents = repository.list_reference_documents()
    stored_chunks = repository.list_reference_chunks()
    assert stored_documents[0].title == "반도체 정책 메모"
    assert stored_documents[0].source_id == DEFAULT_SOURCE_ID
    assert any("전력 공급 지원" in chunk.text for chunk in stored_chunks)


def test_retriever_returns_reference_evidence_for_claims(tmp_path) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "macro.txt").write_text(
        "한국은행 기준금리 인하 시에는 원화 약세와 수입물가 반등 가능성을 함께 점검해야 한다. "
        "에너지 가격 상승이 겹치면 인플레이션 압력이 다시 커질 수 있다.",
        encoding="utf-8",
    )
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))
    ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()
    retriever = EvidenceRetriever(repository=repository)
    article = Article(
        id="a1",
        title="한은 금리 인하 가능성 부각",
        source="Reuters",
        published_at=datetime.now(timezone.utc),
        url="https://example.com/a1",
        content="시장에서는 한국은행 기준금리 인하 가능성과 물가 재상승 위험을 함께 주목하고 있다.",
    )

    evidence = retriever.retrieve_external_for_claim(
        "기준금리 인하가 원화와 수입물가에 미치는 영향",
        [article],
        [article],
    )

    assert evidence
    assert evidence[0].evidence_type == "reference"
    assert evidence[0].document_id is not None
    assert evidence[0].authority_score is not None
    assert "수입물가" in evidence[0].quote


def test_reference_registry_reads_manifest_and_ingestion_uses_source_metadata(tmp_path) -> None:
    docs_dir = tmp_path / "reference_docs"
    (docs_dir / "macro-notes").mkdir(parents=True)
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "macro-notes",
              "name": "Macro Notes",
              "kind": "research",
              "location": "macro-notes",
              "authority_score": 0.84,
              "is_active": true,
              "refresh_minutes": 90,
              "fetch_config": {"respect_robots": true}
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (docs_dir / "macro-notes" / "policy.txt").write_text(
        "기준금리 인하 이후 환율과 수입물가 반응을 함께 점검해야 한다.",
        encoding="utf-8",
    )
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))
    registry = ReferenceSourceRegistry(repository=repository, docs_path=str(docs_dir))

    sources = registry.sync()
    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert any(source.id == "macro-notes" for source in sources)
    stored_sources = repository.list_reference_sources(active_only=True)
    assert any(source.name == "Macro Notes" for source in stored_sources)
    assert any(source.refresh_minutes == 90 for source in stored_sources if source.id == "macro-notes")
    assert any(source.fetch_config.get("respect_robots") is True for source in stored_sources if source.id == "macro-notes")
    assert result.documents[0].source_id == "macro-notes"
    assert result.documents[0].source_type == "research"
    assert result.documents[0].authority_score == 0.84
    sync_runs = repository.list_reference_sync_runs()
    assert any(run["source_id"] == "macro-notes" for run in sync_runs)


def test_reference_ingestion_fetches_remote_sources(tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "bok-web",
              "name": "BOK Web",
              "kind": "government",
              "location": "bok-web",
              "authority_score": 0.97,
              "is_active": true,
              "seed_urls": ["https://example.com/bok-policy"],
              "fetch_config": {
                "respect_robots": true,
                "content_selectors": ["main"]
              }
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        if url == "https://example.com/robots.txt":
            return FakeResponse("User-agent: *\nAllow: /\n")
        assert url == "https://example.com/bok-policy"
        assert timeout > 0
        assert "GroundWire" in headers["User-Agent"]
        return FakeResponse(
            """
            <html>
              <head><title>Policy Update</title></head>
              <body>
                <main>
                  한국은행은 물가안정 목표와 금융안정 여건을 함께 고려해 통화정책을 운영한다.
                  향후 기준금리 판단에서도 환율과 수입물가 흐름을 함께 점검한다.
                </main>
              </body>
            </html>
            """.strip()
        )

    monkeypatch.setattr("app.services.reference_ingestion.requests.get", fake_get)
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))

    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert any(document.source_id == "bok-web" for document in result.documents)
    assert any(document.url == "https://example.com/bok-policy" for document in result.documents)
    assert any(document.content_hash for document in result.documents if document.source_id == "bok-web")
    stored_chunks = repository.list_reference_chunks(active_only=True)
    assert any(chunk.source_id == "bok-web" for chunk in stored_chunks)
    sync_runs = repository.list_reference_sync_runs()
    assert any(run["source_id"] == "bok-web" and run["status"] == "SUCCESS" for run in sync_runs)


def test_reference_ingestion_uses_metadata_and_json_ld_fallbacks(tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "sec-web",
              "name": "SEC Web",
              "kind": "government",
              "location": "sec-web",
              "authority_score": 0.97,
              "is_active": true,
              "seed_urls": ["https://example.com/sec-post"],
              "fetch_config": {
                "respect_robots": true
              }
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        if url == "https://example.com/robots.txt":
            return FakeResponse("User-agent: *\nAllow: /\n")
        return FakeResponse(
            """
            <html>
              <head>
                <meta property="og:title" content="SEC Clarifies Filing Guidance" />
                <script type="application/ld+json">
                  {"headline": "SEC Clarifies Filing Guidance", "articleBody": "The commission clarified filing guidance and explained the updated disclosure timeline in detail for regulated companies and investors."}
                </script>
              </head>
              <body><div>short</div></body>
            </html>
            """.strip()
        )

    monkeypatch.setattr("app.services.reference_ingestion.requests.get", fake_get)
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))

    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert result.documents
    assert result.documents[0].title == "SEC Clarifies Filing Guidance"
    assert "updated disclosure timeline" in result.documents[0].content


def test_reference_ingestion_respects_robots_policy(tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "blocked-web",
              "name": "Blocked Web",
              "kind": "government",
              "location": "blocked-web",
              "authority_score": 0.9,
              "is_active": true,
              "seed_urls": ["https://example.com/blocked"]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        if url == "https://example.com/robots.txt":
            return FakeResponse("User-agent: *\nDisallow: /blocked\n")
        raise AssertionError("blocked page should not be fetched")

    monkeypatch.setattr("app.services.reference_ingestion.requests.get", fake_get)
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))

    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert not result.documents
    sync_runs = repository.list_reference_sync_runs()
    assert any(run["source_id"] == "blocked-web" and run["status"] == "PARTIAL" for run in sync_runs)


def test_reference_document_versions_track_content_changes(tmp_path) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    path = docs_dir / "note.txt"
    path.write_text("첫 번째 버전의 정책 메모 내용입니다. 충분히 긴 문장으로 구성합니다.", encoding="utf-8")
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))
    ingestor = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir))

    first = ingestor.ingest()
    path.write_text("두 번째 버전의 정책 메모 내용입니다. 내용이 바뀌었고 역시 충분히 긴 문장입니다.", encoding="utf-8")
    second = ingestor.ingest()

    assert first.documents and second.documents
    versions = repository.list_reference_document_versions(first.documents[0].id)
    assert len(versions) == 2
    assert versions[0]["content_hash"] != versions[1]["content_hash"]


def test_reference_ingestion_preserves_existing_remote_docs_when_refresh_not_due(tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "fed-rss",
              "name": "Fed RSS",
              "kind": "government",
              "location": "fed-rss",
              "authority_score": 0.98,
              "is_active": true,
              "refresh_minutes": 180,
              "seed_urls": ["https://example.com/feed.xml"],
              "fetch_config": {
                "mode": "rss",
                "respect_robots": false,
                "follow_entry_links": false,
                "max_entries": 2
              }
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class FirstFeed:
        entries = [
            {
                "title": "Policy Statement",
                "link": "https://example.com/policy-1",
                "summary": "<p>Federal Reserve releases a policy statement with enough explanatory text for ingestion.</p>",
            }
        ]

    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))
    monkeypatch.setattr("app.services.reference_ingestion.feedparser.parse", lambda url: FirstFeed())
    ingestor = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir))

    first = ingestor.ingest()
    monkeypatch.setattr("app.services.reference_ingestion.feedparser.parse", lambda url: (_ for _ in ()).throw(AssertionError("refresh should be skipped")))
    second = ingestor.ingest()

    assert len(first.documents) == 1
    assert len(second.documents) == 1
    stored_documents = repository.list_reference_documents()
    assert len(stored_documents) == 1
    assert stored_documents[0].title == "Policy Statement"


def test_reference_ingestion_supports_rss_mode(tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "fed-rss",
              "name": "Fed RSS",
              "kind": "government",
              "location": "fed-rss",
              "authority_score": 0.98,
              "is_active": true,
              "seed_urls": ["https://example.com/feed.xml"],
              "fetch_config": {
                "mode": "rss",
                "respect_robots": false,
                "follow_entry_links": false,
                "max_entries": 2
              }
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class FakeFeed:
        entries = [
            {
                "title": "Policy Statement",
                "link": "https://example.com/policy-1",
                "summary": "<p>Federal Reserve releases a policy statement with enough explanatory text for ingestion.</p>",
            }
        ]

    monkeypatch.setattr("app.services.reference_ingestion.feedparser.parse", lambda url: FakeFeed())
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))

    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert len(result.documents) == 1
    assert result.documents[0].doc_type == "rss"
    assert "policy statement" in result.documents[0].content.lower()


def test_reference_ingestion_rss_falls_back_to_summary_when_link_fetch_blocked(tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "fed-rss",
              "name": "Fed RSS",
              "kind": "government",
              "location": "fed-rss",
              "authority_score": 0.98,
              "is_active": true,
              "seed_urls": ["https://example.com/feed.xml"],
              "fetch_config": {
                "mode": "rss",
                "respect_robots": true,
                "follow_entry_links": true,
                "allow_summary_fallback": true,
                "max_entries": 2
              }
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class FakeFeed:
        entries = [
            {
                "title": "Policy Statement",
                "link": "https://example.com/policy-1",
                "summary": "<p>Federal Reserve releases a policy statement with enough explanatory text for ingestion fallback.</p>",
            }
        ]

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        if url == "https://example.com/robots.txt":
            return FakeResponse("User-agent: *\nDisallow: /policy-1\n")
        raise AssertionError("detail page should not be fetched when robots deny access")

    monkeypatch.setattr("app.services.reference_ingestion.feedparser.parse", lambda url: FakeFeed())
    monkeypatch.setattr("app.services.reference_ingestion.requests.get", fake_get)
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))

    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert len(result.documents) == 1
    assert "policy statement with enough explanatory text" in result.documents[0].content.lower()
    assert result.documents[0].content.startswith("Policy Statement.")


def test_reference_ingestion_truncates_overlong_remote_content(tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "ecb-web",
              "name": "ECB Web",
              "kind": "government",
              "location": "ecb-web",
              "authority_score": 0.97,
              "is_active": true,
              "seed_urls": ["https://example.com/ecb-page"],
              "fetch_config": {
                "mode": "html",
                "respect_robots": false,
                "max_content_chars": 120
              }
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        return FakeResponse(
            "<html><body><main>" + ("long official content " * 50) + "</main></body></html>"
        )

    monkeypatch.setattr("app.services.reference_ingestion.requests.get", fake_get)
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))

    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert len(result.documents) == 1
    assert len(result.documents[0].content) <= 120


def test_reference_ingestion_discovers_links_without_explicit_selectors(tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "imf-list",
              "name": "IMF List",
              "kind": "government",
              "location": "imf-list",
              "authority_score": 0.95,
              "is_active": true,
              "seed_urls": ["https://example.com/newsroom"],
              "fetch_config": {
                "mode": "html_list",
                "respect_robots": false,
                "entry_link_selectors": []
              }
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        if url == "https://example.com/newsroom":
            return FakeResponse(
                """
                <html><body>
                  <a href="/press/2026-05-06-policy-update">Policy update on external balances and market access</a>
                </body></html>
                """.strip()
            )
        return FakeResponse(
            """
            <html><body><main>Policy update on external balances and market access with enough substantive body text for chunking and retrieval.</main></body></html>
            """.strip()
        )

    monkeypatch.setattr("app.services.reference_ingestion.requests.get", fake_get)
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))

    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert len(result.documents) == 1
    assert result.documents[0].url.endswith("/press/2026-05-06-policy-update")


def test_reference_ingestion_supports_html_list_mode(tmp_path, monkeypatch) -> None:
    docs_dir = tmp_path / "reference_docs"
    docs_dir.mkdir()
    (docs_dir / "sources.json").write_text(
        """
        {
          "sources": [
            {
              "id": "imf-list",
              "name": "IMF List",
              "kind": "government",
              "location": "imf-list",
              "authority_score": 0.95,
              "is_active": true,
              "seed_urls": ["https://example.com/list"],
              "fetch_config": {
                "mode": "html_list",
                "respect_robots": false,
                "entry_link_selectors": ["main a"],
                "entry_url_prefixes": ["https://example.com/en/news/articles/"],
                "content_selectors": ["main"]
              }
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        if url == "https://example.com/list":
            return FakeResponse(
                '<html><body><main><a href="/en/news/articles/2026/05/example">Example Release</a></main></body></html>'
            )
        if url == "https://example.com/en/news/articles/2026/05/example":
            return FakeResponse(
                '<html><body><main>This is the full IMF-style article body with enough text to ingest properly for testing.</main></body></html>'
            )
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr("app.services.reference_ingestion.requests.get", fake_get)
    repository = IssueRepository(database_path=str(tmp_path / "rag.db"))

    result = ReferenceCorpusIngestor(repository=repository, docs_path=str(docs_dir)).ingest()

    assert len(result.documents) == 1
    assert result.documents[0].url == "https://example.com/en/news/articles/2026/05/example"
    assert "full imf-style article body" in result.documents[0].content.lower()
