from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import feedparser

from app.config import settings
from app.models import ReferenceChunk, ReferenceDocument, ReferenceSource
from app.repository import IssueRepository
from app.services.reference_registry import DEFAULT_SOURCE_ID, ReferenceSourceRegistry


DOMAIN_FETCH_PROFILES: dict[str, dict[str, object]] = {
    "sec.gov": {
        "entry_link_selectors": [".article-listing a", ".newsroom-article a", "main a", "a"],
        "entry_url_prefixes": ["https://www.sec.gov/news/", "https://www.sec.gov/litigation/"],
        "content_selectors": ["main", ".article__body", ".article-body", ".text-formatted"],
        "remove_selectors": [".share-tools", ".related-materials", ".block-system-main-block .tabs"],
        "title_selectors": ["h1", ".page-title", "meta[property='og:title']"],
    },
    "federalreserve.gov": {
        "content_selectors": ["#article", ".col-xs-12.col-sm-8.col-md-8", "main", "article"],
        "title_selectors": ["h1", ".pageheader h1", "meta[property='og:title']"],
        "remove_selectors": [".social-share", ".feds-related-content", ".breadcrumbs"],
    },
    "imf.org": {
        "content_selectors": [".imf-viewer", ".coveo-result-cell", ".component.content", "main", "article"],
        "title_selectors": ["h1", ".page-title", "meta[property='og:title']"],
        "remove_selectors": [".newsletter-signup", ".social-share", ".related-links"],
    },
    "ecb.europa.eu": {
        "content_selectors": [".ecb-pressContent", ".section", "main", "article"],
        "title_selectors": ["h1", ".title", "meta[property='og:title']"],
        "remove_selectors": [".social", ".relatedTopics", ".shortcut"],
    },
}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if cleaned:
            self.parts.append(cleaned)

    def get_text(self) -> str:
        return " ".join(self.parts)


@dataclass(slots=True)
class IngestionResult:
    documents: list[ReferenceDocument]
    chunks: list[ReferenceChunk]


class ReferenceCorpusIngestor:
    def __init__(self, repository: IssueRepository | None = None, docs_path: str | None = None) -> None:
        self.repository = repository or IssueRepository()
        self.docs_path = Path(docs_path or settings.reference_docs_path)
        self.registry = ReferenceSourceRegistry(repository=self.repository, docs_path=str(self.docs_path))
        self._robots_cache: dict[str, robotparser.RobotFileParser | None] = {}

    def ingest(self) -> IngestionResult:
        sources = self.registry.sync()
        source_ids = [source.id for source in sources]
        if not self.docs_path.exists():
            for source_id in source_ids:
                self.repository.replace_reference_source_data(source_id, [], [])
            return IngestionResult(documents=[], chunks=[])

        existing_documents_by_source: dict[str, list[ReferenceDocument]] = {}
        existing_chunks_by_source: dict[str, list[ReferenceChunk]] = {}
        for item in self.repository.list_reference_documents():
            existing_documents_by_source.setdefault(item.source_id, []).append(item)
        for item in self.repository.list_reference_chunks():
            existing_chunks_by_source.setdefault(item.source_id, []).append(item)

        documents: list[ReferenceDocument] = []
        chunks: list[ReferenceChunk] = []
        documents_by_source: dict[str, list[ReferenceDocument]] = {}
        chunks_by_source: dict[str, list[ReferenceChunk]] = {}
        failures_by_source: dict[str, list[str]] = {}
        seen_hashes: set[tuple[str, str]] = set()
        for path in sorted(self.docs_path.rglob("*")):
            if (
                not path.is_file()
                or path.name == "sources.json"
                or path.suffix.lower() not in settings.reference_doc_extensions
            ):
                continue
            content = self._read_text(path)
            if not content:
                continue
            source = self.registry.source_for_path(path, sources)
            if not source.is_active:
                continue
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            document_id = sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
            title = self._title_for(path, content)
            relative_path = path.relative_to(self.docs_path).as_posix()
            document = ReferenceDocument(
                id=document_id,
                source_id=source.id,
                title=title,
                source=source.name,
                content=content,
                doc_type=path.suffix.lower().lstrip("."),
                updated_at=updated_at,
                file_path=relative_path,
                source_type=source.kind,
                authority_score=source.authority_score,
                content_hash=_content_hash(content),
            )
            if (source.id, document.content_hash) in seen_hashes:
                continue
            seen_hashes.add((source.id, document.content_hash))
            documents.append(document)
            documents_by_source.setdefault(source.id, []).append(document)
            for index, chunk_text in enumerate(_chunk_text(content), start=1):
                chunk = ReferenceChunk(
                    id=f"{document_id}:{index}",
                    document_id=document_id,
                    source_id=source.id,
                    title=title,
                    source=source.name,
                    text=chunk_text,
                    chunk_index=index,
                    updated_at=updated_at,
                    source_type=source.kind,
                    authority_score=source.authority_score,
                    content_hash=_content_hash(chunk_text),
                )
                chunks.append(chunk)
                chunks_by_source.setdefault(source.id, []).append(chunk)

        for source in sources:
            if not source.is_active or not source.seed_urls:
                continue
            if not self._should_refresh_source(source):
                cached_documents = existing_documents_by_source.get(source.id, [])
                cached_chunks = existing_chunks_by_source.get(source.id, [])
                documents.extend(cached_documents)
                chunks.extend(cached_chunks)
                if cached_documents:
                    documents_by_source[source.id] = list(cached_documents)
                if cached_chunks:
                    chunks_by_source[source.id] = list(cached_chunks)
                continue
            remote_documents, remote_chunks, failures = self._ingest_remote_source(source, seen_hashes=seen_hashes)
            if remote_documents:
                documents.extend(remote_documents)
                chunks.extend(remote_chunks)
                documents_by_source[source.id] = list(remote_documents)
                chunks_by_source[source.id] = list(remote_chunks)
            else:
                cached_documents = existing_documents_by_source.get(source.id, [])
                cached_chunks = existing_chunks_by_source.get(source.id, [])
                documents.extend(cached_documents)
                chunks.extend(cached_chunks)
                if cached_documents:
                    documents_by_source[source.id] = list(cached_documents)
                if cached_chunks:
                    chunks_by_source[source.id] = list(cached_chunks)
            if failures:
                failures_by_source[source.id] = failures

        for source_id in source_ids:
            self.repository.replace_reference_source_data(
                source_id,
                documents_by_source.get(source_id, []),
                chunks_by_source.get(source_id, []),
            )
        for source in sources:
            if source.id == DEFAULT_SOURCE_ID or source.is_active:
                source_documents = [item for item in documents if item.source_id == source.id]
                failures = failures_by_source.get(source.id, [])
                self.repository.save_reference_sync_run(
                    source.id,
                    "PARTIAL" if failures else "SUCCESS",
                    {
                        "document_count": len(source_documents),
                        "chunk_count": sum(item.source_id == source.id for item in chunks),
                        "failures": failures,
                    },
                )
        return IngestionResult(documents=documents, chunks=chunks)

    def _should_refresh_source(self, source: ReferenceSource) -> bool:
        if source.last_synced_at is None:
            return True
        last_synced_at = source.last_synced_at
        if last_synced_at.tzinfo is None:
            last_synced_at = last_synced_at.replace(tzinfo=timezone.utc)
        else:
            last_synced_at = last_synced_at.astimezone(timezone.utc)
        elapsed_seconds = (datetime.now(timezone.utc) - last_synced_at).total_seconds()
        return elapsed_seconds >= max(source.refresh_minutes, 1) * 60

    def _ingest_remote_source(
        self,
        source: ReferenceSource,
        seen_hashes: set[tuple[str, str]],
    ) -> tuple[list[ReferenceDocument], list[ReferenceChunk], list[str]]:
        documents: list[ReferenceDocument] = []
        chunks: list[ReferenceChunk] = []
        failures: list[str] = []
        config = self._effective_fetch_config(source)
        mode = str(config.get("mode", "html")).lower()
        if mode == "rss":
            documents, chunks, failures = self._ingest_rss_source(source, seen_hashes=seen_hashes)
            if documents or not config.get("fallback_mode"):
                return documents, chunks, failures
            fallback_mode = str(config.get("fallback_mode", "")).lower()
            fallback_urls = [str(item) for item in config.get("fallback_seed_urls", []) if str(item).strip()]
            if fallback_mode == "html_list" and fallback_urls:
                fallback_source = ReferenceSource(
                    id=source.id,
                    name=source.name,
                    kind=source.kind,
                    location=source.location,
                    authority_score=source.authority_score,
                    is_active=source.is_active,
                    notes=source.notes,
                    last_synced_at=source.last_synced_at,
                    seed_urls=fallback_urls,
                    refresh_minutes=source.refresh_minutes,
                    fetch_config={**config, "mode": "html_list"},
                )
                extra_docs, extra_chunks, extra_failures = self._ingest_html_list_source(fallback_source, seen_hashes=seen_hashes)
                return documents + extra_docs, chunks + extra_chunks, failures + extra_failures
            return documents, chunks, failures
        if mode == "html_list":
            return self._ingest_html_list_source(source, seen_hashes=seen_hashes)
        for url in source.seed_urls:
            try:
                if not self._is_fetch_allowed(url, source):
                    failures.append(f"{url}: blocked by robots policy")
                    continue
                response = requests.get(
                    url,
                    timeout=settings.reference_fetch_timeout_seconds,
                    headers=self._build_request_headers(source),
                )
                response.raise_for_status()
                content, title, doc_type = self._extract_remote_content(response.text, url, source)
                content = _normalize_text(content)
                if not content:
                    failures.append(f"{url}: empty content")
                    continue
                updated_at = datetime.now(timezone.utc)
                document_id = sha1(f"{source.id}:{url}".encode("utf-8")).hexdigest()[:16]
                content_hash = _content_hash(content)
                if (source.id, content_hash) in seen_hashes:
                    failures.append(f"{url}: duplicate content skipped")
                    continue
                seen_hashes.add((source.id, content_hash))
                document = ReferenceDocument(
                    id=document_id,
                    source_id=source.id,
                    title=title,
                    source=source.name,
                    content=content,
                    doc_type=doc_type,
                    updated_at=updated_at,
                    url=url,
                    source_type=source.kind,
                    authority_score=source.authority_score,
                    content_hash=content_hash,
                )
                documents.append(document)
                for index, chunk_text in enumerate(_chunk_text(content), start=1):
                    chunks.append(
                        ReferenceChunk(
                            id=f"{document_id}:{index}",
                            document_id=document_id,
                            source_id=source.id,
                            title=title,
                            source=source.name,
                            text=chunk_text,
                            chunk_index=index,
                            updated_at=updated_at,
                            url=url,
                            source_type=source.kind,
                            authority_score=source.authority_score,
                            content_hash=_content_hash(chunk_text),
                        )
                    )
            except Exception as exc:
                failures.append(f"{url}: {type(exc).__name__}: {exc}")
        return documents, chunks, failures

    def _ingest_rss_source(
        self,
        source: ReferenceSource,
        seen_hashes: set[tuple[str, str]],
    ) -> tuple[list[ReferenceDocument], list[ReferenceChunk], list[str]]:
        documents: list[ReferenceDocument] = []
        chunks: list[ReferenceChunk] = []
        failures: list[str] = []
        config = self._effective_fetch_config(source)
        max_entries = int(config.get("max_entries", 10))
        follow_links = bool(config.get("follow_entry_links", True))
        allow_summary_fallback = bool(config.get("allow_summary_fallback", True))
        for feed_url in source.seed_urls:
            try:
                parsed = feedparser.parse(feed_url)
                entries = list(getattr(parsed, "entries", []))[:max_entries]
                if not entries:
                    failures.append(f"{feed_url}: empty rss feed")
                    continue
                for entry in entries:
                    link = str(_entry_value(entry, "link") or "")
                    title = str(_entry_value(entry, "title") or _title_from_url(link or feed_url)).strip()[:160]
                    summary = str(_entry_value(entry, "summary") or _entry_value(entry, "description") or "").strip()
                    summary_text = _normalize_text(BeautifulSoup(summary, "html.parser").get_text(" ", strip=True))
                    content = summary_text
                    doc_type = "rss"
                    if follow_links and link:
                        try:
                            if not self._is_fetch_allowed(link, source):
                                raise PermissionError("blocked by robots policy")
                            response = requests.get(
                                link,
                                timeout=settings.reference_fetch_timeout_seconds,
                                headers=self._build_request_headers(source),
                            )
                            response.raise_for_status()
                            fetched_content, fetched_title, doc_type = self._extract_remote_content(response.text, link, source)
                            title = fetched_title or title
                            fetched_content = _normalize_text(fetched_content)
                            content = self._prefer_richer_rss_content(summary_text, fetched_content)
                        except Exception as exc:
                            if allow_summary_fallback and summary_text:
                                failures.append(f"{link}: {type(exc).__name__}: {exc} (used rss summary fallback)")
                                content = summary_text
                                doc_type = "rss"
                            else:
                                failures.append(f"{link}: {type(exc).__name__}: {exc}")
                                continue
                    if not content:
                        failures.append(f"{link or feed_url}: empty entry content")
                        continue
                    if len(content) < 120 and title and title not in content:
                        content = f"{title}. {content}".strip()
                    content_hash = _content_hash(content)
                    if (source.id, content_hash) in seen_hashes:
                        continue
                    seen_hashes.add((source.id, content_hash))
                    canonical = link or f"{feed_url}#{sha1(title.encode('utf-8')).hexdigest()[:12]}"
                    document_id = sha1(f"{source.id}:{canonical}".encode("utf-8")).hexdigest()[:16]
                    updated_at = datetime.now(timezone.utc)
                    document = ReferenceDocument(
                        id=document_id,
                        source_id=source.id,
                        title=title,
                        source=source.name,
                        content=content,
                        doc_type=doc_type,
                        updated_at=updated_at,
                        url=link or feed_url,
                        source_type=source.kind,
                        authority_score=source.authority_score,
                        content_hash=content_hash,
                    )
                    documents.append(document)
                    for index, chunk_text in enumerate(_chunk_text(content), start=1):
                        chunks.append(
                            ReferenceChunk(
                                id=f"{document_id}:{index}",
                                document_id=document_id,
                                source_id=source.id,
                                title=title,
                                source=source.name,
                                text=chunk_text,
                                chunk_index=index,
                                updated_at=updated_at,
                                url=link or feed_url,
                                source_type=source.kind,
                                authority_score=source.authority_score,
                                content_hash=_content_hash(chunk_text),
                            )
                        )
            except Exception as exc:
                failures.append(f"{feed_url}: {type(exc).__name__}: {exc}")
        return documents, chunks, failures

    def _ingest_html_list_source(
        self,
        source: ReferenceSource,
        seen_hashes: set[tuple[str, str]],
    ) -> tuple[list[ReferenceDocument], list[ReferenceChunk], list[str]]:
        documents: list[ReferenceDocument] = []
        chunks: list[ReferenceChunk] = []
        failures: list[str] = []
        config = self._effective_fetch_config(source)
        max_entries = int(config.get("max_entries", 10))
        entry_selectors = [str(item) for item in config.get("entry_link_selectors", [])]
        if not entry_selectors:
            entry_selectors = ["a"]
        for list_url in source.seed_urls:
            try:
                if not self._is_fetch_allowed(list_url, source):
                    failures.append(f"{list_url}: blocked by robots policy")
                    continue
                response = requests.get(
                    list_url,
                    timeout=settings.reference_fetch_timeout_seconds,
                    headers=self._build_request_headers(source),
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                links: list[tuple[str, str]] = []
                for selector in entry_selectors:
                    for node in soup.select(selector):
                        href = str(node.get("href", "") or "").strip()
                        text = node.get_text(" ", strip=True)
                        if not href or len(text) < 8:
                            continue
                        absolute = urljoin(list_url, href)
                        links.append((absolute, text))
                    if links:
                        break
                if not links:
                    links = self._discover_candidate_links(soup, list_url)
                deduped: list[tuple[str, str]] = []
                seen_links: set[str] = set()
                allowed_prefixes = [str(item) for item in config.get("entry_url_prefixes", []) if str(item).strip()]
                for href, text in links:
                    if allowed_prefixes and not any(href.startswith(prefix) for prefix in allowed_prefixes):
                        continue
                    if href in seen_links:
                        continue
                    seen_links.add(href)
                    deduped.append((href, text))
                    if len(deduped) >= max_entries:
                        break
                if not deduped:
                    failures.append(f"{list_url}: no entry links matched")
                    continue
                for href, fallback_title in deduped:
                    if not self._is_fetch_allowed(href, source):
                        failures.append(f"{href}: blocked by robots policy")
                        continue
                    article_response = requests.get(
                        href,
                        timeout=settings.reference_fetch_timeout_seconds,
                        headers=self._build_request_headers(source),
                    )
                    article_response.raise_for_status()
                    content, title, doc_type = self._extract_remote_content(article_response.text, href, source)
                    content = _normalize_text(content)
                    if not content:
                        failures.append(f"{href}: empty entry content")
                        continue
                    content_hash = _content_hash(content)
                    if (source.id, content_hash) in seen_hashes:
                        continue
                    seen_hashes.add((source.id, content_hash))
                    document_id = sha1(f"{source.id}:{href}".encode("utf-8")).hexdigest()[:16]
                    updated_at = datetime.now(timezone.utc)
                    document = ReferenceDocument(
                        id=document_id,
                        source_id=source.id,
                        title=(title or fallback_title)[:160],
                        source=source.name,
                        content=content,
                        doc_type=doc_type,
                        updated_at=updated_at,
                        url=href,
                        source_type=source.kind,
                        authority_score=source.authority_score,
                        content_hash=content_hash,
                    )
                    documents.append(document)
                    for index, chunk_text in enumerate(_chunk_text(content), start=1):
                        chunks.append(
                            ReferenceChunk(
                                id=f"{document_id}:{index}",
                                document_id=document_id,
                                source_id=source.id,
                                title=document.title,
                                source=source.name,
                                text=chunk_text,
                                chunk_index=index,
                                updated_at=updated_at,
                                url=href,
                                source_type=source.kind,
                                authority_score=source.authority_score,
                                content_hash=_content_hash(chunk_text),
                            )
                        )
            except Exception as exc:
                failures.append(f"{list_url}: {type(exc).__name__}: {exc}")
        return documents, chunks, failures

    def _read_text(self, path: Path) -> str:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".html":
            parser = _HTMLTextExtractor()
            parser.feed(raw)
            text = parser.get_text()
        else:
            text = raw
        return _normalize_text(text)

    def _title_for(self, path: Path, content: str) -> str:
        first_line = next((line.strip().lstrip("# ").strip() for line in content.splitlines() if line.strip()), "")
        return first_line[:120] if first_line else path.stem.replace("_", " ").replace("-", " ").strip()

    def _extract_remote_content(self, html: str, url: str, source: ReferenceSource) -> tuple[str, str, str]:
        if source.kind in {"remote_text", "text"}:
            title = _title_from_url(url)
            return html, title, "txt"
        soup = BeautifulSoup(html, "html.parser")
        metadata_title = self._extract_title_from_metadata(soup)
        json_ld_title = self._extract_title_from_json_ld(soup)
        json_ld_text = self._extract_text_from_json_ld(soup)
        config = self._effective_fetch_config(source)
        remove_selectors = [str(item) for item in config.get("remove_selectors", [])]
        for selector in remove_selectors:
            for node in soup.select(selector):
                node.decompose()
        for node in soup(["script", "style", "noscript", "footer", "nav", "aside"]):
            node.decompose()
        title = ""
        title_selectors = [str(item) for item in config.get("title_selectors", [])]
        for selector in title_selectors:
            node = soup.select_one(selector)
            if node is None:
                continue
            if selector.startswith("meta["):
                meta_content = str(node.get("content", "") or "").strip()
                if meta_content:
                    title = meta_content
                    break
            else:
                title = node.get_text(" ", strip=True)
                if title:
                    break
        if not title:
            title = metadata_title or json_ld_title
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title:
            heading = soup.find(["h1", "h2"])
            title = heading.get_text(" ", strip=True) if heading else _title_from_url(url)
        selectors = [str(item) for item in config.get("content_selectors", [])]
        selectors.extend(["article", "main", ".content", "#content", ".article", ".article-body", ".view-content"])
        text = ""
        for selector in selectors:
            nodes = soup.select(selector)
            parts = [node.get_text(" ", strip=True) for node in nodes]
            candidate = " ".join(part for part in parts if len(part) >= 40).strip()
            if len(candidate) >= 80:
                text = candidate
                break
        if not text:
            text = json_ld_text
        if not text:
            text = soup.get_text(" ", strip=True)
        max_chars = int(config.get("max_content_chars", 0) or 0)
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0].strip()
        return text, title[:160], "html"

    def _is_fetch_allowed(self, url: str, source: ReferenceSource) -> bool:
        config = self._effective_fetch_config(source)
        if not bool(config.get("respect_robots", True)):
            return True
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = self._robots_cache.get(robots_url)
        if parser is None:
            parser = robotparser.RobotFileParser()
            parser.set_url(robots_url)
            try:
                robots_response = requests.get(
                    robots_url,
                    timeout=settings.reference_fetch_timeout_seconds,
                    headers=self._build_request_headers(source),
                )
                robots_response.raise_for_status()
                parser.parse(robots_response.text.splitlines())
            except Exception:
                self._robots_cache[robots_url] = None
                return False
            self._robots_cache[robots_url] = parser
        if parser is None:
            return False
        return parser.can_fetch("GroundWire/1.0", url)

    def _build_request_headers(self, source: ReferenceSource) -> dict[str, str]:
        headers = {
            "User-Agent": "GroundWire/1.0",
            "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
        }
        custom_headers = self._effective_fetch_config(source).get("request_headers", {})
        if isinstance(custom_headers, dict):
            headers.update({str(key): str(value) for key, value in custom_headers.items()})
        return headers

    def _effective_fetch_config(self, source: ReferenceSource) -> dict[str, object]:
        config = dict(source.fetch_config or {})
        hostnames = [urlparse(url).netloc.lower().replace("www.", "") for url in source.seed_urls if str(url).strip()]
        for hostname in hostnames:
            profile = DOMAIN_FETCH_PROFILES.get(hostname)
            if not profile:
                continue
            for key, value in profile.items():
                config.setdefault(key, value)
        return config

    def _discover_candidate_links(self, soup: BeautifulSoup, list_url: str) -> list[tuple[str, str]]:
        parsed_list = urlparse(list_url)
        list_host = parsed_list.netloc.lower()
        candidates: list[tuple[int, str, str]] = []
        for node in soup.select("a[href]"):
            href = str(node.get("href", "") or "").strip()
            text = node.get_text(" ", strip=True)
            if not href or len(text) < 12:
                continue
            absolute = urljoin(list_url, href)
            parsed = urlparse(absolute)
            if not parsed.scheme.startswith("http") or parsed.netloc.lower() != list_host:
                continue
            score = 0
            path = parsed.path.lower()
            if any(token in path for token in ("press", "release", "news", "statement", "speech", "article", "litigation", "pressrelease")):
                score += 4
            if re.search(r"\d{4}/\d{2}|\d{4}-\d{2}-\d{2}", path):
                score += 3
            if len(text) >= 24:
                score += 2
            if len(text) >= 48:
                score += 1
            if score >= 4:
                candidates.append((score, absolute, text))
        ordered = sorted(candidates, key=lambda item: (-item[0], item[1]))
        return [(href, text) for _, href, text in ordered[:20]]

    def _extract_title_from_metadata(self, soup: BeautifulSoup) -> str:
        for selector in (
            "meta[property='og:title']",
            "meta[name='twitter:title']",
            "meta[name='title']",
        ):
            node = soup.select_one(selector)
            if node:
                content = str(node.get("content", "") or "").strip()
                if content:
                    return content
        return ""

    def _extract_title_from_json_ld(self, soup: BeautifulSoup) -> str:
        for script in soup.select("script[type='application/ld+json']"):
            try:
                payload = script.string or script.get_text()
                if not payload:
                    continue
                parsed = __import__("json").loads(payload)
            except Exception:
                continue
            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                if not isinstance(item, dict):
                    continue
                headline = item.get("headline") or item.get("name")
                if isinstance(headline, str) and headline.strip():
                    return headline.strip()
        return ""

    def _extract_text_from_json_ld(self, soup: BeautifulSoup) -> str:
        for script in soup.select("script[type='application/ld+json']"):
            try:
                payload = script.string or script.get_text()
                if not payload:
                    continue
                parsed = __import__("json").loads(payload)
            except Exception:
                continue
            items = parsed if isinstance(parsed, list) else [parsed]
            for item in items:
                if not isinstance(item, dict):
                    continue
                body = item.get("articleBody") or item.get("description")
                if isinstance(body, str) and len(body.strip()) >= 80:
                    return body.strip()
        return ""

    def _prefer_richer_rss_content(self, summary_text: str, fetched_content: str) -> str:
        if not fetched_content:
            return summary_text
        if not summary_text:
            return fetched_content
        if len(fetched_content) < 260 and len(summary_text) > len(fetched_content):
            return summary_text
        if len(fetched_content) < 180 and len(summary_text) >= 140:
            return f"{summary_text}\n\n{fetched_content}".strip()
        return fetched_content


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            piece = paragraph[start:start + chunk_size].strip()
            if piece:
                chunks.append(piece)
            if start + chunk_size >= len(paragraph):
                break
            start += max(chunk_size - overlap, 1)
        current = ""
    if current:
        chunks.append(current)
    return chunks[:200]


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.rstrip("/").split("/")[-1]
    if slug:
        return slug.replace("-", " ").replace("_", " ").strip()[:120]
    return parsed.netloc


def _content_hash(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()


def _entry_value(entry: object, key: str) -> object:
    if isinstance(entry, dict):
        return entry.get(key)
    return getattr(entry, key, None)
