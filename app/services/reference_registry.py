from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from app.config import settings
from app.models import ReferenceSource
from app.repository import IssueRepository


DEFAULT_SOURCE_ID = "local-reference-docs"


@dataclass(slots=True)
class ReferenceRegistryConfig:
    sources: list[ReferenceSource]


class ReferenceSourceRegistry:
    def __init__(self, repository: IssueRepository | None = None, docs_path: str | None = None) -> None:
        self.repository = repository or IssueRepository()
        self.docs_path = Path(docs_path or settings.reference_docs_path)

    def sync(self) -> list[ReferenceSource]:
        sources = self._load_sources()
        self.repository.save_reference_sources(sources)
        return sources

    def source_for_path(self, path: Path, sources: list[ReferenceSource]) -> ReferenceSource:
        relative_parts = path.relative_to(self.docs_path).parts
        if relative_parts:
            root_name = relative_parts[0]
            for source in sources:
                if source.id == root_name or Path(source.location).name == root_name:
                    return source
        return next(source for source in sources if source.id == DEFAULT_SOURCE_ID)

    def _load_sources(self) -> list[ReferenceSource]:
        manifest_path = self.docs_path / "sources.json"
        sources = [self._default_source()]
        if not manifest_path.exists():
            return sources
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("sources", []):
            sources.append(
                ReferenceSource(
                    id=item["id"],
                    name=item.get("name", item["id"]),
                    kind=item.get("kind", "manual"),
                    location=item.get("location", item["id"]),
                    authority_score=float(item.get("authority_score", 0.8)),
                    is_active=bool(item.get("is_active", True)),
                    notes=item.get("notes"),
                    seed_urls=[str(url) for url in item.get("seed_urls", []) if str(url).strip()],
                    refresh_minutes=int(item.get("refresh_minutes", 60)),
                    fetch_config=item.get("fetch_config", {}),
                )
            )
        unique: dict[str, ReferenceSource] = {}
        for source in sources:
            unique[source.id] = source
        return list(unique.values())

    def _default_source(self) -> ReferenceSource:
        return ReferenceSource(
            id=DEFAULT_SOURCE_ID,
            name="Local Reference Docs",
            kind="manual",
            location=".",
            authority_score=0.8,
            is_active=True,
            notes="Fallback source for unmanaged reference documents.",
            seed_urls=[],
            refresh_minutes=60,
            fetch_config={},
        )
