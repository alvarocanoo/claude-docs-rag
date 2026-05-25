"""Domain models for the RAG pipeline."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocSource(BaseModel):
    """An entry from the llms.txt index — URL pointing to a markdown page."""

    url: str
    title: str
    section: str = ""


class RawPage(BaseModel):
    """A downloaded markdown page, before parsing."""

    url: str
    title: str
    section: str
    content: str
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class Chunk(BaseModel):
    """A retrievable unit: piece of a page with positional metadata."""

    source_url: str
    title: str
    section_path: str
    chunk_index: int
    content: str
    content_tokens: int
    embedding: list[float] | None = None

    def composite_id(self) -> str:
        return f"{self.source_url}#chunk={self.chunk_index}"
