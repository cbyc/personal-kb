"""Pydantic data models shared across all modules."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A loaded document with metadata."""

    content: str
    source: str  # file path or URL
    title: str = ""
    source_type: str = "note"  # "note" or "bookmark"
    url: str | None = None  # original URL for bookmarks


class Chunk(BaseModel):
    """A text chunk from a document."""

    text: str
    source: str
    chunk_index: int
    source_type: str = "note"  # "note" or "bookmark"
    url: str | None = None  # original URL for bookmarks
    metadata: dict = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single result from vector search."""

    chunk: Chunk
    score: float


class QueryResult(BaseModel):
    """The final answer from the RAG agent."""

    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
