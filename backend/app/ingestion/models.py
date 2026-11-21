"""Pydantic models for papers, extracted pages, and semantic chunks."""

from typing import Optional

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Stable identity and bibliographic fields for one uploaded paper."""

    paper_id: str = Field(pattern=r"^P\d{3,}$")
    filename: str = Field(min_length=1)
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = Field(default=None, ge=1000, le=9999)
    page_count: int = Field(ge=1)
    source_path: str = Field(min_length=1)


class PageText(BaseModel):
    """Extracted text for a single PDF page."""

    paper_id: str = Field(pattern=r"^P\d{3,}$")
    page: int = Field(ge=1)
    source_filename: str = Field(min_length=1)
    text: str


class ChunkRecord(BaseModel):
    """One semantic chunk with the metadata needed for later citations."""

    chunk_id: str = Field(pattern=r"^P\d{3,}_C\d{3,}$")
    paper_id: str = Field(pattern=r"^P\d{3,}$")
    title: Optional[str] = None
    section: Optional[str] = None
    page: int = Field(ge=1)
    source_filename: str = Field(min_length=1)
    text: str = Field(min_length=1)


class BreakpointEmbeddingStatus(BaseModel):
    """Which embedding model located semantic chunk boundaries.

    Vectors themselves are not stored in Stage 2.
    """

    mode: str
    model: str
    development_fallback: bool
    reference_equivalent: bool
    vectors_persisted: bool = False


class SemanticChunkingConfig(BaseModel):
    """LlamaIndex semantic splitter settings from the reference paper."""

    buffer_size: int = 1
    breakpoint_percentile_threshold: int = 95
    embedding: BreakpointEmbeddingStatus


class ProcessedPaper(PaperMetadata):
    """Normalized document persisted after text extraction."""

    status: str
    chunk_count: int = Field(ge=0)
    breakpoint_embed_model: str
    semantic_chunking: SemanticChunkingConfig
    pages: list[PageText]


class IngestionResult(BaseModel):
    """Statistics returned after a paper finishes ingestion."""

    paper_id: str
    filename: str
    status: str
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    page_count: int
    chunk_count: int
    processing_time_seconds: float
    source_path: str


class PaperSummary(BaseModel):
    """Paper metadata returned by the list and detail endpoints."""

    paper_id: str
    filename: str
    status: str
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    year: Optional[int] = None
    page_count: int
    chunk_count: int
    source_path: str


class PaperListResponse(BaseModel):
    papers: list[PaperSummary]


class ChunkListResponse(BaseModel):
    paper_id: str
    total: int
    offset: int
    limit: int
    chunks: list[ChunkRecord]
